/**
 * @file can_cybersec.ino
 * @brief CAN Bus Intrusion Detection System — ESP32-WROOM (Arduino Sketch)
 *
 * Hardware: ESP32-WROOM + SN65HVD230 CAN transceiver
 * Framework: ESP-IDF (TWAI peripheral)
 *
 * Pipeline:
 *   1. TWAI init → RX task collects raw frames into a sliding window
 *   2. Feature extractor computes per-window statistics
 *   3. ML inference classifies: NORMAL / DoS / FUZZING / SPOOFING
 *   4. Alert task logs/handles detected attacks
 */


/* -----------------------------------------------------------------------
 * MODEL BACKEND SELECTION
 * -----------------------------------------------------------------------
 * To change the ML model backend, edit model_inference.h line 37:
 *   CONFIG_MODEL_BACKEND_MLP
 *   CONFIG_MODEL_BACKEND_GRU        ← Currently active
 *   CONFIG_MODEL_BACKEND_CNN_LSTM
 * Then recompile and upload.
 *
 * To change the attack simulation mode, edit attack_generator.h line 25:
 *   #define CONFIG_ATTACK_IDLE
 *   #define CONFIG_ATTACK_DOS        ← Currently active
 *   #define CONFIG_ATTACK_FUZZING
 *   #define CONFIG_ATTACK_SPOOFING
 * --------------------------------------------------------------------- */

#include <stdio.h>
#include <string.h>
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "freertos/queue.h"
#include "freertos/semphr.h"
#include "esp_log.h"
#include "esp_err.h"
#include "nvs_flash.h"

#include "can_handler.h"
#include "feature_extractor.h"
#include "model_inference.h"
#include "attack_alert.h"
#include "attack_generator.h"
#include "csv_reporter.h"

static const char *TAG = "MAIN";

/* -----------------------------------------------------------------------
 * FreeRTOS inter-task communication handles
 * --------------------------------------------------------------------- */
QueueHandle_t g_frame_queue    = NULL;   /* raw CAN frames → feature extractor  */
QueueHandle_t g_feature_queue  = NULL;   /* feature vectors  → inference engine */
QueueHandle_t g_alert_queue    = NULL;   /* inference result → alert handler    */

SemaphoreHandle_t g_stats_mutex = NULL;  /* guards shared bus statistics        */

/* -----------------------------------------------------------------------
 * Attack generator state (Core 0)
 * Initialized via attack_generator_init() in setup()
 * --------------------------------------------------------------------- */
static attack_generator_t g_attack_ctx;

/* -----------------------------------------------------------------------
 * Task declarations (implementations in dedicated .c files)
 * --------------------------------------------------------------------- */
extern void task_can_rx(void *pvParam);
extern void task_can_tx_heartbeat(void *pvParam);
extern void task_feature_extractor(void *pvParam);
extern void task_inference(void *pvParam);
extern void task_alert_handler(void *pvParam);

/**
 * @brief Attack generator task (Core 0)
 * Runs state machine that cycles: IDLE → DOS → FUZZING → SPOOFING → REPLAY
 */
void task_attack_generator(void *pvParam);


/* -----------------------------------------------------------------------
 * setup() — Arduino initialization (runs once)
 * --------------------------------------------------------------------- */
void setup()
{
    Serial.begin(115200);
    delay(500);
    
    Serial.println("\n===== CAN-IDS Firmware v1.0 =====");
    ESP_LOGI(TAG, "Initializing...");

    /* NVS — needed by some ESP-IDF drivers */
    esp_err_t ret = nvs_flash_init();
    if (ret == ESP_ERR_NVS_NO_FREE_PAGES ||
        ret == ESP_ERR_NVS_NEW_VERSION_FOUND) {
        ESP_ERROR_CHECK(nvs_flash_erase());
        ret = nvs_flash_init();
    }
    ESP_ERROR_CHECK(ret);

    /* ---- Create inter-task queues ---- */
    g_frame_queue   = xQueueCreate(CAN_FRAME_QUEUE_LEN,   sizeof(can_frame_t));
    g_feature_queue = xQueueCreate(FEATURE_QUEUE_LEN,     sizeof(feature_vector_t));
    g_alert_queue   = xQueueCreate(ALERT_QUEUE_LEN,       sizeof(inference_result_t));
    g_stats_mutex   = xSemaphoreCreateMutex();

    configASSERT(g_frame_queue);
    configASSERT(g_feature_queue);
    configASSERT(g_alert_queue);
    configASSERT(g_stats_mutex);

    /* ---- Initialise TWAI peripheral (software loopback) ---- */
    ESP_ERROR_CHECK(can_handler_init());

    /* ---- Load / initialise ML model ---- */
    ESP_ERROR_CHECK(model_inference_init());
    ESP_LOGI(TAG, "Model: %s", model_get_name());

    /* ---- Initialise attack generator ---- */
    attack_generator_init(&g_attack_ctx);

    /* ---- Print CSV header ---- */
    csv_reporter_print_header();

    /* ---- Spawn FreeRTOS tasks ---- */
    /*                                   name            stack   param  prio  handle */
    xTaskCreate(task_can_rx,           "can_rx",         4096,   NULL,  5,    NULL);
    xTaskCreate(task_can_tx_heartbeat, "can_tx_hb",      2048,   NULL,  3,    NULL);
    xTaskCreate(task_feature_extractor,"feat_ext",       8192,   NULL,  4,    NULL);
    xTaskCreate(task_inference,        "inference",       8192,   NULL,  4,    NULL);
    xTaskCreate(task_alert_handler,    "alert",           3072,   NULL,  3,    NULL);

    /* ---- Spawn attack generator on Core 0 ---- */
    xTaskCreatePinnedToCore(task_attack_generator, "attack_gen", 2048, NULL, 1, NULL, 0);

    Serial.println("===== SETUP COMPLETE =====\n");
    
    /* Enable debug logging */
    esp_log_level_set("*", ESP_LOG_INFO);
}

/* -----------------------------------------------------------------------
 * loop() — Arduino main loop (runs repeatedly after setup)
 * Minimal implementation since FreeRTOS tasks handle all work
 * --------------------------------------------------------------------- */
void loop()
{
    /* FreeRTOS scheduler manages task execution; minimal loop needed */
    vTaskDelay(pdMS_TO_TICKS(100));
}

/* -----------------------------------------------------------------------
 * task_attack_generator — Core 0 attack state machine
 * Runs independently on Core 0; generates attack frames every 10ms
 * Uses software loopback: frames injected directly to frame queue
 * --------------------------------------------------------------------- */
extern esp_err_t can_handler_loopback_frame(const twai_message_t *msg);

void task_attack_generator(void *pvParam)
{
    (void)pvParam;
    twai_message_t msg = {0};
    
    Serial.println("[SETUP] Attack simulator task started");
    
    for (;;) {
        /* Generate next frame in selected attack mode */
        int state = attack_generator_step(&g_attack_ctx, &msg);
        
        if (state >= 0) {
            /* Frame ready — inject via software loopback */
            esp_err_t ret = can_handler_loopback_frame(&msg);
            if (ret != ESP_OK) {
                ESP_LOGW(TAG, "Loopback injection failed: 0x%x", ret);
            }
        }
        
        /* Non-blocking: 10ms cycle */
        vTaskDelay(pdMS_TO_TICKS(10));
    }
}

/* -----------------------------------------------------------------------
 * Getter: expose attack generator context to other tasks
 * --------------------------------------------------------------------- */
attack_generator_t *get_attack_generator_context(void)
{
    return &g_attack_ctx;
}
