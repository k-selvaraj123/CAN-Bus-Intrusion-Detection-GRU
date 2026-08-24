/**
 * @file can_handler.c
 * @brief TWAI driver wrapper — init, TX, RX task, alert broadcast
 */

#include <string.h>
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "freertos/queue.h"
#include "esp_log.h"
#include "esp_timer.h"
#include "driver/twai.h"

#include "can_handler.h"

static const char *TAG = "CAN_HANDLER";

/* -----------------------------------------------------------------------
 * Peripheral initialisation
 * --------------------------------------------------------------------- */
esp_err_t can_handler_init(void)
{
    /* For testing/loopback: skip TWAI hardware initialization
       Frames will be software-looped directly to frame queue by attack_generator */
    
    printf("[TWAI_INIT] TWAI hardware SKIPPED (using software loopback mode)\n");
    printf("[TWAI_INIT] Attack frames will be injected directly to g_frame_queue\n");
    fflush(stdout);
    
    ESP_LOGI(TAG, "Software loopback mode enabled (TWAI hardware bypassed)");
    return ESP_OK;
}

void can_handler_deinit(void)
{
    ESP_LOGI(TAG, "Software loopback mode stopped");
}

/* -----------------------------------------------------------------------
 * Software loopback: inject frame directly to g_frame_queue
 * (simulates what RX task would do with real TWAI)
 * --------------------------------------------------------------------- */
extern QueueHandle_t g_frame_queue;

esp_err_t can_handler_loopback_frame(const twai_message_t *msg)
{
    if (!msg) return ESP_ERR_INVALID_ARG;
    
    can_frame_t frame = {
        .id          = msg->identifier,
        .dlc         = msg->data_length_code,
        .is_extended = (msg->extd  != 0),
        .is_rtr      = (msg->rtr   != 0),
        .timestamp_us = esp_timer_get_time(),
    };
    memcpy(frame.data, msg->data, frame.dlc);

    /* Post to frame queue (non-blocking, drop if full) */
    if (xQueueSend(g_frame_queue, &frame, 0) != pdTRUE) {
        ESP_LOGW(TAG, "Frame queue full — dropped ID=0x%03X", frame.id);
        return ESP_ERR_NO_MEM;
    }
    
    return ESP_OK;
}

/* -----------------------------------------------------------------------
 * Transmit (via software loopback)
 * --------------------------------------------------------------------- */
esp_err_t can_handler_transmit(const can_frame_t *frame)
{
    twai_message_t msg = {
        .identifier       = frame->id,
        .data_length_code = frame->dlc,
        .extd             = frame->is_extended ? 1 : 0,
        .rtr              = frame->is_rtr      ? 1 : 0,
    };
    memcpy(msg.data, frame->data, frame->dlc);

    /* Use software loopback (inject directly to frame queue) */
    return can_handler_loopback_frame(&msg);
}

/* -----------------------------------------------------------------------
 * Alert broadcast — encodes label + confidence into 3 bytes
 * Frame layout:
 *   Byte 0 : attack label (0–3)
 *   Byte 1 : confidence * 100  (0–100)
 *   Byte 2 : sequence counter (wraps at 255)
 *   Bytes 3-6: Unix-style 32-bit ms timestamp (big-endian)
 * --------------------------------------------------------------------- */
void can_handler_send_alert(const inference_result_t *result)
{
    static uint8_t seq = 0;
    int64_t ts_ms = result->timestamp_us / 1000;

    can_frame_t alert = {
        .id          = CAN_ID_ALERT,
        .dlc         = 7,
        .is_extended = false,
        .is_rtr      = false,
        .data = {
            result->label,
            (uint8_t)(result->confidence * 100.0f),
            seq++,
            (uint8_t)((ts_ms >> 24) & 0xFF),
            (uint8_t)((ts_ms >> 16) & 0xFF),
            (uint8_t)((ts_ms >>  8) & 0xFF),
            (uint8_t)( ts_ms        & 0xFF),
        }
    };
    can_handler_transmit(&alert);
}

/* -----------------------------------------------------------------------
 * RX task — software loopback version (no hardware TWAI)
 * --------------------------------------------------------------------- */
void task_can_rx(void *pvParam)
{
    (void)pvParam;
    
    ESP_LOGI(TAG, "RX task started (software loopback mode)");

    for (;;) {
        vTaskDelay(pdMS_TO_TICKS(1000));
    }
}

/* -----------------------------------------------------------------------
 * TX heartbeat task — sends CAN_ID_HEARTBEAT every 1 s
 * --------------------------------------------------------------------- */
void task_can_tx_heartbeat(void *pvParam)
{
    (void)pvParam;
    static uint32_t hb_count = 0;

    ESP_LOGI(TAG, "Heartbeat TX task started");

    for (;;) {
        can_frame_t hb = {
            .id          = CAN_ID_HEARTBEAT,
            .dlc         = 4,
            .is_extended = false,
            .is_rtr      = false,
            .data = {
                (hb_count >> 24) & 0xFF,
                (hb_count >> 16) & 0xFF,
                (hb_count >>  8) & 0xFF,
                 hb_count        & 0xFF,
            }
        };
        can_handler_transmit(&hb);
        hb_count++;
        vTaskDelay(pdMS_TO_TICKS(1000));
    }
}
