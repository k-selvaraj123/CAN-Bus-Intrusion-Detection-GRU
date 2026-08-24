/**
 * @file attack_alert.c
 * @brief Alert handler with per-class cooldown and CAN broadcast
 */

#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "freertos/queue.h"
#include "esp_log.h"
#include "esp_timer.h"

#include "attack_alert.h"
#include "can_handler.h"
#include "attack_generator.h"
#include "csv_reporter.h"
#include "model_inference.h"

static const char *TAG = "ALERT";

extern QueueHandle_t g_alert_queue;

/* Minimum ms between alerts of the same class to prevent bus flooding */
#define ALERT_COOLDOWN_MS   2000

void task_alert_handler(void *pvParam)
{
    (void)pvParam;
    inference_result_t res;

    /* Per-class last-alert timestamp */
    int64_t last_alert_us[ATTACK_COUNT] = {0};

    ESP_LOGI(TAG, "Alert handler started (cooldown=%d ms)", ALERT_COOLDOWN_MS);

    for (;;) {
        if (xQueueReceive(g_alert_queue, &res, portMAX_DELAY) != pdTRUE) {
            continue;
        }

        /* Skip NORMAL detections silently */
        if (res.label == ATTACK_NORMAL) {
            continue;
        }

        /* Cooldown check */
        int64_t now_us    = esp_timer_get_time();
        int64_t elapsed   = now_us - last_alert_us[res.label];
        int64_t cooldown  = (int64_t)ALERT_COOLDOWN_MS * 1000;

        if (elapsed < cooldown) {
            continue;
        }

        last_alert_us[res.label] = now_us;

        /* Log at appropriate severity */
        switch (res.label) {
        case ATTACK_DOS:
            ESP_LOGE(TAG, "⚠ DoS ATTACK DETECTED  conf=%.2f", res.confidence);
            break;
        case ATTACK_FUZZING:
            ESP_LOGE(TAG, "⚠ FUZZING ATTACK DETECTED  conf=%.2f", res.confidence);
            break;
        case ATTACK_SPOOFING:
            ESP_LOGE(TAG, "⚠ SPOOFING/REPLAY ATTACK DETECTED  conf=%.2f",
                     res.confidence);
            break;
        default:
            ESP_LOGW(TAG, "Unknown attack class %d", res.label);
        }

        /* Broadcast alert frame on the CAN bus */
        can_handler_send_alert(&res);

        /* Print CSV alert for plotting */
        attack_generator_t *gen_ctx = get_attack_generator_context();
        if (gen_ctx) {
            const char *state_name = attack_state_name(
                attack_generator_get_current_state(gen_ctx)
            );
            uint32_t timestamp_ms = (uint32_t)(res.timestamp_us / 1000);
            
            /* CSV format: State,FrameID,ModelName,Confidence,Timestamp_ms */
            csv_reporter_print_alert(
                state_name,
                res.last_frame_id,  /* Actual CAN frame ID from last window frame */
                model_get_name(),
                res.confidence,
                timestamp_ms
            );
        }
    }
}
