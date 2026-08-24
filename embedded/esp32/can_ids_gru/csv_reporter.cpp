/**
 * @file csv_reporter.cpp
 * @brief CSV output implementation
 *
 * Outputs to Serial (115200 baud) in CSV format for plotting.
 */

#include <stdio.h>
#include "csv_reporter.h"
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "esp_log.h"

static const char *TAG = "CSV";

/* CSV output synchronization: use log level to ensure UART write completes */
static bool csv_enabled = true;

void csv_reporter_print_header(void)
{
    if (!csv_enabled) return;
    
    /* Print header line (no timestamp tracking needed) */
    printf("State,FrameID,ModelName,Confidence,Timestamp_ms\n");
    fflush(stdout);
    
    ESP_LOGI(TAG, "CSV output started");
}

void csv_reporter_print_alert(const char *attack_state,
                               uint32_t frame_id,
                               const char *model_name,
                               float confidence,
                               uint32_t timestamp_ms)
{
    if (!csv_enabled || !attack_state || !model_name) return;
    
    /* Confidence as percentage (0–100) */
    uint8_t conf_percent = (uint8_t)(confidence * 100.0f);
    
    /* Print in CSV format:
     * State,FrameID,ModelName,Confidence,Timestamp_ms
     * Example:
     *   FUZZING,0x123,GRU,92,1050
     */
    printf("%s,0x%03X,%s,%u,%u\n",
           attack_state,
           frame_id & 0x7FF,
           model_name,
           conf_percent,
           timestamp_ms);
    
    fflush(stdout);
    
    /* Log to ESP log system as well (helps with debugging) */
    ESP_LOGI(TAG, "Alert> %s | ID=0x%03X | %s=%.0f%% | t=%u ms",
             attack_state, frame_id & 0x7FF, model_name, confidence * 100.0f, timestamp_ms);
}
