/**
 * @file csv_reporter.h
 * @brief CSV output formatter for attack detection results
 *
 * Prints detection alerts in CSV format for easy ingestion by plotting tools.
 * Format: State,FrameID,ModelName,Confidence,Timestamp_ms
 */

#ifndef CSV_REPORTER_H
#define CSV_REPORTER_H

#include <stdint.h>
#include "can_handler.h"
#include "model_inference.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief Print CSV header line (call once at startup)
 */
void csv_reporter_print_header(void);

/**
 * @brief Print detection alert in CSV format
 * @param attack_state     Attack state name (e.g., "FUZZING")
 * @param frame_id         CAN message ID
 * @param model_name       ML model backend (e.g., "GRU")
 * @param confidence       Detection confidence (0.0–1.0)
 * @param timestamp_ms     Timestamp in milliseconds (for x-axis)
 */
void csv_reporter_print_alert(const char *attack_state,
                               uint32_t frame_id,
                               const char *model_name,
                               float confidence,
                               uint32_t timestamp_ms);

#ifdef __cplusplus
}
#endif

#endif /* CSV_REPORTER_H */
