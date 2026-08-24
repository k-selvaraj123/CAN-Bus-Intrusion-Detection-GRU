/**
 * @file feature_extractor.h
 * @brief Sliding-window feature extraction from raw CAN frames
 *
 * Computes the 16-element feature vector described in can_handler.h
 * over windows of WINDOW_SIZE frames with WINDOW_STEP stride.
 *
 * Features align with those used in:
 *   Rai et al. "Securing the CAN bus using deep learning for IDS"
 *   Scientific Reports, 2025 — time-domain + ID-entropy features.
 */
#ifdef __cplusplus
extern "C" {
#endif

#ifndef FEATURE_EXTRACTOR_H
#define FEATURE_EXTRACTOR_H

#include "can_handler.h"

/**
 * @brief  FreeRTOS task: reads from g_frame_queue, fills a sliding
 *         window, then posts computed feature_vector_t to g_feature_queue.
 */
void task_feature_extractor(void *pvParam);

/**
 * @brief  Compute features for an arbitrary buffer of frames.
 *         Useful for unit-testing outside of FreeRTOS.
 *
 * @param  frames  Array of can_frame_t, length >= count.
 * @param  count   Number of frames in the buffer.
 * @param  out     Populated feature vector (caller allocates).
 */
void feature_extractor_compute(const can_frame_t *frames,
                                uint32_t           count,
                                feature_vector_t  *out);

#endif /* FEATURE_EXTRACTOR_H */

#ifdef __cplusplus
}
#endif
