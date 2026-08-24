/**
 * @file model_inference.h
 * @brief ML model inference engine for CAN-IDS
 *
 * Supports three model back-ends selectable at compile time:
 *   MODEL_BACKEND_MLP        — lightweight, ~1 ms inference
 *   MODEL_BACKEND_GRU        — moderate, ~5 ms
 *   MODEL_BACKEND_CNN_LSTM   — heaviest, ~15 ms  (default)
 *
 * Set via CMakeLists target_compile_definitions, e.g.:
 *   target_compile_definitions(${COMPONENT_LIB} PUBLIC MODEL_BACKEND_MLP)
 *
 * Weight files are converted from your trained .h5 / .tflite models
 * by the companion Python script tools/export_weights.py, which
 * writes model_weights_<backend>.h with float32 arrays.
 *
 * TFLite Micro integration note:
 *   If using TensorFlow Lite for Microcontrollers, replace the
 *   stub implementations in model_inference.c with TFLite Micro
 *   interpreter calls and link against the tflite-micro ESP-IDF
 *   component (https://github.com/espressif/esp-tflite-micro).
 */

 #ifdef __cplusplus
extern "C" {
#endif

#ifndef MODEL_INFERENCE_H
#define MODEL_INFERENCE_H

#include <stdint.h>
#include "esp_err.h"
#include "can_handler.h"      /* inference_result_t, feature_vector_t */

/* -----------------------------------------------------------------------
 * Backend selection (default: GRU)
 * Edit CONFIG_MODEL_BACKEND below to switch between MLP/GRU/CNN_LSTM
 * --------------------------------------------------------------------- */
#define CONFIG_MODEL_BACKEND_GRU   // ← Change this to MLP or CNN_LSTM or GRU

#if !defined(MODEL_BACKEND_MLP) && \
    !defined(MODEL_BACKEND_GRU) && \
    !defined(MODEL_BACKEND_CNN_LSTM)
  #ifdef CONFIG_MODEL_BACKEND_MLP
    #define MODEL_BACKEND_MLP
  #elif defined(CONFIG_MODEL_BACKEND_GRU)
    #define MODEL_BACKEND_GRU
  #else
    #define MODEL_BACKEND_CNN_LSTM
  #endif
#endif

/* -----------------------------------------------------------------------
 * Feature vector dimensionality
 * Must match the input layer of the exported model.
 * --------------------------------------------------------------------- */
#define MODEL_INPUT_DIM   16   /* matches fields in feature_vector_t   */
#define MODEL_OUTPUT_DIM   4   /* NORMAL / DoS / FUZZING / SPOOFING    */

/* Confidence threshold below which NORMAL is assumed */
#define INFERENCE_CONFIDENCE_THRESHOLD   0.60f

/* -----------------------------------------------------------------------
 * Public API
 * --------------------------------------------------------------------- */

/**
 * @brief  Load weights and initialise the inference engine.
 *         Must be called once before model_inference_run().
 */
esp_err_t model_inference_init(void);

/**
 * @brief  Run inference on a single feature vector.
 *
 * @param  fv    Input features (normalised by the function internally).
 * @param  res   Output: populated with label, confidence, probabilities.
 * @return ESP_OK on success.
 */
esp_err_t model_inference_run(const feature_vector_t *fv,
                               inference_result_t     *res);

/**
 * @brief  Return a human-readable string identifying the loaded model.
 */
const char *model_get_name(void);

/**
 * @brief  FreeRTOS task: reads from g_feature_queue, runs inference,
 *         posts inference_result_t to g_alert_queue.
 */
void task_inference(void *pvParam);

#endif /* MODEL_INFERENCE_H */

#ifdef __cplusplus
}
#endif
