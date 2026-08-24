/**
 * @file model_inference.c
 * @brief ML inference engine — normalisation + forward-pass stubs
 *
 * HOW TO INTEGRATE YOUR TRAINED MODELS
 * =====================================
 * Option A — TFLite Micro (recommended for CNN-LSTM / GRU):
 *   1. Convert: python tools/export_weights.py --format tflite --model gru
 *   2. Link esp-tflite-micro component in CMakeLists.txt
 *   3. Replace run_mlp / run_gru / run_cnn_lstm bodies with TFLite
 *      interpreter calls (see tflite-micro ESP example in README).
 *
 * Option B — CMSIS-NN / bare arrays (for MLP):
 *   1. Export: python tools/export_weights.py --format c --model mlp
 *   2. #include "model_weights_mlp.h"
 *   3. Implement matrix multiply + ReLU + softmax inline.
 *
 * The stubs below use a simple rule-based classifier so the firmware
 * compiles and functions end-to-end while you integrate real weights.
 */

#include <string.h>
#include <math.h>
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "freertos/queue.h"
#include "esp_log.h"
#include "esp_timer.h"

#include "model_inference.h"
#include "can_handler.h"

/* -----------------------------------------------------------------------
 * TFLite Micro headers (optional if TensorFlow Lite is linked)
 * Uncomment if using real TFLite models:
 * --------------------------------------------------------------------- */
/* #include "tensorflow/lite/micro/all_ops_resolver.h"
 * #include "tensorflow/lite/micro/micro_error_reporter.h"
 * #include "tensorflow/lite/micro/micro_interpreter.h"
 * #include "tensorflow/lite/schema/schema_generated.h"
 * using namespace tflite;
 */

/* -----------------------------------------------------------------------
 * Model binary includes (from model header files)
 * Auto-selected by MODEL_BACKEND_* compilation flag
 * --------------------------------------------------------------------- */
#ifdef MODEL_BACKEND_MLP
  #include "mlp_model.h"
  #define MODEL_DATA mlp_model
#elif defined(MODEL_BACKEND_GRU)
  #include "gru_ids_model (1).h"
  #define MODEL_DATA gru_ids_model
#elif defined(MODEL_BACKEND_CNN_LSTM)
  #include "cnn1d_model.h"
  #define MODEL_DATA cnn1d_model
#endif

static const char *TAG = "INFERENCE";

extern QueueHandle_t g_feature_queue;
extern QueueHandle_t g_alert_queue;

/* -----------------------------------------------------------------------
 * TFLite Micro inference engine state
 * (Currently stub-based; ready for real TFLite integration)
 * --------------------------------------------------------------------- */
typedef struct {
    bool initialized;
    /* TODO: Add when TFLite Micro is linked:
     * tflite::MicroInterpreter *interpreter;
     * TfLiteTensor *input_tensor;
     * TfLiteTensor *output_tensor;
     * uint8_t tensor_arena[MODEL_TENSOR_ARENA_SIZE];
     */
} model_state_t;

static model_state_t g_model_state = {0};

/* -----------------------------------------------------------------------
 * Training-set normalisation statistics
 * Replace with actual mean/std computed from your dataset.
 * (tools/export_weights.py writes these automatically.)
 * --------------------------------------------------------------------- */
static const float FEAT_MEAN[MODEL_INPUT_DIM] = {
    /* iat_mean_us  iat_std_us   iat_min_us   iat_max_us  */
       500.0f,      300.0f,      50.0f,       2000.0f,
    /* id_entropy   unique_ids   dom_id_freq  payload_mean */
       6.5f,        40.0f,       0.15f,       128.0f,
    /* payload_std  pay_entropy  zero_ratio   frame_rate  */
       60.0f,       6.0f,        0.1f,        1000.0f,
    /* total_frames rtr_ratio    dlc_mean     dlc_std     */
       32.0f,       0.01f,       6.5f,        1.5f
};

static const float FEAT_STD[MODEL_INPUT_DIM] = {
       400.0f, 250.0f, 40.0f,  1500.0f,
       1.5f,   20.0f,  0.1f,   50.0f,
       30.0f,  1.0f,   0.05f,  500.0f,
       8.0f,   0.005f, 1.0f,   0.8f
};

/* -----------------------------------------------------------------------
 * Feature-vector → float array
 * Must stay in sync with FEAT_MEAN/FEAT_STD ordering above.
 * --------------------------------------------------------------------- */
static void fv_to_array(const feature_vector_t *fv, float *arr)
{
    arr[0]  = fv->iat_mean_us;
    arr[1]  = fv->iat_std_us;
    arr[2]  = fv->iat_min_us;
    arr[3]  = fv->iat_max_us;
    arr[4]  = fv->id_entropy;
    arr[5]  = (float)fv->unique_ids;
    arr[6]  = fv->dominant_id_freq;
    arr[7]  = fv->payload_mean;
    arr[8]  = fv->payload_std;
    arr[9]  = fv->payload_entropy;
    arr[10] = fv->zero_byte_ratio;
    arr[11] = fv->frame_rate_fps;
    arr[12] = (float)fv->total_frames;
    arr[13] = fv->rtr_ratio;
    arr[14] = fv->dlc_mean;
    arr[15] = fv->dlc_std;
}

/* -----------------------------------------------------------------------
 * Z-score normalisation
 * --------------------------------------------------------------------- */
static void normalise(float *arr, uint32_t n)
{
    for (uint32_t i = 0; i < n; i++) {
        arr[i] = (arr[i] - FEAT_MEAN[i]) /
                 (FEAT_STD[i] > 1e-9f ? FEAT_STD[i] : 1.0f);
    }
}

/* -----------------------------------------------------------------------
 * Softmax helper
 * --------------------------------------------------------------------- */
static void softmax(float *arr, uint32_t n)
{
    float max_val = arr[0];
    for (uint32_t i = 1; i < n; i++)
        if (arr[i] > max_val) max_val = arr[i];

    float sum = 0.0f;
    for (uint32_t i = 0; i < n; i++) {
        arr[i] = expf(arr[i] - max_val);
        sum   += arr[i];
    }
    for (uint32_t i = 0; i < n; i++)
        arr[i] /= sum;
}

/* -----------------------------------------------------------------------
 * =====================================================================
 *  STUB CLASSIFIERS — replace with real weight inference
 * =====================================================================
 *
 * These rules approximate detection heuristics from the literature:
 *   DoS     : very high frame rate  (> 5000 fps)
 *   Fuzzing : high ID entropy + high payload entropy
 *   Spoofing: dominant ID ratio very high + low IAT variance
 *   Normal  : everything else
 *
 * Confidence is set proportional to how strongly conditions are met.
 * --------------------------------------------------------------------- */
#ifdef MODEL_BACKEND_MLP
static void run_mlp(const float *feat, float *logits)
{
    /* TODO: replace with CMSIS-NN dense layers using exported weights */
    float rate       = feat[11];   /* frame_rate_fps (normalised) */
    float id_ent     = feat[4];
    float pay_ent    = feat[9];
    float dom_freq   = feat[6];
    float iat_std    = feat[1];

    logits[ATTACK_NORMAL]   =  0.5f;
    logits[ATTACK_DOS]      = (rate     > 2.0f)  ? rate * 1.5f         : -1.0f;
    logits[ATTACK_FUZZING]  = (id_ent   > 1.0f && pay_ent > 1.0f)
                              ? (id_ent + pay_ent) : -1.0f;
    logits[ATTACK_SPOOFING] = (dom_freq > 1.5f && iat_std < -0.5f)
                              ? dom_freq * 1.2f    : -1.0f;
}
#endif

#ifdef MODEL_BACKEND_GRU
static void run_gru(const float *feat, float *logits)
{
    /* TODO: unroll GRU cells with exported weights (W_z,W_r,W_h) */
    float rate     = feat[11];
    float id_ent   = feat[4];
    float pay_ent  = feat[9];
    float dom_freq = feat[6];
    float iat_std  = feat[1];

    logits[ATTACK_NORMAL]   =  0.5f;
    logits[ATTACK_DOS]      = (rate     > 1.8f) ? rate * 1.6f           : -1.0f;
    logits[ATTACK_FUZZING]  = (id_ent   > 0.8f) ? (id_ent + pay_ent)    : -1.0f;
    logits[ATTACK_SPOOFING] = (dom_freq > 1.3f && iat_std < -0.4f)
                              ? dom_freq * 1.3f  : -1.0f;
}
#endif

#ifdef MODEL_BACKEND_CNN_LSTM
static void run_cnn_lstm(const float *feat, float *logits)
{
    /* TODO: invoke TFLite Micro interpreter:
     *   float *input = interpreter->input(0)->data.f;
     *   memcpy(input, feat, MODEL_INPUT_DIM * sizeof(float));
     *   interpreter->Invoke();
     *   memcpy(logits, interpreter->output(0)->data.f,
     *          MODEL_OUTPUT_DIM * sizeof(float));
     */
    float rate     = feat[11];
    float id_ent   = feat[4];
    float pay_ent  = feat[9];
    float dom_freq = feat[6];
    float iat_std  = feat[1];
    float zero_r   = feat[10];

    logits[ATTACK_NORMAL]   =  0.5f;
    logits[ATTACK_DOS]      = (rate     > 1.5f) ? rate * 1.8f           : -1.0f;
    logits[ATTACK_FUZZING]  = (pay_ent  > 0.5f && zero_r < 0.0f)
                              ? (pay_ent + id_ent) * 0.9f : -1.0f;
    logits[ATTACK_SPOOFING] = (dom_freq > 1.0f && iat_std < -0.3f)
                              ? dom_freq * 1.4f  : -1.0f;
}
#endif

/* -----------------------------------------------------------------------
 * Public API
 * --------------------------------------------------------------------- */
esp_err_t model_inference_init(void)
{
    if (g_model_state.initialized) {
        ESP_LOGW(TAG, "Model already initialized");
        return ESP_OK;
    }
    
    ESP_LOGI(TAG, "Initializing model backend: %s", model_get_name());
    ESP_LOGI(TAG, "Model size: %zu bytes (in PROGMEM)", sizeof(MODEL_DATA));
    
    /* ===================================================================
     * OPTION 1: TFLite Micro (when esp-tflite-micro is linked)
     * ===================================================================
     * Uncomment and implement when TensorFlow Lite is available:
     *
     * static tflite::MicroErrorReporter micro_error_reporter;
     * static tflite::AllOpsResolver resolver;
     * static uint8_t tensor_arena[MODEL_TENSOR_ARENA_SIZE];
     * static tflite::MicroInterpreter interpreter(
     *     tflite::GetModel(MODEL_DATA),
     *     resolver, tensor_arena, MODEL_TENSOR_ARENA_SIZE,
     *     &micro_error_reporter);
     *
     * if (interpreter.AllocateTensors() != kTfLiteOk) {
     *     ESP_LOGE(TAG, "AllocateTensors() failed");
     *     return ESP_ERR_NO_MEM;
     * }
     *
     * g_model_state.interpreter = &interpreter;
     * g_model_state.input_tensor = interpreter.input(0);
     * g_model_state.output_tensor = interpreter.output(0);
     * ===================================================================
     *
     * For now, we use heuristic stubs (Rule-based in run_mlp, run_gru, etc.)
     */
    
    g_model_state.initialized = true;
    ESP_LOGI(TAG, "Model initialized ✓ (inference stubs active)");
    return ESP_OK;
}

const char *model_get_name(void)
{
#if   defined(MODEL_BACKEND_MLP)
    return "MLP";
#elif defined(MODEL_BACKEND_GRU)
    return "GRU";
#else
    return "CNN-LSTM";
#endif
}

esp_err_t model_inference_run(const feature_vector_t *fv,
                               inference_result_t     *res)
{
    if (!fv || !res) return ESP_ERR_INVALID_ARG;

    /* 1. Flatten */
    float feat[MODEL_INPUT_DIM];
    fv_to_array(fv, feat);

    /* 2. Normalise */
    normalise(feat, MODEL_INPUT_DIM);

    /* 3. Forward pass → logits */
    float logits[MODEL_OUTPUT_DIM];
#if   defined(MODEL_BACKEND_MLP)
    run_mlp(feat, logits);
#elif defined(MODEL_BACKEND_GRU)
    run_gru(feat, logits);
#else
    run_cnn_lstm(feat, logits);
#endif

    /* 4. Softmax */
    softmax(logits, MODEL_OUTPUT_DIM);

    /* 5. Argmax */
    uint8_t best = 0;
    for (uint8_t i = 1; i < MODEL_OUTPUT_DIM; i++) {
        if (logits[i] > logits[best]) best = i;
    }

    res->label        = best;
    res->confidence   = logits[best];
    res->timestamp_us = esp_timer_get_time();
    res->last_frame_id = fv->last_frame_id;
    memcpy(res->probabilities, logits, sizeof(float) * MODEL_OUTPUT_DIM);

    /* Low-confidence → treat as NORMAL */
    if (res->confidence < INFERENCE_CONFIDENCE_THRESHOLD) {
        res->label      = ATTACK_NORMAL;
        res->confidence = logits[ATTACK_NORMAL];
    }

    return ESP_OK;
}

/* -----------------------------------------------------------------------
 * FreeRTOS task
 * --------------------------------------------------------------------- */
void task_inference(void *pvParam)
{
    (void)pvParam;
    feature_vector_t  fv;
    inference_result_t res;

    ESP_LOGI(TAG, "Inference task started (model: %s)", model_get_name());

    for (;;) {
        if (xQueueReceive(g_feature_queue, &fv, portMAX_DELAY) != pdTRUE) {
            continue;
        }

        if (model_inference_run(&fv, &res) != ESP_OK) {
            ESP_LOGE(TAG, "Inference failed");
            continue;
        }

        ESP_LOGI(TAG, "Result: %-8s  conf=%.2f  [N=%.2f D=%.2f F=%.2f S=%.2f]",
                 ATTACK_LABELS[res.label], res.confidence,
                 res.probabilities[0], res.probabilities[1],
                 res.probabilities[2], res.probabilities[3]);

        if (xQueueSend(g_alert_queue, &res, pdMS_TO_TICKS(5)) != pdTRUE) {
            ESP_LOGW(TAG, "Alert queue full");
        }
    }
}
