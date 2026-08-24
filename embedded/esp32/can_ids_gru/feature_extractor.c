/**
 * @file feature_extractor.c
 * @brief Sliding-window feature computation (IAT, ID entropy, payload stats)
 */

#include <string.h>
#include <math.h>
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "freertos/queue.h"
#include "esp_log.h"

#include "can_handler.h"
#include "feature_extractor.h"

static const char *TAG = "FEAT_EXT";

/* External queue handles (defined in main.c) */
extern QueueHandle_t g_frame_queue;
extern QueueHandle_t g_feature_queue;

/* -----------------------------------------------------------------------
 * Internal helpers
 * --------------------------------------------------------------------- */

/** Shannon entropy of a normalised histogram */
static float entropy_from_hist(const float *hist, uint32_t bins)
{
    float h = 0.0f;
    for (uint32_t i = 0; i < bins; i++) {
        if (hist[i] > 1e-9f) {
            h -= hist[i] * log2f(hist[i]);
        }
    }
    return h;
}

/** Mean of a float array */
static float mean_f(const float *arr, uint32_t n)
{
    float s = 0.0f;
    for (uint32_t i = 0; i < n; i++) s += arr[i];
    return s / (float)n;
}

/** Standard deviation (population) */
static float std_f(const float *arr, uint32_t n, float mu)
{
    float s = 0.0f;
    for (uint32_t i = 0; i < n; i++) {
        float d = arr[i] - mu;
        s += d * d;
    }
    return sqrtf(s / (float)n);
}

/* -----------------------------------------------------------------------
 * Core computation
 * --------------------------------------------------------------------- */
void feature_extractor_compute(const can_frame_t *frames,
                                uint32_t           count,
                                feature_vector_t  *out)
{
    if (!frames || count < 2 || !out) return;

    memset(out, 0, sizeof(*out));
    out->window_start_us = frames[0].timestamp_us;
    out->total_frames    = count;

    /* ---- Inter-arrival times ---- */
    float iat[WINDOW_SIZE];                    /* static max, never overflows */
    uint32_t n_iat = count - 1;

    float iat_min =  1e30f, iat_max = -1e30f;
    for (uint32_t i = 0; i < n_iat; i++) {
        iat[i] = (float)(frames[i+1].timestamp_us - frames[i].timestamp_us);
        if (iat[i] < iat_min) iat_min = iat[i];
        if (iat[i] > iat_max) iat_max = iat[i];
    }
    out->iat_min_us  = iat_min;
    out->iat_max_us  = iat_max;
    out->iat_mean_us = mean_f(iat, n_iat);
    out->iat_std_us  = std_f (iat, n_iat, out->iat_mean_us);

    /* ---- Frame rate ---- */
    float window_duration_s = (float)(frames[count-1].timestamp_us
                                    - frames[0].timestamp_us) / 1e6f;
    out->frame_rate_fps = (window_duration_s > 0.0f)
                        ? (float)count / window_duration_s
                        : 0.0f;

    /* ---- ID entropy — 2048 buckets for 11-bit IDs ---- */
    #define ID_BINS 2048
    static uint32_t id_count[ID_BINS];   /* static: zero-init at program start */
    memset(id_count, 0, sizeof(id_count));

    uint32_t max_id_count = 0;
    uint32_t dominant_id  = 0;

    for (uint32_t i = 0; i < count; i++) {
        uint32_t bucket = frames[i].id & (ID_BINS - 1);
        id_count[bucket]++;
        if (id_count[bucket] > max_id_count) {
            max_id_count = id_count[bucket];
            dominant_id  = bucket;
        }
    }
    (void)dominant_id;   /* suppress unused warning */

    /* Count unique IDs and build normalised histogram */
    uint32_t unique = 0;
    static float id_hist[ID_BINS];
    for (uint32_t i = 0; i < ID_BINS; i++) {
        if (id_count[i] > 0) unique++;
        id_hist[i] = (float)id_count[i] / (float)count;
    }
    out->unique_ids       = unique;
    out->id_entropy       = entropy_from_hist(id_hist, ID_BINS);
    out->dominant_id_freq = (float)max_id_count / (float)count;

    /* ---- Payload statistics ---- */
    uint32_t byte_hist[256] = {0};
    uint32_t total_bytes    = 0;
    uint32_t zero_bytes     = 0;
    uint32_t rtr_count      = 0;
    float    dlc_arr[WINDOW_SIZE];
    float    all_bytes[WINDOW_SIZE * 8];
    uint32_t byte_idx = 0;

    for (uint32_t i = 0; i < count; i++) {
        dlc_arr[i] = (float)frames[i].dlc;
        if (frames[i].is_rtr) rtr_count++;
        for (uint8_t b = 0; b < frames[i].dlc; b++) {
            uint8_t val = frames[i].data[b];
            byte_hist[val]++;
            all_bytes[byte_idx++] = (float)val;
            if (val == 0) zero_bytes++;
            total_bytes++;
        }
    }

    if (byte_idx > 0) {
        out->payload_mean     = mean_f(all_bytes, byte_idx);
        out->payload_std      = std_f (all_bytes, byte_idx, out->payload_mean);
        out->zero_byte_ratio  = (float)zero_bytes / (float)byte_idx;

        float byte_hist_norm[256];
        for (uint32_t b = 0; b < 256; b++) {
            byte_hist_norm[b] = (float)byte_hist[b] / (float)byte_idx;
        }
        out->payload_entropy = entropy_from_hist(byte_hist_norm, 256);
    }

    out->rtr_ratio = (float)rtr_count / (float)count;
    out->dlc_mean  = mean_f(dlc_arr, count);
    out->dlc_std   = std_f (dlc_arr, count, out->dlc_mean);
}

/* -----------------------------------------------------------------------
 * FreeRTOS task
 * --------------------------------------------------------------------- */
void task_feature_extractor(void *pvParam)
{
    (void)pvParam;

    static can_frame_t   window[WINDOW_SIZE];
    static uint32_t      w_head = 0;    /* write position in ring */
    static uint32_t      w_fill = 0;    /* how many valid frames  */
    static uint32_t      step_ctr = 0;  /* frames since last extraction */

    can_frame_t     incoming;
    feature_vector_t fv;

    ESP_LOGI(TAG, "Feature extractor task started (window=%d, step=%d)",
             WINDOW_SIZE, WINDOW_STEP);

    for (;;) {
        /* Block until a frame is available */
        if (xQueueReceive(g_frame_queue, &incoming, portMAX_DELAY) != pdTRUE) {
            continue;
        }

        /* Write into circular buffer */
        window[w_head % WINDOW_SIZE] = incoming;
        w_head++;
        if (w_fill < WINDOW_SIZE) w_fill++;
        step_ctr++;

        /* Trigger extraction once window is full and step reached */
        if (w_fill < WINDOW_SIZE || step_ctr < WINDOW_STEP) continue;
        step_ctr = 0;

        /* Reconstruct chronological linear buffer from ring */
        static can_frame_t ordered[WINDOW_SIZE];
        for (uint32_t i = 0; i < WINDOW_SIZE; i++) {
            ordered[i] = window[(w_head - WINDOW_SIZE + i) % WINDOW_SIZE];
        }

        feature_extractor_compute(ordered, WINDOW_SIZE, &fv);
        /* Track the CAN ID of the last frame in this window */
        fv.last_frame_id = ordered[WINDOW_SIZE - 1].id;

        if (xQueueSend(g_feature_queue, &fv, pdMS_TO_TICKS(5)) != pdTRUE) {
            ESP_LOGW(TAG, "Feature queue full");
        }
    }
}
