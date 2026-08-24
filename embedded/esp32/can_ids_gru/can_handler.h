/**
 * @file can_handler.h
 * @brief TWAI (CAN) peripheral driver abstraction for ESP32-WROOM
 *
 * Wraps ESP-IDF's twai_* API.  All timing, GPIO, and queue
 * parameters are centralised here so nothing is scattered in
 * application code.
 */
#ifdef __cplusplus
extern "C" {
#endif

#ifndef CAN_HANDLER_H
#define CAN_HANDLER_H

#include <stdint.h>
#include <stdbool.h>
#include "esp_err.h"
#include "driver/twai.h"

/* -----------------------------------------------------------------------
 * Hardware pin assignment  (change to match your schematic)
 * --------------------------------------------------------------------- */
#define CAN_TX_GPIO          21    /* ESP32-WROOM → SN65HVD230 TXD */
#define CAN_RX_GPIO          22    /* SN65HVD230 RXD → ESP32-WROOM */

/* -----------------------------------------------------------------------
 * Bus / queue sizing
 * --------------------------------------------------------------------- */
#define CAN_BAUD_RATE        TWAI_TIMING_CONFIG_500KBITS()
#define CAN_FRAME_QUEUE_LEN  64    /* raw frame queue depth            */
#define FEATURE_QUEUE_LEN    16    /* feature-vector queue depth       */
#define ALERT_QUEUE_LEN      8     /* inference-result queue depth     */

/* -----------------------------------------------------------------------
 * Sliding window for feature extraction
 * --------------------------------------------------------------------- */
#define WINDOW_SIZE          32    /* frames per analysis window       */
#define WINDOW_STEP          16    /* stride (50 % overlap)            */

/* -----------------------------------------------------------------------
 * Well-known CAN IDs used by this node
 * --------------------------------------------------------------------- */
#define CAN_ID_HEARTBEAT     0x100  /* periodic keep-alive             */
#define CAN_ID_ALERT         0x7FF  /* attack-alert broadcast          */
#define CAN_ID_ACK           0x101  /* secondary-node acknowledgement  */

/* -----------------------------------------------------------------------
 * Data structures
 * --------------------------------------------------------------------- */

/** Single raw CAN frame with capture timestamp (µs since boot) */
typedef struct {
    uint32_t  id;               /* 11-bit or 29-bit arbitration ID    */
    uint8_t   dlc;              /* data length code (0–8)             */
    uint8_t   data[8];          /* payload bytes                      */
    bool      is_extended;      /* true → 29-bit ID                   */
    bool      is_rtr;           /* true → remote-transmission request */
    int64_t   timestamp_us;     /* esp_timer_get_time() at RX         */
} can_frame_t;

/** Aggregated feature vector for one sliding window */
typedef struct {
    /* Inter-frame timing */
    float     iat_mean_us;      /* mean inter-arrival time            */
    float     iat_std_us;       /* std-dev of inter-arrival time      */
    float     iat_min_us;       /* minimum IAT                        */
    float     iat_max_us;       /* maximum IAT                        */

    /* ID distribution */
    float     id_entropy;       /* Shannon entropy over seen IDs      */
    uint32_t  unique_ids;       /* distinct IDs observed              */
    float     dominant_id_freq; /* fraction of frames from top-1 ID  */

    /* Payload statistics */
    float     payload_mean;     /* mean byte value across all frames  */
    float     payload_std;      /* std-dev of byte values             */
    float     payload_entropy;  /* Shannon entropy of byte histogram  */
    float     zero_byte_ratio;  /* fraction of zero bytes             */

    /* Volume / rate */
    float     frame_rate_fps;   /* frames per second in window        */
    uint32_t  total_frames;     /* raw frame count in window          */

    /* RTR / DLC anomalies */
    float     rtr_ratio;        /* fraction of RTR frames             */
    float     dlc_mean;         /* mean DLC                           */
    float     dlc_std;          /* std-dev DLC                        */

    int64_t   window_start_us;  /* timestamp of first frame           */
    uint32_t  last_frame_id;    /* CAN ID of last frame in window     */
} feature_vector_t;

/** Inference output */
typedef struct {
    uint8_t   label;            /* 0=NORMAL 1=DoS 2=FUZZING 3=SPOOF  */
    float     confidence;       /* probability of predicted class     */
    float     probabilities[4]; /* full softmax output                */
    int64_t   timestamp_us;     /* when inference completed           */
    uint32_t  last_frame_id;    /* CAN ID of last frame in window     */
} inference_result_t;

/** Attack class labels */
typedef enum {
    ATTACK_NORMAL   = 0,
    ATTACK_DOS      = 1,
    ATTACK_FUZZING  = 2,
    ATTACK_SPOOFING = 3,
    ATTACK_COUNT    = 4
} attack_class_t;

static const char *const ATTACK_LABELS[ATTACK_COUNT] = {
    "NORMAL", "DoS", "FUZZING", "SPOOFING"
};

/* -----------------------------------------------------------------------
 * Public API
 * --------------------------------------------------------------------- */

/**
 * @brief  Initialise the TWAI peripheral in normal operating mode.
 * @return ESP_OK on success, ESP_FAIL otherwise.
 */
esp_err_t can_handler_init(void);

/**
 * @brief  De-initialise and stop the TWAI peripheral.
 */
void      can_handler_deinit(void);

/**
 * @brief  Transmit a single CAN frame (blocking, 10 ms timeout).
 * @param  frame  Pointer to frame to send.
 * @return ESP_OK | ESP_ERR_TIMEOUT | ESP_FAIL
 */
esp_err_t can_handler_transmit(const can_frame_t *frame);

/**
 * @brief  Transmit a pre-built attack-alert message on CAN_ID_ALERT.
 * @param  result  Inference result to broadcast.
 */
void      can_handler_send_alert(const inference_result_t *result);

/**
 * @brief  Software loopback: inject a frame directly into g_frame_queue
 *         (simulates what RX task would do with real TWAI peripheral).
 *         Used in loopback mode instead of twai_transmit().
 */
esp_err_t can_handler_loopback_frame(const twai_message_t *msg);

/**
 * @brief  FreeRTOS task: continuously reads TWAI RX FIFO and posts
 *         received frames to g_frame_queue.
 */
void      task_can_rx(void *pvParam);

/**
 * @brief  FreeRTOS task: sends a heartbeat frame every second so the
 *         secondary node can verify the bus is alive.
 */
void      task_can_tx_heartbeat(void *pvParam);

#endif /* CAN_HANDLER_H */

#ifdef __cplusplus
}
#endif

