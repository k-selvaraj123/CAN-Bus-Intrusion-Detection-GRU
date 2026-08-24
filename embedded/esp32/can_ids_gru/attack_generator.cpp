/**
 * @file attack_generator.cpp
 * @brief Single-mode attack simulator (selected via CONFIG_ATTACK_* macro)
 *
 * Generates frames continuously in the selected attack mode.
 * No state cycling — just one mode running forever.
 */

#include <string.h>
#include "esp_log.h"
#include "esp_timer.h"
#include "attack_generator.h"

static const char *TAG = "ATTACK_GEN";

/* Frame generation intervals (milliseconds) */
#define IDLE_INTERVAL_MS        1000  /* Heartbeat every 1s */
#define DOS_INTERVAL_MS         1     /* DOS: every 1ms (high frequency) */
#define FUZZING_INTERVAL_MS     50    /* FUZZING: every 50ms */
#define SPOOFING_INTERVAL_MS    100   /* SPOOFING: every 100ms */

/* CAN frame IDs */
#define CAN_ID_HEARTBEAT        0x100
#define CAN_ID_DOS              0x001

/* Simple pseudo-random number generator */
static uint32_t g_random_seed = 12345;

static uint32_t simple_rand(void) {
    g_random_seed = (g_random_seed * 1103515245U + 12345U) & 0x7fffffffU;
    return g_random_seed;
}

static uint32_t rand_range(uint32_t min_val, uint32_t max_val) {
    if (min_val >= max_val) return min_val;
    return min_val + (simple_rand() % (max_val - min_val + 1));
}

/* Determine attack mode from CONFIG_ATTACK_* macro */
static attack_mode_t get_configured_mode(void) {
#ifdef CONFIG_ATTACK_DOS
    return ATTACK_MODE_DOS;
#elif defined(CONFIG_ATTACK_FUZZING)
    return ATTACK_MODE_FUZZING;
#elif defined(CONFIG_ATTACK_SPOOFING)
    return ATTACK_MODE_SPOOFING;
#else
    return ATTACK_MODE_IDLE;
#endif
}

static const char *mode_name(attack_mode_t mode) {
    switch (mode) {
        case ATTACK_MODE_IDLE:     return "IDLE";
        case ATTACK_MODE_DOS:      return "DOS";
        case ATTACK_MODE_FUZZING:  return "FUZZING";
        case ATTACK_MODE_SPOOFING: return "SPOOFING";
        default:                   return "UNKNOWN";
    }
}

/* -----------------------------------------------------------------------
 * Frame generators (one per attack type)
 * --------------------------------------------------------------------- */

static bool gen_idle_frame(attack_generator_t *ctx, twai_message_t *msg) {
    uint32_t now_ms = esp_timer_get_time() / 1000;
    
    if (now_ms - ctx->last_frame_time_ms < IDLE_INTERVAL_MS) {
        return false;
    }
    
    ctx->last_frame_time_ms = now_ms;
    ctx->frame_counter++;
    
    msg->identifier = CAN_ID_HEARTBEAT;
    msg->data_length_code = 4;
    msg->flags = 0;
    
    uint32_t counter = ctx->frame_counter;
    msg->data[0] = (counter >> 24) & 0xFF;
    msg->data[1] = (counter >> 16) & 0xFF;
    msg->data[2] = (counter >> 8) & 0xFF;
    msg->data[3] = counter & 0xFF;
    
    return true;
}

static bool gen_dos_frame(attack_generator_t *ctx, twai_message_t *msg) {
    uint32_t now_ms = esp_timer_get_time() / 1000;
    
    if (now_ms - ctx->last_frame_time_ms < DOS_INTERVAL_MS) {
        return false;
    }
    
    ctx->last_frame_time_ms = now_ms;
    ctx->frame_counter++;
    
    msg->identifier = CAN_ID_DOS;
    msg->data_length_code = 8;
    msg->flags = 0;
    memset(msg->data, 0xAA, 8);
    
    return true;
}

static bool gen_fuzzing_frame(attack_generator_t *ctx, twai_message_t *msg) {
    uint32_t now_ms = esp_timer_get_time() / 1000;
    
    if (now_ms - ctx->last_frame_time_ms < FUZZING_INTERVAL_MS) {
        return false;
    }
    
    ctx->last_frame_time_ms = now_ms;
    ctx->frame_counter++;
    
    msg->identifier = rand_range(0x001, 0x7FE);
    msg->data_length_code = (uint8_t)rand_range(0, 8);
    msg->flags = 0;
    
    for (int i = 0; i < msg->data_length_code; i++) {
        msg->data[i] = (uint8_t)(simple_rand() & 0xFF);
    }
    
    return true;
}

static bool gen_spoofing_frame(attack_generator_t *ctx, twai_message_t *msg) {
    uint32_t now_ms = esp_timer_get_time() / 1000;
    
    if (now_ms - ctx->last_frame_time_ms < SPOOFING_INTERVAL_MS) {
        return false;
    }
    
    ctx->last_frame_time_ms = now_ms;
    ctx->frame_counter++;
    
    msg->identifier = CAN_ID_HEARTBEAT;
    msg->data_length_code = 4;
    msg->flags = 0;
    
    uint32_t counter = ctx->frame_counter ^ 0xDEADBEEFU;
    msg->data[0] = (counter >> 24) & 0xFF;
    msg->data[1] = (counter >> 16) & 0xFF;
    msg->data[2] = (counter >> 8) & 0xFF;
    msg->data[3] = counter & 0xFF;
    
    return true;
}

/* -----------------------------------------------------------------------
 * Public API
 * --------------------------------------------------------------------- */

int attack_generator_init(attack_generator_t *ctx) {
    if (!ctx) return -1;
    
    memset(ctx, 0, sizeof(attack_generator_t));
    ctx->mode = get_configured_mode();
    ctx->last_frame_time_ms = esp_timer_get_time() / 1000;
    
    ESP_LOGI(TAG, "Attack simulator initialized: MODE=%s", mode_name(ctx->mode));
    return 0;
}

int attack_generator_step(attack_generator_t *ctx, twai_message_t *msg_out) {
    if (!ctx || !msg_out) return -1;
    
    bool frame_ready = false;
    
    switch (ctx->mode) {
        case ATTACK_MODE_IDLE:
            frame_ready = gen_idle_frame(ctx, msg_out);
            break;
        case ATTACK_MODE_DOS:
            frame_ready = gen_dos_frame(ctx, msg_out);
            break;
        case ATTACK_MODE_FUZZING:
            frame_ready = gen_fuzzing_frame(ctx, msg_out);
            break;
        case ATTACK_MODE_SPOOFING:
            frame_ready = gen_spoofing_frame(ctx, msg_out);
            break;
        default:
            return -1;
    }
    
    return frame_ready ? (int)ctx->mode : -1;
}
