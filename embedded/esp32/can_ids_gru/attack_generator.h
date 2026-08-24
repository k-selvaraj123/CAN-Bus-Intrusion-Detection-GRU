/**
 * @file attack_generator.h
 * @brief Single-mode attack simulator for CAN bus (user-selectable)
 *
 * To change the attack mode, edit CONFIG_ATTACK_MODE below.
 * Then recompile and upload.
 */

#ifndef ATTACK_GENERATOR_H
#define ATTACK_GENERATOR_H

#include <stdint.h>
#include <stdbool.h>
#include "driver/twai.h"

#ifdef __cplusplus
extern "C" {
#endif

/* -----------------------------------------------------------------------
 * ATTACK MODE SELECTION
 * -----------------------------------------------------------------------
 * Change this line to select which attack to simulate:
 *   CONFIG_ATTACK_IDLE       (normal heartbeat, baseline)
 *   CONFIG_ATTACK_DOS        (high-frequency spam on ID 0x001)
 *   CONFIG_ATTACK_FUZZING    (random IDs + random payloads)
 *   CONFIG_ATTACK_SPOOFING   (heartbeat ID 0x100 with corrupted payload)
 * --------------------------------------------------------------------- */
#define CONFIG_ATTACK_DOS

/* Attack mode enumeration */
typedef enum {
    ATTACK_MODE_IDLE     = 0,
    ATTACK_MODE_DOS      = 1,
    ATTACK_MODE_FUZZING  = 2,
    ATTACK_MODE_SPOOFING = 3,
} attack_mode_t;

/* Attack generator state */
typedef struct {
    attack_mode_t mode;              /* Which attack mode is active */
    uint32_t frame_counter;          /* Total frames generated */
    uint32_t last_frame_time_ms;     /* For interval-based generation */
} attack_generator_t;

/* -----------------------------------------------------------------------
 * Public API
 * --------------------------------------------------------------------- */

/**
 * @brief Initialize attack generator with selected mode
 */
int attack_generator_init(attack_generator_t *ctx);

/**
 * @brief Generate next frame (call every 10ms from Core 0 task)
 * @param ctx      Generator state
 * @param msg_out  [OUT] CAN message (valid only if return >= 0)
 * @return Attack mode (0-3) if frame generated, -1 if no frame this cycle
 */
int attack_generator_step(attack_generator_t *ctx, twai_message_t *msg_out);

/**
 * @brief Get mode name as string
 */
static inline const char *attack_state_name(attack_mode_t mode) {
    switch (mode) {
        case ATTACK_MODE_IDLE:     return "IDLE";
        case ATTACK_MODE_DOS:      return "DOS";
        case ATTACK_MODE_FUZZING:  return "FUZZING";
        case ATTACK_MODE_SPOOFING: return "SPOOFING";
        default:                   return "UNKNOWN";
    }
}

/**
 * @brief Get current attack mode from generator context
 */
static inline attack_mode_t attack_generator_get_current_state(attack_generator_t *ctx) {
    return ctx ? ctx->mode : ATTACK_MODE_IDLE;
}

/**
 * @brief Get global attack generator context (defined in can_cybersec.ino)
 */
extern attack_generator_t *get_attack_generator_context(void);

#ifdef __cplusplus
}
#endif

#endif // ATTACK_GENERATOR_H
