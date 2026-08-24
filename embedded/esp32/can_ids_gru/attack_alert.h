/**
 * @file attack_alert.h / attack_alert.c
 * @brief Alert handler: logs attacks, broadcasts on CAN, throttles repeats
 */
#ifdef __cplusplus
extern "C" {
#endif

#ifndef ATTACK_ALERT_H
#define ATTACK_ALERT_H

#include "can_handler.h"

/**
 * @brief  FreeRTOS task: reads from g_alert_queue, logs attack events,
 *         calls can_handler_send_alert() for non-NORMAL detections,
 *         and maintains a simple cooldown to avoid alert flooding.
 */
void task_alert_handler(void *pvParam);

#endif /* ATTACK_ALERT_H */


#ifdef __cplusplus
}
#endif
