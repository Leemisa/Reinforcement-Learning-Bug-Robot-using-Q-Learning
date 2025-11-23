# ==================== EXPERIMENT 3A – PURE Q-LEARNING + DISTANCE GOAL ====================

from hub import port, light_matrix, distance_sensor
import motor
import runloop
import random

LEFT_LEGS  = port.A
RIGHT_LEGS = port.B
TILT       = port.C
DIST       = port.F

L_MID, L_FWD = 0, 50
R_MID, R_FWD = 0, -50
C_UP, C_LEVEL = 140, 0
SPEED = 900

NUM_EPISODES = 30
MAX_STEPS = 45
ALPHA, GAMMA = 0.25, 0.9
EPSILON = 0.8

states = [
    "0 Lmid Rmid Lup", "1 Lfwd Rmid Lup", "2 Lfwd Rmid Rup",
    "3 Lmid Rmid Rup", "4 Lmid Rfdw Rup", "5 Lmid Rfdw Lup",
    "6 Lmid Rmid Lup", "7 STUCK"
]
actions = ["Lup", "Rup", "Lfwd", "Lmid", "Rfwd", "Rmid"]

# ALL ZEROS – TRUE LEARNING FROM SCRATCH
Q = [[0.0]*6 for _ in range(8)]
episode_data = []  # For CSV

async def do_Lup():   await motor.run_to_absolute_position(TILT, C_UP, SPEED)
async def do_Rup():   await motor.run_to_absolute_position(TILT, C_LEVEL, SPEED)
async def do_Lfwd():  await motor.run_to_absolute_position(LEFT_LEGS, L_FWD, SPEED)
async def do_Lmid():  await motor.run_to_absolute_position(LEFT_LEGS, L_MID, SPEED)
async def do_Rfwd():  await motor.run_to_absolute_position(RIGHT_LEGS, R_FWD, SPEED)
async def do_Rmid():  await motor.run_to_absolute_position(RIGHT_LEGS, R_MID, SPEED)

action_funcs = [do_Lup, do_Rup, do_Lfwd, do_Lmid, do_Rfwd, do_Rmid]

def get_distance():
    try:
        d = distance_sensor.distance(DIST)
        return d if 20 < d < 2000 else 2000
    except:
        return 2000

def get_state():
    lp = motor.absolute_position(LEFT_LEGS)
    rp = motor.absolute_position(RIGHT_LEGS)
    tp = motor.absolute_position(TILT)
    l_mid = abs(lp) < 25; l_fwd = lp > 20
    r_mid = abs(rp) < 25; r_fwd = rp < -20
    c_up = tp > 80
    if l_mid and r_mid and c_up: return 0
    if l_fwd and r_mid and c_up: return 1
    if l_fwd and r_mid and not c_up: return 2
    if l_mid and r_mid and not c_up: return 3
    if l_mid and r_fwd and not c_up: return 4
    if l_mid and r_fwd and c_up: return 5
    if l_mid and r_mid and c_up: return 6
    return 7

async def train():
    global EPSILON
    await motor.run_to_absolute_position(LEFT_LEGS, 0, SPEED)
    await motor.run_to_absolute_position(RIGHT_LEGS, 0, SPEED)
    await motor.run_to_absolute_position(TILT, 0, SPEED)
    await light_matrix.write("3A")

    print("EXPERIMENT 3A – PURE Q-LEARNING + DISTANCE GOAL (ALL ZEROS)")
    for episode in range(1, NUM_EPISODES + 1):
        await motor.run_to_absolute_position(LEFT_LEGS, 0, SPEED)
        await motor.run_to_absolute_position(RIGHT_LEGS, 0, SPEED)
        await motor.run_to_absolute_position(TILT, 0, SPEED)
        await runloop.sleep_ms(800)

        state = 0
        total_reward = 0.0
        start_dist = get_distance()

        for step in range(MAX_STEPS):
            a = random.randint(0, 5) if random.random() < EPSILON else Q[state].index(max(Q[state]))
            await action_funcs[a]()
            await runloop.sleep_ms(320)

            dist = get_distance()
            reward = 0

            # BIG REWARD: Getting closer
            if dist < 60 and dist > 30:
                reward += 20
            elif dist < 100:
                reward += 8
            elif dist < 150:
                reward += 3

            # CRASH PENALTY
            if dist <= 30:
                reward -= 30

            # Gait cycle bonus
            next_state = get_state()
            if state == 0 and next_state == 6:
                reward += 8

            total_reward += reward
            Q[state][a] += ALPHA * (reward + GAMMA * max(Q[next_state]) - Q[state][a])
            state = next_state

        final_dist = get_distance()
        episode_data.append((episode, total_reward, final_dist))
        EPSILON = max(0.1, EPSILON * 0.92)
        await light_matrix.write(str(episode % 10))

        print(f"EPISODE {episode} | Reward: {total_reward:+.1f} | Final Dist: {final_dist}mm")
        for i in range(8):
            row = "  ".join(f"{Q[i][j]:5.2f}" for j in range(6))
            print(f"  {states[i]:20} | {row}")

    await light_matrix.write("OK")

    # FINAL CSV + POLICY
    print("\nEpisode,Reward,Final_Distance_mm")
    for ep, r, d in episode_data:
        print(f"{ep},{r:.1f},{d}")

    print("\nLearned Policy:")
    for i in range(8):
        best = Q[i].index(max(Q[i]))
        print(f"{states[i]} → {actions[best]}")

    print("\nFinal Q-Table:")
    print("State,Lup,Rup,Lfwd,Lmid,Rfwd,Rmid")
    for i in range(8):
        row = ",".join(f"{Q[i][j]:.2f}" for j in range(6))
        print(f"{states[i].replace(' ','_')},{row}")

    print("\nWALKING TOWARD OBJECT FOREVER...")
    while True:
        state = get_state()
        if state == 7: state = 0
        best = Q[state].index(max(Q[state]))
        await action_funcs[best]()
        await runloop.sleep_ms(300)

async def main():
    await train()

runloop.run(main())