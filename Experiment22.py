# ==================== EXPERIMENT 2 – LEARNS + WALKS FOREVER (REALLY!) ====================

from hub import port, light_matrix
import motor
import runloop
import random

# YOUR PORTS
LEFT_LEGS= port.A
RIGHT_LEGS = port.B
TILT    = port.C

# POSITIONS
L_MID = 0
L_FWD = 50
R_MID = 0
R_FWD = -50
C_UP = 140
C_LEVEL = 0
SPEED = 1000

NUM_EPISODES = 25
MAX_STEPS = 40
ALPHA = 0.3
GAMMA = 0.9
EPSILON = 0.7

states = [
    "0 Lmid Rmid Lup",
    "1 Lfwd Rmid Lup",
    "2 Lfwd Rmid Rup",
    "3 Lmid Rmid Rup",
    "4 Lmid Rfdw Rup",
    "5 Lmid Rfdw Lup",
    "6 Lmid Rmid Lup",
    "7 STUCK"
]

actions = ["Lup", "Rup", "Lfwd", "Lmid", "Rfwd", "Rmid"]

# YOUR INITIAL Q-TABLE (one 1 per row)
Q = [
    [1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    [0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
    [0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
    [0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
    [1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    [0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
    [0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
]

episode_rewards = []

async def do_Lup():await motor.run_to_absolute_position(TILT, C_UP, SPEED)
async def do_Rup():await motor.run_to_absolute_position(TILT, C_LEVEL, SPEED)
async def do_Lfwd():await motor.run_to_absolute_position(LEFT_LEGS, L_FWD, SPEED)
async def do_Lmid():await motor.run_to_absolute_position(LEFT_LEGS, L_MID, SPEED)
async def do_Rfwd():await motor.run_to_absolute_position(RIGHT_LEGS, R_FWD, SPEED)
async def do_Rmid():await motor.run_to_absolute_position(RIGHT_LEGS, R_MID, SPEED)

action_funcs = [do_Lup, do_Rup, do_Lfwd, do_Lmid, do_Rfwd, do_Rmid]

def get_state():
    lp = motor.absolute_position(LEFT_LEGS)
    rp = motor.absolute_position(RIGHT_LEGS)
    tp = motor.absolute_position(TILT)

    l_mid = abs(lp - L_MID) < 25
    l_fwd = lp > 20
    r_mid = abs(rp - R_MID) < 25
    r_fwd = rp < -20
    c_up= tp > 80

    if l_mid and r_mid and c_up:return 0
    if l_fwd and r_mid and c_up:return 1
    if l_fwd and r_mid and not c_up: return 2
    if l_mid and r_mid and not c_up: return 3
    if l_mid and r_fwd and not c_up: return 4
    if l_mid and r_fwd and c_up:return 5
    if l_mid and r_mid and c_up:return 6
    return 7

def print_q_table(episode_num):
    print("\n" + "="*100)
    if episode_num == 0:
        print("INITIAL Q-TABLE")
    else:
        print("EPISODE {} – REWARD: {:+} | ON HUB: {}".format(
            episode_num, int(episode_rewards[-1]), episode_num))
    print("State                |LupRup Lfwd Lmid Rfwd Rmid")
    print("-"*100)
    for i in range(8):
        row = "".join("{:5.2f}".format(Q[i][j]) for j in range(6))
        print("{:20} | {}".format(states[i], row))
    print("-"*100)

async def train():
    global EPSILON

    await motor.run_to_absolute_position(LEFT_LEGS, 0, SPEED)
    await motor.run_to_absolute_position(RIGHT_LEGS, 0, SPEED)
    await motor.run_to_absolute_position(TILT, 0, SPEED)
    await light_matrix.write("E2")

    print_q_table(0)

    for episode in range(1, NUM_EPISODES + 1):
        await motor.run_to_absolute_position(LEFT_LEGS, 0, SPEED)
        await motor.run_to_absolute_position(RIGHT_LEGS, 0, SPEED)
        await motor.run_to_absolute_position(TILT, 0, SPEED)
        await runloop.sleep_ms(600)

        state = 0
        total_reward = 0.0
        cycle_count = 0

        for step in range(MAX_STEPS):
            if random.random() < EPSILON:
                a = random.randint(0, 5)
            else:
                a = Q[state].index(max(Q[state]))

            await action_funcs[a]()
            await runloop.sleep_ms(300)

            next_state = get_state()
            reward = 0
            if state == 0 and next_state == 6:
                reward = 10
                cycle_count += 1
            elif next_state == 6:
                reward = 5
            elif a in [2, 4]:
                reward = 1
            elif next_state == 7:
                reward = -3

            total_reward += reward
            Q[state][a] += ALPHA * (reward + GAMMA * max(Q[next_state]) - Q[state][a])
            state = next_state

        episode_rewards.append(total_reward)
        EPSILON = max(0.1, EPSILON * 0.9)
        await light_matrix.write(str(episode % 10))
        print_q_table(episode)

    await light_matrix.write("OK")
    print("\nTRAINING DONE → NOW WALKING FOREVER!")

# FIXED: Now uses await so it really walks forever!
async def walk_forever():
    print("\n" + "="*100)
    print("WALKING FOREVER WITH LEARNED 8-PHASE GAIT!")
    for i in range(8):
        best = Q[i].index(max(Q[i]))
        print("{} → {}".format(states[i], actions[best]))
    print("="*100)

    while True:
        state = get_state()
        if state == 7:
            state = 0
        best = Q[state].index(max(Q[state]))

        await action_funcs[best]()    # ← THIS AWAIT WAS MISSING BEFORE!
        await runloop.sleep_ms(280)# Smooth natural pace

async def main():
    await train()
    await walk_forever()

runloop.run(main())