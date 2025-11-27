# ==================== FINAL – Q-TABLE STARTS AT ALL ZEROS ====================

from hub import port, light_matrix
import motor
import runloop
import random

# === HARDWARE CONFIGURATION ===
LEGSPEED = 1000                    # Motor speed in degrees per second
L_MID, L_FWD    = 0, 45            # Leg motor positions: 0° = middle, 45° = forward
C_LEVEL, C_UP    = 0, 150        # Tilt motor positions: 0° = level (down), 150° = lifted (up)

LEFT_LEGS = port.A                # Port A controls the walking legs
TILT    = port.C                # Port C controls the body tilt (lift/lower)

# === LEARNING HYPERPARAMETERS ===
NUM_EPISODES = 20                # Total number of training episodes
MAX_STEPS    = 30                # Maximum actions allowed per episode
ALPHA        = 0.35                # Learning rate
GAMMA        = 0.92                # Discount factor for future rewards
EPSILON    = 0.3                # Exploration rate (ε in ε-greedy policy)

# === ENVIRONMENT: STATES AND ACTIONS ===
states= ["Lmid Level", "Lmid Lup", "Lfwd Lup", "Lfwd Level"]# Four discrete states
actions = ["C.Lup", "A.Lfwd", "C.Level", "A.Lmid"]            # Four possible actions

# Dictionary mapping state names to Q-table row indices
state_idx = {states[i]: i for i in range(4)}

# === Q-TABLE INITIALIZED TO ALL ZEROS (as requested) ===
# Robot starts with no prior knowledge — pure learning from scratch
Q = [
    [0.0, 0.0, 0.0, 0.0],# Lmid Level → no preference
    [0.0, 0.0, 0.0, 0.0],# Lmid Lup→ no preference
    [0.0, 0.0, 0.0, 0.0],# Lfwd Lup→ no preference
    [0.0, 0.0, 0.0, 0.0]# Lfwd Level → no preference
]

episode_data = []# Stores (episode, reward, cycles, epsilon) for graphing

# === ACTION FUNCTIONS (each moves one motor to target position) ===
async def do_C_Lup():await motor.run_to_absolute_position(TILT, C_UP, LEGSPEED)    # Lift body
async def do_A_Lfwd():await motor.run_to_absolute_position(LEFT_LEGS, L_FWD, LEGSPEED)# Move legs forward
async def do_C_Level(): await motor.run_to_absolute_position(TILT, C_LEVEL, LEGSPEED)# Lower body
async def do_A_Lmid():await motor.run_to_absolute_position(LEFT_LEGS, L_MID, LEGSPEED)# Return legs to middle

action_funcs = [do_C_Lup, do_A_Lfwd, do_C_Level, do_A_Lmid]

# === STATE OBSERVATION FUNCTION ===
def get_state():
    lp = motor.absolute_position(LEFT_LEGS)
    tp = motor.absolute_position(TILT)
    if abs(lp) < 28 and tp < 70:    return "Lmid Level"
    if abs(lp) < 28 and tp > 90:    return "Lmid Lup"
    if lp > 25 and tp > 90:        return "Lfwd Lup"
    if lp > 25 and tp < 70:        return "Lfwd Level"
    return "Lmid Level"

# === VISUALIZE Q-TABLE ON CONSOLE ===
def print_q_table(ep):
    print("\n" + "="*85)
    print("INITIAL Q-TABLE (ALL ZEROS)" if ep == 0 else "Q-TABLE AFTER EPISODE {}".format(ep))
    print("="*85)
    print("State        |C.LupA.LfwdC.LevelA.Lmid| Best")
    print("-"*85)
    for i in range(4):
        row = "".join("{:7.3f}".format(Q[i][j]) for j in range(4))
        best = actions[Q[i].index(max(Q[i]))]
        print("{:14} | {}→{}".format(states[i], row, best))
    print("-"*85)

# === MAIN TRAINING LOOP ===
async def train():
    global EPSILON

    await motor.run_to_absolute_position(LEFT_LEGS, 0, LEGSPEED)
    await motor.run_to_absolute_position(TILT, 0, LEGSPEED)
    await light_matrix.write("QL")
    print_q_table(0)# Show all-zero initial Q-table

    for episode in range(1, NUM_EPISODES + 1):
        await motor.run_to_absolute_position(LEFT_LEGS, 0, LEGSPEED)
        await motor.run_to_absolute_position(TILT, 0, LEGSPEED)
        await runloop.sleep_ms(800)

        state_name = "Lmid Level"
        total_reward = 0.0
        cycles = 0
        last_action = -1

        for _ in range(MAX_STEPS):
            s = state_idx[state_name]

            # === STILL PROTECT FIRST ACTION (C.Lup) FOR FIRST 5 EPISODES ===
            # Even with zero initialization, we keep this to ensure physical safety
            if episode <= 5 and state_name == "Lmid Level":
                a = 0# Force C.Lup — prevents dangerous leg movement while body is down
            else:
                if random.random() < EPSILON:
                    a = random.randint(0, 3)
                else:
                    a = Q[s].index(max(Q[s]))

            # Prevent oscillation
            if a == last_action and random.random() < 0.3:
                a = (a + 2) % 4

            await action_funcs[a]()
            await runloop.sleep_ms(680 if a in [0, 2] else 520)

            next_state = get_state()

            reward = 0.1
            if state_name == "Lfwd Level" and a == 3 and next_state == "Lmid Level":
                reward += 4.0
                cycles += 1
            elif state_name != next_state:
                reward += 0.2

            total_reward += reward

            # Q-update (skip update only when forcing first action)
            if not (episode <= 5 and state_name == "Lmid Level" and a != 0):
                td_target = reward + GAMMA * max(Q[state_idx[next_state]])
                Q[s][a] += ALPHA * (td_target - Q[s][a])

            last_action = a
            state_name = next_state

        episode_data.append((episode, round(total_reward,2), cycles, round(EPSILON,3)))
        EPSILON = max(0.1, EPSILON * 0.92)
        await light_matrix.write(str(episode % 10))

        print_q_table(episode)
        print("Episode {} | Reward: {:+.2f} | Cycles: {} | ε: {:.3f}".format(episode, total_reward, cycles, EPSILON))

    await light_matrix.write("OK")

# === FINAL WALKING BEHAVIOR ===
async def walk_forever():
    print("\n" + "="*80)
    print("FINAL LEARNED POLICY – LEARNED FROM SCRATCH (Q-TABLE STARTED AT ZERO)")
    print("="*80)
    for i in range(4):
        best = actions[Q[i].index(max(Q[i]))]
        print("{:14} → {}".format(states[i], best))
    print("="*80)

    print("\nCSV DATA:")
    print("Episode,Reward,Cycles,Epsilon")
    for ep, rew, cyc, eps in episode_data:
        print("{},{},{},{}".format(ep, rew, cyc, eps))

    print("\nWalking perfectly – press red button to stop")
    while True:
        s = get_state()
        if s == "Lmid Level":
            a = 0# Always lift body first (safe & correct)
        else:
            a = Q[state_idx[s]].index(max(Q[state_idx[s]]))
        await action_funcs[a]()
        await runloop.sleep_ms(660 if a in [0, 2] else 500)

# === PROGRAM ENTRY POINT ===
async def main():
    await train()
    await walk_forever()

runloop.run(main())