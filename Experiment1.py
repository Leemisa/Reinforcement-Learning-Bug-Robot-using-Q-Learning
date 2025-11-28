# ==================== Q-LEARNING FIXED WALKING ====================
# LEGO SPIKE Prime robot learns a walking gait using Q-learning.
# Robot has 4 states and 4 actions. Rewards encourage completing the correct gait cycle.

from hub import port, light_matrix
import motor
import runloop
import random

# === MOTOR CONFIGURATION ===
LEGSPEED = 1000           # Motor speed in degrees per second
L_MID, L_FWD = 0, 45      # Left leg positions: middle and forward
C_LEVEL, C_UP = 0, 150    # Tilt motor positions: level (down) and lifted (up)

LEFT_LEGS = port.A         # Port controlling walking legs
TILT = port.C              # Port controlling body tilt

# === LEARNING PARAMETERS ===
NUM_EPISODES = 20          # Total training episodes
MAX_STEPS = 30             # Max actions per episode
ALPHA = 0.35               # Learning rate
GAMMA = 0.92               # Discount factor for future rewards
EPSILON = 0.3              # Exploration rate for ε-greedy policy

# === STATES AND ACTIONS ===
states = ["Lmid Level", "Lmid Lup", "Lfwd Lup", "Lfwd Level"]  # Discrete robot states
actions = ["C.Lup", "A.Lfwd", "C.Level", "A.Lmid"]            # Possible motor actions
state_idx = {states[i]: i for i in range(4)}                 # Map state name to Q-table row

# === Q-TABLE INITIALIZATION ===
Q = [[0.0 for _ in range(4)] for _ in range(4)]              # Start with all zeros (learning from scratch)
episode_data = []                                            # Stores reward, cycles, epsilon for each episode

# === MOTOR ACTION FUNCTIONS ===
async def do_C_Lup(): await motor.run_to_absolute_position(TILT, C_UP, LEGSPEED)      # Lift body
async def do_A_Lfwd(): await motor.run_to_absolute_position(LEFT_LEGS, L_FWD, LEGSPEED)  # Move legs forward
async def do_C_Level(): await motor.run_to_absolute_position(TILT, C_LEVEL, LEGSPEED)    # Lower body
async def do_A_Lmid(): await motor.run_to_absolute_position(LEFT_LEGS, L_MID, LEGSPEED)  # Return legs to middle

action_funcs = [do_C_Lup, do_A_Lfwd, do_C_Level, do_A_Lmid]   # List for easy indexing by Q-table

# === STATE DETECTION FUNCTION ===
def get_state():
    lp = motor.absolute_position(LEFT_LEGS)  # Left leg position
    tp = motor.absolute_position(TILT)       # Tilt position
    if abs(lp) < 28 and tp < 70: return "Lmid Level"
    if abs(lp) < 28 and tp > 90: return "Lmid Lup"
    if lp > 25 and tp > 90: return "Lfwd Lup"
    if lp > 25 and tp < 70: return "Lfwd Level"
    return "Lmid Level"

# === Q-TABLE VISUALIZATION ===
def print_q_table(ep):
    print("\n" + "="*85)
    print("INITIAL Q-TABLE" if ep == 0 else "Q-TABLE AFTER EPISODE {}".format(ep))
    print("="*85)
    print("State        |C.LupA.LfwdC.LevelA.Lmid| Best")
    print("-"*85)
    for i in range(4):
        row = "".join("{:7.3f}".format(Q[i][j]) for j in range(4))
        best = actions[Q[i].index(max(Q[i]))]
        print("{:14} | {}→{}".format(states[i], row, best))
    print("-"*85)

# === TRAINING LOOP ===
async def train():
    global EPSILON

    # Reset robot to start position
    await motor.run_to_absolute_position(LEFT_LEGS, 0, LEGSPEED)
    await motor.run_to_absolute_position(TILT, 0, LEGSPEED)
    await light_matrix.write("QL")
    print_q_table(0)

    for episode in range(1, NUM_EPISODES + 1):
        # Reset positions at start of each episode
        await motor.run_to_absolute_position(LEFT_LEGS, 0, LEGSPEED)
        await motor.run_to_absolute_position(TILT, 0, LEGSPEED)
        await runloop.sleep_ms(800)

        state_name = "Lmid Level"
        total_reward = 0.0
        cycles = 0
        last_action = -1

        for _ in range(MAX_STEPS):
            s = state_idx[state_name]

            # ε-greedy action selection
            if random.random() < EPSILON:
                a = random.randint(0, 3)         # Explore random action
            else:
                a = Q[s].index(max(Q[s]))        # Exploit best known action

            # Safety: only allow moving legs forward if body is lifted
            if state_name in ["Lmid Level", "Lfwd Level"] and a == 1:  # A.Lfwd
                a = 0  # lift body first

            # Prevent repeated oscillation
            if a == last_action and random.random() < 0.3:
                a = (a + 2) % 4

            await action_funcs[a]()
            await runloop.sleep_ms(680 if a in [0, 2] else 520)

            next_state = get_state()

            # Reward system
            reward = -0.2  # Small penalty for useless moves
            if state_name == "Lmid Level" and a == 0 and next_state == "Lmid Lup":
                reward = 3.0
            elif state_name == "Lmid Lup" and a == 1 and next_state == "Lfwd Lup":
                reward = 4.0
            elif state_name == "Lfwd Lup" and a == 2 and next_state == "Lfwd Level":
                reward = 5.0
            elif state_name == "Lfwd Level" and a == 3 and next_state == "Lmid Level":
                reward = 6.0
                cycles += 1  # Completed one full gait cycle

            total_reward += reward

            # Q-Learning update
            next_s = state_idx[next_state]
            Q[s][a] += ALPHA * (reward + GAMMA * max(Q[next_s]) - Q[s][a])

            last_action = a
            state_name = next_state

        # Record episode data
        episode_data.append((episode, round(total_reward,2), cycles, round(EPSILON,3)))
        EPSILON = max(0.1, EPSILON * 0.97)  # Gradual exploration decay
        await light_matrix.write(str(episode % 10))

        print_q_table(episode)
        print("Episode {} | Reward: {:+.2f} | Cycles: {} | ε: {:.3f}".format(
            episode, total_reward, cycles, EPSILON))

    await light_matrix.write("OK")

# === FINAL WALKING POLICY ===
async def walk_forever():
    print("\n" + "="*80)
    print("FINAL LEARNED POLICY")
    print("="*80)
    for i in range(4):
        best = actions[Q[i].index(max(Q[i]))]
        print("{:14} → {}".format(states[i], best))
    print("="*80)

    print("\nEpisode,Reward,Cycles,Epsilon")
    for ep, rew, cyc, eps in episode_data:
        print("{},{},{},{}".format(ep, rew, cyc, eps))

    # Repeat learned gait indefinitely
    while True:
        s = get_state()
        a = Q[state_idx[s]].index(max(Q[state_idx[s]]))
        await action_funcs[a]()
        await runloop.sleep_ms(660 if a in [0, 2] else 500)

# === PROGRAM ENTRY POINT ===
async def main():
    await train()
    await walk_forever()

runloop.run(main())
