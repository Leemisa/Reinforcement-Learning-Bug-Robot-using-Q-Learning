# ==================== EXPERIMENT 1B – SEEDED Q-TABLE (C.Lup, A.Lfwd, C.Level, A.Lmid = 1) ====================

from hub import port, light_matrix
import motor
import runloop
import random

# SETTINGS
LEGSPEED = 1000
L_MID, L_FWD = 0, 45
C_LEVEL, C_UP = 0, 150
LEFT_LEGS = port.A
TILT = port.C

NUM_EPISODES = 20
MAX_STEPS = 25
ALPHA = 0.3
GAMMA = 0.85
EPSILON = 0.4          # Lower exploration because of good seed

states = ["left-up", "right-up", "right-down", "left-down"]
actions = ["L-mid", "L-fwd", "C-level", "C-up"]
state_idx = {"left-up":0, "right-up":1, "right-down":2, "left-down":3}

# YOUR SEEDED Q-TABLE (exactly as requested)
Q = [
    [0.0, 0.0, 0.0, 1.0],  # left-up    → C-up (C.Lup)
    [0.0, 0.0, 1.0, 0.0],  # right-up   → C-level (C.Level)
    [1.0, 0.0, 0.0, 0.0],  # right-down → L-mid (A.Lmid)
    [0.0, 1.0, 0.0, 0.0]   # left-down  → L-fwd (A.Lfwd)
]

episode_rewards = []

async def move_leg_mid():  await motor.run_to_absolute_position(LEFT_LEGS, L_MID, LEGSPEED)
async def move_leg_fwd():  await motor.run_to_absolute_position(LEFT_LEGS, L_FWD, LEGSPEED)
async def tilt_level():    await motor.run_to_absolute_position(TILT, C_LEVEL, LEGSPEED)
async def tilt_up():       await motor.run_to_absolute_position(TILT, C_UP, LEGSPEED)

action_funcs = [move_leg_mid, move_leg_fwd, tilt_level, tilt_up]

previous_state = "left-down"
stuck_counter = 0

def get_state():
    global previous_state, stuck_counter
    lp = motor.absolute_position(LEFT_LEGS)
    tp = motor.absolute_position(TILT)

    if tp > 80 and lp < 20:
        state = "left-up"
    elif tp > 80 and lp > 30:
        state = "right-up"
    elif tp < 40 and lp > 30:
        state = "right-down"
    else:
        state = "left-down"

    # Anti-stuck mechanism
    if state == previous_state:
        stuck_counter += 1
        if stuck_counter > 3:
            stuck_counter = 0
            state = random.choice(["right-up", "left-down", "right-down"])
    else:
        stuck_counter = 0

    previous_state = state
    return state

async def train():
    global EPSILON

    await motor.run_to_absolute_position(LEFT_LEGS, 0, LEGSPEED)
    await motor.run_to_absolute_position(TILT, 0, LEGSPEED)
    await light_matrix.write("1B")

    print("\n" + "="*90)
    print("EXPERIMENT 1B – SEEDED Q-TABLE")
    print("C.Lup, A.Lfwd, C.Level, A.Lmid = 1.0 → Fast & Smooth Learning!")
    print("="*90)

    print("INITIAL SEEDED Q-TABLE:")
    print("State         | L-mid  L-fwd C-level  C-up")
    print("-"*50)
    for i in range(4):
        row = "  ".join(f"{Q[i][j]:6.1f}" for j in range(4))
        print(f"{states[i]:13} | {row}")
    print("-"*50)

    for episode in range(1, NUM_EPISODES + 1):
        await motor.run_to_absolute_position(LEFT_LEGS, 0, LEGSPEED)
        await motor.run_to_absolute_position(TILT, 0, LEGSPEED)
        await runloop.sleep_ms(500)

        state_name = "left-down"
        total_reward = 0.0
        cycle_count = 0

        for _ in range(MAX_STEPS):
            s = state_idx[state_name]
            a = random.randint(0, 3) if random.random() < EPSILON else Q[s].index(max(Q[s]))

            await action_funcs[a]()
            await runloop.sleep_ms(250)

            # Reward shaping (same as before)
            reward = 0
            if state_name == "left-down" and a == 0:   # L-mid → full cycle
                reward = 3
                cycle_count += 1
            elif a == 1 and "down" in state_name:      # L-fwd from down
                reward = 2
            elif a == 3 and "left" in state_name:      # C-up from left
                reward = 1
            elif a == 2 and "right" in state_name:     # C-level from right
                reward = 1
            else:
                reward = -1

            total_reward += reward
            next_state = get_state()
            Q[s][a] += ALPHA * (reward + GAMMA * max(Q[state_idx[next_state]]) - Q[s][a])
            state_name = next_state

        episode_rewards.append(total_reward)
        EPSILON = max(0.1, EPSILON * 0.85)
        await light_matrix.write(str(episode % 10))

        print(f"\nEPISODE {episode} – REWARD: {total_reward:+.0f} ({cycle_count} cycles)")
        for i in range(4):
            row = "  ".join(f"{Q[i][j]:6.2f}" for j in range(4))
            print(f"{states[i]:13} | {row}")
        print("-"*50)

    await light_matrix.write("OK")
    print("\nTRAINING COMPLETE!")

async def walk_forever():
    print("\n" + "="*90)
    print("FINAL LEARNED POLICY – WALKING FOREVER!")
    for i in range(4):
        best = Q[i].index(max(Q[i]))
        print(f"{states[i]} → {actions[best]}")
    print("="*90)

    print("\nEpisode,Reward")
    for i, r in enumerate(episode_rewards, 1):
        print(f"{i},{int(r)}")

    print("\nFinal Q-Table (CSV format):")
    print("State,L-mid,L-fwd,C-level,C-up")
    for i in range(4):
        row = ",".join(f"{Q[i][j]:.2f}" for j in range(4))
        print(f"{states[i]},{row}")

    print("\nWALKING SMOOTHLY FOREVER – Press red STOP button to end")
    print("="*90)

    while True:
        state = get_state()
        best = Q[state_idx[state]].index(max(Q[state_idx[state]]))
        await action_funcs[best]()
        await runloop.sleep_ms(250)

async def main():
    await train()
    await walk_forever()

runloop.run(main())