# ==================== EXPERIMENT 2 – YOUR PERFECT 6-STEP GAIT (FIXED & PROTECTED) ====================

from hub import port, light_matrix
import motor
import runloop
import random

# =================================== HARDWARE CONFIGURATION ===================================
MOTOR_SPEED = 1000                                  # Motor speed in degrees/second

LEFT_LEG_MOTOR  = port.A                            # Port A controls left leg (forward = +50°)
RIGHT_LEG_MOTOR = port.B                            # Port B controls right leg (forward = -50°)
BODY_TILT_MOTOR = port.C                            # Port C controls body lift/lower

# Target motor positions (degrees)
LEG_MIDDLE     = 0                                  # Both legs centered
LEG_FORWARD    = 50                                 # Left leg forward position
LEG_BACKWARD   = -50                                # Right leg forward position (negative direction)
BODY_UP        = 140                                # Body lifted (off the ground)
BODY_DOWN      = 0                                  # Body lowered (weight on legs)

# =================================== LEARNING PARAMETERS ===================================
NUM_EPISODES   = 30                                 # Total training episodes
MAX_STEPS      = 40                                 # Maximum actions per episode
LEARNING_RATE  = 0.55                               # α – fast but stable learning (tuned for real robot)
DISCOUNT       = 0.9                                # γ – importance of future rewards
EXPLORATION    = 0.7                                # ε – initial exploration rate (decays over time)

# =================================== ENVIRONMENT: STATES ===================================
# Exactly matching your hand-designed gait table
states = [
    "0 Lmid Rmid Lup",   # Start: both legs middle, body up
    "1 Lfwd Rmid Lup",   # Left leg forward, body still up
    "2 Lfwd Rmid Rup",   # Body lowered onto left leg
    "3 Lmid Rmid Rup",   # Left leg returned to middle
    "4 Lmid Rfdw Rup",   # Right leg pushed forward
    "5 Lmid Rfdw Lup",   # Body lifted again
    "6 Lmid Rmid Lup",   # Right leg back → full cycle complete
    "7 STUCK"            # Fallback state if sensors are confused
]

# =================================== ACTIONS – EXACTLY YOUR TABLE ORDER ===================================
actions = ["A.Lfwd", "C.Rup", "A.Lmid", "B.Rfwd", "C.Lup", "B.Rmid"]
# Index:      0         1         2         3         4         5

# =================================== Q-TABLE: STRONG EXPERT SEEDING ===================================
# We seed your perfect 6-step sequence with high confidence (1.5)
Q = [[0.0]*6 for _ in range(8)]
Q[0][0] = 1.5   # State 0 → A.Lfwd  (move left leg forward)
Q[1][1] = 1.5   # State 1 → C.Rup   (lower body)
Q[2][2] = 1.5   # State 2 → A.Lmid  (return left leg)
Q[3][3] = 1.5   # State 3 → B.Rfwd  (move right leg forward)
Q[4][4] = 1.5   # State 4 → C.Lup   (lift body)
Q[5][5] = 1.5   # State 5 → B.Rmid  (return right leg)

episode_stats = []  # Stores (episode, reward, cycles, epsilon) for CSV export

# =================================== ACTION FUNCTIONS ===================================
async def a_lfwd(): await motor.run_to_absolute_position(LEFT_LEG_MOTOR, 50, MOTOR_SPEED)     # Left leg forward
async def c_rup():  await motor.run_to_absolute_position(BODY_TILT_MOTOR, BODY_DOWN, MOTOR_SPEED)  # Lower body
async def a_lmid(): await motor.run_to_absolute_position(LEFT_LEG_MOTOR, 0, MOTOR_SPEED)      # Left leg middle
async def b_rfwd(): await motor.run_to_absolute_position(RIGHT_LEG_MOTOR, -50, MOTOR_SPEED)   # Right leg forward
async def c_lup():  await motor.run_to_absolute_position(BODY_TILT_MOTOR, BODY_UP, MOTOR_SPEED)    # Lift body
async def b_rmid(): await motor.run_to_absolute_position(RIGHT_LEG_MOTOR, 0, MOTOR_SPEED)     # Right leg middle

action_functions = [a_lfwd, c_rup, a_lmid, b_rfwd, c_lup, b_rmid]

# =================================== STATE DETECTION ===================================
def get_current_state():
    lp = motor.absolute_position(LEFT_LEG_MOTOR)
    rp = motor.absolute_position(RIGHT_LEG_MOTOR)
    tp = motor.absolute_position(BODY_TILT_MOTOR)

    l_mid = abs(lp) < 30
    l_fwd = lp > 20
    r_mid = abs(rp) < 30
    r_fwd = rp < -20
    body_up = tp > 80

    if l_mid and r_mid and body_up:     return 0
    if l_fwd and r_mid and body_up:     return 1
    if l_fwd and r_mid and not body_up: return 2
    if l_mid and r_mid and not body_up: return 3
    if l_mid and r_fwd and not body_up: return 4
    if l_mid and r_fwd and body_up:     return 5
    if l_mid and r_mid and body_up:     return 6
    return 7  # STUCK – safety fallback

# =================================== VISUALIZE Q-TABLE ===================================
def print_q_table(ep):
    print("\n" + "="*110)
    if ep == 0:
        print("INITIAL Q-TABLE – YOUR PERFECT EXPERT POLICY (PROTECTED)")
    else:
        r = episode_stats[-1][1]
        c = episode_stats[-1][2]
        e = episode_stats[-1][3]
        print("EPISODE {} | Reward: {:+.1f} | Cycles: {} | ε: {:.3f}".format(ep, r, c, e))
    print("="*110)
    print("State            | A.Lfwd C.Rup A.Lmid B.Rfwd C.Lup B.Rmid | Best")
    print("-"*110)
    for i in range(8):
        row = "".join("{:6.3f}".format(v) for v in Q[i])
        best = actions[Q[i].index(max(Q[i]))]
        print("{:19} | {}→{}".format(states[i], row, best))
    print("-"*110)

# =================================== TRAINING LOOP – PROTECTS YOUR GAIT ===================================
async def train():
    global EXPLORATION

    # Reset robot to known starting position
    await motor.run_to_absolute_position(LEFT_LEG_MOTOR, 0, MOTOR_SPEED)
    await motor.run_to_absolute_position(RIGHT_LEG_MOTOR, 0, MOTOR_SPEED)
    await motor.run_to_absolute_position(BODY_TILT_MOTOR, 0, MOTOR_SPEED)
    await light_matrix.write("E2")
    print_q_table(0)

    for episode in range(1, NUM_EPISODES + 1):
        # Reset position at start of each episode
        await motor.run_to_absolute_position(LEFT_LEG_MOTOR, 0, MOTOR_SPEED)
        await motor.run_to_absolute_position(RIGHT_LEG_MOTOR, 0, MOTOR_SPEED)
        await motor.run_to_absolute_position(BODY_TILT_MOTOR, 0, MOTOR_SPEED)
        await runloop.sleep_ms(700)

        state = 0
        total_reward = 0.0
        cycles = 0

        for _ in range(MAX_STEPS):
            # For first 15 episodes: strictly follow your perfect hand-designed sequence
            if episode <= 15 and state <= 5:
                a = state                    # Forces exact gait: 0→0, 1→1, 2→2, 3→3, 4→4, 5→5
            else:
                # After episode 15: allow normal Q-learning (but expert actions stay strong)
                if random.random() < EXPLORATION:
                    a = random.randint(0, 5)
                else:
                    a = Q[state].index(max(Q[state]))

            await action_functions[a]()
            await runloop.sleep_ms(380)        # Wait for motors to settle

            next_state = get_current_state()

            # Reward shaping – strongly encourage full cycles
            reward = 0.0
            if state == 0 and next_state == 6:
                reward = 20.0                    # Huge reward for completing a full walking cycle
                cycles += 1
            elif next_state == 6:
                reward = 10.0
            elif a in [0, 3]:                    # Reward forward leg movements
                reward = 2.0
            elif next_state == 7:                # Penalty for getting stuck
                reward = -8.0

            total_reward += reward

            # Standard Q-learning update
            td = reward + DISCOUNT * max(Q[next_state]) - Q[state][a]
            Q[state][a] += LEARNING_RATE * td

            # Keep your expert actions dominant (never overwritten)
            if state <= 5:
                Q[state][state] = max(Q[state][state], 1.4)

            state = next_state

        # Record episode statistics
        episode_stats.append((episode, round(total_reward,2), cycles, round(EXPLORATION,3)))
        EXPLORATION = max(0.1, EXPLORATION * 0.93)   # Decay exploration
        await light_matrix.write(str(episode % 10))
        print_q_table(episode)

    await light_matrix.write("OK")

# =================================== FINAL WALKING – GUARANTEED YOUR GAIT ===================================
async def walk_forever():
    print("\n" + "="*110)
    print("YOUR PERFECT 6-STEP GAIT – LOCKED FOREVER")
    print("="*110)
    correct = ["A.Lfwd", "C.Rup", "A.Lmid", "B.Rfwd", "C.Lup", "B.Rmid", "A.Lfwd"]
    for i in range(7):
        print("{:19} → {}".format(states[i], correct[i]))
    print("="*110)

    # Export data for graphs
    print("\nCSV DATA:")
    print("Episode,Reward,Cycles,Epsilon")
    for ep, r, c, e in episode_stats:
        print("{},{},{},{}".format(ep, r, c, e))
    print("="*110)

    print("\nWALKING FOREVER WITH YOUR EXACT GAIT")
    # Hard-coded perfect sequence – no chance of deviation
    while True:
        s = get_current_state()
        if s == 7: 
            s = 0
        if s <= 5:
            a = s          # Follow your perfect sequence
        else:
            a = 0          # From state 6 → restart with A.Lfwd
        await action_functions[a]()
        await runloop.sleep_ms(330)   # Smooth, natural pace

# =================================== MAIN ===================================
async def main():
    await train()
    await walk_forever()

runloop.run(main())