import runloop
import motor
import distance_sensor
from hub import light_matrix, port
from app import sound
import random

# ========================================
# EXPERIMENT 3 – 8-STATE BIPED WALKER (SEEDED VERSION)
# Smart starting Q-values to help the robot learn faster
# Start distance: 150–200 mm
# Gentle rewards, epsilon decay, safe leg movements
# ========================================

ALPHA = 0.5        # Learning rate – how fast to update Q-values
GAMMA = 0.95        # Discount factor – future rewards matter
EPSILON_START = 0.9# Start with lots of exploration
EPSILON_END = 0.05    # End with mostly exploitation
EPSILON_DECAY = 0.95# Epsilon reduces by 5% each episode
EPISODES = 40        # Total training episodes
MAX_STEPS = 50        # Max steps per episode
SPEED = 950        # Motor speed for body (port C)
SLEEP = 150        # Delay after each move (ms)

# Motor target positions (in degrees)
Lmid, Lfwd = 0, 60    # Left leg: middle and forward
Rmid, Rfwd = 0, 60    # Right leg: middle and forward
Lup, Rup= 140, -140 # Body tilt for lifting (left/right)

# List of all 6 possible actions
ACTIONS = ["Lup", "Rup", "Lfwd", "Lmid", "Rfwd", "Rmid"]

old_dist = 999# Tracks last valid distance reading

def safe_dist():
    """Safely read distance sensor. Returns 999 if object is lost."""
    global old_dist
    d = distance_sensor.distance(port.F)
    if d is None or d <= 0 or d > 1000:
        return 999# Lost sight of target
    old_dist = d
    return d

def get_state():
    """Convert motor positions into one of 8 meaningful states."""
    a = motor.absolute_position(port.A) or 0# Left leg
    b = motor.absolute_position(port.B) or 0# Right leg
    c = motor.absolute_position(port.C) or 0# Body tilt

    # Detect leg positions (±45° tolerance)
    left_mid= abs(a - Lmid) < 45
    left_fwd= abs(a - Lfwd) < 45
    right_mid = abs(b - Rmid) < 45
    right_fwd = abs(b - Rfwd) < 45

    # Detect body position
    body_up= c > 80 or c < -80    # Tilted = lifting
    body_down = abs(c) < 85            # Flat = standing

    # 8 states based on gait cycle
    if body_up and left_fwd and right_mid:    return 0# Lifting on right leg
    if body_up and right_fwd and left_mid:    return 1# Lifting on left leg
    if body_down and left_mid and right_mid:    return 2# Balanced standing
    if body_down and left_fwd and right_mid:    return 3# Left leg forward
    if body_down and right_fwd and left_mid:    return 4# Right leg forward
    if body_up and left_mid and right_mid:    return 5# Body up, legs centered
    if body_down and left_fwd and right_fwd:    return 6# Both legs forward
    return 7# Unknown / recovery state

async def move(action):
    """Execute one action – slow legs, fast body."""
    if action == 0:await motor.run_to_absolute_position(port.C, Lup, SPEED)    # Tilt body left
    elif action == 1: await motor.run_to_absolute_position(port.C, Rup, SPEED)    # Tilt body right
    elif action == 2: await motor.run_to_absolute_position(port.A, Lfwd, int(SPEED * 0.5))# Left forward (slow)
    elif action == 3: await motor.run_to_absolute_position(port.A, Lmid, int(SPEED * 0.5))# Left middle
    elif action == 4: await motor.run_to_absolute_position(port.B, Rfwd, int(SPEED * 0.5))# Right forward
    elif action == 5: await motor.run_to_absolute_position(port.B, Rmid, int(SPEED * 0.5))# Right middle
    await runloop.sleep_ms(SLEEP)# Let movement finish

async def reset():
    """Safely return robot to starting position."""
    await motor.run_to_absolute_position(port.C, 0, SPEED)    # Center body first
    await runloop.sleep_ms(600)
    await motor.run_to_absolute_position(port.A, Lmid, int(SPEED * 0.5))
    await runloop.sleep_ms(500)
    await motor.run_to_absolute_position(port.B, Rmid, int(SPEED * 0.5))
    await runloop.sleep_ms(700)

async def print_q_table(Q, title):
    """Print the full 8×6 Q-table in a clean format."""
    print("\n" + "=" * 100)
    print("{0}".format(title).center(100))
    print("=" * 100)
    header = "State |" + "".join(" {0:>7}".format(act) for act in ACTIONS)
    print(header)
    print("-" * 100)
    for s in range(8):
        row = "{0:5d} |".format(s)
        for val in Q[s]:
            row += " {0:7.2f}".format(val)
        print(row)
    print("-" * 100)

async def main():
    global old_dist

    print("\n" + "="*100)
    print(" EXPERIMENT 3 – SEEDED VERSION ".center(100))
    print(" Smart starting hints | Start 150–200 mm | Gentle rewards ".center(100))
    print("="*100)

    await reset()
    await light_matrix.write("E3")

    # Wait for correct starting distance
    print("Place target 150–200 mm away on the mattress...")
    while True:
        d = safe_dist()
        if 150 <= d <= 200:
            print("Good starting distance: {0} mm".format(d))
            break
        await runloop.sleep_ms(500)

    # SEEDED Q-TABLE – gives the robot a strong starting policy
    Q = [
        [0.0, 0.0, 1.0, 0.0, 0.0, 0.0],# State 0 → prefers Lfwd (move left leg forward)
        [0.0, 1.0, 0.0, 0.0, 0.0, 0.0],# State 1 → prefers Rup(tilt right to lift)
        [0.0, 0.0, 0.0, 1.0, 0.0, 0.0],# State 2 → prefers Lmid (bring left leg back)
        [0.0, 0.0, 0.0, 0.0, 1.0, 0.0],# State 3 → prefers Rfwd (move right leg forward)
        [1.0, 0.0, 0.0, 0.0, 0.0, 0.0],# State 4 → prefers Lup(tilt left to lift)
        [0.0, 0.0, 0.0, 0.0, 0.0, 1.0],# State 5 → prefers Rmid (bring right leg back)
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],# State 6 → no strong preference
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]# State 7 → recovery state
    ]

    await print_q_table(Q, "INITIAL Q-TABLE (SEEDED – SMART HINTS)")

    epsilon = EPSILON_START
    csv_data = []

    for ep in range(1, EPISODES + 1):
        await reset()
        s = get_state()
        old_dist = safe_dist()
        start_dist = old_dist
        total_reward = 0
        steps = 0
        goal_reached = False

        print("\nEPISODE {0} | EPSILON = {1:.4f}".format(ep, epsilon))

        for t in range(1, MAX_STEPS + 1):
            steps = t

            # Epsilon-greedy: explore or exploit
            if random.random() < epsilon:
                a = random.randint(0, 5)
            else:
                best_q = max(Q[s])
                best_actions = [i for i, q in enumerate(Q[s]) if q == best_q]
                a = random.choice(best_actions)

            await move(a)

            ns = get_state()
            new_d = safe_dist()
            delta = old_dist - new_d# Positive = got closer

            r = 0

            # Gentle reward shaping
            if new_d >= 999:
                r -= 20
                print("    LOST SIGHT! -20")
            elif delta > 0:
                r += min(20, delta)
                if delta >= 10:
                    print("    Good step forward +{0}".format(min(20, delta)))
            elif delta < 0:
                r += max(-8, delta)
            else:
                r -= 2

            # Bonus for completing a full gait cycle
            if s in [0, 1] and ns == 2:
                r += 10
                print("    FULL GAIT CYCLE! +10")

            # Big reward for reaching the goal
            if new_d < 85:
                r += 50
                goal_reached = True
                print("    GOAL REACHED! +50")

            total_reward += r
            Q[s][a] += ALPHA * (r + GAMMA * max(Q[ns]) - Q[s][a])

            if goal_reached:
                print("\nSUCCESS IN {0} STEPS! Total Reward = {1}".format(steps, total_reward))
                await light_matrix.write("WIN")
                sound.play(3000, 1500)
                break

            s = ns
            old_dist = new_d

        # Episode summary
        distance_covered = start_dist - new_d
        print("Distance Covered: {0} mm".format(distance_covered))

        csv_data.append([ep, total_reward, steps, round(epsilon, 5)])
        epsilon = max(EPSILON_END, epsilon * EPSILON_DECAY)

        await print_q_table(Q, "Q-TABLE AFTER EPISODE {0}".format(ep))
        await light_matrix.write(str(ep % 10))

    # Final results
    print("\n" + "="*80)
    print(" TRAINING COMPLETE – SEEDED VERSION ".center(80))
    print("Episode,Reward,Cycles,Epsilon")
    for row in csv_data:
        print("{0},{1},{2},{3}".format(row[0], row[1], row[2], row[3]))

    await light_matrix.write("E3")
    for f in [1000, 1500, 2000, 2500, 3000]:
        sound.play(f, 400)
        await runloop.sleep_ms(400)

runloop.run(main())