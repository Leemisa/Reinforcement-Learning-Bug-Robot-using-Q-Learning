# ==================== EXPERIMENT 3 – WALK TO OBJECT & STOP AT ~60mm (USING .format() ONLY) ====================

from hub import port, light_matrix
import motor
import runloop
import random

# =================================== HARDWARE ===================================
MOTOR_SPEED = 1000
LEFT_LEG_MOTOR= port.A
RIGHT_LEG_MOTOR = port.B
BODY_TILT_MOTOR = port.C
DISTANCE_SENSOR = port.F                            # Ultrasonic sensor facing forward

LEG_MIDDLE    = 0
LEG_FORWARD    = 50
LEG_BACKWARD= -50
BODY_UP        = 140
BODY_DOWN    = 0

# =================================== PARAMETERS ===================================
NUM_EPISODES= 40
MAX_STEPS    = 45
LEARNING_RATE= 0.55
DISCOUNT    = 0.9
EXPLORATION    = 0.8

# =================================== STATES ===================================
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

actions = ["A.Lfwd", "C.Rup", "A.Lmid", "B.Rfwd", "C.Lup", "B.Rmid"]

# =================================== ACTION FUNCTIONS ===================================
async def a_lfwd(): await motor.run_to_absolute_position(LEFT_LEG_MOTOR, 50, MOTOR_SPEED)
async def c_rup():await motor.run_to_absolute_position(BODY_TILT_MOTOR, BODY_DOWN, MOTOR_SPEED)
async def a_lmid(): await motor.run_to_absolute_position(LEFT_LEG_MOTOR, 0, MOTOR_SPEED)
async def b_rfwd(): await motor.run_to_absolute_position(RIGHT_LEG_MOTOR, -50, MOTOR_SPEED)
async def c_lup():await motor.run_to_absolute_position(BODY_TILT_MOTOR, BODY_UP, MOTOR_SPEED)
async def b_rmid(): await motor.run_to_absolute_position(RIGHT_LEG_MOTOR, 0, MOTOR_SPEED)

action_functions = [a_lfwd, c_rup, a_lmid, b_rfwd, c_lup, b_rmid]

# =================================== Q-TABLE & STATS ===================================
Q = [[0.0] * 6 for _ in range(8)]
episode_stats = []

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

    if l_mid and r_mid and body_up:    return 0
    if l_fwd and r_mid and body_up:    return 1
    if l_fwd and r_mid and not body_up: return 2
    if l_mid and r_mid and not body_up: return 3
    if l_mid and r_fwd and not body_up: return 4
    if l_mid and r_fwd and body_up:    return 5
    if l_mid and r_mid and body_up:    return 6
    return 7

# =================================== DISTANCE SENSOR ===================================
def get_distance_mm():
    try:
        d = DISTANCE_SENSOR.distance()
        return d if d is not None else 999
    except:
        return 999

def get_distance_reward(d):
    if d < 30:        return -100.0
    if 50 <= d <= 70:return +120.0
    if 70 < d <= 120:return +30.0
    if 120 < d <= 250:return +5.0
    return -15.0

# =================================== SHOW EPISODE ON MATRIX ===================================
def show_episode_number(ep):
    if ep <= 9:
        light_matrix.write(str(ep))
    else:
        tens = ep // 10
        ones = ep % 10
        light_matrix.write(str(tens))
        runloop.sleep_ms(700)
        light_matrix.write(str(ones))

# =================================== PRINT FULL Q-TABLE ===================================
def print_full_q_table(episode_num):
    print("\n" + "="*120)
    print("EPISODE {} COMPLETE – FULL Q-TABLE".format(episode_num))
    print("="*120)
    print("State            |A.LfwdC.RupA.LmidB.RfwdC.LupB.Rmid| Best Action")
    print("-"*120)
    for i in range(8):
        row = "".join("{:8.3f}".format(Q[i][j]) for j in range(6))
        best_act = actions[Q[i].index(max(Q[i]))]
        print("{:19} | {}→{}".format(states[i], row, best_act))
    print("-"*120)

# =================================== TRAINING ===================================
async def train():
    global EXPLORATION

    await motor.run_to_absolute_position(LEFT_LEG_MOTOR, 0, MOTOR_SPEED)
    await motor.run_to_absolute_position(RIGHT_LEG_MOTOR, 0, MOTOR_SPEED)
    await motor.run_to_absolute_position(BODY_TILT_MOTOR, 0, MOTOR_SPEED)
    await light_matrix.write("E3")
    await runloop.sleep_ms(1000)
    print_full_q_table(0)

    for episode in range(1, NUM_EPISODES + 1):
        await motor.run_to_absolute_position(LEFT_LEG_MOTOR, 0, MOTOR_SPEED)
        await motor.run_to_absolute_position(RIGHT_LEG_MOTOR, 0, MOTOR_SPEED)
        await motor.run_to_absolute_position(BODY_TILT_MOTOR, 0, MOTOR_SPEED)
        await runloop.sleep_ms(1000)

        show_episode_number(episode)

        state = 0
        total_reward = 0.0
        cycles = 0
        final_dist = 999

        for _ in range(MAX_STEPS):
            if random.random() < EXPLORATION:
                a = random.randint(0, 5)
            else:
                a = Q[state].index(max(Q[state]))

            await action_functions[a]()
            await runloop.sleep_ms(380)

            next_state = get_current_state()
            dist = get_distance_mm()
            final_dist = dist

            gait_reward = 2.0 if a in [0, 3] else 0.0
            if state == 0 and next_state == 6:
                gait_reward += 15.0
                cycles += 1

            dist_reward = get_distance_reward(dist)
            reward = gait_reward + dist_reward
            total_reward += reward

            td = reward + DISCOUNT * max(Q[next_state]) - Q[state][a]
            Q[state][a] += LEARNING_RATE * td

            state = next_state

            if 50 <= dist <= 70:
                total_reward += 80
                break

        success = "YES" if 50 <= final_dist <= 70 else "NO "
        episode_stats.append((episode, round(total_reward,1), cycles, final_dist, success))
        EXPLORATION = max(0.1, EXPLORATION * 0.95)

        print_full_q_table(episode)
        print(">>> EPISODE {:2d} | Reward: {:+6.1f} | Cycles: {:2d} | Dist: {:4d}mm | SUCCESS: {}".format(
            episode, total_reward, cycles, final_dist, success))

    await light_matrix.write("OK")

# =================================== FINAL WALK ===================================
async def walk_to_object():
    print("\n" + "="*120)
    print("FINAL LEARNED POLICY – APPROACH & STOP AT 60mm")
    print("="*120)
    for i in range(7):
        best = actions[Q[i].index(max(Q[i]))]
        print("{:19} → {}".format(states[i], best))
    print("="*120)

    print("\nCSV DATA:")
    print("Episode,Reward,Cycles,Final_Distance_mm,Success")
    for ep, r, c, d, s in episode_stats:
        print("{},{},{},{},{}".format(ep, r, c, d, s))
    print("="*120)

    print("\nAPPROACHING OBJECT...")
    await light_matrix.write("GO")
    while True:
        s = get_current_state()
        if s == 7: s = 0
        a = Q[s].index(max(Q[s]))
        await action_functions[a]()
        await runloop.sleep_ms(330)

        dist = get_distance_mm()
        print("Distance: {} mm".format(dist if dist < 1000 else ">1000"))

        if 50 <= dist <= 70:
            print("SUCCESS! Stopped safely at {} mm".format(dist))
            await light_matrix.write("OK")
            break
        if dist < 30:
            print("COLLISION at {} mm!".format(dist))
            await light_matrix.write("NO")
            break

# =================================== MAIN ===================================
async def main():
    await train()
    await walk_to_object()

runloop.run(main())