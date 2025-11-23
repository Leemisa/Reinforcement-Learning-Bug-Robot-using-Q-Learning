"""
EXPERIMENT 1: Q-Learning Bug Robot Walk (Left Side Focus)
Copy and paste this entire code into LEGO Spike Prime App

Goal: Robot learns to walk efficiently using Q-learning
States: 4 states representing left leg position and side position
Actions: 4 actions (lift side, move legs forward, lower side, reset legs)
"""

import runloop, motor, distance_sensor
from hub import light_matrix, port
import random

# ============================================================================
# Q-LEARNING PARAMETERS
# ============================================================================
NUM_EPISODES = 20           # Number of learning episodes
MAX_STEPS_PER_EPISODE = 10  # Max walking cycles per episode
LEARNING_RATE = 0.3         # Alpha (how fast to learn)
DISCOUNT_FACTOR = 0.8       # Gamma (value future rewards)
EXPLORATION_RATE = 1.0      # Epsilon (start with full exploration)
EXPLORATION_DECAY = 0.9     # Decay per episode
MIN_EXPLORATION = 0.1       # Minimum exploration

# ============================================================================
# ROBOT MOTOR SETTINGS
# ============================================================================
legspeed = 720      # degrees per second
Lfwd = 30           # Left legs forward
Lmid = 0            # Left legs middle
Lup = 150           # Left side up
Level = 0           # Level position

# ============================================================================
# Q-LEARNING SETUP
# ============================================================================
# 4 States (left side position + leg position)
# State 0: Side down, legs back
# State 1: Side up, legs back
# State 2: Side up, legs forward
# State 3: Side down, legs forward

# 4 Actions
# Action 0: Lift left side
# Action 1: Move left legs forward
# Action 2: Lower left side
# Action 3: Move left legs back (reset)

STATE_SPACE = 4
ACTION_SPACE = 4

# Initialize Q-table (4 states x 4 actions)
q_table = [[0.0] * ACTION_SPACE for _ in range(STATE_SPACE)]

# Storage for learning data
episode_rewards = []
episode_steps = []
exploration_rates = []

async def main():
    global q_table, EXPLORATION_RATE
    
    # Initialize motors to zero position
    await light_matrix.write("Init")
    motor.run_to_absolute_position(port.A, 0, legspeed)  # Left legs
    motor.run_to_absolute_position(port.C, 0, legspeed)  # Side motor
    await runloop.sleep_ms(1000)
    
    await light_matrix.write("Ready")
    
    # Wait for distance sensor trigger (hand in front)
    distance = distance_sensor.distance(port.F)
    while distance > 100:
        await runloop.sleep_ms(100)
        distance = distance_sensor.distance(port.F)
    
    await light_matrix.write("Go!")
    await runloop.sleep_ms(500)
    
    # ========================================================================
    # Q-LEARNING TRAINING LOOP
    # ========================================================================
    for episode in range(NUM_EPISODES):
        await light_matrix.write(str(episode + 1))
        
        state = 0  # Start at: side down, legs back
        total_reward = 0
        steps = 0
        
        # Reset to starting position
        await motor.run_to_absolute_position(port.C, Level, legspeed)
        await motor.run_to_absolute_position(port.A, Lmid, legspeed)
        
        # Episode loop
        for step in range(MAX_STEPS_PER_EPISODE):
            # Choose action using epsilon-greedy
            if random.random() < EXPLORATION_RATE:
                action = random.randint(0, ACTION_SPACE - 1)  # Explore
            else:
                # Exploit: choose best action from Q-table
                action = max(range(ACTION_SPACE), key=lambda a: q_table[state][a])
            
            # Execute action and get reward
            next_state, reward = await execute_action(state, action)
            
            # Q-learning update
            current_q = q_table[state][action]
            max_next_q = max(q_table[next_state])
            new_q = current_q + LEARNING_RATE * (reward + DISCOUNT_FACTOR * max_next_q - current_q)
            q_table[state][action] = new_q
            
            # Update for next iteration
            total_reward += reward
            state = next_state
            steps += 1
        
        # Record episode data
        episode_rewards.append(total_reward)
        episode_steps.append(steps)
        exploration_rates.append(EXPLORATION_RATE)
        
        # Decay exploration
        EXPLORATION_RATE = max(MIN_EXPLORATION, EXPLORATION_RATE * EXPLORATION_DECAY)
    
    # ========================================================================
    # DISPLAY RESULTS
    # ========================================================================
    await light_matrix.write("Done")
    await runloop.sleep_ms(1000)
    
    # Print Q-table
    print("=" * 50)
    print("LEARNED Q-TABLE:")
    print("=" * 50)
    print("States: 0=Down/Back, 1=Up/Back, 2=Up/Fwd, 3=Down/Fwd")
    print("Actions: 0=Lift, 1=LegsFwd, 2=Lower, 3=LegsBack")
    print("-" * 50)
    for s in range(STATE_SPACE):
        print("State", s, ":", [round(q_table[s][a], 2) for a in range(ACTION_SPACE)])
    
    # Print rewards per episode
    print("\n" + "=" * 50)
    print("REWARDS PER EPISODE:")
    print("=" * 50)
    for ep, reward in enumerate(episode_rewards):
        print("Episode", ep + 1, ":", round(reward, 2))
    
    # Print summary statistics
    print("\n" + "=" * 50)
    print("LEARNING SUMMARY:")
    print("=" * 50)
    print("Total Episodes:", NUM_EPISODES)
    print("Average Reward:", round(sum(episode_rewards) / len(episode_rewards), 2))
    print("Best Reward:", round(max(episode_rewards), 2))
    print("Worst Reward:", round(min(episode_rewards), 2))
    print("First Episode Reward:", round(episode_rewards[0], 2))
    print("Last Episode Reward:", round(episode_rewards[-1], 2))
    print("Improvement:", round(episode_rewards[-1] - episode_rewards[0], 2))
    print("Final Exploration Rate:", round(EXPLORATION_RATE, 3))
    
    # Print data for graphing (CSV format)
    print("\n" + "=" * 50)
    print("DATA FOR GRAPHS (CSV):")
    print("=" * 50)
    print("Episode,Reward,Exploration")
    for i in range(len(episode_rewards)):
        print(f"{i+1},{round(episode_rewards[i], 2)},{round(exploration_rates[i], 3)}")
    
    print("\n" + "=" * 50)
    print("EXPERIMENT COMPLETE!")
    print("=" * 50)


async def execute_action(state, action):
    """
    Execute action and return next state and reward
    
    State transitions:
    State 0 (Down/Back) -> Action 0 (Lift) -> State 1 (Up/Back)
    State 1 (Up/Back)   -> Action 1 (Fwd)  -> State 2 (Up/Fwd)
    State 2 (Up/Fwd)    -> Action 2 (Lower) -> State 3 (Down/Fwd)
    State 3 (Down/Fwd)  -> Action 3 (Back) -> State 0 (Down/Back)
    """
    
    reward = 0.0
    next_state = state
    
    # Action 0: Lift left side
    if action == 0:
        await motor.run_to_absolute_position(port.C, Lup, legspeed)
        if state == 0:
            next_state = 1
            reward = 10.0  # Correct sequence
        else:
            reward = -5.0  # Wrong sequence
    
    # Action 1: Move left legs forward
    elif action == 1:
        await motor.run_to_absolute_position(port.A, Lfwd, legspeed)
        if state == 1:
            next_state = 2
            reward = 10.0  # Correct sequence
        else:
            reward = -5.0  # Wrong sequence
    
    # Action 2: Lower left side
    elif action == 2:
        await motor.run_to_absolute_position(port.C, Level, legspeed)
        if state == 2:
            next_state = 3
            reward = 15.0  # Good! This completes forward walk
        else:
            reward = -5.0  # Wrong sequence
    
    # Action 3: Move left legs back (reset)
    elif action == 3:
        await motor.run_to_absolute_position(port.A, Lmid, legspeed)
        if state == 3:
            next_state = 0
            reward = 5.0  # Reset for next cycle
        else:
            reward = -5.0  # Wrong sequence
    
    # Small delay for stability
    await runloop.sleep_ms(100)
    
    # Measure distance to reward forward movement
    distance = distance_sensor.distance(port.F)
    if distance is not None and distance < 100:
        reward += 5.0  # Bonus if moving towards something
    
    return next_state, reward


# Run the experiment
runloop.run(main())
