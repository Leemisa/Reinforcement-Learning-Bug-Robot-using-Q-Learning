# ==================== EXPERIMENT 3B – SEEDED Q-TABLE + DISTANCE GOAL ====================

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
EPSILON = 0.6  # Less exploration due to good seed

states = [
    "0 Lmid Rmid Lup", "1 Lfwd Rmid Lup", "2 Lfwd Rmid Rup",
    "3 Lmid Rmid Rup", "4 Lmid Rfdw Rup", "5 Lmid Rfdw Lup",
    "6 Lmid Rmid Lup", "7 STUCK"
]
actions = ["Lup", "Rup", "Lfwd", "Lmid", "Rfwd", "Rmid"]

# YOUR SEEDED Q-TABLE (ONE 1 PER ROW)
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

episode_data = []

# Same motor functions and get_state() as 3A
# ... (copy from 3A or just run this full version below)

# FULL CODE FOR 3B (identical structure, only Q and EPSILON differ)
# → Just change the Q table and EPSILON = 0.6

# (Full code same as 3A but with the seeded Q and EPSILON=0.6)
# I’ll give you the full 3B if you want — just say "3B full"