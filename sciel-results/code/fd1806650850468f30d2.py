import math
import numpy as np
import time
from scipy.optimize import linprog, minimize, minimize_scalar

N = 11
APOTHEM = math.sqrt(3) / 2.0
START_TIME = time.time()
TIME_LIMIT = 150.0
SAFETY_MARGIN = 5.0
EPS = 1e-9

def construct():
    if time.time() - START_TIME > TIME_LIMIT - SAFETY_MARGIN:
        return [[0.0, 0.0, 0.0]], [0.0, 0.0], 1.0, 0.0

    # Define candidate point sets (axial coordinates)
    candidates = []

    # Ring + 4 outer
    candidates.append([
        (0,0),(1,0),(0,1),(-1,1),(-1,0),(0,-1),(1,-1),
        (1,1),(-1,2),(-2,1),(2,-1)
    ])
    # Ring + 4 axial
    candidates.append([
        (0,0),(1,0),(0,1),(-1,1),(-1,0),(0,-1),(1,-1),
        (2,0),(-2,0),(0,2),(0,-2)
    ])
    # 3x3 + 2
    candidates.append([
        (0,0),(1,0),(2,0),(0,1),(1,1),(2,1),(0,2),(1,2),(2,2),
        (-1,1),(3,1)
    ])
    # 4x2 + 3
    candidates.append([
        (0,0),(1,0),(2,0),(3,0),(0,1),(1,1),(2,1),(3,1),(0,2),(1,2),(2,2)
    ])
    # 2x3 + 3
    candidates.append([
        (0,0),(1,0),(0,1),(1,1),(2,1),(0,2),(1,2),(2,2),(0,3),(1,3),(2,3)
    ])
    # 3x3 + 2 bottom
    candidates.append([
        (0,0),(1,0),(2,0),(0,1),(1,1),(2,1),(0,2),(1,2),(2,2),(0,3),(1,3)
    ])
    # Cross
    candidates.append([
        (0,0),(1,0),(-1,0),(0,1),(0,-1),(1,1),(-1,1),(2,0),(-2,0),(0,2),(0,-2)
    ])
    # 5+6
    candidates.append([
        (0,0),(1,0),(2,0),(3,0),(4,0),(0,1),(1,1),(2,1),(3,1),(4,1),(5,1)
    ])
    # Ring + 4 diagonal
    candidates.append([
        (0,0),(1,0),(0,1),(-1,1),(-1,0),(0,-1),(1,-1),(2,1),(-2,2),(-1,3),(1,-2)
    ])
    # 3x3 + 2 top
    candidates.append([
        (0,0),(1,0),(2,0),(0,1),(1,1),(2,1),(0,2),(1,2),(2,2),(1,3),(2,3)
    ])
    # 2x4 + 1
    candidates.append([
        (0,0),(1,0),(0,1),(1,1),(2,1),(3,1),(0,2),(1,2),(2,2),(3,2),(1,3)
    ])
    # 4x1 + 2+2+1
    candidates.append([
        (0,0),(1,0),(2,0),(3,0),(0,1),(1,1),(0,2),(1,2),(0,3),(1,3),(0,4)
    ])
    # Ring + 4 mixed
    candidates.append([
        (0,0),(1,0),(0,1),(-1,1),(-1,0),(0,-1),(1,-1),(2,0),(-2,0),(0,2),(1,2)
    ])
    # Ring + 4 alt
    candidates.append([
        (0,0),(1,0),(0,1),(-1,1),(-1,0),(0,-1),(1,-1),(2,0),(-2,0),(1,1),(-1,-1)
    ])
    # 3x2 + 3x1
    candidates.append([
        (0,0),(1,0),(2,0),(1,1),(2,1),(0,2),(1,2),(2,2),(-1,1),(3,1),(0,-1)
    ])
    # 2x2 + 4x1
    candidates.append([
        (0,0),(1,0),(0,1),(1,1),(2,1),(1,2),(2,2),(0,3),(1,3),(2,3),(3,2)
    ])
    # Ring + 4 top
    candidates.append([
        (0,0),(1,0),(0,1),(-1,1),(-1,0),(0,-1),(1,-1),(2,1),(1,2),(-1,2),(0,3)
    ])
    # 4x1 + 3x1 + 3x1
    candidates.append([
        (0,0),(1,0),(2,0),(3,0),(0,1),(1,1),(2,1),(1,2),(2,2),(3,2)
    ])
    # 2x2 + 4x1 + 1
    candidates.append([
        (0,0),(1,0),(0,1),(1,1),(2,1),(1,2),(2,2),(0,3),(1,3),(2,3),(3,3)
    ])
    # Ring + 4
    candidates.append([
        (0,0),(1,0),(0,1),(-1,1),(-1,0),(0,-1),(1,-1),(2,0),(-2,0),(1,1),(-1,1)
    ])
    # 4x2 + 3x1
    candidates.append([
        (0,0),(1,0),(2,0),(3,0),(0,1),(1,1),(2,1),(3,1),(1,2),(2,2),(3,2)
    ])
    # 3x2 + 4x1 + 1
    candidates.append([
        (0,0),(1,0),(2,0),(0,1),(1,1),(2,1),(3,1),(0,2),(1,2),(2,2),(1,3)
    ])
    # Ring + 4 (axial outer)
    candidates.append([
        (0,0),(1,0),(0,1),(-1,1),(-1,0),(0,-1),(1,-1),(2,0),(-2,0),(0,2),(0,-2)
    ])
    # Compact: 3x3 + 2 staggered
    candidates.append([
        (0,0),(1,0),(2,0),(0,1),(1,1),(2,1),(0,2),(1,2),(2,2),
        (-1,1),(3,1)
    ])
    # Compact: ring + 4 in 2x2 block
    candidates.append([
        (0,0),(1,0),(0,1),(-1,1),(-1,0),(0,-1),(1,-1),
        (1,1),(2,1),(2,0),(1,-1)
    ])
    # Compact: 2x3 + 2x2 offset
    candidates.append([
        (0,0),(1,0),(0,1),(1,1),(2,1),(0,2),(1,2),(2,2),
        (3,0),(3,1),(3,2)
    ])
    # Compact: 3x2 + 3x1 + 2
    candidates.append([
        (0,0),(1,0),(2,0),(0,1),(1,1),(2,1),(0,2),(1,2),(2,2),
        (3,0),(3,1),(3,2)
    ])
    # 2x2 + 3x1 + 2x1
    candidates.append([
        (0,0),(1,0),(0,1),(1,1),(0,2),(1,2),(2,2),(0,3),(1,3),(0,4)
    ])
    # Compact: 3x2 + 3x1 + 2x1
    candidates.append([
        (0,0),(1,0),(2,0),(0,1),(1,1),(2,1),(0,2),(1,2),(2,2),
        (0,3),(1,3)
    ])
    # Compact: 2x3 + 2x2 + 1
    candidates.append([
        (0,0),(1,0),(0,1),(1,1),(2,1),(0,2),(1,2),(2,2),
        (3,0),(3,1)
    ])
    # 3x2 + 3x1
    candidates.append([
        (0,0),(1,0),(2,0),(0,1),(1,1),(0,2),(1,2),(2,2),(0,3),(1,3),(2,3)
    ])
    # Compact: 4x1 + 3x1 + 2x1 + 2x1
    candidates.append([
        (0,0),(1,0),(2,0),(3,0),(0,1),(1,1),(2,1),
        (0,2),(1,2),(2,2)
    ])
    # Compact: 3x3 + 2 in L
    candidates.append([
        (0,0),(1,0),(2,0),(0,1),(1,1),(2,1),(0,2),(1,2),(2,2),
        (3,0),(0,3)
    ])
    # Compact: 4x2 + 3 in L-shape
    candidates.append([
        (0,0),(1,0),(2,0),(3,0),(0,1),(1,1),(2,1),(3,1),
        (0,2),(1,2),(2,2)
    ])
    # Compact: 3x3 + 2 in corners
    candidates.append([
        (0,0),(1,0),(2,0),(0,1),(1,1),(2,1),(0,2),(1,2),(2,2),
        (-1,2),(3,0)
    ])
    # Compact: 2x4 + 3
    candidates.append([
        (0,0),(1,0),(0,1),(1,1),(2,1),(3,1),(0,2),(1,2),(2,2),(3,2),
        (0,3),(1,3)
    ])
    # 4x2 + 2x1
    candidates.append([
        (0,0),(1,0),(2,0),(3,0),(1,1),(2,1),(3,1),(0,2),(1,2),(2,2),(3,2),(1,3),(2,3)
    ])
    # Compact: 3x3 + 2 in corners (other)
    candidates.append([
        (0,0),(1,0),(2,0),(0,1),(1,1),(2,1),(0,2),(1,2),(2,2),
        (-1,1),(2,3)
    ])
    # Compact: 2x3 + 2x2 + 1 (other)
    candidates.append([
        (0,0),(1,0),(0,1),(1,1),(2,1),(0,2),(1,2),(2,2),
        (3,1),(3,2)
    ])
    # Ring + 4
    candidates.append([
        (0,0),(1,0),(0,1),(-1,1),(-1,0),(0,-1),(1,-1),(2,0),(-2,0),(0,2),(1,-2)
    ])
    # Compact: 3x3 + 2 bottom staggered
    candidates.append([
        (0,0),(1,0),(2,0),(0,1),(1,1),(2,1),(0,2),(1,2),(2,2),
        (0,-1),(1,-1)
    ])
    # Compact: 2x4 + 3
    candidates.append([
        (0,0),(1,0),(0,1),(1,1),(2,1),(3,1),(0,2),(1,2),(2,2),(3,2),
        (1,3),(2,3)
    ])
    # 2x2 + 3x1 + 2x1
    candidates.append([
        (0,0),(1,0),(0,1),(1,1),(2,1),(0,2),(1,2),(0,3),(1,3),(2,3),(1,4),(2,4)
    ])
    # Compact: 3x3 + 2 in V
    candidates.append([
        (0,0),(1,0),(2,0),(0,1),(1,1),(2,1),(0,2),(1,2),(2,2),
        (-1,2),(3,2)
    ])
    # Compact: 4x2 + 3 (other)
    candidates.append([
        (0,0),(1,0),(2,0),(3,0),(0,1),(1,1),(2,1),(3,1),
        (0,2),(1,2),(3,2)
    ])
    # Compact: 5x1 + 3x1 + 3x1 staggered
    candidates.append([
        (0,0),(1,0),(2,0),(3,0),(4,0),
        (1,1),(2,1),(3,1),
        (0,2),(1,2),(2,2)
    ])
    # Compact: 4x1 + 4x1 + 3x1
    candidates.append([
        (0,0),(1,0),(2,0),(3,0),
        (0,1),(1,1),(2,1),(3,1),
        (0,2),(1,2),(2,2)
    ])
    # Compact: 3x3 + 2 diagonal
    candidates.append([
        (0,0),(1,0),(2,0),(0,1),(1,1),(2,1),(0,2),(1,2),(2,2),
        (3,1),(1,3)
    ])

    # New diverse candidates
    # Full hex ring of 6 + center + 4 more = 11 (different outer)
    candidates.append([
        (0,0),(1,0),(0,1),(-1,1),(-1,0),(0,-1),(1,-1),
        (2,0),(0,2),(-2,2),(2,-2)
    ])
    # Compact: 3x3 + 2 in X
    candidates.append([
        (0,0),(1,0),(2,0),(0,1),(1,1),(2,1),(0,2),(1,2),(2,2),
        (-1,1),(3,1)
    ])
    # Compact: 2x3 + 2x2 + 1 (other)
    candidates.append([
        (0,0),(1,0),(0,1),(1,1),(2,1),(0,2),(1,2),(2,2),
        (3,0),(3,2)
    ])
    # 3x3 block + 2 offset
    candidates.append([
        (0,0),(1,0),(2,0),(0,1),(1,1),(2,1),(0,2),(1,2),(2,2),
        (1,3),(2,3)
    ])
    # Compact: 3x3 + 2 in Y
    candidates.append([
        (0,0),(1,0),(2,0),(0,1),(1,1),(2,1),(0,2),(1,2),(2,2),
        (0,3),(2,3)
    ])
    # Compact: 2x4 + 3 (other)
    candidates.append([
        (0,0),(1,0),(0,1),(1,1),(2,1),(3,1),(0,2),(1,2),(2,2),(3,2),
        (1,3),(2,3)
    ])
    # 2x3 + 2x2
    candidates.append([
        (0,0),(1,0),(0,1),(1,1),(2,1),(0,2),(1,2),(2,2),(3,0),(3,1)
    ])
    # Compact: 3x3 + 2 in Z
    candidates.append([
        (0,0),(1,0),(2,0),(0,1),(1,1),(2,1),(0,2),(1,2),(2,2),
        (-1,0),(3,0)
    ])
    # Compact: 4x2 + 3 (other)
    candidates.append([
        (0,0),(1,0),(2,0),(3,0),(0,1),(1,1),(2,1),(3,1),
        (1,2),(2,2),(3,2)
    ])
    # Elongated 5+4
    candidates.append([
        (0,0),(1,0),(2,0),(3,0),(4,0),(0,1),(1,1),(2,1),(3,1),(1,2),(2,2)
    ])
    # Compact: 2x2 + 2x2 + 2x2 + 1
    candidates.append([
        (0,0),(1,0),(0,1),(1,1),(2,0),(3,0),(2,1),(3,1),(4,0),(5,0),(4,1)
    ])
    # Compact: 3x3 + 2 side
    candidates.append([
        (0,0),(1,0),(2,0),(0,1),(1,1),(2,1),(0,2),(1,2),(2,2),
        (3,0),(3,1)
    ])
    # 3x3 + 2 diagonal
    candidates.append([
        (0,0),(1,0),(2,0),(0,1),(1,1),(2,1),(0,2),(1,2),(2,2),
        (-1,0),(3,2)
    ])
    # Compact: 3x3 + 2 in V (other)
    candidates.append([
        (0,0),(1,0),(2,0),(0,1),(1,1),(2,1),(0,2),(1,2),(2,2),
        (-1,2),(3,2)
    ])
    # Compact: 4x2 + 3 (other)
    candidates.append([
        (0,0),(1,0),(2,0),(3,0),(0,1),(1,1),(2,1),(3,1),
        (0,2),(2,2),(1,2)
    ])
    # Compact: 3x3 + 2 in W (other)
    candidates.append([
        (0,0),(1,0),(2,0),(0,1),(1,1),(2,1),(0,2),(1,2),(2,2),
        (0,3),(2,-1)
    ])
    # Compact: 2x3 + 2x2 + 1 (other)
    candidates.append([
        (0,0),(1,0),(0,1),(1,1),(2,1),(0,2),(1,2),(2,2),
        (3,1),(3,0)
    ])
    # Compact: 3x3 + 2 in X (other)
    candidates.append([
        (0,0),(1,0),(2,0),(0,1),(1,1),(2,1),(0,2),(1,2),(2,2),
        (-1,2),(3,2)
    ])
    # Compact: 4x2 + 3 (other)
    candidates.append([
        (0,0),(1,0),(2,0),(3,0),(0,1),(1,1),(2,1),(3,1),
        (1,2),(2,2),(0,2)
    ])
    # Compact: 3x3 + 2 in Y (other)
    candidates.append([
        (0,0),(1,0),(2,0),(0,1),(1,1),(2,1),(0,2),(1,2),(2,2),
        (0,3),(2,3)
    ])
    # Compact: 2x3 + 2x2 + 1 (other)
    candidates.append([
        (0,0),(1,0),(0,1),(1,1),(2,1),(0,2),(1,2),(2,2),
        (3,2),(3,1)
    ])
    # Compact: 3x3 + 2 in Z (other)
    candidates.append([
        (0,0),(1,0),(2,0),(0,1),(1,1),(2,1),(0,2),(1,2),(2,2),
        (-1,1),(3,0)
    ])
    # Compact: 4x2 + 3 (other)
    candidates.append([
        (0,0),(1,0),(2,0),(3,0),(0,1),(1,1),(2,1),(3,1),
        (0,2),(1,2),(3,2)
    ])
    # Compact: 3x3 + 2 in A
    candidates.append([
        (0,0),(1,0),(2,0),(0,1),(1,1),(2,1),(0,2),(1,2),(2,2),
        (1,3),(1,-1)
    ])
    # Compact: 2x3 + 2x2 + 1 (other)
    candidates.append([
        (0,0),(1,0),(0,1),(1,1),(2,1),(0,2),(1,2),(2,2),
        (3,0),(3,1)
    ])
    # Compact: 3x3 + 2 in B
    candidates.append([
        (0,0),(1,0),(2,0),(0,1),(1,1),(2,1),(0,2),(1,2),(2,2),
        (3,2),(3,1)
    ])
    # Compact: 4x2 + 3 (other)
    candidates.append([
        (0,0),(1,0),(2,0),(3,0),(0,1),(1,1),(2,1),(3,1),
        (2,2),(3,2),(1,2)
    ])
    # Compact: 3x3 + 2 in C
    candidates.append([
        (0,0),(1,0),(2,0),(0,1),(1,1),(2,1),(0,2),(1,2),(2,2),
        (-1,2),(-1,1)
    ])
    # Compact: 2x3 + 2x2 + 1 (other)
    candidates.append([
        (0,0),(1,0),(0,1),(1,1),(2,1),(0,2),(1,2),(2,2),
        (3,1),(3,2)
    ])
    # Compact: 3x3 + 2 in D
    candidates.append([
        (0,0),(1,0),(2,0),(0,1),(1,1),(2,1),(0,2),(1,2),(2,2),
        (3,1),(3,0)
    ])
    # Compact: 4x2 + 3 (other)
    candidates.append([
        (0,0),(1,0),(2,0),(3,0),(0,1),(1,1),(2,1),(3,1),
        (0,2),(1,2),(2,2)
    ])
    # Compact: 3x3 + 2 in E
    candidates.append([
        (0,0),(1,0),(2,0),(0,1),(1,1),(2,1),(0,2),(1,2),(2,2),
        (-1,0),(-1,1)
    ])
    # Compact: 2x3 + 2x2 + 1 (other)
    candidates.append([
        (0,0),(1,0),(0,1),(1,1),(2,1),(0,2),(1,2),(2,2),
        (3,2),(3,0)
    ])
    # Compact: 3x3 + 2 in F
    candidates.append([
        (0,0),(1,0),(2,0),(0,1),(1,1),(2,1),(0,2),(1,2),(2,2),
        (0,-1),(2,-1)
    ])
    # Compact: 4x2 + 3 (other)
    candidates.append([
        (0,0),(1,0),(2,0),(3,0),(0,1),(1,1),(2,1),(3,1),
        (1,2),(3,2),(0,2)
    ])
    # Compact: 3x3 + 2 in G
    candidates.append([
        (0,0),(1,0),(2,0),(0,1),(1,1),(2,1),(0,2),(1,2),(2,2),
        (3,0),(3,1)
    ])
    # Compact: 2x3 + 2x2 + 1 (other)
    candidates.append([
        (0,0),(1,0),(0,1),(1,1),(2,1),(0,2),(1,2),(2,2),
        (3,0),(3,2)
    ])
    # Compact: 3x3 + 2 in H
    candidates.append([
        (0,0),(1,0),(2,0),(0,1),(1,1),(2,1),(0,2),(1,2),(2,2),
        (-1,1),(3,2)
    ])
    # Compact: 4x2 + 3 (other)
    candidates.append([
        (0,0),(1,0),(2,0),(3,0),(0,1),(1,1),(2,1),(3,1),
        (0,2),(2,2),(3,2)
    ])
    # Compact: 3x3 + 2 in I
    candidates.append([
        (0,0),(1,0),(2,0),(0,1),(1,1),(2,1),(0,2),(1,2),(2,2),
        (1,-1),(1,3)
    ])
    # Compact: 2x3 + 2x2 + 1 (other)
    candidates.append([
        (0,0),(1,0),(0,1),(1,1),(2,1),(0,2),(1,2),(2,2),
        (3,1),(3,0)
    ])
    # Compact: 3x3 + 2 in J
    candidates.append([
        (0,0),(1,0),(2,0),(0,1),(1,1),(2,1),(0,2),(1,2),(2,2),
        (0,3),(2,-1)
    ])
    # Compact: 4x2 + 3 (other)
    candidates.append([
        (0,0),(1,0),(2,0),(3,0),(0,1),(1,1),(2,1),(3,1),
        (1,2),(2,2),(3,2)
    ])
    # Compact: 3x3 + 2 in K
    candidates.append([
        (0,0),(1,0),(2,0),(0,1),(1,1),(2,1),(0,2),(1,2),(2,2),
        (-1,2),(3,1)
    ])
    # Compact: 2x3 + 2x2 + 1 (other)
    candidates.append([
        (0,0),(1,0),(0,1),(1,1),(2,1),(0,2),(1,2),(2,2),
        (3,2),(3,1)
    ])
    # Compact: 3x3 + 2 in L
    candidates.append([
        (0,0),(1,0),(2,0),(0,1),(1,1),(2,1),(0,2),(1,2),(2,2),
        (3,1),(3,2)
    ])
    # Compact: 4x2 + 3 (other)
    candidates.append([
        (0,0),(1,0),(2,0),(3,0),(0,1),(1,1),(2,1),(3,1),
        (0,2),(1,2),(2,2)
    ])
    # Compact: 3x3 + 2 in M
    candidates.append([
        (0,0),(1,0),(2,0),(0,1),(1,1),(2,1),(0,2),(1,2),(2,2),
        (0,-1),(2,3)
    ])
    # Compact: 2x3 + 2x2 + 1 (other)
    candidates.append([
        (0,0),(1,0),(0,1),(1,1),(2,1),(0,2),(1,2),(2,2),
        (3,0),(3,2)
    ])
    # Compact: 3x3 + 2 in N
    candidates.append([
        (0,0),(1,0),(2,0),(0,1),(1,1),(2,1),(0,2),(1,2),(2,2),
        (-1,0),(3,2)
    ])
    # Compact: 4x2 + 3 (other)
    candidates.append([
        (0,0),(1,0),(2,0),(3,0),(0,1),(1,1),(2,1),(3,1),
        (2,2),(3,2),(0,2)
    ])
    # Compact: 3x3 + 2 in O
    candidates.append([
        (0,0),(1,0),(2,0),(0,1),(1,1),(2,1),(0,2),(1,2),(2,2),
        (-1,1),(3,1)
    ])
    # Compact: 2x3 + 2x2 + 1 (other)
    candidates.append([
        (0,0),(1,0),(0,1),(1,1),(2,1),(0,2),(1,2),(2,2),
        (3,1),(3,2)
    ])
    # Compact: 3x3 + 2 in P
    candidates.append([
        (0,0),(1,0),(2,0),(0,1),(1,1),(2,1),(0,2),(1,2),(2,2),
        (2,-1),(2,3)
    ])
    # Compact: 4x2 + 3 (other)
    candidates.append([
        (0,0),(1,0),(2,0),(3,0),(0,1),(1,1),(2,1),(3,1),
        (0,2),(1,2),(3,2)
    ])
    # Compact: 3x3 + 2 in Q
    candidates.append([
        (0,0),(1,0),(2,0),(0,1),(1,1),(2,1),(0,2),(1,2),(2,2),
        (-1,2),(3,0)
    ])
    # Compact: 2x3 + 2x2 + 1 (other)
    candidates.append([
        (0,0),(1,0),(0,1),(1,1),(2,1),(0,2),(1,2),(2,2),
        (3,0),(3,1)
    ])
    # Compact: 3x3 + 2 in R
    candidates.append([
        (0,0),(1,0),(2,0),(0,1),(1,1),(2,1),(0,2),(1,2),(2,2),
        (3,0),(3,2)
    ])
    # Compact: 4x2 + 3 (other)
    candidates.append([
        (0,0),(1,0),(2,0),(3,0),(0,1),(1,1),(2,1),(3,1),
        (1,2),(2,2),(0,2)
    ])
    # Compact: 3x3 + 2 in S
    candidates.append([
        (0,0),(1,0),(2,0),(0,1),(1,1),(2,1),(0,2),(1,2),(2,2),
        (1,-1),(1,3)
    ])
    # Compact: 2x3 + 2x2 + 1 (other)
    candidates.append([
        (0,0),(1,0),(0,1),(1,1),(2,1),(0,2),(1,2),(2,2),
        (3,2),(3,0)
    ])
    # Compact: 3x3 + 2 in U
    candidates.append([
        (0,0),(1,0),(2,0),(0,1),(1,1),(2,1),(0,2),(1,2),(2,2),
        (0,3),(2,3)
    ])
    # Compact: 4x2 + 3 (other)
    candidates.append([
        (0,0),(1,0),(2,0),(3,0),(0,1),(1,1),(2,1),(3,1),
        (0,2),(2,2),(3,2)
    ])
    # Compact: 3x3 + 2 in W
    candidates.append([
        (0,0),(1,0),(2,0),(0,1),(1,1),(2,1),(0,2),(1,2),(2,2),
        (0,3),(2,-1)
    ])
    # Compact: 2x3 + 2x2 + 1 (other)
    candidates.append([
        (0,0),(1,0),(0,1),(1,1),(2,1),(0,2),(1,2),(2,2),
        (3,1),(3,0)
    ])
    # Compact: 2x3 + 2x2 + 1 offset
    candidates.append([
        (0,0),(1,0),(0,1),(1,1),(2,1),(0,2),(1,2),(2,2),
        (3,1),(3,2),(3,0)
    ])
    # Compact: 4x2 + 3 staggered
    candidates.append([
        (0,0),(1,0),(2,0),(3,0),(0,1),(1,1),(2,1),(3,1),
        (1,2),(2,2),(3,2)
    ])
    # Compact: 2x3 + 2x2 + 1
    candidates.append([
        (0,0),(1,0),(0,1),(1,1),(2,1),(0,2),(1,2),(2,2),
        (3,0),(3,1),(3,2)
    ])
    # Compact: 3x2 + 4x1
    candidates.append([
        (0,0),(1,0),(2,0),(0,1),(1,1),(2,1),(0,2),(1,2),(2,2),
        (3,0),(3,1),(3,2)
    ])

    def axial_to_cartesian(axial_list):
        pts = []
        for q, r in axial_list:
            x = math.sqrt(3) * (q + r / 2.0)
            y = 1.5 * r
            pts.append((x, y))
        return np.array(pts)

    def exact_S_for_beta(pts, beta):
        """Fast exact computation: find optimal center and D for given beta via LP."""
        if time.time() - START_TIME > TIME_LIMIT - SAFETY_MARGIN:
            return float('inf'), None, None
        normals = np.array([
            [math.cos(beta + math.pi / 6.0 + k * math.pi / 3.0),
             math.sin(beta + math.pi / 6.0 + k * math.pi / 3.0)]
            for k in range(6)
        ])
        projs = normals @ pts.T
        max_projs = projs.max(axis=1)
        min_projs = projs.min(axis=1)
        
        c_obj = np.array([0.0, 0.0, 1.0])
        A_ub = []
        b_ub = []
        for k in range(6):
            A_ub.append([-normals[k,0], -normals[k,1], -1.0])
            b_ub.append(-max_projs[k])
            A_ub.append([normals[k,0], normals[k,1], -1.0])
            b_ub.append(min_projs[k])
        
        A_ub = np.array(A_ub)
        b_ub = np.array(b_ub)
        bounds = [(None, None), (None, None), (0, None)]
        
        try:
            res = linprog(c_obj, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
            if res.status == 0:
                D = res.x[2]
                cx = res.x[0]
                cy = res.x[1]
                S = (D + APOTHEM) / APOTHEM
                return S, (cx, cy), D
            else:
                return float('inf'), None, None
        except Exception:
            return float('inf'), None, None

    def exact_S_direct(pts, beta, cx, cy):
        """Direct evaluation without LP."""
        normals = np.array([
            [math.cos(beta + math.pi / 6.0 + k * math.pi / 3.0),
             math.sin(beta + math.pi / 6.0 + k * math.pi / 3.0)]
            for k in range(6)
        ])
        centered = pts - np.array([cx, cy])
        projs = normals @ centered.T  # (6, n)
        max_proj = np.max(np.abs(projs))
        S = (max_proj + APOTHEM) / APOTHEM
        return S

    def smooth_calc_S(pts, beta, cx, cy, temperature=0.1):
        """Smooth surrogate: soft-max of |proj| with exact max as floor."""
        normals = np.array([
            [math.cos(beta + math.pi / 6.0 + k * math.pi / 3.0),
             math.sin(beta + math.pi / 6.0 + k * math.pi / 3.0)]
            for k in range(6)
        ])
        pts_centered = pts - np.array([cx, cy])
        projs = pts_centered @ normals.T  # (n, 6)
        abs_projs = np.abs(projs)  # (n, 6)
        flat = abs_projs.ravel()
        exact_max = float(np.max(flat))
        if exact_max > 100:
            return float('inf')
        # Soft-max: temperature * log(sum(exp((v - exact_max)/temperature))) + exact_max
        # This is always >= exact_max and approaches it as temperature -> 0
        shifted = (flat - exact_max) / temperature
        shifted = np.clip(shifted, -50, 50)
        log_sum = np.log(np.sum(np.exp(shifted)))
        D = exact_max + temperature * log_sum
        S = (D + APOTHEM) / APOTHEM
        return S

    def optimize_shape(pts):
        best_S = float('inf')
        best_beta = 0.0
        best_center = None
        
        if time.time() - START_TIME > TIME_LIMIT - SAFETY_MARGIN:
            return float('inf'), 0.0, None
        
        # Coarse grid search over beta (36 intervals)
        for i in range(37):
            beta = i * (math.pi / 6) / 36
            S, center, D = exact_S_for_beta(pts, beta)
            if S < best_S:
                best_S = S
                best_beta = beta
                best_center = center
        
        # Refine beta
        try:
            if time.time() - START_TIME < TIME_LIMIT - SAFETY_MARGIN:
                res = minimize_scalar(lambda b: exact_S_for_beta(pts, b)[0], 
                                     bounds=(0, math.pi / 6), method='bounded',
                                     options={'maxiter': 200, 'xatol': 1e-12})
                if res.fun < best_S:
                    best_S = res.fun
                    best_beta = res.x
                    _, best_center, _ = exact_S_for_beta(pts, best_beta)
        except Exception:
            pass
        
        # Direct Nelder-Mead on exact objective (cx, cy, beta)
        try:
            if time.time() - START_TIME < TIME_LIMIT - SAFETY_MARGIN:
                def objective(params):
                    cx, cy, beta = params
                    beta = ((beta % (math.pi/6)) + math.pi/6) % (math.pi/6)
                    return exact_S_direct(pts, beta, cx, cy)
                
                starts = [
                    [best_center[0], best_center[1], best_beta],
                    [best_center[0], best_center[1], best_beta + 0.05],
                    [best_center[0], best_center[1], best_beta - 0.05],
                    [best_center[0] + 0.1, best_center[1], best_beta],
                    [best_center[0] - 0.1, best_center[1], best_beta],
                    [best_center[0], best_center[1] + 0.1, best_beta],
                    [best_center[0], best_center[1] - 0.1, best_beta],
                ]
                
                for init in starts:
                    if time.time() - START_TIME > TIME_LIMIT - SAFETY_MARGIN:
                        break
                    res = minimize(objective, init, method='Nelder-Mead',
                                 options={'maxiter': 3000, 'xatol': 1e-12, 'fatol': 1e-12})
                    if res.fun < best_S:
                        best_S = res.fun
                        best_center = (res.x[0], res.x[1])
                        best_beta = ((res.x[2] % (math.pi/6)) + math.pi/6) % (math.pi/6)
        except Exception:
            pass
        
        # Smooth surrogate refinement
        try:
            if time.time() - START_TIME < TIME_LIMIT - SAFETY_MARGIN:
                temperatures = [0.2, 0.1, 0.05, 0.02]
                for temp in temperatures:
                    if time.time() - START_TIME > TIME_LIMIT - SAFETY_MARGIN:
                        break
                    def smooth_obj(params, t=temp):
                        cx, cy, beta = params
                        beta = ((beta % (math.pi/6)) + math.pi/6) % (math.pi/6)
                        return smooth_calc_S(pts, beta, cx, cy, temperature=t)
                    
                    res = minimize(smooth_obj, [best_center[0], best_center[1], best_beta],
                                 method='Nelder-Mead',
                                 options={'maxiter': 2000, 'xatol': 1e-11, 'fatol': 1e-11})
                    if res.fun < 1e10:
                        cx_r, cy_r, beta_r = res.x
                        beta_r = ((beta_r % (math.pi/6)) + math.pi/6) % (math.pi/6)
                        S_exact = exact_S_direct(pts, beta_r, cx_r, cy_r)
                        if S_exact < best_S:
                            best_S = S_exact
                            best_center = (cx_r, cy_r)
                            best_beta = beta_r
        except Exception:
            pass
        
        # Final high-precision polish: coordinate descent on center + fine beta sweep
        try:
            if time.time() - START_TIME < TIME_LIMIT - SAFETY_MARGIN:
                # Fine beta sweep around best_beta
                for db in [-0.02, -0.01, -0.005, 0.0, 0.005, 0.01, 0.02]:
                    if time.time() - START_TIME > TIME_LIMIT - SAFETY_MARGIN:
                        break
                    b_try = ((best_beta + db) % (math.pi/6) + math.pi/6) % (math.pi/6)
                    S_try, c_try, _ = exact_S_for_beta(pts, b_try)
                    if S_try < best_S:
                        best_S = S_try
                        best_beta = b_try
                        best_center = c_try
                
                # Coordinate descent on (cx, cy) for fixed beta
                # For fixed beta, the problem is: minimize max_k |proj_k| where proj_k = n_k . (p - c)
                # This is a convex problem in (cx, cy). Use iterative refinement.
                for _cd_iter in range(50):
                    if time.time() - START_TIME > TIME_LIMIT - SAFETY_MARGIN:
                        break
                    S_cd, c_cd, D_cd = exact_S_for_beta(pts, best_beta)
                    if S_cd >= best_S - 1e-15:
                        break
                    best_S = S_cd
                    best_center = c_cd
                    # Try small perturbations
                    improved = False
                    for dx, dy in [(0.01,0),(-0.01,0),(0,0.01),(0,-0.01),(0.005,0.005),(-0.005,-0.005),(0.005,-0.005),(-0.005,0.005)]:
                        cx_p = best_center[0] + dx
                        cy_p = best_center[1] + dy
                        S_p = exact_S_direct(pts, best_beta, cx_p, cy_p)
                        if S_p < best_S - 1e-12:
                            best_S = S_p
                            best_center = (cx_p, cy_p)
                            improved = True
                    if not improved:
                        break
                
                # Nelder-Mead polish on all three variables
                def objective(params):
                    cx, cy, beta = params
                    beta = ((beta % (math.pi/6)) + math.pi/6) % (math.pi/6)
                    return exact_S_direct(pts, beta, cx, cy)
                
                for init in [[best_center[0], best_center[1], best_beta],
                             [best_center[0], best_center[1], best_beta + 0.005],
                             [best_center[0], best_center[1], best_beta - 0.005]]:
                    if time.time() - START_TIME > TIME_LIMIT - SAFETY_MARGIN:
                        break
                    res = minimize(objective, init, method='Nelder-Mead',
                                 options={'maxiter': 20000, 'xatol': 1e-15, 'fatol': 1e-15})
                    if res.fun < best_S:
                        best_S = res.fun
                        best_center = (res.x[0], res.x[1])
                        best_beta = ((res.x[2] % (math.pi/6)) + math.pi/6) % (math.pi/6)
        except Exception:
            pass
            
        return best_S, best_beta, best_center

    best_global_S = float('inf')
    best_global_pts = None
    best_global_beta = 0.0
    best_global_center = None

    for axial in candidates:
        if time.time() - START_TIME > TIME_LIMIT - SAFETY_MARGIN:
            break
        pts = axial_to_cartesian(axial)
        S, beta, center = optimize_shape(pts)
        if S < best_global_S:
            best_global_S = S
            best_global_pts = pts
            best_global_beta = beta
            best_global_center = center

    if best_global_center is None or best_global_S == float('inf'):
        best_global_pts = axial_to_cartesian(candidates[0])
        best_global_center = (float(np.mean(best_global_pts[:, 0])), float(np.mean(best_global_pts[:, 1])))
        best_global_beta = 0.0
        max_proj = 0
        for k in range(6):
            ang = math.pi / 6.0 + k * math.pi / 3.0
            nx, ny = math.cos(ang), math.sin(ang)
            projs = best_global_pts[:, 0] * nx + best_global_pts[:, 1] * ny
            max_proj = max(max_proj, float(np.max(projs)), float(-np.min(projs)))
        best_global_S = (max_proj + APOTHEM) / APOTHEM

    cx, cy = best_global_center
    angle_deg = math.degrees(best_global_beta)
    
    SCALE = 1.0 + 1e-12
    pts_scaled = []
    for x, y in best_global_pts:
        sx = cx + (x - cx) * SCALE
        sy = cy + (y - cy) * SCALE
        pts_scaled.append((sx, sy))
    pts_scaled = np.array(pts_scaled)
    
    max_proj = 0
    for k in range(6):
        ang = best_global_beta + math.pi / 6.0 + k * math.pi / 3.0
        nx, ny = math.cos(ang), math.sin(ang)
        projs = (pts_scaled[:, 0] - cx) * nx + (pts_scaled[:, 1] - cy) * ny
        max_proj = max(max_proj, float(np.max(projs)), float(-np.min(projs)))
    
    D_scaled = max_proj
    S_scaled = (D_scaled + APOTHEM) / APOTHEM
    
    inner = []
    for x, y in pts_scaled:
        inner.append([float(x), float(y), float(angle_deg)])
        
    center = [float(cx), float(cy)]
    side = float(S_scaled)
    rotation = float(angle_deg)
    
    return inner, center, side, rotation
