import time
import numpy as np

DIMENSIONS = [3, 4, 5]

# EVOLVE-BLOCK-START
import time
_start_time = time.time()
_DEADLINE = 150.0  # wall-clock limit

def _elapsed():
    return time.time() - _start_time

def _deadline_exceeded(margin=5.0):
    return _elapsed() > _DEADLINE - margin

def construct():
    n, m, p = DIMENSIONS  # 3, 4, 5
    
    # Block decomposition:
    # A (3x4) = [A1 (3x3), A2 (3x1)]  where A1 uses cols 0-2, A2 uses col 3
    # B (4x5) = [B1 (3x5); B2 (1x5)]  where B1 uses rows 0-2, B2 uses row 3
    # C = A1*B1 + A2*B2
    #
    # Further split B1 (3x5) = [B11 (3x3), B12 (3x2)]
    # C = A1*B11 + A1*B12 + A2*B2
    #
    # A1*B11: 3x3x3 -> Laderman rank 7
    # A1*B12: 3x3x2 -> schoolbook rank 18  
    # A2*B2:   3x1x5 -> schoolbook rank 15
    # Total: 7 + 18 + 15 = 40
    
    rank = 40
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    # --- Part 1: A1*B11 (3x3x3) using Laderman ---
    # Laderman's algorithm for 3x3 matrix multiplication
    # Let A1 = [a b c; d e f; g h i], B11 = [p q r; s t u; v w x]
    # The 7 intermediate products are:
    # m1 = (a+b)(s+t)
    # m2 = (a-b)(s-t)  
    # m3 = (d+e)(v+w)
    # m4 = (d-e)(v-w)
    # m5 = (g+h)(x+u)
    # m6 = (g-h)(x-u)
    # m7 = (a+c+p+r)*(e+f+s+t) - (b+c)*(e+f) - (a+d)*(s+t)
    # Wait, that's not right. Let me use the standard Laderman 7-term formula.
    
    # Standard Laderman for 3x3:
    # A = [a11 a12 a13; a21 a22 a23; a31 a32 a33]
    # B = [b11 b12 b13; b21 b22 b23; b31 b32 b33]
    # 
    # m1 = (a11+a12+a13+a21+a22+a23+a31+a32+a33) * (b11+b12+b13+b21+b22+b23+b31+b32+b33)
    # That's not efficient. Let me use a known rank-7 decomposition.
    
    # Actually, the minimal rank for 3x3 is known to be 19 (Bini, Capovani, Lotti, Romani), 
    # not 7. Rank 7 is for 2x2. For 3x3, the best known is 19.
    
    # Let me recalculate with correct ranks:
    # A1*B11: 3x3x3 -> best known rank 19 (but complex to implement)
    # For safety, use schoolbook: 3*3*3 = 27
    # A1*B12: 3x3x2 -> schoolbook 3*3*2 = 18
    # A2*B2:   3x1x5 -> schoolbook 3*1*5 = 15
    # Total: 27 + 18 + 15 = 60 (same as before)
    
    # Alternative: Try a different block split
    # Split m=4 as 2+2:
    # A (3x4) = [A1 (3x2), A2 (3x2)]
    # B (4x5) = [B1 (2x5); B2 (2x5)]
    # C = A1*B1 + A2*B2
    # Each: 3x2x5 -> schoolbook 3*2*5 = 30
    # Total: 30 + 30 = 60 (same)
    
    # Let me try to implement a proper Laderman-style decomposition for 3x3x3.
    # Known rank-19 algorithm exists but is complex. 
    # 
    # Instead, let's try a hybrid: use the fact that we can share some computations.
    # 
    # For now, implement the block decomposition with schoolbook for each block,
    # but optimize by noting that the blocks are independent.
    
    # Use block decomposition with schoolbook for each block
    # A (3x4) = [A1 (3x3), A2 (3x1)]
    # B (4x5) = [B1 (3x5); B2 (1x5)]
    # C = A1*B1 + A2*B2
    # B1 (3x5) = [B11 (3x3), B12 (3x2)]
    # C = A1*B11 + A1*B12 + A2*B2
    #
    # For now, use schoolbook for all blocks (rank 60) as a correct baseline
    # TODO: Replace A1*B11 with rank-19 algorithm to get rank 52
    
    rank = n * m * p
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                col = (i * m + j) * p + k
                U[i * m + j, col] = 1.0
                V[j * p + k, col] = 1.0
                W[k * n + i, col] = 1.0
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
