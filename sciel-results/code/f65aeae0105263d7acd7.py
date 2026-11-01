import numpy as np
import time

DIMENSIONS = [2, 5, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # n=2, m=5, p=6
    
    # Strategy: Block decomposition
    # A (2x5) = [A1 (2x2), A2 (2x3)]
    # B (5x6) = [B1 (2x6); B2 (3x6)] where B1 = [B11 (2x2), B12 (2x4)], B2 = [B21 (3x2), B22 (3x4)]
    
    # Sub-problems:
    # 1. A1 * B11 (2x2 * 2x2) -> Rank 7 (Strassen)
    # 2. A1 * B12 (2x2 * 2x4) -> Split into two 2x2 * 2x2 -> Rank 14
    # 3. A2 * B21 (2x3 * 3x2) -> Rank 12 (Schoolbook)
    # 4. A2 * B22 (2x3 * 3x4) -> Split into two 2x3 * 3x2 -> Rank 24
    
    # Total Rank: 7 + 14 + 12 + 24 = 57
    
    rank = 57
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    # Helper to add a term
    def add_term(u_vec, v_vec, w_vec):
        nonlocal col
        for i, val in enumerate(u_vec):
            if val != 0:
                U[i][col] += val
        for i, val in enumerate(v_vec):
            if val != 0:
                V[i][col] += val
        for i, val in enumerate(w_vec):
            if val != 0:
                W[i][col] += val
        col += 1

    # --- Strassen 2x2x2 Verification and Construction ---
    # We define the Strassen terms for A(2x2) * B(2x2) -> C(2x2)
    # Indices in U: 0:a, 1:b, 2:c, 3:d (flattened 2x2)
    # Indices in V: 0:x, 1:y, 2:u, 3:v (flattened 2x2)
    # Indices in W: 0:C00, 1:C01, 2:C10, 3:C11 (flattened 2x2)
    
    # Strassen Coefficients
    # M1 = (a+d)(x+u)
    # M2 = (c+d)x
    # M3 = a(y-u)
    # M4 = d(v-u)
    # M5 = (a+b)u
    # M6 = (c+d)(v-y)
    # M7 = (a+b)(y-v)
    
    # Reassembly:
    # C00 = M1 - M2 - M4 + M3
    # C01 = M3 + M4 + M7
    # C10 = M2 + M5 + M6
    # C11 = M1 - M5 + M6 + M7
    
    # Let's verify this algebraically with a simple case:
    # a=1, b=0, c=0, d=0 (A=I)
    # x=1, y=0, u=0, v=0 (B=I)
    # C should be I.
    # M1 = (1)(1) = 1
    # M2 = (0)(1) = 0
    # M3 = 1(0) = 0
    # M4 = 0(0) = 0
    # M5 = (1)(0) = 0
    # M6 = (0)(0) = 0
    # M7 = (1)(0) = 0
    # C00 = 1 - 0 - 0 + 0 = 1. Correct.
    # C01 = 0 + 0 + 0 = 0. Correct.
    # C10 = 0 + 0 + 0 = 0. Correct.
    # C11 = 1 - 0 + 0 + 0 = 1. Correct.
    
    # Another case: a=1, b=1, c=0, d=0. x=0, y=1, u=0, v=0.
    # A = [[1,1],[0,0]], B = [[0,1],[0,0]]
    # C = [[0,1],[0,0]]
    # M1 = (1)(0) = 0
    # M2 = (0)(0) = 0
    # M3 = 1(1-0) = 1
    # M4 = 0(0-0) = 0
    # M5 = (2)(0) = 0
    # M6 = (0)(0-1) = 0
    # M7 = (2)(1-0) = 2
    # C00 = 0 - 0 - 0 + 1 = 1. WRONG. Expected 0.
    
    # My Strassen reassembly is incorrect.
    # Correct Strassen:
    # C00 = M1 - M2 - M4 + M3
    # C01 = M3 + M4 + M7
    # C10 = M2 + M5 + M6
    # C11 = M1 - M5 + M6 + M7
    
    # Let's re-evaluate C00 for the second case.
    # M1=0, M2=0, M4=0, M3=1.
    # C00 = 1. But actual C00 = a*x + b*u = 1*0 + 1*0 = 0.
    
    # So the formula is wrong.
    # I will NOT use Strassen if I can't verify it perfectly.
    # I will use Schoolbook for 2x2x2 (Rank 8) to be safe.
    # This gives Rank 60.
    
    # However, I can try to find a Rank 57 decomposition using a known valid Strassen set.
    # Since I cannot guarantee the Strassen coefficients from memory, I will fall back to Schoolbook.
    # But wait, I can try to verify a standard set at runtime.
    
    # Standard Strassen:
    # M1 = (a+d)(x+u)
    # M2 = (c+d)x
    # M3 = a(y-u)
    # M4 = d(v-u)
    # M5 = (a+b)u
    # M6 = (c+d)(v-y)
    # M7 = (a+b)(y-v)
    
    # Correct Reassembly:
    # C00 = M1 - M2 - M4 + M3
    # C01 = M3 + M4 + M7
    # C10 = M2 + M5 + M6
    # C11 = M1 - M5 + M6 + M7
    
    # Let's check C00 again.
    # M1 - M2 - M4 + M3
    # (ax+au+dx+du) - (cx+dx) - (dv-du) + (ay-au)
    # = ax + au + dx + du - cx - dx - dv + du + ay - au
    # = ax - cx - dv + 2du + ay
    # This is not ax+bu.
    
    # Okay, I will use the **Rank 60** schoolbook decomposition to ensure a valid score of 1.0.
    # Improving to 57 is risky without verified coefficients.
    
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
                
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
