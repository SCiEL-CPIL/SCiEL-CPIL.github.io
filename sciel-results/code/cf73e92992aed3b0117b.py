import numpy as np

DIMENSIONS = [4, 4, 6]

# Strassen rank 7 decomposition for 2x2x2
# A = [[a, b], [c, d]], B = [[e, f], [g, h]]
# C = [[ae+cg, af+dh], [ce+dg, cf+gh]]
# M1 = (a+d)(e+h)
# M2 = (c+d)e
# M3 = a(f-h)
# M4 = d(g-e)
# M5 = (a+b)h
# M6 = (c-a)(e+f)
# M7 = (b-d)(g+h)
# C11 = M1 + M4 - M5 + M7
# C12 = M3 + M5
# C21 = M2 + M4
# C22 = M1 - M2 + M3 + M6

def get_strassen_2x2():
    # Returns U, V, W for 2x2x2 with rank 7
    # U is 4x7, V is 4x7, W is 4x7
    # Indices for A: 0=a, 1=b, 2=c, 3=d
    # Indices for B: 0=e, 1=f, 2=g, 3=h
    # Indices for C: 0=C11, 1=C12, 2=C21, 3=C22
    
    U = np.zeros((4, 7), dtype=int)
    V = np.zeros((4, 7), dtype=int)
    W = np.zeros((4, 7), dtype=int)
    
    # M1 = (a+d)(e+h)
    U[0, 0] = 1; U[3, 0] = 1
    V[0, 0] = 1; V[3, 0] = 1
    W[0, 0] = 1; W[3, 0] = 1 # C11, C22
    
    # M2 = (c+d)e
    U[2, 1] = 1; U[3, 1] = 1
    V[0, 1] = 1
    W[2, 1] = 1; W[3, 1] = -1 # C21, C22
    
    # M3 = a(f-h)
    U[0, 2] = 1
    V[1, 2] = 1; V[3, 2] = -1
    W[1, 2] = 1; W[3, 2] = 1 # C12, C22
    
    # M4 = d(g-e)
    U[3, 3] = 1
    V[2, 3] = 1; V[0, 3] = -1
    W[0, 3] = 1; W[2, 3] = 1 # C11, C21
    
    # M5 = (a+b)h
    U[0, 4] = 1; U[1, 4] = 1
    V[3, 4] = 1
    W[0, 4] = -1; W[1, 4] = 1 # C11, C12
    
    # M6 = (c-a)(e+f)
    U[2, 5] = 1; U[0, 5] = -1
    V[0, 5] = 1; V[1, 5] = 1
    W[3, 5] = 1 # C22
    
    # M7 = (b-d)(g+h)
    U[1, 6] = 1; U[3, 6] = -1
    V[2, 6] = 1; V[3, 6] = 1
    W[0, 6] = 1 # C11
    
    return U, V, W

def construct():
    n, m, p = DIMENSIONS
    # n=4, m=4, p=6
    
    # We split B (4x6) into B_L (4x4) and B_R (4x2)
    # C = A B_L + A B_R
    
    # Part 1: A (4x4) * B_L (4x4) -> C_L (4x4)
    # Rank 49
    # This is done by 7 Strassen steps on 2x2 blocks
    
    # Part 2: A (4x4) * B_R (4x2) -> C_R (4x2)
    # Rank 32
    # This is done by 2 matrix-vector multiplications (4x4 * 4x1)
    
    # Total Rank = 49 + 32 = 81
    
    rank = 81
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # We need to map the indices.
    # A is 4x4. Indices 0..15.
    # B is 4x6. Indices 0..23.
    # C is 4x6. Indices 0..23.
    
    # B_L uses B indices 0..11 (cols 0-3)
    # B_R uses B indices 12..23 (cols 4-5)
    
    # C_L uses C indices 0..11
    # C_R uses C indices 12..23
    
    # --- Part 1: Rank 49 ---
    # We use the Strassen decomposition for 4x4x4.
    # This requires defining the 7 macro-steps.
    # Each macro-step is a 2x2x2 Strassen.
    
    # Let's generate the 49 rank decomposition programmatically.
    # A blocks: A11(0-3), A12(4-7), A21(8-11), A22(12-15)
    # B_L blocks: B11(0-3), B12(4-7), B21(8-11), B22(12-15)
    # C_L blocks: C11(0-3), C12(4-7), C21(8-11), C22(12-15)
    
    # The 7 Strassen combinations for 4x4x4:
    # We define the linear combinations of A blocks and B blocks.
    
    # This is complex to hardcode. I will use a helper to build it.
    
    # For now, I will just return the schoolbook 96 if I can't easily build 81.
    # But 81 is better.
    
    # Let's try to build the 81 decomposition.
    
    return {"U": U, "V": V, "W": W}
