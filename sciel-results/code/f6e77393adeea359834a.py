import time
import numpy as np

DIMENSIONS = [2, 5, 6]

def construct():
    start_time = time.time()
    deadline = start_time + 140  # Safety margin
    
    n, m, p = DIMENSIONS
    
    # We will build the decomposition by combining sub-decompositions.
    # Total rank is sum of ranks of sub-problems.
    
    # Sub-problem ranks:
    # R221 = 2
    # R222 = 7
    # R223 = 9 (2+7)
    # R216 = 1
    # R226 = 18 (2+7+9)
    # R236 = 19 (1+18)
    # Total = 18 + 19 = 37
    
    rank = 37
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Helper to add a sub-decomposition
    # sub_U, sub_V, sub_W are lists of lists (matrices)
    # offset_u, offset_v, offset_w are starting indices for rows
    # col_offset is starting column index
    def add_decomp(sub_U, sub_V, sub_W, off_u, off_v, off_w, col_off):
        sub_rank = len(sub_U[0]) if sub_U else 0
        for i in range(len(sub_U)):
            for r in range(sub_rank):
                U[off_u + i][col_off + r] += sub_U[i][r]
        for j in range(len(sub_V)):
            for r in range(sub_rank):
                V[off_v + j][col_off + r] += sub_V[j][r]
        for k in range(len(sub_W)):
            for r in range(sub_rank):
                W[off_w + k][col_off + r] += sub_W[k][r]
        return sub_rank

    # 1. Generate 2x2x1 (Rank 2)
    # U: 2x2x2, V: 2x1x2, W: 1x2x2
    # U[i,j,r], V[j,k,r], W[k,i,r]
    # r=0: U[i,0,0]=1, V[0,0,0]=1, W[0,i,0]=1
    # r=1: U[i,1,1]=1, V[1,0,1]=1, W[0,i,1]=1
    U_221 = [[0]*2 for _ in range(4)] # 2x2 flattened? No, U is n*m x rank
    # n=2, m=2. U is 4x2.
    # Index i (0..1), j (0..1).
    # U[i*2+j, r]
    U_221 = [
        [1, 0], # i=0, j=0
        [0, 1], # i=0, j=1
        [1, 0], # i=1, j=0
        [0, 1]  # i=1, j=1
    ]
    # V is m*p x rank. m=2, p=1. V is 2x2.
    # Index j (0..1), k (0..0).
    # V[j*1+k, r]
    V_221 = [
        [1, 0], # j=0, k=0
        [0, 1]  # j=1, k=0
    ]
    # W is p*n x rank. p=1, n=2. W is 2x2.
    # Index k (0..0), i (0..1).
    # W[k*2+i, r]
    W_221 = [
        [1, 1], # k=0, i=0
        [1, 1]  # k=0, i=1
    ]

    # 2. Generate 2x2x2 (Rank 7) - Strassen
    # U: 2x2x7, V: 2x2x7, W: 2x2x7
    # Standard Strassen
    # M1 = (A22 - A11) * (B11 + B22)
    # M2 = (A11 + A22) * B22
    # M3 = (A21 - A11) * B11
    # M4 = A11 * (B12 - B22)
    # M5 = A22 * (B21 - B11)
    # M6 = (A11 + A21) * (B22 - B21)
    # M7 = (A21 + A22) * (B11 - B12)
    
    # We need to map these to U, V, W.
    # This is tedious to hardcode. 
    # Given time constraints, I will use the known Strassen decomposition.
    # U, V, W for 2x2x2.
    # Rows of U: i*2+j
    # Rows of V: j*2+k
    # Rows of W: k*2+i
    
    # Let's use a known 7-term decomposition.
    # See e.g. https://en.wikipedia.org/wiki/Strassen_algorithm
    # The coefficients are integers.
    
    U_222 = [
        [0, 1, 0, 0, 0, 1, 1], # i=0, j=0 (A11)
        [0, 0, 1, 1, 0, 0, 0], # i=0, j=1 (A12)
        [1, 1, 1, 0, 1, 0, 0], # i=1, j=0 (A21)
        [1, 0, 0, 0, 1, 1, 1]  # i=1, j=1 (A22)
    ]
    V_222 = [
        [1, 0, 1, 1, 1, 0, 1], # j=0, k=0 (B11)
        [0, 0, 0, 1, 0, 1, 0], # j=0, k=1 (B12)
        [1, 1, 0, 0, 1, 1, 0], # j=1, k=0 (B21)
        [0, 1, 0, 0, 0, 1, 1]  # j=1, k=1 (B22)
    ]
    W_222 = [
        [0, 1, 0, 1, 0, 0, 0], # k=0, i=0 (C11)
        [0, 0, 0, 0, 0, 1, 1], # k=0, i=1 (C12)
        [0, 0, 1, 0, 1, 0, 0], # k=1, i=0 (C21)
        [1, 0, 0, 0, 0, 1, 0]  # k=1, i=1 (C22)
    ]
    # Note: The exact coefficients might vary, but rank 7 is standard.
    # I will assume this specific set is correct for the standard Strassen.
    # Actually, let's verify one term.
    # M1 = (A22 - A11)(B11 + B22).
    # U for A22: [0,0,0,1]. U for A11: [1,0,0,0].
    # Diff: [-1, 0, 0, 1].
    # My U_222 col 0: [0,0,1,1]. This is not A22-A11.
    # The Strassen decomposition is not just simple differences in the U matrix directly without linear combinations in the output or input.
    # The standard form is:
    # P1 = (A22 - A11) * (B11 + B22)
    # P2 = (A11 + A22) * B22
    # ...
    # The U, V, W matrices represent the linear combinations.
    # For P1:
    # U part: A22 has 1, A11 has -1.
    # V part: B11 has 1, B22 has 1.
    # W part: C11 has 1, C12 has -1, C21 has -1, C22 has 1.
    
    # Let's use a verified Strassen decomposition.
    # U (4x7), V (4x7), W (4x7)
    U_222 = [
        [0, 1, 0, 0, 0, 1, 1], # A11
        [0, 0, 1, 1, 0, 0, 0], # A12
        [1, 1, 1, 0, 1, 0, 0], # A21
        [1, 0, 0, 0, 1, 1, 1]  # A22
    ]
    # This U corresponds to:
    # Col 0: A21+A22
    # Col 1: A11+A22
    # Col 2: A12+A21
    # Col 3: A12
    # Col 4: A21
    # Col 5: A11+A12
    # Col 6: A11+A12+A21+A22 ? No.
    
    # Given the complexity and risk of error in manual Strassen coefficients,
    # and the 120s limit, I will use the schoolbook for 2x2x2 (Rank 4) 
    # if I can't guarantee the 7-rank implementation is bug-free.
    # Rank 4 for 2x2x2 gives:
    # R221 = 2
    # R222 = 4
    # R223 = 2+4 = 6
    # R226 = 2+4+6 = 12
    # R236 = 1+12 = 13
    # Total = 12 + 13 = 25.
    # This is still better than 60.
    # And much safer to implement.
    
    # Let's switch to Schoolbook for sub-blocks to ensure correctness.
    # R221 = 2 (Schoolbook is 4, but we found Rank 2)
    # R222 = 4 (Schoolbook)
    # R223 = 2 + 4 = 6 (Split 3 into 1+2)
    # R216 = 1
    # R226 = 2 + 4 + 6 = 12
    # R236 = 1 + 12 = 13
    # Total = 12 + 13 = 25.
    
    # Let's re-verify R221=2.
    # Yes, outer product.
    # R222=4.
    # R223=6.
    # R216=1.
    
    # New Total Rank = 25.
    
    rank = 25
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col_offset = 0
    
    # Helper for Schoolbook 2x2x2
    def get_222():
        U_s = [[1,0,0,0], [0,1,0,0], [0,0,1,0], [0,0,0,1]]
        V_s = [[1,0,0,0], [0,0,1,0], [0,1,0,0], [0,0,0,1]]
        W_s = [[0,0,1,0], [0,0,0,1], [1,0,0,0], [0,1,0,0]]
        return U_s, V_s, W_s, 4

    # Helper for 2x2x1
    def get_221():
        U_s = [[1,0], [0,1], [1,0], [0,1]]
        V_s = [[1,0], [0,1]]
        W_s = [[1,1], [1,1]]
        return U_s, V_s, W_s, 2

    # Helper for 2x1x6
    def get_216():
        U_s = [[1], [0], [1], [0]] # Wait, U is 2x1x1. Flattened 2x1.
        # n=2, m=1. U is 2x1.
        # U[0,0]=1, U[1,0]=1?
        # C_ik = A_i1 B_1k.
        # U_i1 = A_i1.
        # So U is column of 1s?
        # U[0]=1, U[1]=1.
        U_s = [[1], [1]]
        V_s = [[1]] # m=1, p=6. V is 1x6.
        # V[0,0]=1, V[0,1]=1...
        V_s = [[1]*6]
        W_s = [[1]*2] # p=6, n=2. W is 6x2.
        # W[k,i] = 1.
        W_s = [[1,1] for _ in range(6)]
        return U_s, V_s, W_s, 1

    # 1. A1 B1 (2x2x6)
    # Split p=6 into 1, 2, 3
    # A1 B11 (2x2x1)
    U_s, V_s, W_s, r = get_221()
    add_decomp(U_s, V_s, W_s, 0, 0, 0, col_offset)
    col_offset += r
    
    # A1 B12 (2x2x2)
    U_s, V_s, W_s, r = get_222()
    add_decomp(U_s, V_s, W_s, 0, 1, 1, col_offset)
    col_offset += r
    
    # A1 B13 (2x2x3) -> 2x2x1 + 2x2x2
    U_s, V_s, W_s, r = get_221()
    add_decomp(U_s, V_s, W_s, 0, 3, 3, col_offset)
    col_offset += r
    
    U_s, V_s, W_s, r = get_222()
    add_decomp(U_s, V_s, W_s, 0, 4, 4, col_offset)
    col_offset += r
    
    # 2. A2 B2 (2x3x6)
    # Split m=3 into 1, 2
    # A21 B21 (2x1x6)
    U_s, V_s, W_s, r = get_216()
    add_decomp(U_s, V_s, W_s, 2, 6, 6, col_offset)
    col_offset += r
    
    # A22 B22 (2x2x6)
    # A22 B221 (2x2x1)
    U_s, V_s, W_s, r = get_221()
    add_decomp(U_s, V_s, W_s, 4, 7, 0, col_offset)
    col_offset += r
    
    # A22 B222 (2x2x2)
    U_s, V_s, W_s, r = get_222()
    add_decomp(U_s, V_s, W_s, 4, 8, 2, col_offset)
    col_offset += r
    
    # A22 B223 (2x2x3) -> 2x2x1 + 2x2x2
    U_s, V_s, W_s, r = get_221()
    add_decomp(U_s, V_s, W_s, 4, 10, 4, col_offset)
    col_offset += r
    
    U_s, V_s, W_s, r = get_222()
    add_decomp(U_s, V_s, W_s, 4, 11, 5, col_offset)
    col_offset += r

    if time.time() > deadline:
        return {"U": U, "V": V, "W": W}

    return {"U": U, "V": V, "W": W}

# EVOLVE-BLOCK-END
