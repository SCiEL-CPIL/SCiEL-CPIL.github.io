"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time

DIMENSIONS = [3, 4, 8]

# EVOLVE-BLOCK-START
def construct():
    """
    Constructs a rank-minimal (or near-minimal) exact tensor decomposition for 3x4 * 4x8.
    
    Strategy:
    1. Schoolbook fallback: Rank 96.
    2. Winograd-like block decomposition: Rank 88.
       - Split B (4x8) into four 4x2 blocks.
       - For each 3x4 * 4x2 subproblem:
         * Split A (3x4) into A1 (2x4) and A2 (1x4).
         * C1 = A1 * B_block (2x2): Use Strassen on two 2x2*2x2 products (rank 14).
         * C2 = A2 * B_block (1x2): Schoolbook (rank 8).
         * Total per subproblem: rank 22.
       - Total: 4 * 22 = 88.
    3. Verify and return the best valid decomposition.
    """
    n, m, p = DIMENSIONS  # 3, 4, 8
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin
    
    # --- 1. Schoolbook Fallback (Rank 96) ---
    rank_sb = n * m * p
    U_sb = [[0] * rank_sb for _ in range(n * m)]
    V_sb = [[0] * rank_sb for _ in range(m * p)]
    W_sb = [[0] * rank_sb for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i * m + j][col] = 1
                V_sb[j * p + k][col] = 1
                W_sb[k * n + i][col] = 1
                col += 1
    
    best_U, best_V, best_W = U_sb, V_sb, W_sb
    best_rank = rank_sb
    
    # --- 2. Winograd-like Block Decomposition (Rank 88) ---
    # Strassen factors for 2x2 * 2x2
    U_s = [
        [1, 0, 0, 1],
        [0, 0, 1, 1],
        [1, -1, 0, 0],
        [0, 0, 0, 1],
        [1, 0, 1, 0],
        [0, 1, 0, -1],
        [1, 0, -1, 0]
    ]
    V_s = [
        [1, 0, 0, 1],
        [1, 0, 0, 0],
        [0, 0, 1, 1],
        [0, 1, 0, -1],
        [0, 0, 0, 1],
        [1, 0, 1, 0],
        [0, 1, 0, 0]
    ]
    W_s = [
        [1, 0, 0, 1],
        [0, 0, 1, -1],
        [0, 1, 0, 1],
        [1, 0, 1, 0],
        [-1, 1, 0, 0],
        [0, 0, 0, 1],
        [1, 0, 0, 0]
    ]
    
    # Global dimensions
    nA = n * m  # 12
    nB = m * p  # 32
    nC = p * n  # 24
    
    rank_w = 88
    U_w = [[0] * rank_w for _ in range(nA)]
    V_w = [[0] * rank_w for _ in range(nB)]
    W_w = [[0] * rank_w for _ in range(nC)]
    
    col_offset = 0
    
    # Split B (4x8) into four 4x2 blocks: B[:, 0:2], B[:, 2:4], B[:, 4:6], B[:, 6:8]
    # For each block, solve 3x4 * 4x2
    
    for block_idx in range(4):
        # B block columns: block_idx*2 and block_idx*2+1
        b_col_base = block_idx * 2
        
        # --- C1 = A1 * B_block (2x2) using Strassen ---
        # A1 is first 2 rows of A (2x4)
        # Split A1 into A11 (cols 0-1) and A12 (cols 2-3)
        # Split B_block into B_top (rows 0-1) and B_bot (rows 2-3)
        
        # A11 global indices (A rows 0-1, cols 0-1):
        # A is 3x4, row-major: A[i,j] at index i*4+j
        # A11: (0,0)=0, (0,1)=1, (1,0)=4, (1,1)=5
        A11_idx = [0, 1, 4, 5]
        
        # A12 global indices (A rows 0-1, cols 2-3):
        # A12: (0,2)=2, (0,3)=3, (1,2)=6, (1,3)=7
        A12_idx = [2, 3, 6, 7]
        
        # B_block is 4x2, rows 0-3, cols b_col_base, b_col_base+1
        # B_top: rows 0-1, B_bot: rows 2-3
        # B is 4x8, row-major: B[i,j] at index i*8+j
        # B_top: (0, b_col_base)=b_col_base, (0, b_col_base+1)=b_col_base+1,
        #        (1, b_col_base)=8+b_col_base, (1, b_col_base+1)=8+b_col_base+1
        B_top_idx = [b_col_base, b_col_base + 1, 8 + b_col_base, 8 + b_col_base + 1]
        
        # B_bot: (2, b_col_base)=16+b_col_base, (2, b_col_base+1)=16+b_col_base+1,
        #        (3, b_col_base)=24+b_col_base, (3, b_col_base+1)=24+b_col_base+1
        B_bot_idx = [16 + b_col_base, 16 + b_col_base + 1, 24 + b_col_base, 24 + b_col_base + 1]
        
        # C1 is 2x2, part of C (3x8)
        # C rows 0-1, cols b_col_base, b_col_base+1
        # C is 3x8, but W uses trace convention W[k*n+i, r]
        # C[i,j] corresponds to W[j*n + i, r] = W[j*3 + i, r]
        # C1: (0, b_col_base), (0, b_col_base+1), (1, b_col_base), (1, b_col_base+1)
        # W indices:
        # C[0, b_col_base] -> W[b_col_base*3 + 0, r] = W[3*b_col_base, r]
        # C[0, b_col_base+1] -> W[(b_col_base+1)*3 + 0, r] = W[3*b_col_base+3, r]
        # C[1, b_col_base] -> W[b_col_base*3 + 1, r] = W[3*b_col_base+1, r]
        # C[1, b_col_base+1] -> W[(b_col_base+1)*3 + 1, r] = W[3*b_col_base+4, r]
        
        C1_idx = [
            3 * b_col_base,        # C[0, b_col_base]
            3 * (b_col_base + 1),  # C[0, b_col_base+1]
            3 * b_col_base + 1,    # C[1, b_col_base]
            3 * (b_col_base + 1) + 1  # C[1, b_col_base+1]
        ]
        
        # C1 = A11 * B_top + A12 * B_bot
        # First product: A11 * B_top (2x2 * 2x2)
        for k in range(7):
            for local_idx, val in enumerate(U_s[k]):
                if val != 0:
                    U_w[A11_idx[local_idx]][col_offset] += val
            for local_idx, val in enumerate(V_s[k]):
                if val != 0:
                    V_w[B_top_idx[local_idx]][col_offset] += val
            for local_idx, val in enumerate(W_s[k]):
                if val != 0:
                    W_w[C1_idx[local_idx]][col_offset] += val
            col_offset += 1
        
        # Second product: A12 * B_bot (2x2 * 2x2)
        for k in range(7):
            for local_idx, val in enumerate(U_s[k]):
                if val != 0:
                    U_w[A12_idx[local_idx]][col_offset] += val
            for local_idx, val in enumerate(V_s[k]):
                if val != 0:
                    V_w[B_bot_idx[local_idx]][col_offset] += val
            for local_idx, val in enumerate(W_s[k]):
                if val != 0:
                    W_w[C1_idx[local_idx]][col_offset] += val
            col_offset += 1
        
        # --- C2 = A2 * B_block (1x2) using schoolbook ---
        # A2 is row 2 of A (1x4): indices 8, 9, 10, 11
        A2_idx = [8, 9, 10, 11]
        
        # C2: C[2, b_col_base] and C[2, b_col_base+1]
        # W indices:
        # C[2, b_col_base] -> W[b_col_base*3 + 2, r] = W[3*b_col_base+2, r]
        # C[2, b_col_base+1] -> W[(b_col_base+1)*3 + 2, r] = W[3*b_col_base+5, r]
        
        C2_idx = [
            3 * b_col_base + 2,      # C[2, b_col_base]
            3 * (b_col_base + 1) + 2  # C[2, b_col_base+1]
        ]
        
        # C2[0] = A2[0]*B[0, b_col_base] + A2[1]*B[1, b_col_base] + A2[2]*B[2, b_col_base] + A2[3]*B[3, b_col_base]
        # C2[1] = A2[0]*B[0, b_col_base+1] + A2[1]*B[1, b_col_base+1] + A2[2]*B[2, b_col_base+1] + A2[3]*B[3, b_col_base+1]
        
        # 8 schoolbook terms
        for i in range(4):  # A2 row index (0-3)
            for j in range(2):  # B block col index (0-1)
                U_w[A2_idx[i]][col_offset] = 1
                V_w[(i * 8) + b_col_base + j][col_offset] = 1
                W_w[C2_idx[j]][col_offset] = 1
                col_offset += 1
    
    # Verify the Winograd decomposition
    # Check that col_offset == 88
    if col_offset != rank_w:
        # Fall back to schoolbook
        return {"U": best_U, "V": best_V, "W": best_W}
    
    # Simple verification: check a few random cases
    import numpy as np
    np.random.seed(42)
    A_test = np.random.randint(-10, 10, size=(3, 4))
    B_test = np.random.randint(-10, 10, size=(4, 8))
    C_expected = A_test @ B_test
    
    # Compute C from factors
    C_computed = np.zeros((3, 8), dtype=int)
    for r in range(rank_w):
        u = np.array([U_w[i * 4 + j][r] for i in range(3) for j in range(4)])
        v = np.array([V_w[i * 8 + j][r] for i in range(4) for j in range(8)])
        w = np.array([W_w[j * 3 + i][r] for i in range(3) for j in range(8)])
        # u is flattened A, v is flattened B, w is flattened C
        # C += outer of (u * A_flat) and (v * B_flat) projected by w
        # Actually: C_computed += (sum_i U[i,r]*A[i]) * (sum_j V[j,r]*B[j]) * W pattern
        # More precisely: for each factor r, we compute:
        # a_r = sum_i U[i,r] * A_flat[i]
        # b_r = sum_j V[j,r] * B_flat[j]
        # c_r = a_r * b_r
        # Then C += c_r * w_r (where w_r is the W vector for factor r)
        
        A_flat = A_test.flatten()
        B_flat = B_test.flatten()
        a_r = sum(U_w[i][r] * A_flat[i] for i in range(12))
        b_r = sum(V_w[j][r] * B_flat[j] for j in range(32))
        c_r = a_r * b_r
        for i in range(3):
            for j in range(8):
                C_computed[i, j] += c_r * W_w[j * 3 + i][r]
    
    if not np.allclose(C_computed, C_expected, atol=1e-6):
        # Fall back to schoolbook
        return {"U": best_U, "V": best_V, "W": best_W}
    
    # Winograd decomposition is valid
    best_U, best_V, best_W = U_w, V_w, W_w
    best_rank = rank_w
    
    return {"U": best_U, "V": best_V, "W": best_W}
# EVOLVE-BLOCK-END
