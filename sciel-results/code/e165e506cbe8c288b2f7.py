"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 4, 8]

# EVOLVE-BLOCK-START
import time
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin
    
    n, m, p = DIMENSIONS
    # Decompose into blocks
    # n=3 -> 1, 2
    # m=4 -> 2, 2
    # p=8 -> 2, 2, 2, 2
    
    n1, n2 = 1, 2
    m1, m2 = 2, 2
    p_blocks = 4
    p_block_size = 2
    
    rank = 0
    U = []
    V = []
    W = []
    
    # Helper to add a term
    def add_term(u_idx, v_idx, w_idx, u_val, v_val, w_val):
        nonlocal rank
        if rank < len(U):
            U[rank][u_idx] += u_val
            V[rank][v_idx] += v_val
            W[rank][w_idx] += w_val
        else:
            U.append([0] * (n * m))
            V.append([0] * (m * p))
            W.append([0] * (p * n))
            U[rank][u_idx] += u_val
            V[rank][v_idx] += v_val
            W[rank][w_idx] += w_val
            rank += 1

    # Initialize lists
    # We don't know the exact rank yet, so we will build it dynamically
    # Or we can pre-calculate:
    # Row 1: 4 blocks of 1x2x2. Rank 4 each? Total 16.
    # Row 2: 4 blocks of 2x2x2. Rank 7 each? Total 28.
    # Total 44.
    
    # Let's use a fixed rank 44 and fill it.
    total_rank = 16 + 28
    U = [[0] * total_rank for _ in range(n * m)]
    V = [[0] * total_rank for _ in range(m * p)]
    W = [[0] * total_rank for _ in range(p * n)]
    
    col = 0
    
    # Row 1: n1=1, m1=2, m2=2
    # A[0:1, 0:2] and A[0:1, 2:4]
    # B[0:2, 0:2], B[0:2, 2:4], B[0:2, 4:6], B[0:2, 6:8]
    # B[2:4, 0:2], B[2:4, 2:4], B[2:4, 4:6], B[2:4, 6:8]
    
    for b in range(4):
        # Block b of p
        p_start = b * 2
        # Term 1: A[0,0] * B[0, p_start]
        U[0 * 4 + 0][col] = 1
        V[0 * 8 + p_start][col] = 1
        W[p_start * 3 + 0][col] = 1
        col += 1
        # Term 2: A[0,0] * B[0, p_start+1]
        U[0 * 4 + 0][col] = 1
        V[0 * 8 + p_start + 1][col] = 1
        W[(p_start + 1) * 3 + 0][col] = 1
        col += 1
        # Term 3: A[0,1] * B[1, p_start]
        U[0 * 4 + 1][col] = 1
        V[1 * 8 + p_start][col] = 1
        W[p_start * 3 + 0][col] = 1
        col += 1
        # Term 4: A[0,1] * B[1, p_start+1]
        U[0 * 4 + 1][col] = 1
        V[1 * 8 + p_start + 1][col] = 1
        W[(p_start + 1) * 3 + 0][col] = 1
        col += 1

    # Row 2: n2=2, m1=2, m2=2
    # A[1:3, 0:2] and A[1:3, 2:4]
    # B[0:2, 0:2], ...
    
    for b in range(4):
        p_start = b * 2
        # Strassen for 2x2x2
        # A is 2x2: A[1:3, 0:2]
        # B is 2x2: B[0:2, p_start:p_start+2]
        
        # M1 = (A11 + A22) (B11 + B22)
        # A11 is at (1,0), A22 is at (2,1)
        # B11 is at (0, p_start), B22 is at (1, p_start+1)
        
        # U for M1: A11 + A22
        U[1 * 4 + 0][col] = 1
        U[2 * 4 + 1][col] = 1
        # V for M1: B11 + B22
        V[0 * 8 + p_start][col] = 1
        V[1 * 8 + p_start + 1][col] = 1
        # W for M1: C11 + C22 - M2 - M5?
        # Standard Strassen reconstruction:
        # C11 = M1 - M4 - M5 + M7
        # C12 = M3 + M5
        # C21 = M2 + M4
        # C22 = M1 - M2 + M3 + M6
        
        # We need to map these to the global W indices.
        # C11 is at (1, p_start)
        # C12 is at (1, p_start+1)
        # C21 is at (2, p_start)
        # C22 is at (2, p_start+1)
        
        # M1 contributes to C11, C22
        W[p_start * 3 + 1][col] = 1
        W[(p_start + 1) * 3 + 2][col] = -1 # Wait, C11 = M1 - M4 - M5 + M7
        # This requires combining terms.
        # It is easier to just define the 7 products and then define the linear combinations in W?
        # No, W is part of the decomposition.
        # The decomposition is:
        # C = sum_r U_r V_r W_r^T ?
        # No, C_{i,k} = sum_r U_{i,j,r} V_{j,k,r} W_{k,i,r} ?
        # The schoolbook code sets W[k*n+i, r] = 1.
        # This implies the term is added to C_{i,k}.
        # So if we define M1 as a product, we must distribute it to C.
        
        # Let's stick to the schoolbook for the 2x2x2 blocks to ensure correctness,
        # as implementing Strassen correctly in this tensor format is error-prone.
        # Rank of 2x2x2 schoolbook is 8.
        # Total rank: 16 (row 1) + 4 * 8 (row 2) = 16 + 32 = 48.
        # Score: 96 / 48 = 2.0.
        
        # This is a significant improvement over 1.0.
        
        # Term 1: A[1,0] * B[0, p_start]
        U[1 * 4 + 0][col] = 1
        V[0 * 8 + p_start][col] = 1
        W[p_start * 3 + 1][col] = 1
        col += 1
        # Term 2: A[1,0] * B[0, p_start+1]
        U[1 * 4 + 0][col] = 1
        V[0 * 8 + p_start + 1][col] = 1
        W[(p_start + 1) * 3 + 1][col] = 1
        col += 1
        # Term 3: A[1,1] * B[1, p_start]
        U[1 * 4 + 1][col] = 1
        V[1 * 8 + p_start][col] = 1
        W[p_start * 3 + 1][col] = 1
        col += 1
        # Term 4: A[1,1] * B[1, p_start+1]
        U[1 * 4 + 1][col] = 1
        V[1 * 8 + p_start + 1][col] = 1
        W[(p_start + 1) * 3 + 1][col] = 1
        col += 1
        # Term 5: A[2,0] * B[0, p_start]
        U[2 * 4 + 0][col] = 1
        V[0 * 8 + p_start][col] = 1
        W[p_start * 3 + 2][col] = 1
        col += 1
        # Term 6: A[2,0] * B[0, p_start+1]
        U[2 * 4 + 0][col] = 1
        V[0 * 8 + p_start + 1][col] = 1
        W[(p_start + 1) * 3 + 2][col] = 1
        col += 1
        # Term 7: A[2,1] * B[1, p_start]
        U[2 * 4 + 1][col] = 1
        V[1 * 8 + p_start][col] = 1
        W[p_start * 3 + 2][col] = 1
        col += 1
        # Term 8: A[2,1] * B[1, p_start+1]
        U[2 * 4 + 1][col] = 1
        V[1 * 8 + p_start + 1][col] = 1
        W[(p_start + 1) * 3 + 2][col] = 1
        col += 1

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
