"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 4, 6]

# EVOLVE-BLOCK-START
import time
import numpy as np
from fractions import Fraction
from itertools import product
from scipy.optimize import least_squares

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # safety margin
    
    n, m, p = DIMENSIONS  # 2, 4, 6
    
    # Function to find rank-8 decomposition of 2x2x3 matrix multiplication
    def find_rank8_223():
        # U: 4x8, V: 6x8, W: 6x8
        # T[i,j,k] = 1 for i,j,k in range
        # Tensor size: 4 * 6 * 6 = 144 entries
        # Variables: 8 * (4 + 6 + 6) = 160
        # We want sum_r U[:,r] ⊗ V[:,r] ⊗ W[:,r] = T
        
        # Initialize with random values
        np.random.seed(42)
        U_init = np.random.randn(4, 8)
        V_init = np.random.randn(6, 8)
        W_init = np.random.randn(6, 8)
        
        x0 = np.concatenate([U_init.flatten(), V_init.flatten(), W_init.flatten()])
        
        def residuals(x):
            U = x[:32].reshape(4, 8)
            V = x[32:80].reshape(6, 8)
            W = x[80:128].reshape(6, 8)
            
            # Compute tensor
            T = np.zeros((4, 6, 6))
            for r in range(8):
                T += np.einsum('i,j,k->ijk', U[:, r], V[:, r], W[:, r])
            
            # Target tensor: T[i,j,k] = 1
            # Wait, the tensor for MM is not all 1s.
            # T[i,j,k] is non-zero only if the indices correspond to a valid multiplication path?
            # No, the tensor is defined by the structure constants.
            # For M_{2,2,3}, the tensor is:
            # T_{(i,j), (j,k), (i,k)} = 1
            # Let's map indices:
            # U index: i*2 + j (0..3)
            # V index: j*3 + k (0..5)
            # W index: k*2 + i (0..5)
            
            target = np.zeros((4, 6, 6))
            for i in range(2):
                for j in range(2):
                    for k in range(3):
                        u_idx = i * 2 + j
                        v_idx = j * 3 + k
                        w_idx = k * 2 + i
                        target[u_idx, v_idx, w_idx] = 1.0
            
            return (T - target).flatten()
        
        try:
            result = least_squares(residuals, x0, method='lm', max_nfev=10000)
            if result.cost < 1e-8:
                U = result.x[:32].reshape(4, 8)
                V = result.x[32:80].reshape(6, 8)
                W = result.x[80:128].reshape(6, 8)
                # Round to nearest integer/half-integer if possible
                # Often these decompositions have integer entries
                U = np.round(U * 2) / 2
                V = np.round(V * 2) / 2
                W = np.round(W * 2) / 2
                # Verify
                T_check = np.zeros((4, 6, 6))
                for r in range(8):
                    T_check += np.einsum('i,j,k->ijk', U[:, r], V[:, r], W[:, r])
                target_check = np.zeros((4, 6, 6))
                for i in range(2):
                    for j in range(2):
                        for k in range(3):
                            u_idx = i * 2 + j
                            v_idx = j * 3 + k
                            w_idx = k * 2 + i
                            target_check[u_idx, v_idx, w_idx] = 1.0
                if np.allclose(T_check, target_check, atol=1e-5):
                    return U, V, W
        except Exception:
            pass
        return None

    # Try to get rank 32 solution
    rank8_decomp = find_rank8_223()
    
    if rank8_decomp is not None:
        U_223, V_223, W_223 = rank8_decomp
        rank = 32
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        
        # 4 blocks:
        # Block 0: A_L, B_1_T (Cols 0-2 of B, Rows 0-1) -> C Cols 0-2
        # Block 1: A_R, B_1_B (Cols 0-2 of B, Rows 2-3) -> C Cols 0-2
        # Block 2: A_L, B_2_T (Cols 3-5 of B, Rows 0-1) -> C Cols 3-5
        # Block 3: A_R, B_2_B (Cols 3-5 of B, Rows 2-3) -> C Cols 3-5
        
        # Indices in U (8x32)
        # A_L: 0,1,4,5
        # A_R: 2,3,6,7
        
        # Indices in V (24x32)
        # B_1_T: Rows 0,1; Cols 0,1,2
        #   j=0: k=0,1,2 -> 0,1,2
        #   j=1: k=0,1,2 -> 6,7,8
        # B_1_B: Rows 2,3; Cols 0,1,2
        #   j=2: k=0,1,2 -> 12,13,14
        #   j=3: k=0,1,2 -> 18,19,20
        # B_2_T: Rows 0,1; Cols 3,4,5
        #   j=0: k=3,4,5 -> 3,4,5
        #   j=1: k=3,4,5 -> 9,10,11
        # B_2_B: Rows 2,3; Cols 3,4,5
        #   j=2: k=3,4,5 -> 15,16,17
        #   j=3: k=3,4,5 -> 21,22,23
        
        # Indices in W (12x32)
        # C Cols 0-2:
        #   k=0: i=0,1 -> 0,1
        #   k=1: i=0,1 -> 2,3
        #   k=2: i=0,1 -> 4,5
        # C Cols 3-5:
        #   k=3: i=0,1 -> 6,7
        #   k=4: i=0,1 -> 8,9
        #   k=5: i=0,1 -> 10,11
        
        def add_block(u_base, v_indices, w_indices, start_col):
            # u_base: 4 indices into U
            # v_indices: 6 indices into V (j*3+k)
            # w_indices: 6 indices into W (k*2+i)
            for r in range(8):
                c = start_col + r
                # U
                for i in range(2):
                    for j in range(2):
                        u_idx = i * 2 + j
                        U[u_base[u_idx]][c] += U_223[i * 2 + j, r]
                # V
                for j in range(2):
                    for k in range(3):
                        v_idx = j * 3 + k
                        V[v_indices[v_idx]][c] += V_223[j * 3 + k, r]
                # W
                for k in range(3):
                    for i in range(2):
                        w_idx = k * 2 + i
                        W[w_indices[w_idx]][c] += W_223[k * 2 + i, r]

        # Block 0: A_L, B_1_T
        u_L = [0, 1, 4, 5]
        v_1T = [0, 1, 2, 6, 7, 8]
        w_0 = [0, 1, 2, 3, 4, 5]
        add_block(u_L, v_1T, w_0, 0)

        # Block 1: A_R, B_1_B
        u_R = [2, 3, 6, 7]
        v_1B = [12, 13, 14, 18, 19, 20]
        w_0 = [0, 1, 2, 3, 4, 5]
        add_block(u_R, v_1B, w_0, 8)

        # Block 2: A_L, B_2_T
        u_L = [0, 1, 4, 5]
        v_2T = [3, 4, 5, 9, 10, 11]
        w_1 = [6, 7, 8, 9, 10, 11]
        add_block(u_L, v_2T, w_1, 16)

        # Block 3: A_R, B_2_B
        u_R = [2, 3, 6, 7]
        v_2B = [15, 16, 17, 21, 22, 23]
        w_1 = [6, 7, 8, 9, 10, 11]
        add_block(u_R, v_2B, w_1, 24)

        # Verify
        def verify_decomp(U, V, W, rank):
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        total = 0
                        for r in range(rank):
                            total += U[i*m+j][r] * V[j*p+k][r] * W[k*n+i][r]
                        if total != 1:
                            return False
            return True

        if verify_decomp(U, V, W, rank):
            return {"U": U, "V": V, "W": W}
        
    # Fallback to Rank 42
    rank = 42
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    def add_strassen_block(a_idx, b_idx, w_idx, start_col):
        c = start_col
        U[a_idx[0]][c] += 1; U[a_idx[3]][c] += 1
        V[b_idx[0]][c] += 1; V[b_idx[3]][c] += 1
        W[w_idx[0]][c] += 1; W[w_idx[3]][c] += 1
        U[a_idx[2]][c+1] += 1; U[a_idx[3]][c+1] += 1
        V[b_idx[0]][c+1] += 1
        W[w_idx[2]][c+1] += 1; W[w_idx[3]][c+1] -= 1
        U[a_idx[0]][c+2] += 1
        V[b_idx[1]][c+2] += 1; V[b_idx[3]][c+2] -= 1
        W[w_idx[1]][c+2] += 1; W[w_idx[3]][c+2] += 1
        U[a_idx[3]][c+3] += 1
        V[b_idx[2]][c+3] += 1; V[b_idx[0]][c+3] -= 1
        W[w_idx[0]][c+3] += 1; W[w_idx[2]][c+3] += 1
        U[a_idx[0]][c+4] += 1; U[a_idx[1]][c+4] += 1
        V[b_idx[3]][c+4] += 1
        W[w_idx[0]][c+4] -= 1; W[w_idx[1]][c+4] += 1
        U[a_idx[2]][c+5] += 1; U[a_idx[0]][c+5] -= 1
        V[b_idx[0]][c+5] += 1; V[b_idx[1]][c+5] += 1
        W[w_idx[3]][c+5] += 1
        U[a_idx[1]][c+6] += 1; U[a_idx[3]][c+6] -= 1
        V[b_idx[2]][c+6] += 1; V[b_idx[3]][c+6] += 1
        W[w_idx[0]][c+6] += 1

    a_L = [0, 1, 4, 5]
    a_R = [2, 3, 6, 7]
    b_1_T = [0, 1, 6, 7]
    b_1_B = [12, 13, 18, 19]
    w_1 = [0, 2, 1, 3]
    add_strassen_block(a_L, b_1_T, w_1, 0)
    add_strassen_block(a_R, b_1_B, w_1, 7)
    
    b_2_T = [2, 3, 8, 9]
    b_2_B = [14, 15, 20, 21]
    w_2 = [4, 6, 5, 7]
    add_strassen_block(a_L, b_2_T, w_2, 14)
    add_strassen_block(a_R, b_2_B, w_2, 21)
    
    b_3_T = [4, 5, 10, 11]
    b_3_B = [16, 17, 22, 23]
    w_3 = [8, 10, 9, 11]
    add_strassen_block(a_L, b_3_T, w_3, 28)
    add_strassen_block(a_R, b_3_B, w_3, 35)
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
