"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 6]

# EVOLVE-BLOCK-START
import time
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 140  # safety margin before 150s limit
    
    n, m, p = DIMENSIONS
    # A: n x m = 2x3, B: m x p = 3x6, C: n x p = 2x6
    # U: (n*m) x rank = 6 x rank
    # V: (m*p) x rank = 18 x rank  
    # W: (p*n) x rank = 12 x rank
    # Convention: W[k*n+i, r]
    
    rank = 33
    U = [[0] * rank for _ in range(n * m)]   # 6 x 33
    V = [[0] * rank for _ in range(m * p)]   # 18 x 33
    W = [[0] * rank for _ in range(p * n)]   # 12 x 33
    
    # Define the 11 terms for a 2x3x2 block
    # A (2x3): flat index i*3+j, i in 0..1, j in 0..2
    # B (3x2): flat index j*2+k, j in 0..2, k in 0..1
    # C (2x2): flat index i*2+k, i in 0..1, k in 0..1
    
    # Strassen terms for the 2x2 x 2x2 sub-problem (j=0,1 columns of B)
    # A_L = A[:, 0:2] (2x2), B_T = B[0:2, :] (2x2)
    # A[0,0]=0, A[0,1]=1, A[1,0]=3, A[1,1]=4
    # B[0,0]=0, B[0,1]=1, B[1,0]=2, B[1,1]=3
    
    terms = []
    
    # M1 = (A00+A11)(B00+B11)
    terms.append({
        'u': {0: 1, 4: 1},
        'v': {0: 1, 3: 1},
        'w': {0: 1, 3: 1}
    })
    # M2 = (A10+A11)B00
    terms.append({
        'u': {3: 1, 4: 1},
        'v': {0: 1},
        'w': {2: 1, 3: -1}
    })
    # M3 = A00(B01-B11)
    terms.append({
        'u': {0: 1},
        'v': {1: 1, 3: -1},
        'w': {1: 1, 3: 1}
    })
    # M4 = A11(B10-B00)
    terms.append({
        'u': {4: 1},
        'v': {2: 1, 0: -1},
        'w': {0: 1, 2: 1}
    })
    # M5 = (A00+A01)B11
    terms.append({
        'u': {0: 1, 1: 1},
        'v': {3: 1},
        'w': {0: -1, 1: 1}
    })
    # M6 = (A10-A00)(B00+B01)
    terms.append({
        'u': {3: 1, 0: -1},
        'v': {0: 1, 1: 1},
        'w': {3: 1}
    })
    # M7 = (A01-A11)(B10+B11)
    terms.append({
        'u': {1: 1, 4: -1},
        'v': {2: 1, 3: 1},
        'w': {0: 1}
    })
    
    # Scalar terms for the j=2 column/row (outer product A[:,2] * B[2,:])
    # A[0,2]=2, A[1,2]=5
    # B[2,0]=4, B[2,1]=5
    
    # K1 = A02 * B20 -> C00
    terms.append({
        'u': {2: 1},
        'v': {4: 1},
        'w': {0: 1}
    })
    # K2 = A02 * B21 -> C01
    terms.append({
        'u': {2: 1},
        'v': {5: 1},
        'w': {1: 1}
    })
    # K3 = A12 * B20 -> C10
    terms.append({
        'u': {5: 1},
        'v': {4: 1},
        'w': {2: 1}
    })
    # K4 = A12 * B21 -> C11
    terms.append({
        'u': {5: 1},
        'v': {5: 1},
        'w': {3: 1}
    })
    
    # Fill the matrices for 3 blocks
    for block in range(3):
        for t_idx, term in enumerate(terms):
            col = block * 11 + t_idx
            
            # U: index is i*3+j (0..5)
            for idx, val in term['u'].items():
                U[idx][col] = val
            
            # V: local B index j*2+k, global is j*6 + (block*2 + k)
            for l_idx, val in term['v'].items():
                j_local = l_idx // 2
                k_local = l_idx % 2
                g_idx = j_local * p + (block * 2 + k_local)
                V[g_idx][col] = val
            
            # W: local C index i*2+k, global is (block*2+k)*n + i
            for l_idx, val in term['w'].items():
                i_local = l_idx // 2
                k_local = l_idx % 2
                g_idx = (block * 2 + k_local) * n + i_local
                W[g_idx][col] = val
    
    # Verify the decomposition with random test vectors
    if time.time() < deadline - 5:
        np.random.seed(42)
        A_test = np.random.randn(n, m)
        B_test = np.random.randn(m, p)
        C_expected = A_test @ B_test
        
        U_arr = np.array(U, dtype=float)
        V_arr = np.array(V, dtype=float)
        W_arr = np.array(W, dtype=float)
        
        # Compute C from decomposition
        # C[i,k] = sum_r W[k*n+i, r] * (sum_j U[i*m+j, r] * A[i,j]) * (sum_j V[j*p+k, r] * B[j,k])
        # Actually the bilinear form is:
        # For each term r: U_r (vector of length n*m) dotted with A flattened,
        #                   V_r (vector of length m*p) dotted with B flattened,
        #                   W_r (vector of length p*n) selects which C entry
        # C[i,k] = sum_r W[k*n+i, r] * (U_r . A_flat) * (V_r . B_flat)
        
        A_flat = A_test.flatten()  # length n*m = 6
        B_flat = B_test.flatten()  # length m*p = 18
        
        C_computed = np.zeros((n, p))
        for r in range(rank):
            u_val = np.dot(U_arr[:, r], A_flat)
            v_val = np.dot(V_arr[:, r], B_flat)
            for k in range(p):
                for i in range(n):
                    C_computed[i, k] += W_arr[k * n + i, r] * u_val * v_val
        
        # Check if decomposition is correct
        if not np.allclose(C_computed, C_expected, atol=1e-9):
            # If verification fails, fall back to naive rank-36
            rank_naive = n * m * p
            U_naive = [[0] * rank_naive for _ in range(n * m)]
            V_naive = [[0] * rank_naive for _ in range(m * p)]
            W_naive = [[0] * rank_naive for _ in range(p * n)]
            
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        r = i * m * p + j * p + k
                        U_naive[i * m + j][r] = 1
                        V_naive[j * p + k][r] = 1
                        W_naive[k * n + i][r] = 1
            
            return {"U": U_naive, "V": V_naive, "W": W_naive}
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
