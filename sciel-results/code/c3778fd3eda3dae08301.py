"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 6]

# EVOLVE-BLOCK-START
import time
import numpy as np

def _build_rank33_decomposition():
    """Build a verified rank-33 decomposition for 2x3 @ 3x6 matrix multiplication.
    
    Uses 3 independent blocks of 2x3 @ 3x2, each using rank 11.
    """
    n, m, p = 2, 3, 6
    
    # Terms for a single 2x3 @ 3x2 block (rank 11)
    # Indices for U: i*m + j (0..5)
    # Indices for V: j*p + k (0..17), but local k is 0..1, so local j*p + k_local
    # Indices for W: k*n + i (0..11), but local k is 0..1, so local k_local*n + i
    
    # The following terms are a known rank-11 decomposition for 2x3 @ 3x2.
    # u: dict (i*m+j) -> val
    # v: dict (j*2+k_local) -> val  (Note: local p=2)
    # w: dict (k_local*n+i) -> val  (Note: local n=2)
    
    terms = [
        {'u': {0: 1, 4: 1}, 'v': {0: 1, 3: 1}, 'w': {0: 1, 3: 1}},
        {'u': {3: 1, 4: 1}, 'v': {0: 1}, 'w': {2: 1, 3: -1}},
        {'u': {0: 1}, 'v': {1: 1, 3: -1}, 'w': {1: 1, 3: 1}},
        {'u': {4: 1}, 'v': {2: 1, 0: -1}, 'w': {0: 1, 2: 1}},
        {'u': {0: 1, 1: 1}, 'v': {3: 1}, 'w': {0: -1, 1: 1}},
        {'u': {3: 1, 0: -1}, 'v': {0: 1, 1: 1}, 'w': {3: 1}},
        {'u': {1: 1, 4: -1}, 'v': {2: 1, 3: 1}, 'w': {0: 1}},
        {'u': {2: 1}, 'v': {4: 1}, 'w': {0: 1}},
        {'u': {2: 1}, 'v': {5: 1}, 'w': {1: 1}},
        {'u': {5: 1}, 'v': {4: 1}, 'w': {2: 1}},
        {'u': {5: 1}, 'v': {5: 1}, 'w': {3: 1}},
    ]
    
    rank = 33
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    for block in range(3):
        for t_idx, term in enumerate(terms):
            col = block * 11 + t_idx
            
            # Fill U
            for idx, val in term['u'].items():
                U[idx][col] = val
            
            # Fill V
            # Local indices: j in 0..2, k_local in 0..1
            # Global index: j * p + (block * 2 + k_local)
            for l_idx, val in term['v'].items():
                j_local = l_idx // 2
                k_local = l_idx % 2
                g_idx = j_local * p + (block * 2 + k_local)
                V[g_idx][col] = val
            
            # Fill W
            # Local indices: k_local in 0..1, i in 0..1
            # Global index: (block * 2 + k_local) * n + i
            for l_idx, val in term['w'].items():
                i_local = l_idx // 2
                k_local = l_idx % 2
                g_idx = (block * 2 + k_local) * n + i_local
                W[g_idx][col] = val
    
    return U, V, W

def _verify_decomposition(U, V, W, n, m, p):
    """Verify that the tensor decomposition is correct."""
    rank = len(U[0])
    U_arr = np.array(U, dtype=np.float64)
    V_arr = np.array(V, dtype=np.float64)
    W_arr = np.array(W, dtype=np.float64)
    
    np.random.seed(42)
    A = np.random.randn(n, m)
    B = np.random.randn(m, p)
    C_expected = A @ B
    
    C_computed = np.zeros((n, p))
    for r in range(rank):
        u_r = U_arr[:, r].reshape(n, m)
        v_r = V_arr[:, r].reshape(m, p)
        w_r = W_arr[:, r].reshape(p, n)
        
        # Efficient computation using broadcasting or einsum
        # C_computed += np.einsum('ij,jk,ki->ik', u_r, v_r, w_r) * A * B? No.
        # The term is sum_j u_r[i,j] A[i,j] v_r[j,k] B[j,k] w_r[k,i]
        # This is not a simple einsum because A and B are outside the sum over r.
        # Actually, C_computed[i,k] = sum_r w_r[k,i] * (sum_j u_r[i,j] A[i,j] v_r[j,k] B[j,k])
        # Let M_r = u_r * A  (element-wise)
        # Let N_r = v_r * B  (element-wise)
        # Then sum_j M_r[i,j] N_r[j,k] is (M_r @ N_r)[i,k]
        # So C_computed += w_r.T @ ( (u_r * A) @ (v_r * B) )? No.
        # w_r is p x n. We need to multiply element-wise with the result?
        # The term is w_r[k,i] * ((u_r * A) @ (v_r * B))[i,k]
        # So C_computed += (w_r.T * ((u_r * A) @ (v_r * B)).T).T
        # Or simply:
        prod = (u_r * A) @ (v_r * B)
        C_computed += prod * w_r.T
        
    return np.allclose(C_expected, C_computed, atol=1e-10)

def construct():
    start_time = time.time()
    deadline = start_time + 115
    
    n, m, p = DIMENSIONS
    
    U, V, W = _build_rank33_decomposition()
    
    if not _verify_decomposition(U, V, W, n, m, p):
        # Fallback: schoolbook rank 36
        rank = 36
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        
        idx = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][idx] = 1
                    V[j * p + k][idx] = 1
                    W[k * n + i][idx] = 1
                    idx += 1
        
        return {"U": U, "V": V, "W": W}
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
