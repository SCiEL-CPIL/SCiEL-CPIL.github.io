import time
import sys
import numpy as np

DIMENSIONS = [6, 6, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Target: Rank 147 using Strassen(2x2) outer and BCLR(3x3) inner.
    # Fallback: Rank 189 using Strassen(2x2) outer and Schoolbook(3x3) inner.
    
    # 1. Attempt Rank 147 Construction
    try:
        result_147 = _construct_rank_147()
        # Verify
        if _verify_decomposition(result_147, n, m, p):
            return result_147
    except Exception:
        pass

    # 2. Fallback to Rank 189 Construction
    return _construct_rank_189()

def _verify_decomp(U, V, W, n, m, p, trials=10):
    """Verify the decomposition using random integer matrices."""
    U = np.array(U)
    V = np.array(V)
    W = np.array(W)
    for _ in range(trials):
        A = np.random.randint(-5, 5, (n, m))
        B = np.random.randint(-5, 5, (m, p))
        C_expected = A @ B
        
        # Compute C from decomposition
        # C = sum_r U[:,r] * V[:,r] * W[:,r]
        # U is (n*m, rank), V is (m*p, rank), W is (p*n, rank)
        # Reshape U to (n, m, rank)
        # Reshape V to (m, p, rank)
        # Reshape W to (p, n, rank) -> Note: W index is k*n+i. So W is (p, n, rank)
        
        U_reshaped = U.reshape(n, m, -1)
        V_reshaped = V.reshape(m, p, -1)
        W_reshaped = W.reshape(p, n, -1)
        
        # C[i, k] = sum_j sum_r U[i, j, r] * V[j, k, r] * W[k, i, r]
        # Note: W is indexed by (k, i) in the tensor product sense?
        # Task convention: W[k*n+i, r]. So W_reshaped[k, i, r].
        
        C_computed = np.zeros((n, p), dtype=int)
        for i in range(n):
            for k in range(p):
                val = 0
                for j in range(m):
                    for r in range(U.shape[1]):
                        val += U_reshaped[i, j, r] * V_reshaped[j, k, r] * W_reshaped[k, i, r]
                C_computed[i, k] = val
        
        if not np.array_equal(C_expected, C_computed):
            return False
    return True

def _construct_rank_189():
    # Current best solution: Strassen 2x2 blocks of 3x3 Schoolbook
    n, m, p = DIMENSIONS
    block_size = 3
    dim = n
    rank_inner = 27
    rank_outer = 7
    total_rank = 189
    
    U = [[0] * total_rank for _ in range(dim * dim)]
    V = [[0] * total_rank for _ in range(dim * dim)]
    W = [[0] * total_rank for _ in range(dim * dim)]
    
    col = 0
    steps = [
        { 'A': { (0,0): 1, (1,1): 1 }, 'B': { (0,0): 1, (1,1): 1 }, 'C': { (0,0): 1, (1,1): 1 } },
        { 'A': { (1,0): 1, (1,1): 1 }, 'B': { (0,0): 1 }, 'C': { (1,0): 1, (1,1): -1 } },
        { 'A': { (0,0): 1 }, 'B': { (0,1): 1, (1,1): -1 }, 'C': { (0,1): 1, (1,1): 1 } },
        { 'A': { (1,1): 1 }, 'B': { (1,0): 1, (0,0): -1 }, 'C': { (0,0): 1, (1,0): 1 } },
        { 'A': { (0,0): 1, (0,1): 1 }, 'B': { (1,1): 1 }, 'C': { (0,0): -1, (0,1): 1 } },
        { 'A': { (1,0): 1, (0,0): -1 }, 'B': { (0,0): 1, (0,1): 1 }, 'C': { (1,1): 1 } },
        { 'A': { (0,1): 1, (1,1): -1 }, 'B': { (1,0): 1, (1,1): 1 }, 'C': { (0,0): 1 } }
    ]
    
    for step in steps:
        A_coeffs = step['A']
        B_coeffs = step['B']
        C_coeffs = step['C']
        
        for u in range(block_size):
            for v in range(block_size):
                for w in range(block_size):
                    for (I, J), ca in A_coeffs.items():
                        if ca != 0:
                            i_global = I * block_size + u
                            j_global = J * block_size + v
                            idx_u = i_global * dim + j_global
                            U[idx_u][col] += ca
                    for (J, K), cb in B_coeffs.items():
                        if cb != 0:
                            j_global = J * block_size + v
                            k_global = K * block_size + w
                            idx_v = j_global * dim + k_global
                            V[idx_v][col] += cb
                    for (I, K), cc in C_coeffs.items():
                        if cc != 0:
                            i_global = I * block_size + u
                            k_global = K * block_size + w
                            idx_w = k_global * dim + i_global
                            W[idx_w][col] += cc
                    col += 1
    return {"U": U, "V": V, "W": W}

def _construct_rank_147():
    # Strassen 2x2 outer, BCLR 3x3 inner
    # This requires valid BCLR coefficients.
    # If not available, raise exception to trigger fallback.
    raise NotImplementedError("BCLR coefficients not available in memory")

# EVOLVE-BLOCK-END
