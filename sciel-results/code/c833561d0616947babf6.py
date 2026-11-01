"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

DIMENSIONS = [2, 2, 6]

# EVOLVE-BLOCK-START
import time
import numpy as np
from itertools import product

def _verify_decomposition(U, V, W, n, m, p):
    """Verify that U, V, W exactly decompose the matrix multiplication tensor."""
    n, m, p = int(n), int(m), int(p)
    rank = len(U[0]) if U else 0
    
    # Build the tensor: T[i, j, k] = sum_r U[i,r] * V[j,r] * W[k,r]
    # where i is row of A (0..n-1), j is col of A / row of B (0..m-1), k is col of B (0..p-1)
    # Output C[i, k] = sum_j A[i,j] * B[j,k]
    # So T[i, j, k] should be 1 if this term appears in C[i,k], which is always the case for matrix mult.
    # Actually, the tensor is T[i, j, k] = delta_{ik} in some convention... 
    # Let me think: C[i,k] = sum_j A[i,j] B[j,k]
    # The multiplication tensor has T[i,j,k] = 1 for all i,j,k (it's the all-ones tensor in this context)
    # No wait: the tensor is defined such that sum_j A[i,j] B[j,k] = sum_r (sum_i U[i,r] A[i,j]) (sum_k V[j,r] B[j,k]) W[k,r]... 
    # 
    # Standard definition: The multiplication tensor M_{n,m,p} has entries T[i,j,k] where:
    # C = A @ B, C[i,k] = sum_j A[i,j] B[j,k]
    # Tensor form: T[i,j,k] = 1 (all ones)
    # Decomposition: T = sum_r U_r ⊗ V_r ⊗ W_r
    # where U_r[i] = U[i,r], V_r[j] = V[j,r], W_r[k] = W[k,r]
    # So we need: sum_r U[i,r] * V[j,r] * W[k,r] = 1 for all i,j,k
    
    U_arr = np.array(U, dtype=np.float64)
    V_arr = np.array(V, dtype=np.float64)
    W_arr = np.array(W, dtype=np.float64)
    
    # Compute sum_r U[i,r] * V[j,r] * W[k,r] for all i,j,k
    # This is a tensor product
    result = np.einsum('ir,jr,kr->ijk', U_arr, V_arr, W_arr)
    
    # Check if result is all ones
    if not np.allclose(result, 1.0, atol=1e-10):
        return False
    return True

def _build_strassen_base(n, m, p):
    """Build the base Strassen decomposition for the first two blocks (14 terms)."""
    strassen = [
        ({0: 1, 3: 1}, {0: 1, 3: 1}, {(0, 0): 1, (1, 1): 1}),
        ({2: 1, 3: 1}, {0: 1}, {(1, 0): 1, (1, 1): -1}),
        ({0: 1}, {1: 1, 3: -1}, {(0, 1): 1, (1, 1): 1}),
        ({3: 1}, {2: 1, 0: -1}, {(0, 0): 1, (1, 0): 1}),
        ({0: 1, 1: 1}, {3: 1}, {(0, 0): -1, (0, 1): 1}),
        ({2: 1, 0: -1}, {0: 1, 1: 1}, {(1, 1): 1}),
        ({1: 1, 3: -1}, {2: 1, 3: 1}, {(0, 0): 1})
    ]
    
    U = [[0] * 14 for _ in range(n * m)]
    V = [[0] * 14 for _ in range(m * p)]
    W = [[0] * 14 for _ in range(p * n)]
    
    r = 0
    for b in range(2):  # First two 2x2 blocks
        for t in range(7):
            u_coeffs, v_coeffs, w_coeffs = strassen[t]
            for idx, val in u_coeffs.items():
                U[idx][r] = val
            for l, val in v_coeffs.items():
                if l == 0:
                    v_idx = 0 * p + 2 * b
                elif l == 1:
                    v_idx = 0 * p + 2 * b + 1
                elif l == 2:
                    v_idx = 1 * p + 2 * b
                elif l == 3:
                    v_idx = 1 * p + 2 * b + 1
                V[v_idx][r] = val
            for (i, k_local), val in w_coeffs.items():
                k_global = 2 * b + k_local
                w_idx = k_global * n + i
                W[w_idx][r] = val
            r += 1
    
    return U, V, W

def _try_rank20_search(n, m, p, deadline):
    """Try to find a rank-20 decomposition using a search approach."""
    # Start with Strassen for all 3 blocks (rank 21)
    strassen = [
        ({0: 1, 3: 1}, {0: 1, 3: 1}, {(0, 0): 1, (1, 1): 1}),
        ({2: 1, 3: 1}, {0: 1}, {(1, 0): 1, (1, 1): -1}),
        ({0: 1}, {1: 1, 3: -1}, {(0, 1): 1, (1, 1): 1}),
        ({3: 1}, {2: 1, 0: -1}, {(0, 0): 1, (1, 0): 1}),
        ({0: 1, 1: 1}, {3: 1}, {(0, 0): -1, (0, 1): 1}),
        ({2: 1, 0: -1}, {0: 1, 1: 1}, {(1, 1): 1}),
        ({1: 1, 3: -1}, {2: 1, 3: 1}, {(0, 0): 1})
    ]
    
    # Build rank-21 Strassen
    U21 = [[0] * 21 for _ in range(n * m)]
    V21 = [[0] * 21 for _ in range(m * p)]
    W21 = [[0] * 21 for _ in range(p * n)]
    
    r = 0
    for b in range(3):
        for t in range(7):
            u_coeffs, v_coeffs, w_coeffs = strassen[t]
            for idx, val in u_coeffs.items():
                U21[idx][r] = val
            for l, val in v_coeffs.items():
                if l == 0:
                    v_idx = 0 * p + 2 * b
                elif l == 1:
                    v_idx = 0 * p + 2 * b + 1
                elif l == 2:
                    v_idx = 1 * p + 2 * b
                elif l == 3:
                    v_idx = 1 * p + 2 * b + 1
                V21[v_idx][r] = val
            for (i, k_local), val in w_coeffs.items():
                k_global = 2 * b + k_local
                w_idx = k_global * n + i
                W21[w_idx][r] = val
            r += 1
    
    # Try to find rank-20 by removing one term and adjusting
    # Strategy: For each pair of terms (r1, r2) from different blocks, try to merge them
    # or find a linear combination that reduces rank
    
    # Alternative: Use the fact that the three blocks share the same A matrix
    # The key insight: terms involving A coefficients can be combined across blocks
    
    # Let's try a direct algebraic approach:
    # Parameterize rank-20 with specific structure and solve
    
    # For now, return the rank-21 solution as fallback
    return U21, V21, W21

def construct():
    """
    Returns a decomposition for M_{2,2,6} with minimal rank.
    
    U: shape (n*m, rank) = (4, rank)
    V: shape (m*p, rank) = (12, rank)
    W: shape (p*n, rank) = (12, rank)
    """
    start_time = time.time()
    deadline = start_time + 110  # Safety margin for 120s limit
    
    n, m, p = DIMENSIONS  # 2, 2, 6
    
    # Try to find rank-20 decomposition
    U, V, W = _try_rank20_search(n, m, p, deadline)
    
    # Verify the decomposition
    if _verify_decomposition(U, V, W, n, m, p):
        return {"U": U, "V": V, "W": W}
    
    # Fallback: return rank-21 Strassen
    strassen = [
        ({0: 1, 3: 1}, {0: 1, 3: 1}, {(0, 0): 1, (1, 1): 1}),
        ({2: 1, 3: 1}, {0: 1}, {(1, 0): 1, (1, 1): -1}),
        ({0: 1}, {1: 1, 3: -1}, {(0, 1): 1, (1, 1): 1}),
        ({3: 1}, {2: 1, 0: -1}, {(0, 0): 1, (1, 0): 1}),
        ({0: 1, 1: 1}, {3: 1}, {(0, 0): -1, (0, 1): 1}),
        ({2: 1, 0: -1}, {0: 1, 1: 1}, {(1, 1): 1}),
        ({1: 1, 3: -1}, {2: 1, 3: 1}, {(0, 0): 1})
    ]
    
    rank = 21
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    r = 0
    for b in range(3):
        for t in range(7):
            u_coeffs, v_coeffs, w_coeffs = strassen[t]
            for idx, val in u_coeffs.items():
                U[idx][r] = val
            for l, val in v_coeffs.items():
                if l == 0:
                    v_idx = 0 * p + 2 * b
                elif l == 1:
                    v_idx = 0 * p + 2 * b + 1
                elif l == 2:
                    v_idx = 1 * p + 2 * b
                elif l == 3:
                    v_idx = 1 * p + 2 * b + 1
                V[v_idx][r] = val
            for (i, k_local), val in w_coeffs.items():
                k_global = 2 * b + k_local
                w_idx = k_global * n + i
                W[w_idx][r] = val
            r += 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
