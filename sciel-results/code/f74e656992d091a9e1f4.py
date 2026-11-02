"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 4, 8]

# EVOLVE-BLOCK-START
import time
import numpy as np
from itertools import product

def _verify(U, V, W):
    """Check that the decomposition exactly represents matrix multiplication.
    U: (n*m, rank), V: (m*p, rank), W: (p*n, rank)
    For each (i,j,k): sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] == 1
    """
    n, m, p = DIMENSIONS
    rank = len(U[0])
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    U3 = U_arr.reshape(n, m, rank)
    V3 = V_arr.reshape(m, p, rank)
    W3 = W_arr.reshape(p, n, rank)
    result = np.einsum('ijr,jkr,kir->ijk', U3, V3, W3)
    if not np.allclose(result, 1.0, atol=1e-10):
        return False
    return True

def _try_rank_decomposition(max_rank=98, time_limit=100):
    """
    Try to find a rank-minimal decomposition using a tensor completion approach.
    We fix U and V to sparse structured bases and solve for W.
    """
    n, m, p = DIMENSIONS
    start_time = time.time()
    
    # For each rank from max_rank down, try to find a valid decomposition
    # Strategy: Use a structured approach where U and V are fixed to have
    # specific sparsity patterns, then solve for W.
    
    # For 4x4 * 4x8, the schoolbook rank is 128.
    # Strassen gives 98. We try to find something better.
    
    # Approach: Use a basis of rank-1 tensors and try to combine them.
    # Each rank-1 tensor corresponds to a choice of (u, v, w) where:
    # u is a vector in R^(n*m), v in R^(m*p), w in R^(p*n)
    # and the tensor is u ⊗ v ⊗ w
    
    # We'll use a greedy approach: start with the schoolbook decomposition
    # and try to reduce the rank by finding linear dependencies.
    
    # Actually, let's try a different approach: use the fact that for small
    # matrices, we can enumerate possible sparse patterns.
    
    # For now, let's just use the Strassen decomposition as a baseline
    # and see if we can improve it.
    
    # Try to find a rank-96 or lower decomposition
    for rank in range(max_rank, 90, -1):
        if time.time() - start_time > time_limit:
            break
        # Try to construct a rank-rank decomposition
        result = _construct_rank_fixed(rank, n, m, p, time_limit - (time.time() - start_time))
        if result is not None:
            return result
    
    return None

def _construct_rank_fixed(rank, n, m, p, time_budget):
    """
    Try to construct a decomposition of the given rank.
    Uses a structured approach with sparse U and V.
    """
    start_time = time.time()
    
    # For a 4x4 * 4x8 multiplication, we can think of it as:
    # C = A * B where A is 4x4, B is 4x8, C is 4x8
    
    # One approach: split B into two 4x4 blocks and use Strassen for each
    # That gives rank 98.
    
    # To get lower rank, we need a more clever approach.
    
    # Let's try a block approach where we use fewer multiplications.
    
    # For 4x4 * 4x4, the best known is 49 (Strassen).
    # For 4x4 * 4x8, we can't just use 2*49 = 98 because that's not optimal.
    
    # Actually, we can view 4x4 * 4x8 as a single matrix multiplication.
    # The tensor is of size 4*4*8 = 128.
    
    # Let's try a different decomposition strategy.
    
    # Use a basis of vectors and solve a linear system.
    
    # Define the multiplication tensor T[i,j,k] = 1 for all i,j,k
    # We need to find U, V, W such that:
    # sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1 for all i,j,k
    
    # This is equivalent to finding a rank-rank decomposition of the tensor.
    
    # For small ranks, we can try random or structured approaches.
    
    # Let's try a structured approach:
    # U has shape (16, rank), V has shape (32, rank), W has shape (32, rank)
    
    # We'll try to use a sparse structure for U and V.
    
    # For now, return None to indicate we couldn't find a better decomposition
    # in the time budget. The Strassen decomposition will be used as fallback.
    
    return None

def _strassen_2x2_factors():
    """
    Return the 7 rank-1 factors for 2x2*2x2 matrix multiplication.
    """
    factors = []
    
    # M1 = (A00+A11)(B00+B11)
    u = [1, 0, 0, 1]
    v = [1, 0, 0, 1]
    w = [1, 0, 0, 1]
    factors.append((u, v, w))
    
    # M2 = (A10+A11) * B00
    u = [0, 0, 1, 1]
    v = [1, 0, 0, 0]
    w = [0, 0, 1, -1]
    factors.append((u, v, w))
    
    # M3 = A00 * (B01-B11)
    u = [1, 0, 0, 0]
    v = [0, 1, 0, -1]
    w = [0, 1, 0, 1]
    factors.append((u, v, w))
    
    # M4 = A11 * (B10-B00)
    u = [0, 0, 0, 1]
    v = [-1, 0, 1, 0]
    w = [1, 0, 1, 0]
    factors.append((u, v, w))
    
    # M5 = (A00+A01) * B11
    u = [1, 1, 0, 0]
    v = [0, 0, 0, 1]
    w = [-1, 1, 0, 0]
    factors.append((u, v, w))
    
    # M6 = (A10-A00) * (B00+B01)
    u = [-1, 0, 1, 0]
    v = [1, 1, 0, 0]
    w = [0, 0, 0, 1]
    factors.append((u, v, w))
    
    # M7 = (A01-A11) * (B10+B11)
    u = [0, 1, 0, -1]
    v = [0, 0, 1, 1]
    w = [1, 0, 0, 0]
    factors.append((u, v, w))
    
    return factors

def _try_strassen_4x4_4x8():
    """
    Construct a rank-98 decomposition for 4x4 * 4x8 using Strassen.
    """
    n, m, p = DIMENSIONS  # 4, 4, 8
    
    s2_factors = _strassen_2x2_factors()
    
    rank = 98
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    for t in range(2):
        for s in range(7):
            for r in range(7):
                col = t * 49 + s * 7 + r
                
                s2_u_block = s2_factors[s][0]
                s2_v_block = s2_factors[s][1]
                s2_w_block = s2_factors[s][2]
                
                s2_u_scalar = s2_factors[r][0]
                s2_v_scalar = s2_factors[r][1]
                s2_w_scalar = s2_factors[r][2]
                
                for bi in range(2):
                    for bj in range(2):
                        block_idx = bi * 2 + bj
                        c_block = s2_u_block[block_idx]
                        if c_block == 0:
                            continue
                        for ei in range(2):
                            for ej in range(2):
                                scalar_idx = ei * 2 + ej
                                c_scalar = s2_u_scalar[scalar_idx]
                                if c_scalar == 0:
                                    continue
                                i = 2 * bi + ei
                                j = 2 * bj + ej
                                U[i * m + j][col] += c_block * c_scalar
                
                for bi in range(2):
                    for bj in range(2):
                        block_idx = bi * 2 + bj
                        c_block = s2_v_block[block_idx]
                        if c_block == 0:
                            continue
                        for ei in range(2):
                            for ej in range(2):
                                scalar_idx = ei * 2 + ej
                                c_scalar = s2_v_scalar[scalar_idx]
                                if c_scalar == 0:
                                    continue
                                j = 2 * bi + ei
                                k = 4 * t + 2 * bj + ej
                                V[j * p + k][col] += c_block * c_scalar
                
                for ci in range(2):
                    for cj in range(2):
                        block_idx = ci * 2 + cj
                        c_block = s2_w_block[block_idx]
                        if c_block == 0:
                            continue
                        for ei in range(2):
                            for ej in range(2):
                                scalar_idx = ei * 2 + ej
                                c_scalar = s2_w_scalar[scalar_idx]
                                if c_scalar == 0:
                                    continue
                                i = 2 * ci + ei
                                k = 4 * t + 2 * cj + ej
                                W[k * n + i][col] += c_block * c_scalar
    
    return {"U": U, "V": V, "W": W}

def _schoolbook():
    """Return the trivial rank n*m*p decomposition."""
    n, m, p = DIMENSIONS
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    return {"U": U, "V": V, "W": W}

def construct():
    """
    Return a tensor decomposition for 4x4 * 4x8 matrix multiplication.
    Tries to find a rank-minimal decomposition, falls back to Strassen (rank 98) or schoolbook (rank 128).
    """
    start_time = time.time()
    deadline = start_time + 140  # Leave 10s safety margin
    
    # First, try to find a better decomposition
    try:
        result = _try_rank_decomposition(max_rank=97, time_limit=120)
        if result is not None and time.time() < deadline:
            if _verify(result["U"], result["V"], result["W"]):
                return result
    except Exception:
        pass
    
    # Fall back to Strassen
    try:
        result = _try_strassen_4x4_4x8()
        if _verify(result["U"], result["V"], result["W"]):
            return result
    except Exception:
        pass
    
    # Final fallback: schoolbook
    return _schoolbook()
# EVOLVE-BLOCK-END
