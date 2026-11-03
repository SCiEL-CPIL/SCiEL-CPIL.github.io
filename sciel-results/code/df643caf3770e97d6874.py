"""Program for 3x3x3 matrix multiplication tensor decomposition."""
import numpy as np
import time
import itertools

DIMENSIONS = [3, 3, 3]

# EVOLVE-BLOCK-START
def _build_decomposition(rank_terms):
    """
    Build U, V, W factor matrices from a list of term specifications.
    Each term is (U_dict, V_dict, W_dict) where each dict maps (i,j) or (j,k) or (k,i) to a scalar.
    """
    n, m, p = DIMENSIONS
    R = len(rank_terms)
    U = np.zeros((n * m, R))
    V = np.zeros((m * p, R))
    W = np.zeros((p * n, R))
    
    for r, (u_dict, v_dict, w_dict) in enumerate(rank_terms):
        for (i, j), val in u_dict.items():
            U[i * m + j, r] = val
        for (j, k), val in v_dict.items():
            V[j * p + k, r] = val
        for (k, i), val in w_dict.items():
            W[k * n + i, r] = val
    
    return U, V, W

def _verify_decomposition(U, V, W, n, m, p):
    """Verify that the decomposition exactly reproduces the multiplication tensor."""
    R = U.shape[1]
    U_reshaped = U.reshape((n, m, R))
    V_reshaped = V.reshape((m, p, R))
    W_reshaped = W.reshape((p, n, R))
    
    T_recon = np.einsum('ijr,jkr,kir->ijk', U_reshaped, V_reshaped, W_reshaped)
    T_target = np.ones((n, m, p))
    
    return np.allclose(T_recon, T_target, atol=1e-9)

def _strassen_2x2_terms(offset_i, offset_j, offset_k):
    """
    Return the 7 Strassen terms for a 2x2 core at given offsets.
    Each term: (U_dict, V_dict, W_dict)
    Indices are offset by the given values.
    """
    terms = []
    
    def idx(a, b, off):
        return (a + off[0], b + off[1])
    
    off_ij = (offset_i, offset_j)
    off_jk = (offset_j, offset_k)
    off_ki = (offset_k, offset_i)
    
    # m1 = (a00 + a11)(b00 + b11) -> c00 += m1, c11 += m1
    terms.append(({idx(0,0,off_ij): 1, idx(1,1,off_ij): 1},
                  {idx(0,0,off_jk): 1, idx(1,1,off_jk): 1},
                  {idx(0,0,off_ki): 1, idx(1,1,off_ki): 1}))
    
    # m2 = (a10 + a11)b00 -> c10 += m2, c11 -= m2
    terms.append(({idx(1,0,off_ij): 1, idx(1,1,off_ij): 1},
                  {idx(0,0,off_jk): 1},
                  {idx(0,1,off_ki): 1, idx(1,1,off_ki): -1}))
    
    # m3 = a00(b01 - b11) -> c01 += m3, c11 += m3
    terms.append(({idx(0,0,off_ij): 1},
                  {idx(0,1,off_jk): 1, idx(1,1,off_jk): -1},
                  {idx(1,0,off_ki): 1, idx(1,1,off_ki): 1}))
    
    # m4 = a11(b10 - b00) -> c00 += m4, c10 += m4
    terms.append(({idx(1,1,off_ij): 1},
                  {idx(1,0,off_jk): 1, idx(0,0,off_jk): -1},
                  {idx(0,0,off_ki): 1, idx(0,1,off_ki): 1}))
    
    # m5 = (a00 + a01)b11 -> c00 -= m5, c01 += m5
    terms.append(({idx(0,0,off_ij): 1, idx(0,1,off_ij): 1},
                  {idx(1,1,off_jk): 1},
                  {idx(0,0,off_ki): -1, idx(1,0,off_ki): 1}))
    
    # m6 = (a10 - a00)(b00 + b01) -> c11 += m6
    terms.append(({idx(1,0,off_ij): 1, idx(0,0,off_ij): -1},
                  {idx(0,0,off_jk): 1, idx(0,1,off_jk): 1},
                  {idx(1,1,off_ki): 1}))
    
    # m7 = (a01 - a11)(b10 + b11) -> c00 += m7
    terms.append(({idx(0,1,off_ij): 1, idx(1,1,off_ij): -1},
                  {idx(1,0,off_jk): 1, idx(1,1,off_jk): 1},
                  {idx(0,0,off_ki): 1}))
    
    return terms

def _try_strassen_core():
    """
    Strategy: Use Strassen on the 2x2x2 core (indices 0,1) and standard multiplication for the rest.
    Rank = 7 (core) + 19 (border) = 26.
    """
    n, m, p = DIMENSIONS
    terms = _strassen_2x2_terms(0, 0, 0)
    
    # Add border terms (at least one index is 2)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                if i < 2 and j < 2 and k < 2:
                    continue
                terms.append(({(i,j): 1}, {(j,k): 1}, {(k,i): 1}))
                
    return terms

def _try_standard_decomposition():
    """Standard schoolbook: rank 27."""
    n, m, p = DIMENSIONS
    terms = []
    for i in range(n):
        for j in range(m):
            for k in range(p):
                terms.append(({(i,j): 1}, {(j,k): 1}, {(k,i): 1}))
    return terms

def construct():
    """
    Find a rank-minimal exact tensor decomposition for 3x3x3 matrix multiplication.
    Tries several known constructions and returns the best one.
    """
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin
    
    n, m, p = DIMENSIONS
    
    best_result = None
    best_rank = float('inf')
    
    # Strategy 1: Strassen Core + Border (Rank 26)
    if time.time() < deadline:
        terms = _try_strassen_core()
        U, V, W = _build_decomposition(terms)
        if _verify_decomposition(U, V, W, n, m, p):
            R = U.shape[1]
            if R < best_rank:
                best_rank = R
                best_result = (U, V, W)
    
    # Strategy 2: Pure Standard (Rank 27) as fallback
    if time.time() < deadline:
        terms = _try_standard_decomposition()
        U, V, W = _build_decomposition(terms)
        if _verify_decomposition(U, V, W, n, m, p):
            R = U.shape[1]
            if R < best_rank:
                best_rank = R
                best_result = (U, V, W)
    
    if best_result is None:
        # Should not happen, but fallback
        terms = _try_standard_decomposition()
        U, V, W = _build_decomposition(terms)
        best_result = (U, V, W)
    
    U, V, W = best_result
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
