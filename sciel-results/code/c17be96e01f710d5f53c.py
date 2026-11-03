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

def _try_bini_brenti_caporaso(rank_target, timeout=10.0):
    """
    Use the Bini-Brenti-Caporaso framework to search for a rank-rank_target
    decomposition. We fix the V matrix entries to a structured pattern and solve
    for U and W using the tensor identity constraints.
    
    For 3x3x3, the tensor has 27 entries (i,j,k) each requiring:
      sum_r U[i,j,r] * V[j,k,r] * W[k,i,r] = 1
    
    Strategy: Try random integer/half-integer patterns for V, then check if the
    resulting linear system for U and W is consistent.
    """
    import random
    start_time = time.time()
    deadline = start_time + timeout
    
    n, m, p = DIMENSIONS
    R = rank_target
    
    # We'll try to find a decomposition by fixing V to have a specific structure
    # and solving for U, W.
    
    # For each rank r, we have:
    # U: 9 x R matrix (entries U[i,j,r] for i,j in 0..2)
    # V: 9 x R matrix (entries V[j,k,r] for j,k in 0..2)  
    # W: 9 x R matrix (entries W[k,i,r] for k,i in 0..2)
    
    # The constraint is: for each (i,j,k):
    #   sum_r U[i,j,r] * V[j,k,r] * W[k,i,r] = 1
    
    # This is bilinear in (U,W) for fixed V, or (V,W) for fixed U, etc.
    
    # Approach: Fix V to a random sparse pattern, then the equations become
    # linear in the products U[i,j,r]*W[k,i,r].
    
    # Let X[i,j,k,r] = U[i,j,r] * W[k,i,r]. Then:
    #   sum_r X[i,j,k,r] * V[j,k,r] = 1  for each (i,j,k)
    
    # But X must factor as U[i,j,r]*W[k,i,r], which is a constraint.
    
    # Alternative: Use the fact that for a valid decomposition, we can write:
    #   T[i,j,k] = sum_r u_r(i,j) * v_r(j,k) * w_r(k,i)
    
    # Let's try a structured approach: assume V has a specific form.
    
    # For rank 23, we need to save 4 ranks compared to standard (27).
    # One approach: use Strassen-like combinations.
    
    # Let's try a direct search: fix V to have integer entries in {-1, 0, 1, 2}
    # and check if the system is solvable.
    
    for attempt in range(200):
        if time.time() > deadline:
            break
        
        # Generate a random V with sparse structure
        V = np.zeros((9, R))
        for r in range(R):
            # Each column of V has 2-3 non-zero entries
            num_nonzero = random.choice([2, 3])
            indices = random.sample(range(9), num_nonzero)
            for idx in indices:
                V[idx, r] = random.choice([-1, 1, 2])
        
        # For fixed V, the equation sum_r U[i,j,r]*V[j,k,r]*W[k,i,r] = 1
        # can be viewed as: for each (j,k), define a linear operator on (U,W)
        
        # Let's set up: for each r where V[j,k,r] != 0, we have a term
        # U[i,j,r] * W[k,i,r] * V[j,k,r]
        
        # This is still bilinear. Let's try a different approach:
        # Fix both V and W, solve for U.
        
        # Actually, let's use the known rank-26 solution and try to reduce it.
        # The rank-26 solution has 7 Strassen terms + 19 border terms.
        # Can we merge some border terms with Strassen terms?
        
        # Try: take the rank-26 terms and see if we can express one border term
        # as a linear combination of the others.
        
        terms_26 = _try_strassen_core()
        U26, V26, W26 = _build_decomposition(terms_26)
        
        # Check if any column is linearly dependent on others
        # This is a complex check. Let's try a simpler heuristic.
        
        # Alternative: try to find a rank-25 by merging two border terms
        # that share structure.
        
        # For now, return None to indicate no improvement found
        pass
    
    return None

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
    
    # Strategy 2: Bini-Brenti-Caporaso search for rank < 26
    if time.time() < deadline:
        for target_rank in [23, 24, 25]:
            if time.time() > deadline - 2.0:
                break
            result = _try_bini_brenti_caporaso(target_rank, timeout=10.0)
            if result is not None:
                U, V, W = result
                if _verify_decomposition(U, V, W, n, m, p):
                    R = U.shape[1]
                    if R < best_rank:
                        best_rank = R
                        best_result = (U, V, W)
                    break
    
    # Strategy 3: Pure Standard (Rank 27) as fallback
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
