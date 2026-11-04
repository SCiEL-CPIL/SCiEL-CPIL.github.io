"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import sys

DIMENSIONS = [3, 5, 5]

# EVOLVE-BLOCK-START
import numpy as np
import random
import itertools
from fractions import Fraction

def _verify(U, V, W, n, m, p):
    """Verify that U, V, W correctly decompose the n x m x p multiplication tensor."""
    rank = len(U[0])
    # Build tensor T[i, j, k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    # T should equal 1 iff this is a valid multiplication term
    T = np.zeros((n, m, p), dtype=object)
    for c in range(rank):
        uc = np.array([U[i*m+j][c] for i in range(n) for j in range(m)]).reshape(n, m)
        vc = np.array([V[j*p+k][c] for j in range(m) for k in range(p)]).reshape(m, p)
        wc = np.array([W[k*n+i][c] for k in range(p) for i in range(n)]).reshape(p, n)
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    T[i, j, k] += uc[i, j] * vc[j, k] * wc[k, i]
    
    # For matrix multiplication, T[i,j,k] should be 1 for all i,j,k
    # Actually no - the tensor for matrix multiplication C = A*B has T[i,j,k] = delta_{j,k}... 
    # Wait, let me reconsider. The multiplication tensor M_{n,m,p} has entries:
    # M[i, j, k] = 1 if we're computing C[i,k] = sum_j A[i,j]*B[j,k]
    # So the tensor is indexed by (i, j, k) where i is row of A/C, j is inner index, k is col of B/C
    # M[i,j,k] = 1 for all valid i,j,k - no wait, that's not right either.
    
    # The matrix multiplication tensor: (A ⊗ B)_{i,k} = sum_j A_{i,j} B_{j,k}
    # As a 3-tensor T[i, j, k] = delta_{i,i} delta_{j,j} delta_{k,k}... 
    # Actually T[i,j,k] = 1 for all i in [n], j in [m], k in [p]
    # Because the tensor is the sum over j of e_i ⊗ e_j ⊗ e_k^T
    # No wait - T = sum_{i,j,k} e_i ⊗ e_j ⊗ e_k is the "all ones" tensor? No.
    
    # Let me think again. The multiplication tensor for C = A*B where A is n×m, B is m×p:
    # C_{i,k} = sum_{j=1}^{m} A_{i,j} * B_{j,k}
    # The tensor M has M[i,j,k] = 1 for all i,j,k. That's the standard definition.
    # A rank-r decomposition: M = sum_{c=1}^{r} u_c ⊗ v_c ⊗ w_c
    # means M[i,j,k] = sum_c U[i,j,c] * V[j,k,c] * W[k,i,c]... 
    
    # Hmm, but the interface says W has shape (p*n, rank) with index W[k*n+i, c]
    # So W[k*n+i, c] corresponds to the (k,i) entry of the third factor.
    # The decomposition is: sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1 for all i,j,k
    
    # So yes, T[i,j,k] should equal 1 for all i, j, k.
    for i in range(n):
        for j in range(m):
            for k in range(p):
                if T[i, j, k] != 1:
                    return False
    return True

def _verify_fast(U, V, W, n, m, p):
    """Fast verification using numpy."""
    rank = len(U[0])
    # T[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    # We need T[i,j,k] = 1 for all i,j,k
    
    # Reshape U to (n, m, rank), V to (m, p, rank), W to (p, n, rank)
    U_arr = np.array([[U[i*m+j][c] for c in range(rank)] for i in range(n) for j in range(m)]).reshape(n, m, rank)
    V_arr = np.array([[V[j*p+k][c] for c in range(rank)] for j in range(m) for k in range(p)]).reshape(m, p, rank)
    W_arr = np.array([[W[k*n+i][c] for c in range(rank)] for k in range(p) for i in range(n)]).reshape(p, n, rank)
    
    # T[i,j,k] = sum_c U_arr[i,j,c] * V_arr[j,k,c] * W_arr[k,i,c]
    # This is a bit tricky to do in pure numpy without loops over i,j,k
    # But for small dimensions it's fine
    
    T = np.zeros((n, m, p), dtype=object)
    for c in range(rank):
        uc = U_arr[:, :, c]  # (n, m)
        vc = V_arr[:, :, c]  # (m, p)
        wc = W_arr[:, :, c]  # (p, n)
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    T[i, j, k] += uc[i, j] * vc[j, k] * wc[k, i]
    
    return np.all(T == 1)

def _schoolbook(n, m, p):
    """Return the schoolbook (rank n*m*p) decomposition."""
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
    return U, V, W

def _random_candidate(n, m, p, rank, rng):
    """Generate a random candidate decomposition of given rank."""
    # Use small integers: -1, 0, 1
    U = [[rng.choice([-1, 0, 1]) for _ in range(rank)] for _ in range(n * m)]
    V = [[rng.choice([-1, 0, 1]) for _ in range(rank)] for _ in range(m * p)]
    W = [[rng.choice([-1, 0, 1]) for _ in range(rank)] for _ in range(p * n)]
    return U, V, W

def _greedy_construct(n, m, p, max_rank, time_limit=100, seed=42):
    """Try to find a decomposition with rank <= max_rank using greedy approach."""
    rng = random.Random(seed)
    best_rank = max_rank
    best_decomp = None
    
    # First get schoolbook
    U_sb, V_sb, W_sb = _schoolbook(n, m, p)
    best_rank = n * m * p
    best_decomp = (U_sb, V_sb, W_sb)
    
    start_time = time.time()
    
    # Try various ranks from high to low
    for target_rank in range(min(max_rank, n*m*p - 1), 0, -1):
        if time.time() - start_time > time_limit:
            break
        if target_rank >= best_rank:
            continue
            
        # Try random search for this rank
        max_attempts = 2000 if target_rank <= 70 else 5000
        for attempt in range(max_attempts):
            if time.time() - start_time > time_limit:
                break
                
            U, V, W = _random_candidate(n, m, p, target_rank, rng)
            
            # Quick check: verify
            if _verify_fast(U, V, W, n, m, p):
                best_rank = target_rank
                best_decomp = (U, V, W)
                break
    
    return best_rank, best_decomp

def construct():
    n, m, p = DIMENSIONS
    
    # Deadline guard
    start_time = time.time()
    deadline = start_time + 110  # 110s limit, 10s margin
    
    # Get schoolbook as baseline
    U_sb, V_sb, W_sb = _schoolbook(n, m, p)
    best_rank = n * m * p
    best_decomp = (U_sb, V_sb, W_sb)
    
    # Try to find lower rank
    # Known lower bounds: rank >= max(n*m, m*p, p*n) = max(15, 25, 15) = 25
    # Upper bound from Strassen-like: for 5x5x5, best known is around 48-49
    # For 3x5x5, we can potentially do better than 75
    
    # Try greedy search
    try:
        rank_found, decomp_found = _greedy_construct(n, m, p, max_rank=74, time_limit=100, seed=42)
        if decomp_found is not None and rank_found < best_rank:
            best_rank = rank_found
            best_decomp = decomp_found
    except Exception:
        pass
    
    # If time allows, try more sophisticated approach
    if time.time() < deadline - 5:
        try:
            # Try with different seeds
            for seed in [123, 456, 789, 111, 222]:
                if time.time() > deadline - 5:
                    break
                rank_found, decomp_found = _greedy_construct(n, m, p, max_rank=best_rank - 1, time_limit=20, seed=seed)
                if decomp_found is not None and rank_found < best_rank:
                    best_rank = rank_found
                    best_decomp = decomp_found
        except Exception:
            pass
    
    U, V, W = best_decomp
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
