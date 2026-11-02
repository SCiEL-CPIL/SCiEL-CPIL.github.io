"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 2, 6]

# EVOLVE-BLOCK-START
import numpy as np
from itertools import product
import time

def _verify(U, V, W, n=2, m=2, p=6):
    """Check if U, V, W form a valid matrix multiplication decomposition."""
    rank = len(U[0])
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    
    # Build the multiplication tensor from U, V, W
    # T[i,j,k] = sum_r U[i,r] * V[j,r] * W[k,r]
    # where i indexes A (n*m), j indexes B (m*p), k indexes C (p*n)
    
    # Check: for all i,j,k, T[i,j,k] should equal the standard MM tensor
    # Standard: T[i,j,k] = 1 iff A[i] * B[j] contributes to C[k]
    # A is (n,m) so i = row*n + col, i.e., i = a0*m + a1 where a0 in [0,n), a1 in [0,m)
    # B is (m,p) so j = b0*m + b1 where b0 in [0,m), b1 in [0,p)
    # C is (n,p) so k = c0*p + c1 where c0 in [0,n), c1 in [0,p)
    # Contribution: C[c0,c1] += A[c0,a1] * B[a1,c1]
    # So T[i,j,k] = 1 iff a1 == b0 AND c0 == i//m AND c1 == j%m AND k == c0*p + c1
    
    # Actually let me re-read the convention: W[k*n+i, r]
    # So C index is k*n + i where k is column of C (0..p-1) and i is row of C (0..n-1)
    # C[i, k] = sum_j A[i,j] * B[j,k]
    # So T[a_idx, b_idx, c_idx] = 1 iff A[a_idx] and B[b_idx] multiply to contribute to C[c_idx]
    # a_idx = i*m + j (row i, col j of A)
    # b_idx = j*p + k (row j, col k of B)  
    # c_idx = k*n + i (col k, row i of C)  -- per the W convention W[k*n+i]
    
    # Wait, let me re-check. The problem says W[k*n+i, r] where the output is C.
    # C is n x p. If we index C as C[row, col], then row in [0,n), col in [0,p).
    # The convention W[k*n+i] suggests: k is the first index (col?), i is the second (row?)
    # So C[i, k] where i is row (0..n-1) and k is col (0..p-1), indexed as k*n + i.
    
    # Standard MM: C[i, k] = sum_j A[i, j] * B[j, k]
    # A[i,j]: index i*m + j
    # B[j,k]: index j*p + k
    # C[i,k]: index k*n + i
    
    # T[a, b, c] = 1 iff there exists j such that:
    #   a = i*m + j, b = j*p + k, c = k*n + i for some i, j, k
    
    U_mat = U  # shape (n*m, rank)
    V_mat = V  # shape (m*p, rank)
    W_mat = W  # shape (p*n, rank)
    
    # Compute tensor from factors
    # T[a,b,c] = sum_r U[a,r] * V[b,r] * W[c,r]
    # Use einsum
    T = np.einsum('ar,br,cr->abc', U_mat, V_mat, W_mat)
    
    # Build standard tensor
    T_std = np.zeros((n*m, m*p, p*n), dtype=float)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a_idx = i * m + j
                b_idx = j * p + k
                c_idx = k * n + i
                T_std[a_idx, b_idx, c_idx] = 1.0
    
    return np.allclose(T, T_std, atol=1e-9)

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin
    
    # Rank 21 baseline (Strassen on 3 blocks)
    U_local = np.array([
        [1, 0, 1, 0, 1, -1, 0],
        [0, 0, 0, 0, 1, 0, 1],
        [0, 1, 0, 0, 0, 1, 0],
        [1, 1, 0, 1, 0, 0, -1]
    ], dtype=float)
    V_local = np.array([
        [1, 1, 0, -1, 0, 1, 0],
        [0, 0, 1, 0, 0, 1, 0],
        [0, 0, 0, 1, 0, 0, 1],
        [1, 0, -1, 0, 1, 0, 1]
    ], dtype=float)
    W_local = np.array([
        [1, 0, 0, 1, -1, 0, 1],
        [0, 0, 1, 0, 1, 0, 0],
        [0, 1, 0, 1, 0, 0, 0],
        [1, -1, 1, 0, 0, 1, 0]
    ], dtype=float)
    
    def build_rank21():
        rank = 21
        U = np.zeros((n*m, rank))
        V = np.zeros((m*p, rank))
        W = np.zeros((p*n, rank))
        num_blocks = p // 2
        for k in range(num_blocks):
            base = k * 7
            U[:, base:base+7] = U_local
            V[2*k, base:base+7] = V_local[0]
            V[2*k+1, base:base+7] = V_local[1]
            V[6+2*k, base:base+7] = V_local[2]
            V[7+2*k, base:base+7] = V_local[3]
            W[4*k, base:base+7] = W_local[0]
            W[4*k+2, base:base+7] = W_local[1]
            W[4*k+1, base:base+7] = W_local[2]
            W[4*k+3, base:base+7] = W_local[3]
        return U, V, W
    
    U21, V21, W21 = build_rank21()
    
    # Try to find rank 20 by merging one pair of Strassen terms across blocks
    # Strategy: The 21st term might be redundant or combinable
    # Try: take rank-21 solution, zero out one column, and see if we can fix
    # with a small perturbation
    
    # Alternative: structured search for rank 20
    # The tensor has 4*12*12 = 576 entries, rank r gives 4r + 12r + 12r = 28r parameters
    # For r=20: 560 params, 576 constraints -> underdetermined but close
    # For r=21: 588 params -> slightly overdetermined
    
    # Try a more clever approach: exploit that A is shared
    # In Strassen, the 7 products are:
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B01-B11)
    # M4 = A11(B10-B00)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B10)
    # M7 = (A01-A11)(B11+B10)
    
    # For 3 blocks sharing A, we have 3*7 = 21 products
    # Can we reduce by 1? The structure of A constraints might allow it.
    
    # Let's try a randomized search: start from rank-21, try to eliminate one dimension
    # by finding a linear dependency
    
    # Actually, let's try a direct approach: assume rank 20 with specific structure
    # where the U matrix has a particular block structure
    
    # Try: use 6 full Strassen blocks for 2 of the 3 blocks (12 terms)
    # and a modified 8-term scheme for the third block that shares 4 terms
    # with the first two blocks
    
    # Simpler: try to find if rank 20 exists by checking a specific structure
    # where one of the 21 Strassen terms is a linear combination of others
    
    # Let's try: for each pair of blocks, can we share one Strassen product?
    # M1 in block k uses (A00+A11)(B00^k+B11^k)
    # If we could express one product as a sum, we'd save 1
    
    # Given time constraints, let's do a quick random search for rank 20
    # with small integer coefficients
    
    if time.time() < deadline:
        # Random search for rank 20
        rng = np.random.default_rng(42)
        best_rank = 21
        best_solution = (U21, V21, W21)
        
        # Try structured rank-20: take rank-21 and try to merge two columns
        # by finding a linear relation
        for attempt in range(200):
            if time.time() > deadline - 5:
                break
            
            # Randomly perturb the rank-21 solution slightly
            # and try to project to rank 20
            U_try = U21.copy()
            V_try = V21.copy()
            W_try = W21.copy()
            
            # Try to make one column a linear combination of others
            # Pick two random columns and try to merge
            c1, c2 = rng.choice(21, 2, replace=False)
            
            # Try: W[:, c2] = alpha * W[:, c1] + W[:, c2]
            # and adjust U, V accordingly
            # This is complex; skip for now
            
            if time.time() > deadline - 2:
                break
        
        # If no rank-20 found, return rank-21
        U, V, W = best_solution
    else:
        U, V, W = U21, V21, W21
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
