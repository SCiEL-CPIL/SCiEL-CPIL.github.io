"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time
import random
from itertools import product

DIMENSIONS = [2, 2, 11]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 110  # Safety margin
    
    # Build the target tensor T of shape (n*m, m*p, p*n)
    # T[i*m+j, j*p+k, k*n+i] = 1
    T = np.zeros((n*m, m*p, p*n), dtype=int)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i*m + j, j*p + k, k*n + i] = 1
    
    # Try to find a decomposition with rank r
    # U: (n*m, r), V: (m*p, r), W: (p*n, r)
    # Such that sum_c U[a,c]*V[b,c]*W[c_idx,c] == T[a,b,c_idx]
    
    best_result = None
    best_rank = n * m * p  # schoolbook
    
    # First, get the schoolbook decomposition as fallback
    rank_school = n * m * p
    U = [[0] * rank_school for _ in range(n * m)]
    V = [[0] * rank_school for _ in range(m * p)]
    W = [[0] * rank_school for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m + j][col] = 1
                V[j*p + k][col] = 1
                W[k*n + i][col] = 1
                col += 1
    best_result = {"U": U, "V": V, "W": W}
    best_rank = rank_school
    
    # Try lower ranks using a structured approach
    # For 2x2 * 2xp, try rank = 2p + 2, 2p+1, 2p, ...
    # Actually, let's try a more systematic approach with random initialization and gradient-free search
    
    # Try ranks from a reasonable upper bound down
    # Upper bound: min(n*m*p, some heuristic)
    # For small n,m and large p, rank might be around 2*p + n*m - 1 or similar
    
    max_rank_to_try = min(best_rank - 1, 2*p + n + m)  # Try up to 2*11+4 = 26
    
    for target_rank in range(24, 0, -1):
        if time.time() > deadline:
            break
        
        # Try to find a decomposition of this rank
        success = _try_find_decomposition(T, n, m, p, target_rank, deadline)
        if success is not None:
            U_list, V_list, W_list = success
            # Verify
            if _verify_decomposition(U_list, V_list, W_list, T, n, m, p):
                best_result = {"U": U_list, "V": V_list, "W": W_list}
                best_rank = target_rank
                break  # Found the minimum for this search
    
    return best_result

def _verify_decomposition(U, V, W, T, n, m, p):
    """Verify that sum_c U[a,c]*V[b,c]*W[c_idx,c] == T[a,b,c_idx]"""
    rank = len(U[0]) if U else 0
    for a in range(n*m):
        for b in range(m*p):
            for c_idx in range(p*n):
                total = 0
                for c in range(rank):
                    total += U[a][c] * V[b][c] * W[c_idx][c]
                if total != T[a, b, c_idx]:
                    return False
    return True

def _try_find_decomposition(T, n, m, p, target_rank, deadline):
    """Try to find a rank-1 decomposition of the tensor."""
    # Use a structured approach: since the tensor has a specific pattern,
    # try to exploit the linearity
    
    # For 2x2 * 2xp: 
    # Input A is 2x2, B is 2xp
    # Output C = A*B is 2xp
    # C[i,k] = sum_j A[i,j] * B[j,k]
    
    # The tensor T has T[i*2+j, j*p+k, k*2+i] = 1
    
    # Strategy: try to use the fact that for each (i,j) pair in A and each k in B,
    # we need T[i*2+j, j*p+k, k*2+i] = 1
    
    # Let's try a construction based on separating the "diagonal" and "off-diagonal" parts
    
    # For n=2, m=2, p=11:
    # We have 4*22*22 = 1936 entries, but only 2*2*11 = 44 are non-zero
    
    # Try a rank-1 decomposition where we group terms
    
    # Approach: For each k (0..p-1), we need to handle the contributions to C[:,k]
    # C[i,k] = A[i,0]*B[0,k] + A[i,1]*B[1,k]
    
    # This is essentially: for each k, we're doing a 2x2 * 2x1 multiplication
    # which has rank at most 4 (schoolbook) but can be done in rank 3 (Strassen)
    
    # But we need to do this for all p=11 k's simultaneously with shared U,V,W
    
    # Let me try a different approach: random search with verification
    
    # Initialize U, V, W randomly with small integers
    np.random.seed(42)
    
    for attempt in range(50):
        if time.time() > deadline:
            return None
        
        # Try different random seeds
        np.random.seed(42 + attempt)
        
        # Initialize with small random integers
        U = np.random.randint(-2, 3, size=(n*m, target_rank))
        V = np.random.randint(-2, 3, size=(m*p, target_rank))
        W = np.random.randint(-2, 3, size=(p*n, target_rank))
        
        # Check if this works (unlikely for random, but let's try)
        if _verify_decomposition_np(U, V, W, T):
            return (U.tolist(), V.tolist(), W.tolist())
    
    # More structured approach: build from the known structure
    # For each (i, j, k), we need a term. Let's try to merge some terms.
    
    # The key insight: T[i*2+j, j*p+k, k*2+i] = 1
    # This means for fixed i,j: the non-zero entries in V are at positions j*p+k for all k
    # and in W at positions k*2+i for all k
    
    # Let's try a construction where we use p terms for the "diagonal" (i=j case doesn't apply here since we're multiplying)
    
    # Actually, let me just return None and rely on the schoolbook or try a smarter construction
    
    # Try a specific construction for 2x2 * 2xp:
    # Use rank = 2*p + 2
    # Terms 0..p-1: handle A[0,0]*B[0,k] -> C[0,k]
    # Terms p..2p-1: handle A[0,1]*B[1,k] -> C[0,k]  
    # etc.
    # This is just schoolbook with rank 4p = 44
    
    # Let me try to find if rank < 44 is possible by checking if the tensor
    # has a lower rank using a simple rank estimation
    
    # For now, return None to indicate no better decomposition found
    return None

def _verify_decomposition_np(U, V, W, T):
    """Vectorized verification using numpy."""
    # Compute sum_c U[a,c]*V[b,c]*W[c_idx,c] for all a,b,c_idx
    # This is a 3-tensor contraction
    # Result[a,b,c_idx] = sum_c U[a,c] * V[b,c] * W[c_idx,c]
    
    # Efficient: for each c, compute outer product of U[:,c], V[:,c], W[:,c]
    # and sum
    result = np.zeros(T.shape, dtype=int)
    for c in range(U.shape[1]):
        uc = U[:, c]
        vc = V[:, c]
        wc = W[:, c]
        # result[a,b,k] += uc[a] * vc[b] * wc[k]
        # This is outer(uc, vc) * wc for each k... 
        # Actually: result += np.einsum('a,b,c->abc', uc, vc, wc)
        result += np.einsum('a,b,c->abc', uc, vc, wc)
    
    return np.array_equal(result, T)

# EVOLVE-BLOCK-END
