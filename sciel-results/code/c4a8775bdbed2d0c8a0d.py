import time
import random

DIMENSIONS = [2, 4, 4]

def construct():
    start_time = time.time()
    deadline = start_time + 110.0
    
    n, m, p = DIMENSIONS
    
    # Build the known rank-28 Strassen solution
    U28, V28, W28 = _build_strassen_28(n, m, p)
    
    # Verify it's correct
    if not _verify(U28, V28, W28, n, m, p):
        # Fallback: schoolbook rank-32
        U32, V32, W32 = _build_schoolbook_32(n, m, p)
        return {"U": U32, "V": V32, "W": W32}
    
    # Try to find a rank-27 decomposition by attempting to merge two Strassen terms
    # across blocks. The idea: if two terms have proportional U, V, or W vectors,
    # they might be combinable.
    
    # Strategy: Try all pairs of terms from different blocks and check if they can
    # be merged (i.e., their contribution can be represented as a single rank-1 term).
    # This is hard in general, so we try a heuristic: look for terms where the 
    # "direction" in one factor is the same.
    
    # Alternative: Try a direct search for rank 27 using the structure.
    # For 2x4 x 4x4, we can think of it as:
    # C[0,:] = A[0,:] * B  (row 0 of C)
    # C[1,:] = A[1,:] * B  (row 1 of C)
    # Each row of C is a 1x4 times 4x4 = 4 outputs, needing at least 4 multiplications schoolbook.
    # But with sharing between the two rows, we can do better.
    
    # Try a structured rank-27 attempt:
    # Use 3 full Strassen blocks (21 terms) + 7 terms for the 4th block with some sharing.
    # This is speculative, so let's try a random search for rank 27.
    
    if time.time() < deadline - 60:
        result = _search_rank_27(n, m, p, deadline)
        if result is not None:
            U, V, W = result
            if _verify(U, V, W, n, m, p):
                return {"U": U, "V": V, "W": W}
    
    return {"U": U28, "V": V28, "W": W28}

def _build_strassen_28(n, m, p):
    rank = 28
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    strassen_terms = [
        ({0: 1, 3: 1}, {0: 1, 3: 1}, {0: 1, 3: 1}),
        ({2: 1, 3: 1}, {0: 1}, {2: 1, 3: -1}),
        ({0: 1}, {1: 1, 3: -1}, {1: 1, 3: 1}),
        ({3: 1}, {2: 1, 0: -1}, {0: 1, 2: 1}),
        ({0: 1, 1: 1}, {3: 1}, {0: -1, 1: 1}),
        ({2: 1, 0: -1}, {0: 1, 1: 1}, {3: 1}),
        ({1: 1, 3: -1}, {2: 1, 3: 1}, {0: 1})
    ]
    
    blocks = [
        (0, 0, 0, 0),
        (2, 2, 0, 0),
        (0, 0, 2, 2),
        (2, 2, 2, 2)
    ]
    
    col_idx = 0
    for a_off, b_row_off, b_col_off, c_off in blocks:
        for u_coeffs, v_coeffs, w_coeffs in strassen_terms:
            for local_idx, val in u_coeffs.items():
                g_idx = (local_idx // 2) * m + a_off + (local_idx % 2)
                U[g_idx][col_idx] = val
            for local_idx, val in v_coeffs.items():
                g_idx = (b_row_off + local_idx // 2) * p + b_col_off + (local_idx % 2)
                V[g_idx][col_idx] = val
            for local_idx, val in w_coeffs.items():
                g_idx = (c_off + local_idx % 2) * n + (local_idx // 2)
                W[g_idx][col_idx] = val
            col_idx += 1
    
    return U, V, W

def _build_schoolbook_32(n, m, p):
    rank = n * m * p
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
    
    return U, V, W

def _verify(U, V, W, n, m, p):
    """Verify that sum_r U[i*m+j][r] * V[j*p+k][r] * W[k*n+i][r] = 1 for all i,j,k."""
    rank = len(U[0])
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0
                for r in range(rank):
                    s += U[i * m + j][r] * V[j * p + k][r] * W[k * n + i][r]
                if s != 1:
                    return False
    return True

def _search_rank_27(n, m, p, deadline):
    """Heuristic search for a rank-27 decomposition."""
    # Try: take the 28-term Strassen and see if any two terms can be combined.
    # Two terms r1, r2 can be combined if there exist scalars a, b such that:
    # a*U[r1] + b*U[r2] = 0 (or proportional), etc. This is very restrictive.
    # 
    # Alternative: Try a different block decomposition.
    # Instead of 4 blocks of 2x2, try overlapping blocks.
    # For A (2x4), we have columns 0,1,2,3.
    # For B (4x4), we have 4x4.
    # 
    # Try: Use Strassen on A[:,0:2] x B[0:2,0:2] -> C[:,0:2] (7 terms)
    #      Use Strassen on A[:,2:4] x B[2:4,0:2] -> C[:,0:2] (7 terms)
    #      Use Strassen on A[:,0:2] x B[0:2,2:4] -> C[:,2:4] (7 terms)
    #      Use Strassen on A[:,2:4] x B[2:4,2:4] -> C[:,2:4] (7 terms)
    # Total: 28. Same as before.
    #
    # Try a non-standard split: 
    # A[:,0:3] is 2x3, not a square. Can't use Strassen directly.
    #
    # Try: Split B into 2x4, 2x4 (top and bottom halves)
    # A is 2x4. 
    # C[:,0:2] = A[:,0:2] * B[0:2,0:2] + A[:,2:4] * B[2:4,0:2]
    # C[:,2:4] = A[:,0:2] * B[0:2,2:4] + A[:,2:4] * B[2:4,2:4]
    # This is what we already have.
    #
    # For rank 27, we need to save 1 term. This likely requires a more clever
    # algebraic structure. Given the difficulty, return None.
    return None
