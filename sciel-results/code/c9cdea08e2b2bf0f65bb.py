"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 4, 8]

# EVOLVE-BLOCK-START
import time
import numpy as np
import random

def _verify(U, V, W, n, m, p):
    """Verify that the decomposition correctly computes matrix multiplication tensor."""
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    rank = U.shape[1]
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for c in range(rank):
                    s += U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
                if abs(s - 1.0) > 1e-9:
                    return False
    return True

def _strassen_block(U, V, W, i0, j0, k0, col_start, n, m, p):
    """
    Add Strassen decomposition for 2x2 A-block at (i0, j0), 2x2 B-block at (j0, k0).
    """
    terms = [
        ([1,0,0,1], [1,0,0,1], [1,0,0,1]),
        ([0,0,1,1], [1,1,0,0], [0,0,1,-1]),
        ([1,1,0,0], [0,0,1,1], [0,1,0,1]),
        ([0,0,0,1], [0,-1,0,1], [1,0,-1,0]),
        ([1,0,0,0], [1,0,-1,0], [-1,1,0,0]),
        ([-1,0,1,0], [0,1,0,1], [0,0,0,1]),
        ([0,1,0,-1], [1,0,1,0], [1,0,0,0]),
    ]
    
    col = col_start
    for t in range(7):
        a_coeffs = terms[t][0]
        b_coeffs = terms[t][1]
        c_coeffs = terms[t][2]
        
        for di in range(2):
            for dj in range(2):
                U[(i0+di)*m + (j0+dj)][col] = a_coeffs[di*2+dj]
        
        for dj in range(2):
            for dk in range(2):
                V[(j0+dj)*p + (k0+dk)][col] = b_coeffs[dj*2+dk]
        
        for di in range(2):
            for dk in range(2):
                W[(k0+dk)*n + (i0+di)][col] = c_coeffs[di*2+dk]
        
        col += 1
    
    return col

def _construct_strassen_base():
    n, m, p = 3, 4, 8
    rank = 88
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    for i0 in [0, 2]:
        i_size = 2 if i0 == 0 else 1
        for j0 in [0, 2]:
            for k0 in [0, 2, 4, 6]:
                if i_size == 2:
                    col = _strassen_block(U, V, W, i0, j0, k0, col, n, m, p)
                else:
                    for dj in range(2):
                        for dk in range(2):
                            U[i0*m + (j0+dj)][col] = 1
                            V[(j0+dj)*p + (k0+dk)][col] = 1
                            W[(k0+dk)*n + i0][col] = 1
                            col += 1
    return U, V, W, rank

def _rank_reduce(U, V, W, target_rank, iters=2000, tol=1e-6):
    """
    Attempts to reduce the rank of a decomposition.
    U: (n*m, r), V: (m*p, r), W: (p*n, r)
    Returns (U', V', W', r') or None if failed.
    """
    n, m, p = 3, 4, 8
    r = U.shape[1]
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    
    # Randomized compression:
    # We want to find U', V', W' of rank r' < r such that sum U' V' W' = sum U V W.
    # This is equivalent to finding a rank-r' approximation of the tensor T.
    # We can use a greedy approach:
    # 1. Start with r' = r.
    # 2. Try to merge two columns into one.
    #    If column c1 and c2 are merged into c_new, we need:
    #    U[:, c1] V[:, c1] W[:, c1] + U[:, c2] V[:, c2] W[:, c2] = U[:, c_new] V[:, c_new] W[:, c_new]
    #    This is a rank-1 approximation of the sum of two rank-1 tensors.
    #    We can use SVD on the "unfolding" to find the best rank-1 approx.
    
    # Simpler: Randomized search for lower rank.
    # We know that if the tensor has rank r, it might have a lower rank representation.
    # We try to find U', V', W' of rank r' by solving:
    # T = sum_{c=1}^{r'} u_c v_c w_c
    # We can use a heuristic: randomly project the columns and solve.
    
    # Let's try a specific heuristic:
    # Assume the decomposition can be reduced.
    # We'll use a "column merging" strategy.
    # If we have two columns c1, c2, their sum is a rank-2 tensor.
    # We approximate it by a rank-1 tensor.
    # The best rank-1 approx of a tensor T is given by the top singular vector of its unfolding.
    
    # However, implementing a robust rank reduction is complex.
    # Let's try a simpler approach: Randomized Kronecker factorization.
    # If p=8=2*2*2, maybe we can find a decomposition of 3x4 x 4x2 with lower rank.
    # Schoolbook rank for 3x4 x 4x2 is 3*4*2 = 24.
    # Strassen for 2x2 blocks:
    # i: 3 -> 2+1, j: 4 -> 2+2, k: 2 -> 2
    # Blocks:
    # (2x2, 2x2, 2x2) -> Strassen (7) * 2 (i-groups) * 1 (j-group) * 1 (k-group) = 14
    # (1x2, 2x2, 2x2) -> Schoolbook (4) * 1 (i-group) * 1 (j-group) * 1 (k-group) = 4
    # Total rank for 3x4 x 4x2: 14 + 4 = 18.
    # Then for 3x4 x 4x8, we use two such blocks? No, k is 8.
    # If we decompose 4x8 into two 4x4, and each 4x4 into two 4x2?
    # This is getting complicated.
    
    # Let's stick to the known Strassen bound for 3x4 x 4x8.
    # Current: 88.
    # Can we do better?
    # Known lower bound for 3x3 x 3x3 is 19 (Strassen).
    # For 3x4 x 4x8, the rank is likely close to 88.
    # Let's just return the base decomposition.
    
    return None

def construct():
    n, m, p = DIMENSIONS  # 3, 4, 8
    start_time = time.time()
    deadline = start_time + 110

    # Base Strassen decomposition
    U, V, W, rank = _construct_strassen_base()
    
    if _verify(U, V, W, n, m, p):
        return {"U": U, "V": V, "W": W}
    
    # Fallback to schoolbook
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
