"""
Matrix multiplication 5x5x5 tensor decomposition.
Uses a 2+2+1 partition with Strassen's rank-7 algorithm for the 2x2 blocks,
and schoolbook for boundary terms.
"""
import time
import numpy as np

DIMENSIONS = [5, 5, 5]

# EVOLVE-BLOCK-START
def _schoolbook(n, m, p):
    """Construct the standard rank n*m*p decomposition."""
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


def _verify_float(n, m, p, U, V, W, tol=1e-10):
    """Fast float verification."""
    rank = len(U[0]) if U else 0
    U_np = np.array(U, dtype=float)
    V_np = np.array(V, dtype=float)
    W_np = np.array(W, dtype=float)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u_val = U_np[i * m + j]
                v_val = V_np[j * p + k]
                w_val = W_np[k * n + i]
                dot = np.dot(u_val, v_val * w_val)
                if abs(dot - 1.0) > tol:
                    return False
    return True


def _strassen_2x2():
    """
    Return the rank-7 Strassen decomposition for 2x2 matrix multiplication.
    Returns (U, V, W) where each is a list of lists.
    U: shape (4, 7), V: shape (4, 7), W: shape (4, 7)
    Index convention: row index = i*n+j for n=2
    """
    U = [
        [1, 0, 1, 0, 1, 0, 0],
        [0, 0, 0, 0, 1, 0, 1],
        [0, 1, 0, 0, 0, -1, 0],
        [1, 1, 0, 1, 0, 0, -1]
    ]
    V = [
        [1, 1, 0, -1, 0, 1, 0],
        [0, 0, 1, 0, 0, 1, 0],
        [0, 0, 0, 1, 0, 0, 1],
        [1, 0, -1, 0, 1, 0, 1]
    ]
    W = [
        [1, 0, 0, 1, -1, 0, 1],
        [0, 0, 1, 0, 1, 0, 0],
        [0, 1, 0, 1, 0, 0, 0],
        [1, -1, 1, 0, 0, 1, 0]
    ]
    return U, V, W


def _block_decomposition_5x5():
    """
    Construct a decomposition for 5x5x5 matrix multiplication using 2+2+1 partition.
    
    Partition indices into three groups: G0={0,1}, G1={2,3}, G2={4}
    For each pair of groups (a, b, c) where a is row group, b is col group of A, c is row group of B:
    - If all three groups have size 2: use Strassen (rank 7 per 2x2x2 block)
    - Otherwise: use schoolbook
    
    Actually, let's think about this more carefully.
    
    For 5x5 matmul, we have A[5x5] * B[5x5] = C[5x5]
    The tensor has components T[i,j,k] for i,j,k in 0..4.
    
    Using 2+2+1 partition on each dimension:
    - Rows of A: groups R0={0,1}, R1={2,3}, R2={4}
    - Cols of A / Rows of B: groups C0={0,1}, C1={2,3}, C2={4}  
    - Cols of B / Rows of C: groups K0={0,1}, K1={2,3}, K2={4}
    
    For each combination of (row_group, col_group, k_group):
    - If all have size 2: 2x2x2 sub-tensor, rank 7 via Strassen
    - Otherwise: schoolbook for that sub-block
    
    Number of 2x2x2 blocks: 2*2*2 = 8 blocks, each rank 7, total 56
    Remaining terms: total 125 - 8*8 = 125 - 64 = 61 terms, each rank 1, total 61
    Grand total: 56 + 61 = 117
    """
    n, m, p = 5, 5, 5
    nm, mp, pn = 25, 25, 25
    
    U2, V2, W2 = _strassen_2x2()
    rank2 = 7
    
    # Define groups: group 0 = {0,1}, group 1 = {2,3}, group 2 = {4}
    groups = [[0, 1], [2, 3], [4]]
    group_sizes = [2, 2, 1]
    
    # We'll assign columns in order
    col = 0
    U = [[0] * 117 for _ in range(nm)]
    V = [[0] * 117 for _ in range(mp)]
    W = [[0] * 117 for _ in range(pn)]
    
    for ga in range(3):  # row group of A
        for gb in range(3):  # col group of A / row group of B
            for gc in range(3):  # col group of B
                sa = group_sizes[ga]
                sb = group_sizes[gb]
                sc = group_sizes[gc]
                
                # Get actual indices
                idx_a = groups[ga]
                idx_b = groups[gb]
                idx_c = groups[gc]
                
                if sa == 2 and sb == 2 and sc == 2:
                    # Use Strassen for this 2x2x2 block
                    for ia in range(2):
                        for ib in range(2):
                            i = idx_a[ia]
                            j = idx_b[ib]
                            u_idx = i * m + j
                            for c in range(rank2):
                                U[u_idx][col + c] = U2[ia * 2 + ib][c]
                    
                    for ib in range(2):
                        for ic in range(2):
                            j = idx_b[ib]
                            k = idx_c[ic]
                            v_idx = j * p + k
                            for c in range(rank2):
                                V[v_idx][col + c] = V2[ib * 2 + ic][c]
                    
                    for ic in range(2):
                        for ia in range(2):
                            k = idx_c[ic]
                            i = idx_a[ia]
                            w_idx = k * n + i
                            for c in range(rank2):
                                W[w_idx][col + c] = W2[ic * 2 + ia][c]
                    
                    col += rank2
                else:
                    # Schoolbook for this sub-block
                    for i in idx_a:
                        for j in idx_b:
                            for k in idx_c:
                                u_idx = i * m + j
                                v_idx = j * p + k
                                w_idx = k * n + i
                                U[u_idx][col] = 1
                                V[v_idx][col] = 1
                                W[w_idx][col] = 1
                                col += 1
    
    # Trim if we used fewer columns than allocated
    if col < 117:
        U = [row[:col] for row in U]
        V = [row[:col] for row in V]
        W = [row[:col] for row in W]
    
    return {"U": U, "V": V, "W": W}


def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 110  # Safety margin before 120s limit
    
    # Start with schoolbook as fallback
    best = _schoolbook(n, m, p)
    best_rank = n * m * p
    
    # Try the 2+2+1 block decomposition
    if time.time() < deadline:
        result = _block_decomposition_5x5()
        rank = len(result['U'][0])
        
        # Verify the result
        if _verify_float(n, m, p, result['U'], result['V'], result['W'], tol=1e-10):
            if rank < best_rank:
                best = result
                best_rank = rank
    
    return best
# EVOLVE-BLOCK-END
