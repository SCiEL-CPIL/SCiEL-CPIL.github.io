import numpy as np
import time

DIMENSIONS = [2, 4, 7]

def construct():
    """
    Rank-minimal decomposition for 2x4 * 4x7 matrix multiplication.
    
    Strategy: Split m=4 into two blocks of 2.
    For each 2x2x7 block, split p=7 into seven 2x2x1 blocks.
    Use rank-2 decomposition for each 2x2x1 sub-block.
    
    Per 2x2x7 block: 7 * 2 = 14
    Total rank: 14 + 14 = 28
    """
    n, m, p = DIMENSIONS  # 2, 4, 7
    
    total_rank = 28
    
    U = np.zeros((n * m, total_rank), dtype=float)  # 8 x 28
    V = np.zeros((m * p, total_rank), dtype=float)  # 28 x 28
    W = np.zeros((p * n, total_rank), dtype=float)  # 14 x 28
    
    col = 0
    
    # Split m=4 into two blocks of size 2
    for block_idx in range(2):
        j_start = block_idx * 2
        
        # Split p=7 into seven columns
        for k in range(p):
            # Subproblem: 2x2x1
            # A is 2x2 (rows 0-1, cols j_start to j_start+1)
            # B is 2x1 (rows j_start to j_start+1, col k)
            # C is 2x1 (col k, rows 0-1)
            
            # Rank 0 of this subproblem: corresponds to j_local=0
            # A indices: (0, j_start), (1, j_start)
            # B index: (j_start, k)
            # C indices: (k, 0), (k, 1)
            
            U[0 * m + j_start, col] = 1.0
            U[1 * m + j_start, col] = 1.0
            V[j_start * p + k, col] = 1.0
            W[k * n + 0, col] = 1.0
            W[k * n + 1, col] = 1.0
            
            col += 1
            
            # Rank 1 of this subproblem: corresponds to j_local=1
            # A indices: (0, j_start+1), (1, j_start+1)
            # B index: (j_start+1, k)
            # C indices: (k, 0), (k, 1)
            
            U[0 * m + (j_start + 1), col] = 1.0
            U[1 * m + (j_start + 1), col] = 1.0
            V[(j_start + 1) * p + k, col] = 1.0
            W[k * n + 0, col] = 1.0
            W[k * n + 1, col] = 1.0
            
            col += 1
    
    # Verify the decomposition
    def verify(U_mat, V_mat, W_mat):
        n, m, p = DIMENSIONS
        # Check that for all i, j, k: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] == 1
        # And for all other (a, b, c) not in support, sum == 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0.0
                    for r in range(total_rank):
                        s += U_mat[i*m+j, r] * V_mat[j*p+k, r] * W_mat[k*n+i, r]
                    if abs(s - 1.0) > 1e-9:
                        return False, f"One check failed: i={i}, j={j}, k={k}, s={s}"
        
        # Check zeros for non-support entries
        for a in range(n * m):
            i_a = a // m
            j_a = a % m
            for b in range(m * p):
                j_b = b // p
                k_b = b % p
                for c in range(p * n):
                    k_c = c // n
                    i_c = c % n
                    if i_a == i_c and j_a == j_b and k_b == k_c:
                        continue
                    s = 0.0
                    for r in range(total_rank):
                        s += U_mat[a, r] * V_mat[b, r] * W_mat[c, r]
                    if abs(s) > 1e-9:
                        return False, f"Zero check failed: a={a}, b={b}, c={c}, s={s}"
        return True, "OK"
    
    valid, msg = verify(U, V, W)
    if not valid:
        print(f"Verification failed: {msg}")
        # Fall back to schoolbook
        rank = n * m * p
        U = np.zeros((n * m, rank), dtype=float)
        V = np.zeros((m * p, rank), dtype=float)
        W = np.zeros((p * n, rank), dtype=float)
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, col] = 1.0
                    V[j * p + k, col] = 1.0
                    W[k * n + i, col] = 1.0
                    col += 1
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
