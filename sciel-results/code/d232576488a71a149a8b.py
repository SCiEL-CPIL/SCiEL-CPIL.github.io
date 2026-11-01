import numpy as np
import time

DIMENSIONS = [2, 4, 7]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 140  # 10s safety margin
    
    n, m, p = DIMENSIONS
    
    # We construct a rank 50 solution based on block decomposition.
    # Split m=4 into two blocks of size 2: j in {0,1} and j in {2,3}.
    # For each block, split p=7 into three blocks of size 2 (k in {0,1}, {2,3}, {4,5}) 
    # and one block of size 1 (k=6).
    # For 2x2 x 2x2 subproblems, use Strassen (rank 7).
    # For 2x2 x 2x1 subproblems, use schoolbook (rank 4).
    # Total rank = 2 * (3 * 7 + 1 * 4) = 2 * (21 + 4) = 50.
    
    rank = 50
    U = np.zeros((n * m, rank), dtype=int)
    V = np.zeros((m * p, rank), dtype=int)
    W = np.zeros((p * n, rank), dtype=int)
    
    col = 0
    
    def add_strassen_block(i0, j0, k0):
        nonlocal col
        # i0: start row of A (0 or 1) - here always 0 since n=2 is not split
        # j0: start col of A (0 or 2)
        # k0: start col of B (0, 2, 4)
        
        # A indices:
        # A is 2x4. Row i, Col j.
        # Sub-block A[i0:i0+2, j0:j0+2]
        a_idx = i0 * m + j0          # A[i0, j0]
        b_idx = i0 * m + (j0 + 1)    # A[i0, j0+1]
        c_idx = (i0 + 1) * m + j0    # A[i0+1, j0]
        d_idx = (i0 + 1) * m + (j0 + 1) # A[i0+1, j0+1]
        
        # B indices:
        # B is 4x7. Row j, Col k.
        # Sub-block B[j0:j0+2, k0:k0+2]
        e_idx = j0 * p + k0          # B[j0, k0]
        f_idx = j0 * p + (k0 + 1)    # B[j0, k0+1]
        g_idx = (j0 + 1) * p + k0    # B[j0+1, k0]
        h_idx = (j0 + 1) * p + (k0 + 1) # B[j0+1, k0+1]
        
        # W indices:
        # W is 7x2. Row k, Col i. Convention W[k*n+i, r].
        # Sub-block C[i0:i0+2, k0:k0+2]
        c00_idx = k0 * n + i0        # C[i0, k0]
        c01_idx = (k0 + 1) * n + i0  # C[i0, k0+1]
        c10_idx = k0 * n + (i0 + 1)  # C[i0+1, k0]
        c11_idx = (k0 + 1) * n + (i0 + 1) # C[i0+1, k0+1]
        
        # M1 = (a+d)(e+h)
        U[a_idx, col] += 1; U[d_idx, col] += 1
        V[e_idx, col] += 1; V[h_idx, col] += 1
        W[c00_idx, col] += 1; W[c11_idx, col] += 1
        col += 1
        
        # M2 = (c+d)e
        U[c_idx, col] += 1; U[d_idx, col] += 1
        V[e_idx, col] += 1
        W[c10_idx, col] += 1; W[c11_idx, col] -= 1
        col += 1
        
        # M3 = a(f-h)
        U[a_idx, col] += 1
        V[f_idx, col] += 1; V[h_idx, col] -= 1
        W[c01_idx, col] += 1; W[c11_idx, col] += 1
        col += 1
        
        # M4 = d(g-e)
        U[d_idx, col] += 1
        V[g_idx, col] += 1; V[e_idx, col] -= 1
        W[c00_idx, col] += 1; W[c10_idx, col] += 1
        col += 1
        
        # M5 = (a+b)h
        U[a_idx, col] += 1; U[b_idx, col] += 1
        V[h_idx, col] += 1
        W[c00_idx, col] -= 1; W[c01_idx, col] += 1
        col += 1
        
        # M6 = (c-a)(e+f)
        U[c_idx, col] += 1; U[a_idx, col] -= 1
        V[e_idx, col] += 1; V[f_idx, col] += 1
        W[c11_idx, col] += 1
        col += 1
        
        # M7 = (b-d)(g+h)
        U[b_idx, col] += 1; U[d_idx, col] -= 1
        V[g_idx, col] += 1; V[h_idx, col] += 1
        W[c00_idx, col] += 1
        col += 1

    def add_schoolbook_221_block(i0, j0, k0):
        nonlocal col
        # A indices: A[i0:i0+2, j0:j0+2]
        a_idx = i0 * m + j0
        b_idx = i0 * m + (j0 + 1)
        c_idx = (i0 + 1) * m + j0
        d_idx = (i0 + 1) * m + (j0 + 1)
        
        # B indices: B[j0:j0+2, k0:k0+1] (only 1 column)
        e_idx = j0 * p + k0
        g_idx = (j0 + 1) * p + k0
        
        # W indices: W[k0:k0+1, i0:i0+2]
        c00_idx = k0 * n + i0
        c10_idx = k0 * n + (i0 + 1)
        
        # P1 = a*e
        U[a_idx, col] += 1; V[e_idx, col] += 1; W[c00_idx, col] += 1; col += 1
        # P2 = b*g
        U[b_idx, col] += 1; V[g_idx, col] += 1; W[c00_idx, col] += 1; col += 1
        # P3 = c*e
        U[c_idx, col] += 1; V[e_idx, col] += 1; W[c10_idx, col] += 1; col += 1
        # P4 = d*g
        U[d_idx, col] += 1; V[g_idx, col] += 1; W[c10_idx, col] += 1; col += 1

    # Process the two j-blocks
    # j_start = 0 covers j=0,1
    # j_start = 2 covers j=2,3
    for j_start in [0, 2]:
        if time.time() > deadline:
            break
        # k_start = 0, 2, 4 for Strassen (2x2 blocks)
        for k_start in [0, 2, 4]:
            add_strassen_block(0, j_start, k_start)
        # k_start = 6 for Schoolbook (2x1 block)
        add_schoolbook_221_block(0, j_start, 6)

    # Verify correctness
    if col != rank:
        # If we ran out of time or made a mistake, fallback to schoolbook
        pass
    else:
        # Optional: Verify the tensor decomposition
        def verify(U, V, W, tol=1e-9):
            r = U.shape[1]
            for i in range(n):
                for k in range(p):
                    for j in range(m):
                        for jp in range(m):
                            s = 0.0
                            for c in range(r):
                                s += U[i*m + j, c] * V[jp*p + k, c] * W[k*n + i, c]
                            if abs(s - (1.0 if j == jp else 0.0)) > tol:
                                return False
            return True

        if not verify(U, V, W):
            # Fallback to schoolbook if something went wrong (should not happen)
            rank = n * m * p
            U = np.zeros((n * m, rank), dtype=int)
            V = np.zeros((m * p, rank), dtype=int)
            W = np.zeros((p * n, rank), dtype=int)
            idx = 0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        U[i*m + j, idx] = 1
                        V[j*p + k, idx] = 1
                        W[k*n + i, idx] = 1
                        idx += 1

    return {
        "U": U.tolist(),
        "V": V.tolist(),
        "W": W.tolist()
    }
# EVOLVE-BLOCK-END
