"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [6, 6, 6]

# EVOLVE-BLOCK-START
import time

def construct():
    start_time = time.time()
    deadline = 120.0  # safety margin
    
    n, m, p = DIMENSIONS
    
    # Use 2x2 block partitioning of 6x6 into 3x3 grid of 2x2 blocks
    # Each 2x2 block multiplication uses Strassen (rank 7)
    # Total rank = 27 * 7 = 189
    rank = 189
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Strassen terms for 2x2 matrix multiplication
    # M1 = (A00 + A11)(B00 + B11)
    # M2 = (A10 + A11)B00
    # M3 = A00(B01 - B11)
    # M4 = A11(B10 - B00)
    # M5 = (A00 + A01)B11
    # M6 = (A10 - A00)(B00 + B01)
    # M7 = (A01 - A11)(B10 + B11)
    # C00 = M1 + M4 - M5 - M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    strassen_terms = [
        # Term 0: M1 = (A00 + A11)(B00 + B11)
        {
            'U': [(0, 0, 1), (1, 1, 1)],
            'V': [(0, 0, 1), (1, 1, 1)],
            'W': [(0, 0, 1), (1, 1, 1)]
        },
        # Term 1: M2 = (A10 + A11)B00
        {
            'U': [(1, 0, 1), (1, 1, 1)],
            'V': [(0, 0, 1)],
            'W': [(1, 0, 1), (1, 1, -1)]
        },
        # Term 2: M3 = A00(B01 - B11)
        {
            'U': [(0, 0, 1)],
            'V': [(0, 1, 1), (1, 1, -1)],
            'W': [(0, 1, 1), (1, 1, 1)]
        },
        # Term 3: M4 = A11(B10 - B00)
        {
            'U': [(1, 1, 1)],
            'V': [(1, 0, 1), (0, 0, -1)],
            'W': [(0, 0, 1), (1, 0, 1)]
        },
        # Term 4: M5 = (A00 + A01)B11
        {
            'U': [(0, 0, 1), (0, 1, 1)],
            'V': [(1, 1, 1)],
            'W': [(0, 0, -1), (0, 1, 1)]
        },
        # Term 5: M6 = (A10 - A00)(B00 + B01)
        {
            'U': [(1, 0, 1), (0, 0, -1)],
            'V': [(0, 0, 1), (0, 1, 1)],
            'W': [(1, 1, 1)]
        },
        # Term 6: M7 = (A01 - A11)(B10 + B11)
        {
            'U': [(0, 1, 1), (1, 1, -1)],
            'V': [(1, 0, 1), (1, 1, 1)],
            'W': [(0, 0, 1)]
        }
    ]
    
    # Iterate over all block positions
    for i in range(3): # Row block of A, Row block of C
        for j in range(3): # Col block of B, Col block of C
            for k in range(3): # Col block of A, Row block of B
                for t in range(7):
                    col_idx = (i * 9 + j * 3 + k) * 7 + t
                    term = strassen_terms[t]
                    
                    # Update U
                    for lr, lc, coeff in term['U']:
                        u_row = (2 * i + lr) * m + (2 * k + lc)
                        U[u_row][col_idx] += coeff
                        
                    # Update V
                    for lr, lc, coeff in term['V']:
                        v_row = (2 * k + lr) * p + (2 * j + lc)
                        V[v_row][col_idx] += coeff
                        
                    # Update W
                    for lr, lc, coeff in term['W']:
                        w_row = (2 * j + lc) * n + (2 * i + lr)
                        W[w_row][col_idx] += coeff

    # Verify correctness with a simple test
    if time.time() < deadline - 5.0:
        import numpy as np
        U_np = np.array(U, dtype=float)
        V_np = np.array(V, dtype=float)
        W_np = np.array(W, dtype=float)
        
        # Test with random matrices
        np.random.seed(42)
        A_test = np.random.randint(-5, 5, (n, m))
        B_test = np.random.randint(-5, 5, (m, p))
        C_expected = A_test @ B_test
        
        # Compute using tensor decomposition
        # C[i,j] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
        # But wait, the convention is:
        # U has shape (n*m, rank) - indexed by (i, j) where i is row of A, j is col of A
        # V has shape (m*p, rank) - indexed by (j, k) where j is row of B, k is col of B
        # W has shape (p*n, rank) - indexed by (k, i) where k is col of C, i is row of C
        # C[i,k] = sum_j A[i,j]*B[j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
        
        C_computed = np.zeros((n, p), dtype=float)
        for i in range(n):
            for k in range(p):
                total = 0.0
                for j in range(m):
                    for r in range(rank):
                        total += U_np[i*m+j, r] * V_np[j*p+k, r] * W_np[k*n+i, r]
                C_computed[i, k] = total
        
        if not np.allclose(C_computed, C_expected):
            # Fallback to schoolbook if verification fails
            rank_school = n * m * p
            U_school = [[0] * rank_school for _ in range(n * m)]
            V_school = [[0] * rank_school for _ in range(m * p)]
            W_school = [[0] * rank_school for _ in range(p * n)]
            
            idx = 0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        U_school[i*m+j][idx] = 1
                        V_school[j*p+k][idx] = 1
                        W_school[k*n+i][idx] = 1
                        idx += 1
            
            return {"U": U_school, "V": V_school, "W": W_school}
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
