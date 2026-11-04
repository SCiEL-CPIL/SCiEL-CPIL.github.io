"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time

DIMENSIONS = [4, 4, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # 4, 4, 5
    # Schoolbook rank
    rank_schoolbook = n * m * p  # 80
    
    # Build schoolbook decomposition
    rank = rank_schoolbook
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1
    
    best_U, best_V, best_W, best_rank = U.copy(), V.copy(), W.copy(), rank
    
    # Deadline guard: 140s safety margin within 150s limit
    start_time = time.time()
    deadline = start_time + 140
    
    # Strategy: Try to find a lower-rank decomposition using structured search.
    # For 4x4x5, known best is 65 (Strassen for 4x4x4 + schoolbook for 4x4x1).
    # Let's try to find rank 65 first, then try lower.
    
    # Helper: verify a decomposition
    def verify(U_try, V_try, W_try, tol=1e-9):
        n, m, p = DIMENSIONS
        # Check on-support entries: for each (i,j,k), sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] should be 1
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    a = i * m + j
                    b = j * p + k
                    c = k * n + i
                    val = np.dot(U_try[a, :], V_try[b, :] * W_try[c, :])
                    if abs(val - 1.0) > tol:
                        return False
        # Check off-support entries: for random samples, should be 0
        rng = np.random.RandomState(42)
        for _ in range(200):
            a = rng.randint(0, n * m)
            b = rng.randint(0, m * p)
            c = rng.randint(0, p * n)
            i1, j1 = divmod(a, m)
            j2, k2 = divmod(b, p)
            i2, k1 = divmod(c, n)
            if j1 == j2 and i1 == i2 and k1 == k2:
                continue  # on support
            val = np.dot(U_try[a, :], V_try[b, :] * W_try[c, :])
            if abs(val) > tol:
                return False
        return True
    
    # Try to construct rank-65 decomposition using Strassen for 4x4x4 part + schoolbook for 4x4x1
    # C[4x5] = A[4x4] * B[4x5]
    # Split B into B1[4x4] and B2[4x1]
    # C = A*B1 + A*B2
    # A*B1: use Strassen rank 49 for 4x4x4
    # A*B2: use schoolbook rank 16 for 4x4x1
    # Total: 65
    
    # Strassen 2x2x2 decomposition:
    # M1 = (A11 - A21)(B21 - B22)
    # M2 = (A11 + A12)B22
    # M3 = A11(B11 - B22)
    # M4 = A22(B11 + B22)
    # M5 = (A12 - A22)(B11 + B21)
    # M6 = (A11 + A21)(B22 - B21)
    # M7 = (A21 - A11)(B11 + B12)
    # C11 = M1 + M4 - M5 + M7
    # C12 = M2 + M3
    # C21 = M1 + M6
    # C22 = M3 + M4 - M2 + M5
    
    # For 4x4 matrices, we can use block Strassen with 2x2 blocks.
    # But the rank of 4x4x4 multiplication is 49 (not 28).
    # Let's use the known rank-49 decomposition for 4x4x4.
    
    # Actually, let's just try random search for ranks from 60 to 80, with structured initialization.
    
    # Try rank 65 with a structured approach
    for test_rank in [65, 60, 55, 50, 45, 40, 35, 30, 25, 20]:
        if time.time() > deadline:
            break
        
        found = False
        for attempt in range(10):
            if time.time() > deadline:
                break
            
            # Initialize with some structure
            U_try = np.zeros((n * m, test_rank))
            V_try = np.zeros((m * p, test_rank))
            W_try = np.zeros((p * n, test_rank))
            
            # Fill first min(test_rank, 80) columns with schoolbook pattern if possible
            cols_to_fill = min(test_rank, n * m * p)
            for c in range(cols_to_fill):
                i = c // (m * p)
                j = (c // p) % m
                k = c % p
                U_try[i * m + j, c] = 1
                V_try[j * p + k, c] = 1
                W_try[k * n + i, c] = 1
            
            # For remaining columns, use random values
            for c in range(cols_to_fill, test_rank):
                U_try[:, c] = np.random.randn(n * m)
                V_try[:, c] = np.random.randn(m * p)
                W_try[:, c] = np.random.randn(p * n)
            
            if verify(U_try, V_try, W_try):
                best_U, best_V, best_W, best_rank = U_try.copy(), V_try.copy(), W_try.copy(), test_rank
                found = True
                break
        
        if found:
            break
    
    # If we didn't find a lower rank, return the schoolbook decomposition
    result = {
        "U": best_U.tolist(),
        "V": best_V.tolist(),
        "W": best_W.tolist()
    }
    return result
# EVOLVE-BLOCK-END
