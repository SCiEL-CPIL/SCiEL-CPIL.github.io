"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 10]

# EVOLVE-BLOCK-START
import time
import numpy as np

def _verify(U, V, W, n, m, p, tol=1e-9):
    """Verify that the decomposition is exact using numpy for speed."""
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    rank = U_arr.shape[1]
    for i in range(n):
        for j in range(m):
            for k in range(p):
                total = np.dot(U_arr[i*m+j], V_arr[j*p+k] * W_arr[k*n+i])
                if abs(total - 1.0) > tol:
                    return False
    return True

def _schoolbook(n, m, p):
    """Return schoolbook decomposition of rank n*m*p."""
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

def construct():
    start_time = time.time()
    deadline = 145
    
    n, m, p = DIMENSIONS
    
    # Try rank 50 (5 terms per column)
    # For each k, we have 5 terms.
    # Terms for M_2,3,1:
    # 1. A00 B0 -> C0
    # 2. A01 B1 -> C0
    # 3. A02 B2 -> C0
    # 4. (A10-A00) B0 -> C1
    # 5. (A11-A01) B1 -> C1
    # And C1 also gets A12 B2? 
    # If rank is 5, one term must do double duty or be combined.
    # Let's try the 6-term (rank 60) first as it is guaranteed.
    # But we need to beat 60.
    # I will implement the 5-term hypothesis.
    
    rank = 50
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    for k in range(p):
        # Term 1: A[0,0] B[0,k] -> C[0,k]
        U[0][col] = 1
        V[0*p+k][col] = 1
        W[k*n+0][col] = 1
        # Also contribute to C[1,k]? 
        # If we add to C[1,k], we need to subtract later.
        # Let's assume the 5-term decomposition is:
        # T1: A00 B0 -> C0, C1
        # T2: A01 B1 -> C0, C1
        # T3: A02 B2 -> C0
        # T4: (A10-A00) B0 -> C1
        # T5: (A11-A01) B1 -> C1
        # Missing: A12 B2 for C1.
        # If T3 also contributes to C1?
        # Then C1 = T1+T2+T3+T4+T5 = A00B0+A01B1+A02B2 + A10B0-A00B0 + A11B1-A01B1 = A10B0+A11B1+A02B2.
        # We need A12B2. So we need A12B2 - A02B2 = (A12-A02)B2.
        # This would be a 6th term.
        # So rank 50 might not be possible with this simple structure.
        
        # Let's just use schoolbook rank 60.
        pass
        
    # Fallback to schoolbook
    sb = _schoolbook(n, m, p)
    return sb
# EVOLVE-BLOCK-END
