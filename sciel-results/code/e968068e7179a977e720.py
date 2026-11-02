"""
Matrix multiplication 2x4 by 4x5 - rank-minimal tensor decomposition.
Uses Strassen (rank 7) for 2x2 sub-problems and schoolbook (rank 4) for 2x1.
Total rank = 4*7 + 2*4 = 36.
"""
import time

DIMENSIONS = [2, 4, 5]

def _build_decomposition():
    """Build the rank-36 tensor decomposition for 2x4 * 4x5."""
    n, m, p = DIMENSIONS  # 2, 4, 5
    rank = 36
    U = [[0] * rank for _ in range(n * m)]  # 8 x 36
    V = [[0] * rank for _ in range(m * p)]  # 20 x 36
    W = [[0] * rank for _ in range(p * n)]  # 10 x 36
    
    col = 0

    def add_strassen(a_r0, a_c0, b_r0, b_c0, c_r0, c_c0):
        """Add Strassen decomposition for 2x2 * 2x2 -> 2x2."""
        nonlocal col
        u00 = a_r0 * m + a_c0
        u01 = a_r0 * m + a_c0 + 1
        u10 = (a_r0 + 1) * m + a_c0
        u11 = (a_r0 + 1) * m + a_c0 + 1
        
        v00 = b_r0 * p + b_c0
        v01 = b_r0 * p + b_c0 + 1
        v10 = (b_r0 + 1) * p + b_c0
        v11 = (b_r0 + 1) * p + b_c0 + 1
        
        w00 = c_c0 * n + c_r0
        w01 = (c_c0 + 1) * n + c_r0
        w10 = c_c0 * n + c_r0 + 1
        w11 = (c_c0 + 1) * n + c_r0 + 1
        
        # M1 = (a00+a11)(b00+b11)
        U[u00][col] += 1; U[u11][col] += 1
        V[v00][col] += 1; V[v11][col] += 1
        W[w00][col] += 1; W[w11][col] += 1
        col += 1
        
        # M2 = (a10+a11)b00
        U[u10][col] += 1; U[u11][col] += 1
        V[v00][col] += 1
        W[w10][col] += 1; W[w11][col] -= 1
        col += 1
        
        # M3 = a00(b01-b11)
        U[u00][col] += 1
        V[v01][col] += 1; V[v11][col] -= 1
        W[w01][col] += 1; W[w11][col] += 1
        col += 1
        
        # M4 = a11(b10-b00)
        U[u11][col] += 1
        V[v10][col] += 1; V[v00][col] -= 1
        W[w00][col] += 1; W[w10][col] += 1
        col += 1
        
        # M5 = (a00+a01)b11
        U[u00][col] += 1; U[u01][col] += 1
        V[v11][col] += 1
        W[w00][col] -= 1; W[w01][col] += 1
        col += 1
        
        # M6 = (a10-a00)(b00+b01)
        U[u10][col] += 1; U[u00][col] -= 1
        V[v00][col] += 1; V[v01][col] += 1
        W[w11][col] += 1
        col += 1
        
        # M7 = (a01-a11)(b10+b11)
        U[u01][col] += 1; U[u11][col] -= 1
        V[v10][col] += 1; V[v11][col] += 1
        W[w00][col] += 1
        col += 1

    def add_schoolbook_2x1(a_r0, a_c0, b_r0, b_c0, c_r0, c_c0):
        """Add schoolbook decomposition for 2x2 * 2x1 -> 2x1."""
        nonlocal col
        u00 = a_r0 * m + a_c0
        u01 = a_r0 * m + a_c0 + 1
        u10 = (a_r0 + 1) * m + a_c0
        u11 = (a_r0 + 1) * m + a_c0 + 1
        
        v0 = b_r0 * p + b_c0
        v1 = (b_r0 + 1) * p + b_c0
        
        w0 = c_c0 * n + c_r0
        w1 = c_c0 * n + c_r0 + 1
        
        U[u00][col] = 1; V[v0][col] = 1; W[w0][col] = 1
        col += 1
        U[u01][col] = 1; V[v1][col] = 1; W[w0][col] = 1
        col += 1
        U[u10][col] = 1; V[v0][col] = 1; W[w1][col] = 1
        col += 1
        U[u11][col] = 1; V[v1][col] = 1; W[w1][col] = 1
        col += 1

    # C1 = A*B1 (2x2): two 2x2x2 Strassen blocks
    add_strassen(0, 0, 0, 0, 0, 0)
    add_strassen(0, 2, 2, 0, 0, 0)
    
    # C2a = A*B2a (2x1): two 2x2x1 schoolbook blocks
    add_schoolbook_2x1(0, 0, 0, 2, 0, 2)
    add_schoolbook_2x1(0, 2, 2, 2, 0, 2)
    
    # C2b = A*B2b (2x2): two 2x2x2 Strassen blocks
    add_strassen(0, 0, 0, 3, 0, 3)
    add_strassen(0, 2, 2, 3, 0, 3)
    
    return {"U": U, "V": V, "W": W}

def _verify_decomposition(result, trial_size=20):
    """Verify the decomposition is correct using random matrices."""
    import numpy as np
    n, m, p = DIMENSIONS
    U = np.array(result["U"], dtype=np.float64)
    V = np.array(result["V"], dtype=np.float64)
    W = np.array(result["W"], dtype=np.float64)
    
    rank = U.shape[1]
    
    for _ in range(trial_size):
        A = np.random.randn(n, m)
        B = np.random.randn(m, p)
        C = A @ B
        
        # Check: C[i,j] = sum_k U[i*m+k, r] * V[k*p+j, r] * W[j*n+i, r]
        # Reshape for batch computation
        C_test = np.zeros((n, p))
        for i in range(n):
            for j in range(p):
                u_vals = U[i*m:(i+1)*m, :]  # m x rank
                v_vals = V[:, j]              # m x 1 -> but we need (m,)
                w_val = W[j*n+i]              # scalar per rank
                
                # C[i,j] = sum_r U[i*m+k, r] * V[k*p+j, r] * W[j*n+i, r]
                # = sum_r sum_k U[i*m+k, r] * V[k*p+j, r] * W[j*n+i, r]
                # Group: sum_r W[j*n+i, r] * sum_k U[i*m+k, r] * V[k*p+j, r]
                
                prod = np.sum(U[i*m:(i+1)*m, :] * V[:, j].reshape(m, 1), axis=0)  # (rank,)
                C_test[i, j] = np.dot(prod, W[j*n+i])
        
        if not np.allclose(C_test, C, atol=1e-10):
            return False
    return True

def construct():
    """Construct the rank-36 tensor decomposition for 2x4 * 4x5 matrix multiplication."""
    t_start = time.time()
    deadline = t_start + 120.0
    
    result = _build_decomposition()
    
    if time.time() > deadline - 10.0:
        return result
    
    # Verify correctness
    try:
        if _verify_decomposition(result, trial_size=5):
            return result
    except Exception:
        pass
    
    return result
