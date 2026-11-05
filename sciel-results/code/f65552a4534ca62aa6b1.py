import numpy as np
from fractions import Fraction
import time

def construct():
    n, m, p = 2, 4, 8
    start_time = time.time()
    deadline = start_time + 120.0
    
    # Build the matrix multiplication tensor
    T = np.zeros((n*m, m*p, p*n), dtype=np.float64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u_idx = i * m + j
                v_idx = j * p + k
                w_idx = k * n + i
                T[u_idx, v_idx, w_idx] = 1.0
    
    # Strassen-based decomposition with rank 56
    blocks = [
        (0, 0, 0, 0),
        (0, 0, 2, 2),
        (0, 0, 4, 4),
        (0, 0, 6, 6),
        (2, 2, 0, 0),
        (2, 2, 2, 2),
        (2, 2, 4, 4),
        (2, 2, 6, 6),
    ]
    
    rank = 56
    U = np.zeros((n*m, rank))
    V = np.zeros((m*p, rank))
    W = np.zeros((p*n, rank))
    
    col = 0
    for (a_col_off, b_row_off, b_col_off, c_col_off) in blocks:
        a00 = 0*m + a_col_off
        a01 = 0*m + a_col_off + 1
        a10 = 1*m + a_col_off
        a11 = 1*m + a_col_off + 1
        
        b00 = b_row_off*p + b_col_off
        b01 = b_row_off*p + b_col_off + 1
        b10 = (b_row_off+1)*p + b_col_off
        b11 = (b_row_off+1)*p + b_col_off + 1
        
        c00 = c_col_off*n + 0
        c01 = (c_col_off+1)*n + 0
        c10 = c_col_off*n + 1
        c11 = (c_col_off+1)*n + 1
        
        # P1 = (a00+a11)(b00+b11)
        U[a00, col] += 1
        U[a11, col] += 1
        V[b00, col] += 1
        V[b11, col] += 1
        W[c00, col] += 1
        W[c11, col] += 1
        col += 1
        
        # P2 = (a10+a11)*b00
        U[a10, col] += 1
        U[a11, col] += 1
        V[b00, col] += 1
        W[c10, col] += 1
        W[c11, col] += -1
        col += 1
        
        # P3 = a00*(b01-b11)
        U[a00, col] += 1
        V[b01, col] += 1
        V[b11, col] += -1
        W[c01, col] += 1
        W[c11, col] += 1
        col += 1
        
        # P4 = a11*(b10-b00)
        U[a11, col] += 1
        V[b10, col] += 1
        V[b00, col] += -1
        W[c00, col] += 1
        W[c10, col] += 1
        col += 1
        
        # P5 = (a00+a01)*b11
        U[a00, col] += 1
        U[a01, col] += 1
        V[b11, col] += 1
        W[c00, col] += -1
        W[c01, col] += 1
        col += 1
        
        # P6 = (a10-a00)*(b00+b01)
        U[a10, col] += 1
        U[a00, col] += -1
        V[b00, col] += 1
        V[b01, col] += 1
        W[c11, col] += 1
        col += 1
        
        # P7 = (a01-a11)*(b10+b11)
        U[a01, col] += 1
        U[a11, col] += -1
        V[b10, col] += 1
        V[b11, col] += 1
        W[c00, col] += 1
        col += 1
    
    assert col == rank
    
    # Verify correctness with a few random tests
    for trial in range(3):
        A = np.random.randn(n, m)
        B = np.random.randn(m, p)
        C_expected = A @ B
        C_computed = np.zeros((n, p))
        for r in range(rank):
            a_lin = np.zeros(m)
            for i in range(n):
                for j in range(m):
                    if U[i*m+j, r] != 0:
                        a_lin[j] += U[i*m+j, r] * A[i, j]
            b_lin = np.zeros(p)
            for j in range(m):
                for k in range(p):
                    if V[j*p+k, r] != 0:
                        b_lin[k] += V[j*p+k, r] * B[j, k]
            for i in range(n):
                for k in range(p):
                    C_computed[i, k] += W[k*n+i, r] * a_lin[j] * b_lin[k] if False else 0
        # Actually let me verify more carefully
        C_computed = np.zeros((n, p))
        for r in range(rank):
            a_proj = np.zeros(n)
            b_proj = np.zeros(p)
            for i in range(n):
                for j in range(m):
                    a_proj[i] += U[i*m+j, r] * A[i, j]
            for j in range(m):
                for k in range(p):
                    b_proj[k] += V[j*p+k, r] * B[j, k]
            # This isn't right either. Let me just check the tensor identity directly.
            pass
        
        # Direct tensor check
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    u_idx = i * m + j
                    v_idx = j * p + k
                    w_idx = k * n + i
                    val = 0.0
                    for r in range(rank):
                        val += U[u_idx, r] * V[v_idx, r] * W[w_idx, r]
                    assert abs(val - T[u_idx, v_idx, w_idx]) < 1e-10, f"Tensor check failed at ({u_idx},{v_idx},{w_idx}): got {val}, expected {T[u_idx,v_idx,w_idx]}"
    
    if time.time() > deadline - 5.0:
        return {
            "U": U.tolist(),
            "V": V.tolist(),
            "W": W.tolist()
        }
    
    return {
        "U": U.tolist(),
        "V": V.tolist(),
        "W": W.tolist()
    }
