import numpy as np
import time

def construct():
    n, m, p = 2, 4, 8
    start_time = time.time()
    DEADLINE = start_time + 135  # safety margin before 150s kill
    
    # Build the rank-56 Strassen decomposition
    rank = 56
    U = np.zeros((n * m, rank))  # (8, 56)
    V = np.zeros((m * p, rank))  # (32, 56)
    W = np.zeros((p * n, rank))  # (16, 56)
    
    # The 8 Strassen blocks:
    # Each block: (a_col_off, b_row_off, b_col_off, c_col_off)
    # A submatrix: A[0:2, a_col_off:a_col_off+2]
    # B submatrix: B[b_row_off:b_row_off+2, b_col_off:b_col_off+2]
    # C submatrix: C[0:2, c_col_off:c_col_off+2]
    
    blocks = [
        (0, 0, 0, 0),  # A[:,0:2] * B[0:2,0:2] -> C[:,0:2]
        (2, 2, 0, 0),  # A[:,2:4] * B[2:4,0:2] -> C[:,0:2]
        (0, 0, 2, 2),  # A[:,0:2] * B[0:2,2:4] -> C[:,2:4]
        (2, 2, 2, 2),  # A[:,2:4] * B[2:4,2:4] -> C[:,2:4]
        (0, 0, 4, 4),  # A[:,0:2] * B[0:2,4:6] -> C[:,4:6]
        (2, 2, 4, 4),  # A[:,2:4] * B[2:4,4:6] -> C[:,4:6]
        (0, 0, 6, 6),  # A[:,0:2] * B[0:2,6:8] -> C[:,6:8]
        (2, 2, 6, 6),  # A[:,2:4] * B[2:4,6:8] -> C[:,6:8]
    ]
    
    col = 0
    for (a_col_off, b_row_off, b_col_off, c_col_off) in blocks:
        # A indices: A[i,j] -> i*m + j = i*4 + j
        a00 = 0 * m + a_col_off      # A[0, a_col_off]
        a01 = 0 * m + a_col_off + 1  # A[0, a_col_off+1]
        a10 = 1 * m + a_col_off      # A[1, a_col_off]
        a11 = 1 * m + a_col_off + 1  # A[1, a_col_off+1]
        
        # B indices: B[j,k] -> j*p + k = j*8 + k
        b00 = b_row_off * p + b_col_off
        b01 = b_row_off * p + b_col_off + 1
        b10 = (b_row_off + 1) * p + b_col_off
        b11 = (b_row_off + 1) * p + b_col_off + 1
        
        # C indices: C[i,k] -> k*n + i (trace convention W[k*n+i,r])
        c00 = c_col_off * n + 0  # C[0, c_col_off]
        c01 = (c_col_off + 1) * n + 0  # C[0, c_col_off+1]
        c10 = c_col_off * n + 1  # C[1, c_col_off]
        c11 = (c_col_off + 1) * n + 1  # C[1, c_col_off+1]
        
        # Strassen 7 multiplications for 2x2 @ 2x2:
        # A = [[a00, a01], [a10, a11]], B = [[b00, b01], [b10, b11]]
        
        # P1 = (a00+a11)(b00+b11)
        U[a00, col] += 1
        U[a11, col] += 1
        V[b00, col] += 1
        V[b11, col] += 1
        W[c00, col] += 1   # C00 += P1
        W[c11, col] += 1   # C11 += P1
        col += 1
        
        # P2 = (a10+a11)*b00
        U[a10, col] += 1
        U[a11, col] += 1
        V[b00, col] += 1
        W[c10, col] += 1   # C10 += P2
        W[c11, col] += -1  # C11 -= P2
        col += 1
        
        # P3 = a00*(b01-b11)
        U[a00, col] += 1
        V[b01, col] += 1
        V[b11, col] += -1
        W[c01, col] += 1   # C01 += P3
        W[c11, col] += 1   # C11 += P3
        col += 1
        
        # P4 = a11*(b10-b00)
        U[a11, col] += 1
        V[b10, col] += 1
        V[b00, col] += -1
        W[c00, col] += 1   # C00 += P4
        W[c10, col] += 1   # C10 += P4
        col += 1
        
        # P5 = (a00+a01)*b11
        U[a00, col] += 1
        U[a01, col] += 1
        V[b11, col] += 1
        W[c00, col] += -1  # C00 -= P5
        W[c01, col] += 1   # C01 += P5
        col += 1
        
        # P6 = (a10-a00)*(b00+b01)
        U[a10, col] += 1
        U[a00, col] += -1
        V[b00, col] += 1
        V[b01, col] += 1
        W[c11, col] += 1   # C11 += P6
        col += 1
        
        # P7 = (a01-a11)*(b10+b11)
        U[a01, col] += 1
        U[a11, col] += -1
        V[b10, col] += 1
        V[b11, col] += 1
        W[c00, col] += 1   # C00 += P7
        col += 1
    
    assert col == rank, f"Expected {rank} columns, got {col}"
    
    # Verify correctness with a random test
    A = np.random.randn(n, m)
    B = np.random.randn(m, p)
    C_expected = A @ B
    
    # Reconstruct C from decomposition
    C_recon = np.zeros((n, p))
    for r in range(rank):
        u = U[:, r]  # shape (8,)
        v = V[:, r]  # shape (32,)
        w = W[:, r]  # shape (16,)
        
        # u[i*m+j] * v[j*p+k] * w[k*n+i] summed over i,j,k
        for i in range(n):
            for k in range(p):
                for j in range(m):
                    C_recon[i, k] += u[i * m + j] * v[j * p + k] * w[k * n + i]
    
    if not np.allclose(C_recon, C_expected, atol=1e-10):
        # Fallback: return a valid (but higher rank) schoolbook decomposition
        rank_school = n * m * p  # 64
        U_s = np.zeros((n * m, rank_school))
        V_s = np.zeros((m * p, rank_school))
        W_s = np.zeros((p * n, rank_school))
        col_s = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U_s[i * m + j, col_s] = 1
                    V_s[j * p + k, col_s] = 1
                    W_s[k * n + i, col_s] = 1
                    col_s += 1
        assert col_s == rank_school
        return {"U": U_s.tolist(), "V": V_s.tolist(), "W": W_s.tolist()}
    
    return {
        "U": U.tolist(),
        "V": V.tolist(),
        "W": W.tolist()
    }
