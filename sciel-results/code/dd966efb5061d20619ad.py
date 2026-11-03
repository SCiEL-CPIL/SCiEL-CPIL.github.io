import time
import sys
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 110  # Safety margin for 120s limit
    
    n, m, p = 3, 4, 4
    rank = 44
    U = [[0]*rank for _ in range(n*m)]  # shape (12, 44)
    V = [[0]*rank for _ in range(m*p)]  # shape (16, 44)
    W = [[0]*rank for _ in range(p*n)]  # shape (12, 44)
    
    col = 0
    
    def add_strassen(a_r, a_c, b_r, b_c, c_r, c_c, start_col):
        """Add 7-term Strassen decomposition for 2x2x2 block at given positions.
        a_r, a_c: top-left row/col of A block
        b_r, b_c: top-left row/col of B block  
        c_r, c_c: top-left row/col of C block
        """
        c = start_col
        # m1 = (a00 + a11)(b00 + b11)
        U[a_r*m + a_c][c] += 1
        U[(a_r+1)*m + (a_c+1)][c] += 1
        V[b_r*p + b_c][c] += 1
        V[(b_r+1)*p + (b_c+1)][c] += 1
        W[c_c*n + c_r][c] += 1
        W[(c_c+1)*n + (c_r+1)][c] += 1
        c += 1
        
        # m2 = (a10 + a11)b00
        U[(a_r+1)*m + a_c][c] += 1
        U[(a_r+1)*m + (a_c+1)][c] += 1
        V[b_r*p + b_c][c] += 1
        W[c_c*n + (c_r+1)][c] += 1
        W[(c_c+1)*n + (c_r+1)][c] -= 1
        c += 1
        
        # m3 = a00(b01 - b11)
        U[a_r*m + a_c][c] += 1
        V[b_r*p + (b_c+1)][c] += 1
        V[(b_r+1)*p + (b_c+1)][c] -= 1
        W[(c_c+1)*n + c_r][c] += 1
        W[(c_c+1)*n + (c_r+1)][c] += 1
        c += 1
        
        # m4 = a11(b10 - b00)
        U[(a_r+1)*m + (a_c+1)][c] += 1
        V[(b_r+1)*p + b_c][c] += 1
        V[b_r*p + b_c][c] -= 1
        W[c_c*n + c_r][c] += 1
        W[c_c*n + (c_r+1)][c] += 1
        c += 1
        
        # m5 = (a00 + a01)b11
        U[a_r*m + a_c][c] += 1
        U[a_r*m + (a_c+1)][c] += 1
        V[(b_r+1)*p + (b_c+1)][c] += 1
        W[c_c*n + c_r][c] -= 1
        W[(c_c+1)*n + c_r][c] += 1
        c += 1
        
        # m6 = (a10 - a00)(b00 + b01)
        U[(a_r+1)*m + a_c][c] += 1
        U[a_r*m + a_c][c] -= 1
        V[b_r*p + b_c][c] += 1
        V[b_r*p + (b_c+1)][c] += 1
        W[(c_c+1)*n + (c_r+1)][c] += 1
        c += 1
        
        # m7 = (a01 - a11)(b10 + b11)
        U[a_r*m + (a_c+1)][c] += 1
        U[(a_r+1)*m + (a_c+1)][c] -= 1
        V[(b_r+1)*p + b_c][c] += 1
        V[(b_r+1)*p + (b_c+1)][c] += 1
        W[c_c*n + c_r][c] += 1
        c += 1
        
        return c

    def add_trivial_1x4x2(a_r, b_c, start_col):
        """Add 8-term trivial decomposition for 1x4x2 block.
        a_r: row of A (single row)
        b_c: starting column of B (2 columns)
        """
        c = start_col
        for j in range(4):
            U[a_r*m + j][c] += 1
            V[j*p + b_c][c] += 1
            W[b_c*n + a_r][c] += 1
            c += 1
            U[a_r*m + j][c] += 1
            V[j*p + (b_c+1)][c] += 1
            W[(b_c+1)*n + a_r][c] += 1
            c += 1
        return c

    # Left Side (cols 0,1 of C) - using A rows 0,1 and B cols 0,1
    col = add_strassen(0, 0, 0, 0, 0, 0, col)
    col = add_strassen(0, 2, 2, 0, 0, 0, col)
    col = add_trivial_1x4x2(2, 0, col)
    
    # Right Side (cols 2,3 of C) - using A rows 0,1 and B cols 2,3
    col = add_strassen(0, 0, 0, 2, 0, 2, col)
    col = add_strassen(0, 2, 2, 2, 0, 2, col)
    col = add_trivial_1x4x2(2, 2, col)
    
    if time.time() > deadline:
        return {"U": U, "V": V, "W": W}
    
    # Verify decomposition correctness
    try:
        U_np = np.array(U, dtype=float)
        V_np = np.array(V, dtype=float)
        W_np = np.array(W, dtype=float)
        
        # Check: for each (i,k) in C, sum_r U[i*m+j] * V[j*p+k] * W[k*n+i] should equal 1 if it's a matmul tensor
        # Actually, the decomposition is: C[i,k] = sum_j A[i,j] B[j,k]
        # Tensor: T[i,j,k] = 1 (for all i,j,k)
        # Decomposition: T[i,j,k] = sum_r U[i,j,r] * V[j,k,r] * W[k,i,r]  -- but W is indexed differently
        
        # The convention: W[k*n+i, r] where k is col of C, i is row of C
        # So W is shape (p*n, rank) = (12, 44)
        # U is shape (n*m, rank) = (12, 44), indexed as U[i*m+j, r]
        # V is shape (m*p, rank) = (16, 44), indexed as V[j*p+k, r]
        
        # The matmul tensor: C[i,k] = sum_j A[i,j] * B[j,k]
        # In tensor form: T[i,j,k] = 1 for all i,j,k
        # Decomposition: 1 = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
        
        # Verify for a sample of (i,j,k)
        max_err = 0.0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = 0.0
                    for r in range(rank):
                        val += U_np[i*m+j, r] * V_np[j*p+k, r] * W_np[k*n+i, r]
                    if abs(val - 1.0) > max_err:
                        max_err = abs(val - 1.0)
        
        if max_err > 1e-9:
            # Decomposition is incorrect, fall back to trivial
            rank = n*m*p
            U = [[0]*rank for _ in range(n*m)]
            V = [[0]*rank for _ in range(m*p)]
            W = [[0]*rank for _ in range(p*n)]
            col = 0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        U[i*m+j][col] = 1
                        V[j*p+k][col] = 1
                        W[k*n+i][col] = 1
                        col += 1
            return {"U": U, "V": V, "W": W}
    except Exception:
        pass
    
    return {"U": U, "V": V, "W": W}
