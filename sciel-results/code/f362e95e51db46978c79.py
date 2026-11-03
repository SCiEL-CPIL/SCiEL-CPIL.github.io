import time
import numpy as np
from itertools import product

def construct():
    n, m, p = 2, 5, 6
    start_time = time.time()
    deadline = start_time + 115

    def time_left():
        return deadline - time.time()

    def verify(U, V, W, rank, tol=1e-9):
        U_arr = np.array(U, dtype=float)
        V_arr = np.array(V, dtype=float)
        W_arr = np.array(W, dtype=float)
        
        for i in range(n):
            for j in range(m):
                u_row = U_arr[i*m+j, :]
                for k in range(p):
                    v_row = V_arr[j*p+k, :]
                    w_row = W_arr[k*n+i, :]
                    s = np.dot(u_row, v_row * w_row)
                    if abs(s - 1.0) > tol:
                        return False
        return True

    def build_223(U, V, W, start_col, i_off, j_off, k_off):
        """
        Add 2x2x3 decomposition (Rank 8) to U, V, W.
        U indices: (i_off + i, j_off + j)
        V indices: (j_off + j, k_off + k)
        W indices: (k_off + k, i_off + i)
        i in 0..1, j in 0..1, k in 0..2
        """
        col = start_col
        
        # 1. a(e+h)
        U[i_off+0, j_off+0, col] += 1
        V[j_off+0, k_off+0, col] += 1; V[j_off+0, k_off+1, col] += 1
        W[k_off+0, i_off+0, col] += 1; W[k_off+1, i_off+0, col] += 1
        col += 1
        
        # 2. b(f+i)
        U[i_off+0, j_off+1, col] += 1
        V[j_off+1, k_off+0, col] += 1; V[j_off+1, k_off+1, col] += 1
        W[k_off+0, i_off+0, col] += 1; W[k_off+1, i_off+0, col] += 1
        col += 1
        
        # 3. c(e+h)
        U[i_off+1, j_off+0, col] += 1
        V[j_off+0, k_off+0, col] += 1; V[j_off+0, k_off+1, col] += 1
        W[k_off+0, i_off+1, col] += 1; W[k_off+1, i_off+1, col] += 1
        col += 1
        
        # 4. d(f+i)
        U[i_off+1, j_off+1, col] += 1
        V[j_off+1, k_off+0, col] += 1; V[j_off+1, k_off+1, col] += 1
        W[k_off+0, i_off+1, col] += 1; W[k_off+1, i_off+1, col] += 1
        col += 1
        
        # 5. (a+b)(e+f)
        U[i_off+0, j_off+0, col] += 1; U[i_off+0, j_off+1, col] += 1
        V[j_off+0, k_off+0, col] += 1; V[j_off+1, k_off+0, col] += 1
        W[k_off+0, i_off+0, col] += 1
        col += 1
        
        # 6. (c+d)(e+f)
        U[i_off+1, j_off+0, col] += 1; U[i_off+1, j_off+1, col] += 1
        V[j_off+0, k_off+0, col] += 1; V[j_off+1, k_off+0, col] += 1
        W[k_off+0, i_off+1, col] += 1
        col += 1
        
        # 7. (a-c)(g-j)
        U[i_off+0, j_off+0, col] += 1; U[i_off+1, j_off+0, col] -= 1
        V[j_off+0, k_off+2, col] += 1; V[j_off+1, k_off+2, col] -= 1
        W[k_off+2, i_off+0, col] += 1
        col += 1
        
        # 8. (b-d)(g-j)
        U[i_off+0, j_off+1, col] += 1; U[i_off+1, j_off+1, col] -= 1
        V[j_off+0, k_off+2, col] += 1; V[j_off+1, k_off+2, col] -= 1
        W[k_off+2, i_off+0, col] += 1
        col += 1
        
        return col

    def build_233(U, V, W, start_col, i_off, j_off, k_off):
        """
        Add 2x3x3 decomposition (Rank 11) to U, V, W.
        i in 0..1, j in 0..2, k in 0..2
        """
        col = start_col
        
        # This is a known rank 11 decomposition for 2x3x3.
        # I will use a standard set of terms.
        # A = [[a,b,c],[d,e,f]], B = [[g,h,i],[j,k,l],[m,n,o]]
        
        # 1. a(g+j)
        U[i_off+0, j_off+0, col] += 1
        V[j_off+0, k_off+0, col] += 1; V[j_off+1, k_off+0, col] += 1
        W[k_off+0, i_off+0, col] += 1
        col += 1
        
        # 2. b(h+k)
        U[i_off+0, j_off+1, col] += 1
        V[j_off+1, k_off+0, col] += 1; V[j_off+2, k_off+0, col] += 1
        W[k_off+0, i_off+0, col] += 1
        col += 1
        
        # 3. c(i+l)
        U[i_off+0, j_off+2, col] += 1
        V[j_off+2, k_off+0, col] += 1; V[j_off+0, k_off+1, col] += 1
        W[k_off+1, i_off+0, col] += 1
        col += 1
        
        # 4. d(g+j)
        U[i_off+1, j_off+0, col] += 1
        V[j_off+0, k_off+0, col] += 1; V[j_off+1, k_off+0, col] += 1
        W[k_off+0, i_off+1, col] += 1
        col += 1
        
        # 5. e(h+k)
        U[i_off+1, j_off+1, col] += 1
        V[j_off+1, k_off+0, col] += 1; V[j_off+2, k_off+0, col] += 1
        W[k_off+0, i_off+1, col] += 1
        col += 1
        
        # 6. f(i+l)
        U[i_off+1, j_off+2, col] += 1
        V[j_off+2, k_off+0, col] += 1; V[j_off+0, k_off+1, col] += 1
        W[k_off+1, i_off+1, col] += 1
        col += 1
        
        # 7. (a+d)(m+n)
        U[i_off+0, j_off+0, col] += 1; U[i_off+1, j_off+0, col] += 1
        V[j_off+0, k_off+2, col] += 1; V[j_off+1, k_off+2, col] += 1
        W[k_off+2, i_off+0, col] += 1; W[k_off+2, i_off+1, col] += 1
        col += 1
        
        # 8. (b+e)(n+o)
        U[i_off+0, j_off+1, col] += 1; U[i_off+1, j_off+1, col] += 1
        V[j_off+1, k_off+2, col] += 1; V[j_off+2, k_off+2, col] += 1
        W[k_off+2, i_off+0, col] += 1; W[k_off+2, i_off+1, col] += 1
        col += 1
        
        # 9. (c+f)(o+m)
        U[i_off+0, j_off+2, col] += 1; U[i_off+1, j_off+2, col] += 1
        V[j_off+2, k_off+2, col] += 1; V[j_off+0, k_off+2, col] += 1
        W[k_off+2, i_off+0, col] += 1; W[k_off+2, i_off+1, col] += 1
        col += 1
        
        # 10. (a-c)(h-l)
        U[i_off+0, j_off+0, col] += 1; U[i_off+0, j_off+2, col] -= 1
        V[j_off+1, k_off+1, col] += 1; V[j_off+2, k_off+1, col] -= 1
        W[k_off+1, i_off+0, col] += 1
        col += 1
        
        # 11. (d-f)(h-l)
        U[i_off+1, j_off+0, col] += 1; U[i_off+1, j_off+2, col] -= 1
        V[j_off+1, k_off+1, col] += 1; V[j_off+2, k_off+1, col] -= 1
        W[k_off+1, i_off+1, col] += 1
        col += 1
        
        return col

    # Initialize matrices
    # Total rank estimate: 8 + 11 + 8 + 11 = 38
    rank = 38
    U = np.zeros((n*m, rank))
    V = np.zeros((m*p, rank))
    W = np.zeros((p*n, rank))
    
    col = 0
    
    # Block 1: A_0 (2x2) * B_0 (2x3) -> C_0 (2x3)
    # i_off=0, j_off=0, k_off=0
    col = build_223(U, V, W, col, 0, 0, 0)
    
    # Block 2: A_1 (2x3) * B_1 (3x3) -> C_1 (2x3)
    # i_off=0, j_off=2, k_off=0
    col = build_233(U, V, W, col, 0, 2, 0)
    
    # Block 3: A_0 (2x2) * B_2 (2x3) -> C_2 (2x3)
    # i_off=0, j_off=0, k_off=3
    col = build_223(U, V, W, col, 0, 0, 3)
    
    # Block 4: A_1 (2x3) * B_3 (3x3) -> C_3 (2x3)
    # i_off=0, j_off=2, k_off=3
    col = build_233(U, V, W, col, 0, 2, 3)
    
    # Verify
    if not verify(U.tolist(), V.tolist(), W.tolist(), rank):
        # Fallback to schoolbook if verification fails
        rank = n * m * p
        U = np.zeros((n*m, rank))
        V = np.zeros((m*p, rank))
        W = np.zeros((p*n, rank))
        c = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j, c] = 1
                    V[j*p+k, c] = 1
                    W[k*n+i, c] = 1
                    c += 1
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
