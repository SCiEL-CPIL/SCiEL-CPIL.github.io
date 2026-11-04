import time
import numpy as np

def construct():
    n, m, p = 3, 4, 4
    dim_u = n * m  # 12
    dim_v = m * p  # 16
    dim_w = p * n  # 12
    
    start_time = time.time()
    deadline = start_time + 100  # Safety margin
    
    def build_rank32():
        rank = 32
        U = [[0] * rank for _ in range(dim_u)]
        V = [[0] * rank for _ in range(dim_v)]
        W = [[0] * rank for _ in range(dim_w)]
        
        col = 0
        def add_strassen(X_u, Y_v, Z_w, col):
            # Standard Strassen 7 terms
            # X_u: [i0j0, i0j1, i1j0, i1j1] indices in U
            # Y_v: [j0k0, j0k1, j1k0, j1k1] indices in V
            # Z_w: [k0i0, k0i1, k1i0, k1i1] indices in W
            
            # M1 = (A00 + A11) * (B00 + B11)
            U[X_u[0]][col] += 1; U[X_u[3]][col] += 1
            V[Y_v[0]][col] += 1; V[Y_v[3]][col] += 1
            W[Z_w[0]][col] += 1; W[Z_w[3]][col] += 1
            col += 1
            
            # M2 = (A10 + A11) * B00
            U[X_u[2]][col] += 1; U[X_u[3]][col] += 1
            V[Y_v[0]][col] += 1
            W[Z_w[2]][col] += 1; W[Z_w[3]][col] -= 1
            col += 1
            
            # M3 = A00 * (B10 - B11)
            U[X_u[0]][col] += 1
            V[Y_v[1]][col] += 1; V[Y_v[3]][col] -= 1
            W[Z_w[1]][col] += 1; W[Z_w[3]][col] += 1
            col += 1
            
            # M4 = A11 * (B01 - B00)
            U[X_u[3]][col] += 1
            V[Y_v[2]][col] += 1; V[Y_v[0]][col] -= 1
            W[Z_w[0]][col] += 1; W[Z_w[2]][col] += 1
            col += 1
            
            # M5 = (A00 + A01) * B11
            U[X_u[0]][col] += 1; U[X_u[1]][col] += 1
            V[Y_v[3]][col] += 1
            W[Z_w[0]][col] -= 1; W[Z_w[1]][col] += 1
            col += 1
            
            # M6 = (A10 - A00) * (B00 + B01)
            U[X_u[2]][col] += 1; U[X_u[0]][col] -= 1
            V[Y_v[0]][col] += 1; V[Y_v[1]][col] += 1
            W[Z_w[3]][col] += 1
            col += 1
            
            # M7 = (A01 - A11) * (B10 + B11)
            U[X_u[1]][col] += 1; U[X_u[3]][col] -= 1
            V[Y_v[2]][col] += 1; V[Y_v[3]][col] += 1
            W[Z_w[0]][col] += 1
            col += 1
            
            return col
        
        # Block 1: A[0:2, 0:2] * B[0:2, 0:2] -> C[0:2, 0:2]
        # U indices: 0,1,4,5
        # V indices: 0,1,4,5
        # W indices: 0,3,1,4 (k=0,1; i=0,1)
        col = add_strassen([0,1,4,5], [0,1,4,5], [0,3,1,4], col)
        
        # Block 2: A[0:2, 2:4] * B[2:4, 0:2] -> C[0:2, 0:2]
        # U indices: 2,3,6,7
        # V indices: 8,9,12,13
        # W indices: 0,3,1,4
        col = add_strassen([2,3,6,7], [8,9,12,13], [0,3,1,4], col)
        
        # Block 3: A[0:2, 0:2] * B[0:2, 2:4] -> C[0:2, 2:4]
        # U indices: 0,1,4,5
        # V indices: 2,3,6,7
        # W indices: 6,9,7,10 (k=2,3; i=0,1)
        col = add_strassen([0,1,4,5], [2,3,6,7], [6,9,7,10], col)
        
        # Block 4: A[0:2, 2:4] * B[2:4, 2:4] -> C[0:2, 2:4]
        # U indices: 2,3,6,7
        # V indices: 10,11,14,15
        # W indices: 6,9,7,10
        col = add_strassen([2,3,6,7], [10,11,14,15], [6,9,7,10], col)
        
        # Row 2: A[2, :] * B[:, :] -> C[2, :]
        # We use 4 terms, one for each j in 0..3
        # Term j: U[8+j]=1, V[j*4+k]=1 for all k, W[k*3+2]=1 for all k
        for j in range(4):
            U[8+j][col] = 1
            for k in range(4):
                V[j*4+k][col] = 1
                W[k*3+2][col] = 1
            col += 1
            
        return U, V, W

    U, V, W = build_rank32()
    
    # Verify
    U_np = np.array(U, dtype=float)
    V_np = np.array(V, dtype=float)
    W_np = np.array(W, dtype=float)
    
    def verify(U, V, W, n, m, p):
        for i in range(n):
            for k in range(p):
                for j in range(m):
                    coeff = 0
                    for r in range(U.shape[1]):
                        coeff += U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                    if abs(coeff - 1) > 1e-9:
                        return False
        return True

    if not verify(U_np, V_np, W_np, n, m, p):
        # Fallback to schoolbook
        rank = 48
        U = [[0]*rank for _ in range(dim_u)]
        V = [[0]*rank for _ in range(dim_v)]
        W = [[0]*rank for _ in range(dim_w)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j][col] = 1
                    V[j*p+k][col] = 1
                    W[k*n+i][col] = 1
                    col += 1
        return {"U": U, "V": V, "W": W}
    
    return {"U": U, "V": V, "W": W}
