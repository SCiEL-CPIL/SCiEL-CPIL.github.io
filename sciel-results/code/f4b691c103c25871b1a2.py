import time
import numpy as np

def construct():
    n, m, p = 3, 4, 4
    dim_u = n * m  # 12
    dim_v = m * p  # 16
    dim_w = p * n  # 12
    
    start_time = time.time()
    deadline = start_time + 100  # Safety margin
    
    def build_rank44():
        rank = 44
        U = [[0] * rank for _ in range(dim_u)]
        V = [[0] * rank for _ in range(dim_v)]
        W = [[0] * rank for _ in range(dim_w)]
        
        col = 0
        def add_strassen(X_u, Y_v, Z_w, col):
            U[X_u[0]][col] += 1; U[X_u[3]][col] += 1
            V[Y_v[0]][col] += 1; V[Y_v[3]][col] += 1
            W[Z_w[0]][col] += 1; W[Z_w[3]][col] += 1
            col += 1
            U[X_u[2]][col] += 1; U[X_u[3]][col] += 1
            V[Y_v[0]][col] += 1
            W[Z_w[2]][col] += 1; W[Z_w[3]][col] -= 1
            col += 1
            U[X_u[0]][col] += 1
            V[Y_v[1]][col] += 1; V[Y_v[3]][col] -= 1
            W[Z_w[1]][col] += 1; W[Z_w[3]][col] += 1
            col += 1
            U[X_u[3]][col] += 1
            V[Y_v[2]][col] += 1; V[Y_v[0]][col] -= 1
            W[Z_w[0]][col] += 1; W[Z_w[2]][col] += 1
            col += 1
            U[X_u[0]][col] += 1; U[X_u[1]][col] += 1
            V[Y_v[3]][col] += 1
            W[Z_w[0]][col] -= 1; W[Z_w[1]][col] += 1
            col += 1
            U[X_u[2]][col] += 1; U[X_u[0]][col] -= 1
            V[Y_v[0]][col] += 1; V[Y_v[1]][col] += 1
            W[Z_w[3]][col] += 1
            col += 1
            U[X_u[1]][col] += 1; U[X_u[3]][col] -= 1
            V[Y_v[2]][col] += 1; V[Y_v[3]][col] += 1
            W[Z_w[0]][col] += 1
            col += 1
            return col
        
        col = add_strassen([0,1,4,5], [0,1,4,5], [0,3,1,4], col)
        col = add_strassen([2,3,6,7], [8,9,12,13], [0,3,1,4], col)
        col = add_strassen([0,1,4,5], [2,3,6,7], [6,9,7,10], col)
        col = add_strassen([2,3,6,7], [10,11,14,15], [6,9,7,10], col)
        
        for j in range(4):
            U[8+j][col] += 1; V[j*4+0][col] += 1; W[2][col] += 1
            col += 1
            U[8+j][col] += 1; V[j*4+1][col] += 1; W[5][col] += 1
            col += 1
        for j in range(4):
            U[8+j][col] += 1; V[j*4+2][col] += 1; W[8][col] += 1
            col += 1
            U[8+j][col] += 1; V[j*4+3][col] += 1; W[11][col] += 1
            col += 1
        
        return U, V, W
    
    U, V, W = build_rank44()
    
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
    
    # Try to reduce rank by checking linear dependence
    # Form the matrix M where column r is [U[:,r], V[:,r], W[:,r]]
    M = np.zeros((dim_u + dim_v + dim_w, 44))
    for r in range(44):
        M[:dim_u, r] = U_np[:, r]
        M[dim_u:dim_u+dim_v, r] = V_np[:, r]
        M[dim_u+dim_v:, r] = W_np[:, r]
    
    # Check if any column is in the span of the others
    # For each column r, check if M[:,r] is in the span of M[:,[0..r-1,r+1..43]]
    best_rank = 44
    best_indices = None
    
    for r in range(44):
        if time.time() > deadline:
            break
        # Remove column r
        idx = [i for i in range(44) if i != r]
        M_reduced = M[:, idx]
        # Check if M[:,r] is in the column space of M_reduced
        # Solve M_reduced @ x = M[:,r]
        try:
            x, residuals, rank_r, sv = np.linalg.lstsq(M_reduced, M[:, r], rcond=None)
            if residuals is not None and len(residuals) > 0 and residuals[0] < 1e-6:
                # Column r is in the span of others
                best_rank = 43
                best_indices = idx
                break
        except:
            pass
    
    if best_rank < 44 and best_indices is not None:
        # Reconstruct with 43 terms
        U_new = U_np[:, best_indices]
        V_new = V_np[:, best_indices]
        W_new = W_np[:, best_indices]
        
        # Verify
        if verify(U_new, V_new, W_new, n, m, p):
            return {"U": U_new.tolist(), "V": V_new.tolist(), "W": W_new.tolist()}
    
    return {"U": U, "V": V, "W": W}
