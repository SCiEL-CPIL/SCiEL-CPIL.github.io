import numpy as np
import time

DIMENSIONS = [3, 4, 5]

def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 130
    
    # Target: T[i,j,k] = 1 for all i,j,k
    # T[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    
    def verify(U, V, W, r):
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0
                    for c in range(r):
                        s += U[i*m+j][c] * V[j*p+k][c] * W[k*n+i][c]
                    if abs(s - 1) > 1e-6:
                        return False
        return True
    
    def to_half_int(x):
        return round(x * 2) / 2
    
    best = None
    best_rank = None
    
    # Try increasing ranks
    for r in [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 15, 20, 25, 30, 40, 50, 60]:
        if time.time() > deadline - 10:
            break
        # Multiple random restarts
        for restart in range(50):
            if time.time() > deadline - 10:
                break
            np.random.seed(restart * 1000 + r)
            U = np.random.randn(n*m, r) * 0.1
            V = np.random.randn(m*p, r) * 0.1
            W = np.random.randn(p*n, r) * 0.1
            
            # Alternating minimization
            for it in range(200):
                if time.time() > deadline - 10:
                    break
                # Update U
                for i in range(n):
                    for j in range(m):
                        row = V[j*p:(j+1)*p, :]  # p x r
                        # For each k: U[i*m+j, :] * (row[k,:] * W[k*n+i, :])^T = 1
                        # This is a linear system in U[i*m+j, :]
                        A_mat = np.zeros((p, r))
                        for k in range(p):
                            A_mat[k, :] = row[k, :] * W[k*n+i, :]
                        b = np.ones(p)
                        try:
                            U[i*m+j, :] = np.linalg.lstsq(A_mat, b, rcond=None)[0]
                        except:
                            pass
                
                # Update V
                for j in range(m):
                    for k in range(p):
                        col = U[:j*m + (j+1)*m, :]  # n x r... wait
                        # V[j*p+k, :] * (U[i*m+j, :] * W[k*n+i, :]) = 1 for all i
                        A_mat = np.zeros((n, r))
                        for i in range(n):
                            A_mat[i, :] = U[i*m+j, :] * W[k*n+i, :]
                        b = np.ones(n)
                        try:
                            V[j*p+k, :] = np.linalg.lstsq(A_mat, b, rcond=None)[0]
                        except:
                            pass
                
                # Update W
                for k in range(p):
                    for i in range(n):
                        A_mat = np.zeros((m, r))
                        for j in range(m):
                            A_mat[j, :] = U[i*m+j, :] * V[j*p+k, :]
                        b = np.ones(m)
                        try:
                            W[k*n+i, :] = np.linalg.lstsq(A_mat, b, rcond=None)[0]
                        except:
                            pass
                
                # Check convergence
                max_err = 0
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            s = 0
                            for c in range(r):
                                s += U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
                            max_err = max(max_err, abs(s - 1))
                if max_err < 1e-8:
                    # Round to half-integers
                    U_r = np.array([[to_half_int(U[i*m+j, c]) for c in range(r)] for i in range(n) for j in range(m)])
                    V_r = np.array([[to_half_int(V[j*p+k, c]) for c in range(r)] for j in range(m) for k in range(p)])
                    W_r = np.array([[to_half_int(W[k*n+i, c]) for c in range(r)] for k in range(p) for i in range(n)])
                    
                    # Verify exact
                    exact = True
                    for i in range(n):
                        for j in range(m):
                            for k in range(p):
                                s = 0
                                for c in range(r):
                                    s += U_r[i*m+j, c] * V_r[j*p+k, c] * W_r[k*n+i, c]
                                if abs(s - 1) > 1e-9:
                                    exact = False
                                    break
                            if not exact:
                                break
                        if not exact:
                            break
                    if exact:
                        if best_rank is None or r < best_rank:
                            best_rank = r
                            best = (U_r.tolist(), V_r.tolist(), W_r.tolist())
                        break  # Found valid at this rank, break restarts
                if max_err < 1e-4:
                    break  # Converged enough
            
            if best_rank is not None and best_rank <= r:
                break
    
    if best is not None:
        U_list, V_list, W_list = best
        return {"U": U_list, "V": V_list, "W": W_list}
    
    # Fallback to schoolbook
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m+j][col] = 1
                V[j*p+k][col] = 1
                W[k*n+i][col] = 1
                col += 1
    return {"U": U, "V": V, "W": W}
