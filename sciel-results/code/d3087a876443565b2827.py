import numpy as np
import scipy.optimize as so
import time

DIMENSIONS = [2, 3, 3]

def construct():
    n, m, p = DIMENSIONS
    # Schoolbook fallback
    rank_schoolbook = n * m * p
    U_sb = np.zeros((n * m, rank_schoolbook))
    V_sb = np.zeros((m * p, rank_schoolbook))
    W_sb = np.zeros((p * n, rank_schoolbook))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i * m + j, col] = 1
                V_sb[j * p + k, col] = 1
                W_sb[k * n + i, col] = 1
                col += 1
    
    best_rank = rank_schoolbook
    best_U, best_V, best_W = U_sb, V_sb, W_sb

    # Define the target tensor T
    # T[i, j, k] is not quite right.
    # Indices:
    # U index: i*m + j
    # V index: j*p + k
    # W index: k*n + i
    # T[u, v, w] = 1 if u=i*m+j, v=j*p+k, w=k*n+i for some i,j,k.
    # Else 0.
    
    T = np.zeros((n * m, m * p, p * n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u_idx = i * m + j
                v_idx = j * p + k
                w_idx = k * n + i
                T[u_idx, v_idx, w_idx] = 1.0

    T_flat = T.flatten()

    start_time = time.time()
    deadline = 110 # seconds

    # Try ranks from 13 up to best_rank - 1
    for R in range(13, best_rank):
        if time.time() > deadline:
            break
            
        # Try a few random restarts
        for restart in range(20):
            if time.time() > deadline:
                break
                
            # Random initialization
            U = np.random.randn(n * m, R)
            V = np.random.randn(m * p, R)
            W = np.random.randn(p * n, R)
            
            # Scale to avoid overflow/underflow
            U /= np.sqrt(np.sum(U**2)) if np.sum(U**2) > 0 else 1
            V /= np.sqrt(np.sum(V**2)) if np.sum(V**2) > 0 else 1
            W /= np.sqrt(np.sum(W**2)) if np.sum(W**2) > 0 else 1

            def objective(x):
                U_r = x[:n*m*R].reshape(n*m, R)
                V_r = x[n*m*R:n*m*R + m*p*R].reshape(m*p, R)
                W_r = x[n*m*R + m*p*R:].reshape(p*n, R)
                
                # Compute tensor product sum
                # T_calc[u, v, w] = sum_r U[u,r] V[v,r] W[w,r]
                # This is equivalent to:
                # T_calc = np.einsum('ur,vr,wr->uvw', U_r, V_r, W_r)
                T_calc = np.einsum('ur,vr,wr->uvw', U_r, V_r, W_r)
                return T_calc.flatten() - T_flat

            try:
                x0 = np.concatenate([U.flatten(), V.flatten(), W.flatten()])
                res = so.least_squares(objective, x0, method='lm', max_nfev=1000)
                
                if res.cost < 1e-12:
                    # Found a solution
                    U_sol = res.x[:n*m*R].reshape(n*m, R)
                    V_sol = res.x[n*m*R:n*m*R + m*p*R].reshape(m*p, R)
                    W_sol = res.x[n*m*R + m*p*R:].reshape(p*n, R)
                    
                    # Snap to nearest half-integers
                    U_snap = np.round(U_sol * 2) / 2
                    V_snap = np.round(V_sol * 2) / 2
                    W_snap = np.round(W_sol * 2) / 2
                    
                    # Verify exactness
                    T_check = np.einsum('ur,vr,wr->uvw', U_snap, V_snap, W_snap)
                    if np.allclose(T_check, T, atol=1e-9):
                        best_rank = R
                        best_U = U_snap
                        best_V = V_snap
                        best_W = W_snap
                        # Found a better rank, break out of restarts and try lower rank?
                        # No, we are iterating R upwards. If we find 13, we can stop.
                        break
            except Exception:
                continue
        
        if best_rank == R:
            break

    # Convert to lists of lists for return
    # Ensure entries are integer or half-integer
    def to_list(mat):
        return [[float(x) for x in row] for row in mat]

    return {"U": to_list(best_U), "V": to_list(best_V), "W": to_list(best_W)}
