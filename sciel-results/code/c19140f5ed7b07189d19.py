import numpy as np
import time

DIMENSIONS = [3, 3, 4]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140  # 10s margin

    # Construct T
    T = np.zeros((n*m, m*p, p*n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i*m + j
                b = j*p + k
                c = k*n + i
                T[a, b, c] = 1

    def solve_rank(r, attempts=5):
        for attempt in range(attempts):
            if time.time() > deadline:
                return None, None, None, float('inf')
            
            np.random.seed(42 + attempt)
            U = np.random.randn(n*m, r)
            V = np.random.randn(m*p, r)
            W = np.random.randn(p*n, r)
            
            for it in range(50):
                if time.time() > deadline:
                    return None, None, None, float('inf')
                
                T1 = T.reshape(n*m, m*p * p*n)
                M = np.zeros((m*p * p*n, r))
                for b in range(m*p):
                    for c in range(p*n):
                        M[b * p*n + c, :] = V[b, :] * W[c, :]
                U, _, _, _ = np.linalg.lstsq(M.T, T1.T, rcond=None)
                U = U.T
                
                T2 = T.reshape(m*p, n*m * p*n)
                M2 = np.zeros((n*m * p*n, r))
                for a in range(n*m):
                    for c in range(p*n):
                        M2[a * p*n + c, :] = U[a, :] * W[c, :]
                V, _, _, _ = np.linalg.lstsq(M2.T, T2.T, rcond=None)
                V = V.T
                
                T3 = T.reshape(p*n, n*m * m*p)
                M3 = np.zeros((n*m * m*p, r))
                for a in range(n*m):
                    for b in range(m*p):
                        M3[a * m*p + b, :] = U[a, :] * V[b, :]
                W, _, _, _ = np.linalg.lstsq(M3.T, T3.T, rcond=None)
                W = W.T
            
            # Round to half-integers
            U_half = np.round(U * 2) / 2
            V_half = np.round(V * 2) / 2
            W_half = np.round(W * 2) / 2
            
            T_recon = np.zeros((n*m, m*p, p*n))
            for r_idx in range(r):
                T_recon += np.einsum('a,b,c->abc', U_half[:, r_idx], V_half[:, r_idx], W_half[:, r_idx])
            error = np.linalg.norm(T_recon - T)
            
            if error < 1e-9:
                return U_half, V_half, W_half, error
            else:
                return None, None, None, error

    # Try ranks from 26 to 36
    for r in range(26, 37):
        best_error = float('inf')
        best_U, best_V, best_W = None, None, None
        for attempt in range(5):
            U, V, W, err = solve_rank(r, attempts=1)
            if err < best_error:
                best_error = err
                best_U, best_V, best_W = U, V, W
            if err < 1e-9:
                return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
        
        if best_error < 1e-9:
            return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
            
        if time.time() > deadline:
            break

    # Fallback to schoolbook
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
    return {"U": U, "V": V, "W": W}
