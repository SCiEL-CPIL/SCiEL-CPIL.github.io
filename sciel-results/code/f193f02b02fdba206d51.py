import numpy as np
import time
import random

DIMENSIONS = [3, 3, 3]

def construct():
    n, m, p = DIMENSIONS
    # Schoolbook fallback
    rank_school = n * m * p
    U_school = np.zeros((n * m, rank_school))
    V_school = np.zeros((m * p, rank_school))
    W_school = np.zeros((p * n, rank_school))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * m + j, col] = 1
                V_school[j * p + k, col] = 1
                W_school[k * n + i, col] = 1
                col += 1
    
    # Target Tensor T
    # Indices: u (0..8), v (0..8), w (0..8)
    # u = i*m + j
    # v = j*p + k
    # w = k*n + i
    # T[u,v,w] = 1 if consistent, else 0
    T = np.zeros((n * m, m * p, p * n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u = i * m + j
                v = j * p + k
                w = k * n + i
                T[u, v, w] = 1.0

    def error(U, V, W):
        # Compute sum_r U[u,r] V[v,r] W[w,r]
        # Use einsum
        S = np.einsum('ur,vr,wr->uvw', U, V, W)
        return np.linalg.norm(S - T)

    def rationalize(M, tol=1e-6):
        # Try to convert to integers or half-integers
        # Multiply by 2 and round
        M2 = 2 * M
        M_rounded = np.round(M2)
        if np.linalg.norm(M2 - M_rounded) < tol * np.linalg.norm(M2):
            return M_rounded / 2.0
        # Try integers
        M_int = np.round(M)
        if np.linalg.norm(M - M_int) < tol * np.linalg.norm(M):
            return M_int
        return None

    start_time = time.time()
    deadline = start_time + 110  # 10s margin

    # Try ranks from 23 to 26
    for rank in [23, 24, 25, 26]:
        if time.time() > deadline:
            break
        for restart in range(20):
            if time.time() > deadline:
                break
            np.random.seed(restart * 100 + rank)
            U = np.random.randn(n * m, rank)
            V = np.random.randn(m * p, rank)
            W = np.random.randn(p * n, rank)
            
            # ALS
            for it in range(100):
                if time.time() > deadline:
                    break
                # Update U
                # U[u,r] = sum_{v,w} T[u,v,w] V[v,r] W[w,r] / sum_{v,w} V[v,r]^2 W[w,r]^2
                # Denominator
                den_U = np.einsum('vr,wr->r', V**2, W**2)
                # Numerator
                num_U = np.einsum('uvw,vr,wr->ur', T, V, W)
                U = num_U / (den_U + 1e-9)
                
                # Update V
                den_V = np.einsum('ur,wr->r', U**2, W**2)
                num_V = np.einsum('uvw,ur,wr->vr', T, U, W)
                V = num_V / (den_V + 1e-9)
                
                # Update W
                den_W = np.einsum('ur,vr->r', U**2, V**2)
                num_W = np.einsum('uvw,ur,vr->wr', T, U, V)
                W = num_W / (den_W + 1e-9)
            
            err = error(U, V, W)
            if err < 1e-7:
                # Try to rationalize
                Ur = rationalize(U)
                Vr = rationalize(V)
                Wr = rationalize(W)
                if Ur is not None and Vr is not None and Wr is not None:
                    if error(Ur, Vr, Wr) < 1e-7:
                        return {
                            "U": Ur.tolist(),
                            "V": Vr.tolist(),
                            "W": Wr.tolist()
                        }
                # If rationalization fails, check if float solution is acceptable?
                # Prompt says "integer/half-integer entries".
                # So we must return rationalized.
                # If not rationalized, we cannot return this.
                pass

    # Fallback to schoolbook
    return {
        "U": U_school.tolist(),
        "V": V_school.tolist(),
        "W": W_school.tolist()
    }
