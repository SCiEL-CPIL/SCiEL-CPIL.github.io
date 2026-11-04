import numpy as np
import time

DIMENSIONS = [3, 3, 4]

def construct():
    start_time = time.time()
    deadline = start_time + 110  # safety margin before 120s limit
    
    n, m, p = 3, 3, 4
    N, M, P = n * m, m * p, p * n  # 9, 12, 12
    
    # Build target tensor T[a, b, c] = 1 if a=(i,j), b=(j,k), c=(k,i)
    T = np.zeros((N, M, P))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i * m + j, j * p + k, k * n + i] = 1.0
    
    T_flat = T.reshape(N * M, P)  # (108, 12)
    
    best_rank = 36
    best_U = None
    best_V = None
    best_W = None
    
    # Try ranks from 26 upward (known minimum for 3x3x4)
    for R in range(26, 37):
        if time.time() > deadline:
            break
        
        found = False
        for attempt in range(50):
            if time.time() > deadline:
                break
            
            # Structured initialization: exploit the algebraic structure
            # The tensor has a block structure based on (i,j,k) indices
            np.random.seed(1000 * R + attempt)
            
            # Initialize with structure-aware random values
            U = np.random.randn(N, R) * 0.5
            V = np.random.randn(M, R) * 0.5
            W = np.random.randn(P, R) * 0.5
            
            # Add structured components: for each (i,j,k), add a small
            # perturbation that respects the tensor's support pattern
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        # Randomly assign some of the tensor's "support"
                        # to random rank terms
                        r = np.random.randint(R)
                        U[i * m + j, r] += 0.3
                        V[j * p + k, r] += 0.3
                        W[k * n + i, r] += 0.3
            
            # ALS iterations
            converged = False
            for it in range(30):
                if time.time() > deadline:
                    break
                
                # Solve for W: for each c, T[:, :, c] = sum_r W[c,r] * (U[:,r] outer V[:,r])
                # M_W[:, c*R + r] = (U[:,r] * V[:,r]).flatten()
                M_W = np.zeros((N * M, P * R))
                for c in range(P):
                    for r in range(R):
                        M_W[:, c * R + r] = np.outer(U[:, r], V[:, r]).flatten()
                
                W_new = np.zeros((P, R))
                for c in range(P):
                    W_new[c, :] = np.linalg.lstsq(
                        M_W[:, c * R:(c + 1) * R], T_flat[:, c], rcond=None
                    )[0]
                
                # Solve for U: for each a, T[a, :, :] = sum_r U[a,r] * (V[:,r] outer W[:,r])
                M_U = np.zeros((M * P, N * R))
                for a in range(N):
                    for r in range(R):
                        M_U[:, a * R + r] = np.outer(V[:, r], W_new[:, r]).flatten()
                
                T_flat_U = T.reshape(M * P, N)
                U_new = np.zeros((N, R))
                for a in range(N):
                    U_new[a, :] = np.linalg.lstsq(
                        M_U[:, a * R:(a + 1) * R], T_flat_U[:, a], rcond=None
                    )[0]
                
                # Solve for V: for each b, T[:, b, :] = sum_r V[b,r] * (U[:,r] outer W[:,r])
                M_V = np.zeros((N * P, M * R))
                for b in range(M):
                    for r in range(R):
                        M_V[:, b * R + r] = np.outer(U_new[:, r], W_new[:, r]).flatten()
                
                T_flat_V = T.reshape(N * P, M)
                V_new = np.zeros((M, R))
                for b in range(M):
                    V_new[b, :] = np.linalg.lstsq(
                        M_V[:, b * R:(b + 1) * R], T_flat_V[:, b], rcond=None
                    )[0]
                
                U, V, W = U_new, V_new, W_new
                
                # Check convergence
                error = 0.0
                for c in range(P):
                    pred = np.sum(
                        np.outer(U[:, :], V[:, :]).reshape(N, M, R) * W[c, :], axis=2
                    ).flatten()
                    error += np.sum((pred - T_flat[:, c]) ** 2)
                
                if error < 1e-10:
                    converged = True
                    break
            
            if converged:
                best_rank = R
                best_U = U
                best_V = V
                best_W = W
                found = True
                break
        
        if found:
            break
    
    # Convert to list format
    if best_U is not None:
        return {
            "U": best_U.tolist(),
            "V": best_V.tolist(),
            "W": best_W.tolist()
        }
    else:
        # Fallback to schoolbook
        rank = n * m * p
        U = [[0] * rank for _ in range(N)]
        V = [[0] * rank for _ in range(M)]
        W = [[0] * rank for _ in range(P)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][col] = 1
                    V[j * p + k][col] = 1
                    W[k * n + i][col] = 1
                    col += 1
        return {"U": U, "V": V, "W": W}
