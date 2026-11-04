import numpy as np
import time

DIMENSIONS = [4, 5, 6]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 10  # Very short search time to ensure safety
    
    nm = n * m
    mp = m * p
    pn = p * n
    
    # Build target tensor T
    # T[a, b, c]
    T = np.zeros((nm, mp, pn))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T[a, b, c] = 1.0

    def verify(U, V, W, tol=1e-9):
        U_r = np.round(U * 2) / 2
        V_r = np.round(V * 2) / 2
        W_r = np.round(W * 2) / 2
        
        if not (np.allclose(U, U_r, atol=1e-6) and 
                np.allclose(V, V_r, atol=1e-6) and 
                np.allclose(W, W_r, atol=1e-6)):
            return False
            
        # Check exact decomposition
        for a in range(nm):
            for b in range(mp):
                for c in range(pn):
                    val = np.dot(U_r[a, :], V_r[b, :] * W_r[c, :])
                    if abs(val - T[a, b, c]) > 1e-9:
                        return False
        return True

    def schoolbook():
        rank = n * m * p
        U = np.zeros((nm, rank))
        V = np.zeros((mp, rank))
        W = np.zeros((pn, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    a = i * m + j
                    b = j * p + k
                    c = k * n + i
                    U[a, col] = 1
                    V[b, col] = 1
                    W[c, col] = 1
                    col += 1
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

    fallback = schoolbook()

    def try_als(rank, max_iters=20, num_restarts=2):
        if time.time() > deadline:
            return None
            
        for restart in range(num_restarts):
            if time.time() > deadline:
                return None
                
            np.random.seed(42 + restart)
            U = np.random.randn(nm, rank)
            V = np.random.randn(mp, rank)
            W = np.random.randn(pn, rank)
            
            for it in range(max_iters):
                if time.time() > deadline:
                    return None
                
                # Update U
                for a in range(nm):
                    A = (V[:, None, :] * W[None, :, :]).reshape(-1, rank)
                    t = T[a, :, :].reshape(-1)
                    U[a, :], residuals, rank_val, sv = np.linalg.lstsq(A, t, rcond=None)
                
                # Update V
                for b in range(mp):
                    A = (U[:, None, :] * W[None, :, :]).reshape(-1, rank)
                    t = T[:, b, :].reshape(-1)
                    V[b, :], residuals, rank_val, sv = np.linalg.lstsq(A, t, rcond=None)
                    
                # Update W
                for c in range(pn):
                    A = (U[:, None, :] * V[None, :, :]).reshape(-1, rank)
                    t = T[:, :, c].reshape(-1)
                    W[c, :], residuals, rank_val, sv = np.linalg.lstsq(A, t, rcond=None)
                    
                # Check convergence
                if it % 5 == 0:
                    if verify(U, V, W):
                        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

        return None

    # Try ranks from low to high, but limit the search
    best_result = None
    # We only try a few ranks close to 120 if we think there's a chance
    # Given the time limit, we try 119, 118
    for r in [119, 118, 117]:
        if time.time() > deadline - 2:
            break
        result = try_als(r, max_iters=10, num_restarts=1)
        if result is not None:
            best_result = result
            break
            
    if best_result is None:
        best_result = fallback
        
    return best_result
