import numpy as np
import time

DIMENSIONS = [2, 3, 3]

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
    
    # Precompute index mappings for ALS
    # u = i*m + j
    # v = j*p + k
    # w = k*n + i
    
    # For updating U:
    # For each u (i, j), we have equations for k=0..p-1
    # B_u[k, r] = V[j*p+k, r] * W[k*n+i, r]
    U_indices = []
    for i in range(n):
        for j in range(m):
            u = i * m + j
            # Equations for each k
            eqs = []
            for k in range(p):
                v = j * p + k
                w = k * n + i
                eqs.append((v, w))
            U_indices.append((i, j, u, eqs))
            
    # For updating V:
    # For each v (j, k), we have equations for i=0..n-1
    # C_v[i, r] = U[i*m+j, r] * W[k*n+i, r]
    V_indices = []
    for j in range(m):
        for k in range(p):
            v = j * p + k
            eqs = []
            for i in range(n):
                u = i * m + j
                w = k * n + i
                eqs.append((u, w))
            V_indices.append((j, k, v, eqs))
            
    # For updating W:
    # For each w (k, i), we have equations for j=0..m-1
    # D_w[j, r] = U[i*m+j, r] * V[j*p+k, r]
    W_indices = []
    for k in range(p):
        for i in range(n):
            w = k * n + i
            eqs = []
            for j in range(m):
                u = i * m + j
                v = j * p + k
                eqs.append((u, v))
            W_indices.append((k, i, w, eqs))

    def verify_solution(U, V, W):
        # Check if U, V, W are half-integers
        if not np.allclose(U * 2, np.round(U * 2), atol=1e-9): return False
        if not np.allclose(V * 2, np.round(V * 2), atol=1e-9): return False
        if not np.allclose(W * 2, np.round(W * 2), atol=1e-9): return False
        
        # Check exact decomposition
        # Sum_r U[u,r] V[v,r] W[w,r] == 1
        # We only need to check the 18 non-zero entries
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    u = i * m + j
                    v = j * p + k
                    w = k * n + i
                    s = 0.0
                    for r in range(U.shape[1]):
                        s += U[u, r] * V[v, r] * W[w, r]
                    if not np.isclose(s, 1.0, atol=1e-9):
                        return False
        return True

    def als_decompose(R, max_iters=500, tol=1e-10):
        # Random initialization with diverse strategies
        strategy = np.random.randint(0, 3)
        if strategy == 0:
            U = np.random.randint(-2, 3, (n * m, R)) * 0.5
            V = np.random.randint(-2, 3, (m * p, R)) * 0.5
            W = np.random.randint(-2, 3, (p * n, R)) * 0.5
        elif strategy == 1:
            # Sparse initialization
            U = np.zeros((n * m, R))
            V = np.zeros((m * p, R))
            W = np.zeros((p * n, R))
            for r in range(R):
                for _ in range(3):
                    U[np.random.randint(n * m), r] = np.random.choice([-1, 1]) * 0.5
                for _ in range(3):
                    V[np.random.randint(m * p), r] = np.random.choice([-1, 1]) * 0.5
                for _ in range(3):
                    W[np.random.randint(p * n), r] = np.random.choice([-1, 1]) * 0.5
        else:
            # Gaussian initialization
            U = np.random.randn(n * m, R) * 0.5
            V = np.random.randn(m * p, R) * 0.5
            W = np.random.randn(p * n, R) * 0.5
        
        prev_err = np.inf
        best_err = np.inf
        best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
        
        for it in range(max_iters):
            # Update U
            for i, j, u, eqs in U_indices:
                B = np.zeros((len(eqs), R))
                for idx, (v, w) in enumerate(eqs):
                    B[idx, :] = V[v, :] * W[w, :]
                # Solve B @ U_u = 1
                U_u, _, _, _ = np.linalg.lstsq(B, np.ones(len(eqs)), rcond=None)
                U[u, :] = U_u
                
            # Update V
            for j, k, v, eqs in V_indices:
                C = np.zeros((len(eqs), R))
                for idx, (u, w) in enumerate(eqs):
                    C[idx, :] = U[u, :] * W[w, :]
                V_v, _, _, _ = np.linalg.lstsq(C, np.ones(len(eqs)), rcond=None)
                V[v, :] = V_v
                
            # Update W
            for k, i, w, eqs in W_indices:
                D = np.zeros((len(eqs), R))
                for idx, (u, v) in enumerate(eqs):
                    D[idx, :] = U[u, :] * V[v, :]
                W_w, _, _, _ = np.linalg.lstsq(D, np.ones(len(eqs)), rcond=None)
                W[w, :] = W_w
            
            # Calculate error
            err = 0.0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        u = i * m + j
                        v = j * p + k
                        w = k * n + i
                        s = 0.0
                        for r in range(R):
                            s += U[u, r] * V[v, r] * W[w, r]
                        err += (s - 1.0) ** 2
                        
            if err < best_err:
                best_err = err
                best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
                
            if abs(prev_err - err) < tol:
                break
            prev_err = err
        
        # Use best solution found
        U, V, W = best_U, best_V, best_W
        
        # Round to half-integers and verify
        U_round = np.round(U * 2) / 2
        V_round = np.round(V * 2) / 2
        W_round = np.round(W * 2) / 2
        
        if verify_solution(U_round, V_round, W_round):
            return U_round, V_round, W_round
        
        # Try a polish pass with tighter tolerance
        U_polish = U.copy()
        V_polish = V.copy()
        W_polish = W.copy()
        
        for it in range(100):
            # Update U with tighter solve
            for i, j, u, eqs in U_indices:
                B = np.zeros((len(eqs), R))
                for idx, (v, w) in enumerate(eqs):
                    B[idx, :] = V_polish[v, :] * W_polish[w, :]
                U_u, _, _, _ = np.linalg.lstsq(B, np.ones(len(eqs)), rcond=1e-12)
                U_polish[u, :] = U_u
                
            # Update V with tighter solve
            for j, k, v, eqs in V_indices:
                C = np.zeros((len(eqs), R))
                for idx, (u, w) in enumerate(eqs):
                    C[idx, :] = U_polish[u, :] * W_polish[w, :]
                V_v, _, _, _ = np.linalg.lstsq(C, np.ones(len(eqs)), rcond=1e-12)
                V_polish[v, :] = V_v
                
            # Update W with tighter solve
            for k, i, w, eqs in W_indices:
                D = np.zeros((len(eqs), R))
                for idx, (u, v) in enumerate(eqs):
                    D[idx, :] = U_polish[u, :] * V_polish[v, :]
                W_w, _, _, _ = np.linalg.lstsq(D, np.ones(len(eqs)), rcond=1e-12)
                W_polish[w, :] = W_w
        
        U_round = np.round(U_polish * 2) / 2
        V_round = np.round(V_polish * 2) / 2
        W_round = np.round(W_polish * 2) / 2
        
        if verify_solution(U_round, V_round, W_round):
            return U_round, V_round, W_round
            
        return None

    start_time = time.time()
    deadline = 145 # Safety margin

    # Try rank 16 first with more restarts
    for R in [16, 17, 18]:
        if time.time() > deadline:
            break
            
        # Try multiple restarts
        max_restarts = 150 if R == 16 else 80
        for restart in range(max_restarts):
            if time.time() > deadline:
                break
                
            res = als_decompose(R)
            if res is not None:
                U, V, W = res
                return {
                    "U": U.tolist(),
                    "V": V.tolist(),
                    "W": W.tolist()
                }
                
    # Fallback to schoolbook
    return {
        "U": U_school.tolist(),
        "V": V_school.tolist(),
        "W": W_school.tolist()
    }
