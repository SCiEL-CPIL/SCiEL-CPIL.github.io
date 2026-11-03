import numpy as np
import time

def construct():
    n, m, p = 3, 5, 5
    t0 = time.time()
    deadline = t0 + 115.0  # Safety margin

    # Baseline: schoolbook rank
    rank0 = n * m * p
    U0 = np.zeros((n*m, rank0))
    V0 = np.zeros((m*p, rank0))
    W0 = np.zeros((p*n, rank0))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U0[i*m+j, col] = 1
                V0[j*p+k, col] = 1
                W0[k*n+i, col] = 1
                col += 1
    
    best_rank = rank0
    best_U = U0.copy()
    best_V = V0.copy()
    best_W = W0.copy()
    
    # Try to reduce rank
    # We'll try ranks from rank0 - 1 down to a lower bound
    # Lower bound estimate: max(n*m, m*p, p*n) = 25
    # But realistically, we won't go much below 70.
    # Let's try down to 60 to be safe, but prioritize higher ranks (easier to find).
    
    # To maximize chance of finding *any* improvement, we try ranks in descending order.
    # If we find rank 74, we try 73, etc.
    
    for r in range(rank0 - 1, 59, -1):
        if time.time() > deadline - 10.0:
            break
            
        found = False
        # Number of restarts per rank
        # We have ~115s total. If we try 15 ranks, we have ~7s per rank.
        # Let's do 10 restarts per rank, with 50 ALS iterations each.
        
        for restart in range(10):
            if time.time() > deadline - 10.0:
                break
                
            np.random.seed(12345 + r * 100 + restart)
            U = np.random.randn(n*m, r)
            V = np.random.randn(m*p, r)
            W = np.random.randn(p*n, r)
            
            prev_err = float('inf')
            
            for it in range(100):
                # Update W
                for i in range(n):
                    for k in range(p):
                        A_W = np.zeros((m*m, r))
                        b = np.zeros(m*m)
                        for j1 in range(m):
                            for j2 in range(m):
                                A_W[j1*m+j2, :] = U[i*m+j1, :] * V[j2*p+k, :]
                                b[j1*m+j2] = 1.0 if j1 == j2 else 0.0
                        w, *_ = np.linalg.lstsq(A_W, b, rcond=None)
                        W[k*n+i, :] = w
                        
                # Update U
                for i in range(n):
                    for j1 in range(m):
                        A_U = np.zeros((p*m, r))
                        b = np.zeros(p*m)
                        for k in range(p):
                            for j2 in range(m):
                                A_U[k*m+j2, :] = W[k*n+i, :] * V[j2*p+k, :]
                                b[k*m+j2] = 1.0 if j1 == j2 else 0.0
                        u, *_ = np.linalg.lstsq(A_U, b, rcond=None)
                        U[i*m+j1, :] = u
                        
                # Update V
                for j2 in range(m):
                    for k in range(p):
                        A_V = np.zeros((n*m, r))
                        b = np.zeros(n*m)
                        for i in range(n):
                            for j1 in range(m):
                                A_V[i*m+j1, :] = W[k*n+i, :] * U[i*m+j1, :]
                                b[i*m+j1] = 1.0 if j1 == j2 else 0.0
                        v, *_ = np.linalg.lstsq(A_V, b, rcond=None)
                        V[j2*p+k, :] = v
                        
                # Check convergence (optional, but good for early stopping)
                # Calculate error
                err = 0.0
                for i in range(n):
                    for k in range(p):
                        for j1 in range(m):
                            for j2 in range(m):
                                s = np.sum(W[k*n+i, :] * U[i*m+j1, :] * V[j2*p+k, :])
                                target = 1.0 if j1 == j2 else 0.0
                                err += (s - target)**2
                                
                if err < 1e-8:
                    break
                if err > prev_err * 1.1: # Diverging
                    break
                prev_err = err
                
            # Snap to half-integers
            U_r = np.round(U * 2) / 2
            V_r = np.round(V * 2) / 2
            W_r = np.round(W * 2) / 2
            
            if _verify(U_r, V_r, W_r, n, m, p):
                best_rank = r
                best_U = U_r.copy()
                best_V = V_r.copy()
                best_W = W_r.copy()
                found = True
                break # Found a solution for this rank, try lower rank
                
        # If we found a solution for this rank, we continue to try lower.
        # If not, we continue to next rank.
        
    return {
        "U": best_U.tolist(),
        "V": best_V.tolist(),
        "W": best_W.tolist()
    }

def _verify(U, V, W, n, m, p):
    r = U.shape[1]
    U = np.asarray(U, dtype=float)
    V = np.asarray(V, dtype=float)
    W = np.asarray(W, dtype=float)
    for i in range(n):
        for k in range(p):
            for j1 in range(m):
                for j2 in range(m):
                    s = 0.0
                    for r_ in range(r):
                        s += W[k*n+i, r_] * U[i*m+j1, r_] * V[j2*p+k, r_]
                    target = 1.0 if j1 == j2 else 0.0
                    if abs(s - target) > 1e-6:
                        return False
    return True
