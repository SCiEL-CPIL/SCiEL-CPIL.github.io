"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 4, 5]

# EVOLVE-BLOCK-START
import time
import numpy as np

def construct():
    n, m, p = DIMENSIONS
    t0 = time.time()
    deadline = t0 + 110.0  # safety margin

    def schoolbook():
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][col] = 1
                    V[j * p + k][col] = 1
                    W[k * n + i][col] = 1
                    col += 1
        return {"U": U, "V": V, "W": W}

    best = schoolbook()
    best_rank = n * m * p

    # Tensor T
    T = np.zeros((n*m, m*p, p*n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i*m+j, j*p+k, k*n+i] = 1.0

    # Try to find a lower rank decomposition using ALS
    # We will try ranks from best_rank - 1 down to 40
    for r in range(best_rank - 1, 39, -1):
        if time.time() > deadline - 10:
            break
            
        # Initialize with random values
        np.random.seed(r)
        U = np.random.randn(n*m, r)
        V = np.random.randn(m*p, r)
        W = np.random.randn(p*n, r)
        
        # ALS iterations
        for _ in range(50):
            if time.time() > deadline - 5:
                break
                
            # Update U
            new_U = np.zeros_like(U)
            for a in range(n*m):
                i = a // m
                j = a % m
                for t in range(r):
                    num = 0.0
                    den = 0.0
                    for k in range(p):
                        b = j * p + k
                        c = k * n + i
                        num += T[a, b, c] * V[b, t] * W[c, t]
                        den += V[b, t] * V[b, t] * W[c, t] * W[c, t]
                    if den > 1e-9:
                        new_U[a, t] = num / den
            U = new_U
            
            # Update V
            new_V = np.zeros_like(V)
            for b in range(m*p):
                j = b // p
                k = b % p
                for t in range(r):
                    num = 0.0
                    den = 0.0
                    for i in range(n):
                        a = i * m + j
                        c = k * n + i
                        num += T[a, b, c] * U[a, t] * W[c, t]
                        den += U[a, t] * U[a, t] * W[c, t] * W[c, t]
                    if den > 1e-9:
                        new_V[b, t] = num / den
            V = new_V
            
            # Update W
            new_W = np.zeros_like(W)
            for c in range(p*n):
                k = c // n
                i = c % n
                for t in range(r):
                    num = 0.0
                    den = 0.0
                    for j in range(m):
                        a = i * m + j
                        b = j * p + k
                        num += T[a, b, c] * U[a, t] * V[b, t]
                        den += U[a, t] * U[a, t] * V[b, t] * V[b, t]
                    if den > 1e-9:
                        new_W[c, t] = num / den
            W = new_W
            
            # Check error
            E = np.einsum('at,bt,ct->abc', U, V, W)
            err = np.sum((T - E)**2)
            if err < 1e-6:
                # Snap to half-integers
                U_snap = np.round(U * 2) / 2
                V_snap = np.round(V * 2) / 2
                W_snap = np.round(W * 2) / 2
                
                # Verify exactness with snapped values
                E_snap = np.einsum('at,bt,ct->abc', U_snap, V_snap, W_snap)
                err_snap = np.sum((T - E_snap)**2)
                
                if err_snap < 1e-9:
                    return {"U": U_snap.tolist(), "V": V_snap.tolist(), "W": W_snap.tolist()}

    return best
# EVOLVE-BLOCK-END
