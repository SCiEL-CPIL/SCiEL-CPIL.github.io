import numpy as np
import time

DIMENSIONS = [2, 4, 4]

def construct():
    n, m, p = DIMENSIONS
    # Build tensor T of shape (n*m, m*p, p*n)
    nm, mp, pn = n*m, m*p, p*n
    T = np.zeros((nm, mp, pn), dtype=float)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i*m + j
                b = j*p + k
                c = k*n + i
                T[a, b, c] = 1.0
    
    deadline = time.time() + 140  # safety margin
    
    best = None  # (rank, U, V, W)
    
    def schoolbook():
        rank = n*m*p
        U = np.zeros((nm, rank))
        V = np.zeros((mp, rank))
        W = np.zeros((pn, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j, col] = 1
                    V[j*p+k, col] = 1
                    W[k*n+i, col] = 1
                    col += 1
        return rank, U, V, W
    
    # Try to find low-rank decomposition
    for r in range(16, 32):
        if time.time() > deadline:
            break
        found = try_rank(T, r, nm, mp, pn, deadline)
        if found is not None:
            rank, U, V, W = found
            best = (rank, U, V, W)
            break
    
    if best is None:
        best = schoolbook()
    
    rank, U, V, W = best
    # Convert to lists
    U_list = [[float(x) for x in row] for row in U]
    V_list = [[float(x) for x in row] for row in V]
    W_list = [[float(x) for x in row] for row in W]
    return {"U": U_list, "V": V_list, "W": W_list}

def try_rank(T, r, nm, mp, pn, deadline):
    # ALS to find rank-r decomposition
    T_flat_a = T.reshape(nm, mp*pn)
    T_flat_b = T.reshape(nm*mp, pn)
    T_flat_c = T.reshape(nm*pn, mp)
    
    for restart in range(20):
        if time.time() > deadline:
            return None
        U = np.random.randn(nm, r)
        V = np.random.randn(mp, r)
        W = np.random.randn(pn, r)
        
        for it in range(200):
            if time.time() > deadline:
                return None
            # Update U
            # T[a] = sum_c U[a,c] * (V[:,c] * W[:,c])
            M_U = np.zeros((mp*pn, r))
            for c in range(r):
                M_U[:, c] = (V[:, c] * W[:, c]).reshape(mp*pn)
            U = np.linalg.lstsq(M_U, T_flat_a.T, rcond=None)[0].T
            # Update V
            M_V = np.zeros((nm*pn, r))
            for c in range(r):
                M_V[:, c] = (U[:, c] * W[:, c]).reshape(nm*pn)
            V = np.linalg.lstsq(M_V, T_flat_b.T, rcond=None)[0].T
            # Update W
            M_W = np.zeros((nm*mp, r))
            for c in range(r):
                M_W[:, c] = (U[:, c] * V[:, c]).reshape(nm*mp)
            W = np.linalg.lstsq(M_W, T_flat_c.T, rcond=None)[0].T
        
        # Check residual
        residual = 0.0
        for c in range(r):
            residual_tensor = np.outer(np.outer(U[:, c], V[:, c]), W[:, c])
            residual += np.abs(residual_tensor - T).max()
        # Actually compute properly
        recon = np.zeros_like(T)
        for c in range(r):
            recon += np.einsum('a,b,c->abc', U[:, c], V[:, c], W[:, c])
        err = np.abs(recon - T).max()
        if err < 1e-6:
            # Found a solution, try to rationalize
            return rationalize(U, V, W, r, nm, mp, pn)
    return None

def rationalize(U, V, W, r, nm, mp, pn):
    # Try to find rational approximations
    # Scale to find a common denominator
    for denom in [1, 2, 4, 8, 16, 32, 64, 128]:
        U_r = np.round(U * denom) / denom
        V_r = np.round(V * denom) / denom
        W_r = np.round(W * denom) / denom
        # Verify
        T = np.zeros((nm, mp, pn))
        for c in range(r):
            T += np.einsum('a,b,c->abc', U_r[:, c], V_r[:, c], W_r[:, c])
        if np.abs(T - expected_T()).max() < 1e-9:
            return r, U_r, V_r, W_r
    return None
