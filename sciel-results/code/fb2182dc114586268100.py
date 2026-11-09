"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time

DIMENSIONS = [2, 3, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 138
    
    nm = n * m
    mp = m * p
    pn = p * n
    
    # Nonzero entries
    nz = []
    nz_set = set()
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                nz.append((a, b, c))
                nz_set.add((a, b, c))
    
    target = np.zeros((nm, mp, pn))
    for a, b, c in nz:
        target[a, b, c] = 1.0
    
    def check_half_exact(U, V, W):
        if not (np.allclose(U * 2, np.round(U * 2), atol=1e-9) and
                np.allclose(V * 2, np.round(V * 2), atol=1e-9) and
                np.allclose(W * 2, np.round(W * 2), atol=1e-9)):
            return False
        U_h = np.round(U * 2) / 2.0
        V_h = np.round(V * 2) / 2.0
        W_h = np.round(W * 2) / 2.0
        rank = U_h.shape[1]
        # Check nonzero entries
        for a, b, c in nz:
            s = float(np.dot(U_h[a], V_h[b] * W_h[c]))
            if abs(s - 1.0) > 1e-9:
                return False
        # Check zero entries (sample if too many)
        for a in range(nm):
            for b in range(mp):
                for c in range(pn):
                    if (a, b, c) not in nz_set:
                        s = float(np.dot(U_h[a], V_h[b] * W_h[c]))
                        if abs(s) > 1e-9:
                            return False
        return True
    
    def snap_half(x):
        return np.round(x * 2.0) / 2.0
    
    best_rank = n * m * p
    best_result = None
    
    for rank in range(15, best_rank):
        if time.time() > deadline - 8:
            break
        
        found = False
        max_trials = 15
        for trial in range(max_trials):
            if time.time() > deadline - 5:
                break
            
            rng = np.random.default_rng(trial * 7919 + rank * 104729)
            U = rng.standard_normal((nm, rank))
            V = rng.standard_normal((mp, rank))
            W = rng.standard_normal((pn, rank))
            
            for it in range(80):
                if time.time() > deadline - 5:
                    break
                
                U_new = np.zeros_like(U)
                for r in range(rank):
                    num = np.zeros(nm)
                    for a, b, c in nz:
                        num[a] += V[b, r] * W[c, r]
                    den = np.sum(V[:, r]**2) * np.sum(W[:, r]**2)
                    if den > 1e-12:
                        U_new[:, r] = num / den
                    else:
                        U_new[:, r] = U[:, r]
                U = U_new
                
                V_new = np.zeros_like(V)
                for r in range(rank):
                    num = np.zeros(mp)
                    for a, b, c in nz:
                        num[b] += U[a, r] * W[c, r]
                    den = np.sum(U[:, r]**2) * np.sum(W[:, r]**2)
                    if den > 1e-12:
                        V_new[:, r] = num / den
                    else:
                        V_new[:, r] = V[:, r]
                V = V_new
                
                W_new = np.zeros_like(W)
                for r in range(rank):
                    num = np.zeros(pn)
                    for a, b, c in nz:
                        num[c] += U[a, r] * V[b, r]
                    den = np.sum(U[:, r]**2) * np.sum(V[:, r]**2)
                    if den > 1e-12:
                        W_new[:, r] = num / den
                    else:
                        W_new[:, r] = W[:, r]
                W = W_new
                
                recon = np.einsum('ar,br,cr->abc', U, V, W)
                err = np.max(np.abs(recon - target))
                if err < 1e-7:
                    break
            
            if time.time() > deadline - 5:
                break
            
            U_s = snap_half(U)
            V_s = snap_half(V)
            W_s = snap_half(W)
            
            if check_half_exact(U, V, W):
                best_rank = rank
                best_result = {"U": U_s.tolist(), "V": V_s.tolist(), "W": W_s.tolist()}
                found = True
                break
        
        if found:
            break
    
    if best_result is not None:
        return best_result
    
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
# EVOLVE-BLOCK-END
