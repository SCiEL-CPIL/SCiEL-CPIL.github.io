"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np
from itertools import product

DIMENSIONS = [3, 3, 8]

# EVOLVE-BLOCK-START
def construct():
    """Find a low-rank exact tensor decomposition for 3x3 * 3x8 matrix multiplication."""
    start_time = time.time()
    deadline = start_time + 140  # safety margin before 150s kill
    
    n, m, p = DIMENSIONS
    T = np.ones((n, m, p))
    
    # Schoolbook rank as fallback
    rank_school = n * m * p
    U_school = np.zeros((n*m, rank_school))
    V_school = np.zeros((m*p, rank_school))
    W_school = np.zeros((p*n, rank_school))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i*m+j, col] = 1
                V_school[j*p+k, col] = 1
                W_school[k*n+i, col] = 1
                col += 1
    
    best_U, best_V, best_W = U_school, V_school, W_school
    best_rank = rank_school
    
    for target_rank in range(36, max(36, best_rank) + 1):
        if time.time() > deadline - 30:
            break
            
        success = _greedy_construct(T, n, m, p, target_rank, deadline)
        if success is not None:
            U, V, W = success
            if target_rank < best_rank:
                best_U, best_V, best_W = U, V, W
                best_rank = target_rank
                if best_rank <= 45:
                    break
    
    return {
        "U": best_U.tolist(),
        "V": best_V.tolist(),
        "W": best_W.tolist()
    }

def _greedy_construct(T, n, m, p, target_rank, deadline):
    """
    Greedy + local search to find a rank-`target_rank` decomposition.
    Returns (U, V, W) arrays or None if failed.
    """
    T_residual = T.copy().astype(float)
    U = np.zeros((n*m, target_rank))
    V = np.zeros((m*p, target_rank))
    W = np.zeros((p*n, target_rank))
    
    values = [-2, -1, 0, 1, 2]
    
    for r in range(target_rank):
        if time.time() > deadline - 15:
            return None
        
        best_score = -1e18
        best_u, best_v, best_w = None, None, None
        
        np.random.seed(42 + r * 1000)
        
        for attempt in range(80):
            if time.time() > deadline - 15:
                return None
            
            if attempt < 40:
                u = np.random.choice(values, size=n*m).astype(float)
                v = np.random.choice(values, size=m*p).astype(float)
                w = np.random.choice(values, size=p*n).astype(float)
            elif attempt < 60:
                a = np.random.choice(values, size=n).astype(float)
                b = np.random.choice(values, size=m).astype(float)
                c = np.random.choice(values, size=p).astype(float)
                u = np.kron(a, b)
                v = np.kron(b, c)
                w = np.kron(c, a)
            else:
                u = np.zeros(n*m)
                v = np.zeros(m*p)
                w = np.zeros(p*n)
                nu = max(1, n*m//3)
                nv = max(1, m*p//3)
                nw = max(1, p*n//3)
                u[np.random.choice(n*m, size=nu, replace=False)] = np.random.choice([1,-1], size=nu)
                v[np.random.choice(m*p, size=nv, replace=False)] = np.random.choice([1,-1], size=nv)
                w[np.random.choice(p*n, size=nw, replace=False)] = np.random.choice([1,-1], size=nw)
            
            score = 0.0
            for i in range(n):
                for j in range(m):
                    uval = u[i*m+j]
                    if uval == 0:
                        continue
                    for k in range(p):
                        score += T_residual[i,j,k] * uval * v[j*p+k] * w[k*n+i]
            
            if score > best_score:
                best_score = score
                best_u, best_v, best_w = u.copy(), v.copy(), w.copy()
        
        if best_u is None or best_score <= 1e-10:
            return None
        
        # Local search with fast delta computation
        for ls_iter in range(30):
            if time.time() > deadline - 15:
                break
            
            improved = False
            
            for i in range(n):
                for j in range(m):
                    if time.time() > deadline - 15:
                        break
                    idx = i*m+j
                    old_val = best_u[idx]
                    s = 0.0
                    for k in range(p):
                        s += T_residual[i,j,k] * best_v[j*p+k] * best_w[k*n+i]
                    for new_val in values:
                        if new_val == old_val:
                            continue
                        delta = (new_val - old_val) * s
                        if delta > 1e-10:
                            best_u[idx] = new_val
                            best_score += delta
                            improved = True
                            break
            
            if time.time() > deadline - 15:
                break
            
            for j in range(m):
                for k in range(p):
                    if time.time() > deadline - 15:
                        break
                    idx = j*p+k
                    old_val = best_v[idx]
                    s = 0.0
                    for i in range(n):
                        s += T_residual[i,j,k] * best_u[i*m+j] * best_w[k*n+i]
                    for new_val in values:
                        if new_val == old_val:
                            continue
                        delta = (new_val - old_val) * s
                        if delta > 1e-10:
                            best_v[idx] = new_val
                            best_score += delta
                            improved = True
                            break
            
            if time.time() > deadline - 15:
                break
            
            for k in range(p):
                for i in range(n):
                    if time.time() > deadline - 15:
                        break
                    idx = k*n+i
                    old_val = best_w[idx]
                    s = 0.0
                    for j in range(m):
                        s += T_residual[i,j,k] * best_u[i*m+j] * best_v[j*p+k]
                    for new_val in values:
                        if new_val == old_val:
                            continue
                        delta = (new_val - old_val) * s
                        if delta > 1e-10:
                            best_w[idx] = new_val
                            best_score += delta
                            improved = True
                            break
            
            if not improved:
                break
        
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    T_residual[i,j,k] -= best_u[i*m+j] * best_v[j*p+k] * best_w[k*n+i]
        
        U[:, r] = best_u
        V[:, r] = best_v
        W[:, r] = best_w
        
        if np.allclose(T_residual, 0, atol=1e-10):
            for r2 in range(r+1, target_rank):
                U[:, r2] = 0
                V[:, r2] = 0
                W[:, r2] = 0
            return U[:, :r+1], V[:, :r+1], W[:, :r+1]
    
    if np.allclose(T_residual, 0, atol=1e-10):
        return U, V, W
    
    return None

# EVOLVE-BLOCK-END
