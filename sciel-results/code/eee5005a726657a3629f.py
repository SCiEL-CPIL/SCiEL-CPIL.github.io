"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time
import random
from itertools import product

DIMENSIONS = [3, 4, 7]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    nm, mp, pn = n * m, m * p, p * n
    rank_schoolbook = n * m * p  # 84
    
    # Build the target tensor T[a,b,c] where a=i*m+j, b=j*p+k, c=k*n+i
    # T[i*m+j, j*p+k, k*n+i] = 1
    T = np.zeros((nm, mp, pn), dtype=np.float64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i*m+j, j*p+k, k*n+i] = 1.0
    
    # Schoolbook decomposition as fallback
    U_s = np.zeros((nm, rank_schoolbook))
    V_s = np.zeros((mp, rank_schoolbook))
    W_s = np.zeros((pn, rank_schoolbook))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_s[i*m+j, col] = 1
                V_s[j*p+k, col] = 1
                W_s[k*n+i, col] = 1
                col += 1
    
    best_rank = rank_schoolbook
    best_U, best_V, best_W = U_s.copy(), V_s.copy(), W_s.copy()
    
    deadline = time.time() + 130  # Safety margin
    
    def verify(U, V, W, tol=1e-6):
        """Check if sum_r U[:,r] ⊗ V[:,r] ⊗ W[:,r] == T"""
        S = np.zeros((nm, mp, pn), dtype=np.float64)
        for r in range(U.shape[1]):
            u = U[:, r]
            v = V[:, r]
            w = W[:, r]
            if np.all(u == 0) or np.all(v == 0) or np.all(w == 0):
                continue
            S += np.einsum('a,b,c->abc', u, v, w)
        return np.allclose(S, T, atol=tol)
    
    def objective_fn(params, r):
        """Objective: sum of squared residuals"""
        Ur = params[:nm*r].reshape(nm, r)
        Vr = params[nm*r:nm*r+mp*r].reshape(mp, r)
        Wr = params[nm*r+mp*r:].reshape(pn, r)
        S = np.zeros((nm, mp, pn), dtype=np.float64)
        for ri in range(r):
            S += np.einsum('a,b,c->abc', Ur[:, ri], Vr[:, ri], Wr[:, ri])
        return np.sum((S - T)**2)
    
    def snap_to_rational(U, V, W, max_denom=8):
        """Try to snap values to rational numbers with small denominators"""
        U_snap = U.copy()
        V_snap = V.copy()
        W_snap = W.copy()
        
        for M in [U_snap, V_snap, W_snap]:
            for i in range(M.shape[0]):
                for j in range(M.shape[1]):
                    val = M[i, j]
                    if abs(val) < 1e-10:
                        M[i, j] = 0.0
                    else:
                        best_frac = val
                        best_err = abs(val - best_frac)
                        for d in range(1, max_denom + 1):
                            num = round(val * d)
                            frac = num / d
                            err = abs(val - frac)
                            if err < best_err:
                                best_err = err
                                best_frac = frac
                        if best_err < 1e-6:
                            M[i, j] = best_frac
        
        return U_snap, V_snap, W_snap
    
    def try_optimize(U_init, V_init, W_init, r, max_iters=200):
        """Try to optimize a rank-r decomposition using BFGS"""
        x0 = np.concatenate([U_init.flatten(), V_init.flatten(), W_init.flatten()])
        
        from scipy.optimize import minimize
        
        try:
            res = minimize(objective_fn, x0, args=(r,), method='BFGS',
                          options={'maxiter': max_iters, 'gtol': 1e-8})
            if res.fun < 1e-6:
                Ur = res.x[:nm*r].reshape(nm, r)
                Vr = res.x[nm*r:nm*r+mp*r].reshape(mp, r)
                Wr = res.x[nm*r+mp*r:].reshape(pn, r)
                if verify(Ur, Vr, Wr):
                    # Try to snap to rationals
                    Ur_snap, Vr_snap, Wr_snap = snap_to_rational(Ur, Vr, Wr)
                    if verify(Ur_snap, Vr_snap, Wr_snap):
                        return Ur_snap, Vr_snap, Wr_snap
                    return Ur, Vr, Wr
        except:
            pass
        return None
    
    # Strategy 1: Try to reduce from schoolbook by removing redundant terms
    # The schoolbook has 84 terms. We can try to find linear dependencies.
    
    # First, let's try a more systematic approach:
    # For each output element (i,k), we need sum_j A[i,j]*B[j,k]
    # This is a bilinear form. The tensor rank is the minimum number of rank-1 terms.
    
    # Known lower bounds: rank >= max(n*p, m*p, n*m) = max(21, 28, 12) = 28
    # For 3x4x7, the rank is likely between 28 and 84.
    
    # Let's try various target ranks
    target_ranks = list(range(28, 84))
    
    # Sort by likelihood of success (try lower ranks first, but also try some higher ones)
    # Actually, let's try from low to high, stopping when we find a valid decomposition
    for target_r in target_ranks:
        if time.time() > deadline:
            break
        if target_r >= best_rank:
            continue
            
        # Multiple random initializations
        for attempt in range(10):
            if time.time() > deadline:
                break
            
            # Different initialization strategies
            if attempt == 0:
                # Use subset of schoolbook columns
                cols = np.random.choice(rank_schoolbook, target_r, replace=False)
                U_init = U_s[:, cols].copy()
                V_init = V_s[:, cols].copy()
                W_init = W_s[:, cols].copy()
            elif attempt == 1:
                # Random sparse initialization
                U_init = np.zeros((nm, target_r))
                V_init = np.zeros((mp, target_r))
                W_init = np.zeros((pn, target_r))
                for r in range(target_r):
                    # Each term should involve one i, one j, one k
                    i = random.randint(0, n-1)
                    j = random.randint(0, m-1)
                    k = random.randint(0, p-1)
                    U_init[i*m+j, r] = 1
                    V_init[j*p+k, r] = 1
                    W_init[k*n+i, r] = 1
            elif attempt == 2:
                # Dense random initialization
                U_init = np.random.randn(nm, target_r) * 0.1
                V_init = np.random.randn(mp, target_r) * 0.1
                W_init = np.random.randn(pn, target_r) * 0.1
            elif attempt == 3:
                # Structured initialization based on output elements
                U_init = np.zeros((nm, target_r))
                V_init = np.zeros((mp, target_r))
                W_init = np.zeros((pn, target_r))
                # Distribute terms among output elements
                num_ik = n * p
                terms_per_ik = target_r // num_ik
                rem = target_r % num_ik
                r_idx = 0
                for i in range(n):
                    for k in range(p):
                        num_terms = terms_per_ik + (1 if (i * p + k) < rem else 0)
                        for t in range(num_terms):
                            if r_idx >= target_r:
                                break
                            # Choose a j for this (i,k) term
                            j = (t + i + k) % m
                            U_init[i*m+j, r_idx] = 1
                            V_init[j*p+k, r_idx] = 1
                            W_init[k*n+i, r_idx] = 1
                            r_idx += 1
            elif attempt == 4:
                # Perturbed schoolbook subset
                cols = np.random.choice(rank_schoolbook, min(target_r, rank_schoolbook), replace=False)
                U_init = U_s[:, cols].copy()
                V_init = V_s[:, cols].copy()
                W_init = W_s[:, cols].copy()
                # Add small perturbation
                U_init += np.random.randn(nm, target_r) * 0.01
                V_init += np.random.randn(mp, target_r) * 0.01
                W_init += np.random.randn(pn, target_r) * 0.01
            else:
                # Random with structure
                U_init = np.zeros((nm, target_r))
                V_init = np.zeros((mp, target_r))
                W_init = np.zeros((pn, target_r))
                for r in range(target_r):
                    i = random.randint(0, n-1)
                    j = random.randint(0, m-1)
                    k = random.randint(0, p-1)
                    U_init[i*m+j, r] = random.choice([-1, 1])
                    V_init[j*p+k, r] = random.choice([-1, 1])
                    W_init[k*n+i, r] = random.choice([-1, 1])
            
            result = try_optimize(U_init, V_init, W_init, target_r, max_iters=100)
            if result is not None:
                Ur, Vr, Wr = result
                best_rank = target_r
                best_U, best_V, best_W = Ur, Vr, Wr
                break
        
        # If we found a valid decomposition at this rank, we can stop
        # (since we're going from low to high)
        if best_rank == target_r:
            break
    
    # Convert to lists
    def to_list(M):
        L = []
        for row in M:
            L.append([float(x) for x in row])
        return L
    
    return {"U": to_list(best_U), "V": to_list(best_V), "W": to_list(best_W)}
# EVOLVE-BLOCK-END
