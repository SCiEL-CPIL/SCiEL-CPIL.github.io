"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 3, 8]

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
import time
from fractions import Fraction

def construct():
    deadline = time.time() + 115  # safety margin before 120s limit
    n, m, p = DIMENSIONS
    nm, mp, pn = n * m, m * p, p * n

    def _verify(U, V, W):
        """Verify that U, V, W form a valid tensor decomposition with integer/half-integer entries."""
        if not U or not V or not W:
            return False
        R = len(U[0])
        
        # Convert to Fractions for exact verification
        # This is expensive, so only call when float loss is very low
        try:
            U_frac = [[Fraction(x).limit_denominator(2) for x in row] for row in U]
            V_frac = [[Fraction(x).limit_denominator(2) for x in row] for row in V]
            W_frac = [[Fraction(x).limit_denominator(2) for x in row] for row in W]
        except Exception:
            return False

        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = Fraction(0)
                    for c in range(R):
                        s += U_frac[i * m + j][c] * V_frac[j * p + k][c] * W_frac[k * n + i][c]
                    if s != 1:
                        return False
        return True

    def _schoolbook():
        rank = n * m * p
        U = [[0.0] * rank for _ in range(nm)]
        V = [[0.0] * rank for _ in range(mp)]
        W = [[0.0] * rank for _ in range(pn)]
        column = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][column] = 1.0
                    V[j * p + k][column] = 1.0
                    W[k * n + i][column] = 1.0
                    column += 1
        return U, V, W

    # Start with schoolbook
    best_U, best_V, best_W = _schoolbook()
    best_rank = n * m * p

    def _loss(x, R):
        U = x[:nm * R].reshape(nm, R)
        V = x[nm * R:nm * R + mp * R].reshape(mp, R)
        W = x[nm * R + mp * R:].reshape(pn, R)
        
        U_r = U.reshape(n, m, R)
        V_r = V.reshape(m, p, R)
        W_r = W.reshape(p, n, R)
        
        T = np.einsum('ijc,jkc,kic->ijk', U_r, V_r, W_r)
        return np.sum((T - 1.0) ** 2)

    def _try_rank(R, seed):
        np.random.seed(seed)
        
        # Strategy 1: Prune schoolbook
        U0, V0, W0 = _schoolbook()
        indices = np.random.choice(n * m * p, R, replace=False)
        indices.sort()
        
        U_init = np.zeros((nm, R))
        V_init = np.zeros((mp, R))
        W_init = np.zeros((pn, R))
        
        for new_idx, old_idx in enumerate(indices):
            U_init[:, new_idx] = U0[:, old_idx]
            V_init[:, new_idx] = V0[:, old_idx]
            W_init[:, new_idx] = W0[:, old_idx]
            
        x0 = np.concatenate([U_init.flatten(), V_init.flatten(), W_init.flatten()])
        
        try:
            res = minimize(_loss, x0, args=(R,), method='L-BFGS-B', options={'maxiter': 100, 'ftol': 1e-14, 'gtol': 1e-14})
            if res.success and res.fun < 1e-10:
                U_sol = res.x[:nm * R].reshape(nm, R)
                V_sol = res.x[nm * R:nm * R + mp * R].reshape(mp, R)
                W_sol = res.x[nm * R + mp * R:].reshape(pn, R)
                
                U_rounded = np.round(U_sol * 2) / 2
                V_rounded = np.round(V_sol * 2) / 2
                W_rounded = np.round(W_sol * 2) / 2
                
                if _verify(U_rounded.tolist(), V_rounded.tolist(), W_rounded.tolist()):
                    return U_rounded.tolist(), V_rounded.tolist(), W_rounded.tolist()
        except Exception:
            pass
            
        # Strategy 2: Random sparse init
        # Only try if R is significantly lower
        if R < 70:
            U_init = np.random.randn(nm, R) * 0.1
            V_init = np.random.randn(mp, R) * 0.1
            W_init = np.random.randn(pn, R) * 0.1
            x0 = np.concatenate([U_init.flatten(), V_init.flatten(), W_init.flatten()])
            try:
                res = minimize(_loss, x0, args=(R,), method='L-BFGS-B', options={'maxiter': 100, 'ftol': 1e-14, 'gtol': 1e-14})
                if res.success and res.fun < 1e-10:
                    U_sol = res.x[:nm * R].reshape(nm, R)
                    V_sol = res.x[nm * R:nm * R + mp * R].reshape(mp, R)
                    W_sol = res.x[nm * R + mp * R:].reshape(pn, R)
                    
                    U_rounded = np.round(U_sol * 2) / 2
                    V_rounded = np.round(V_sol * 2) / 2
                    W_rounded = np.round(W_sol * 2) / 2
                    
                    if _verify(U_rounded.tolist(), V_rounded.tolist(), W_rounded.tolist()):
                        return U_rounded.tolist(), V_rounded.tolist(), W_rounded.tolist()
            except Exception:
                pass

        return None

    # Search for lower ranks
    # Start from 71 down to 50
    for R in range(71, 49, -1):
        if time.time() > deadline - 10:
            break
            
        found = False
        # Try more seeds for higher ranks (closer to schoolbook)
        num_seeds = 20 if R > 65 else 10
        
        for seed in range(num_seeds):
            if time.time() > deadline - 5:
                break
            res = _try_rank(R, seed)
            if res:
                best_U, best_V, best_W = res
                best_rank = R
                found = True
                break
        
        # If we found a solution for this R, we can stop if we want the minimal one?
        # The task asks for "rank-minimal". We should keep searching downwards if possible.
        # But if we found one, we have a better score.
        # We continue to see if we can go lower.

    return {"U": best_U, "V": best_V, "W": best_W}
# EVOLVE-BLOCK-END
