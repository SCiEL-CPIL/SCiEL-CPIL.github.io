"""Tensor rank minimization for 4x4x5 matrix multiplication via structured search."""
DIMENSIONS = [4, 4, 5]
import numpy as np
import time
from itertools import product

# EVOLVE-BLOCK-START
def _schoolbook():
    n, m, p = DIMENSIONS
    rank = n * m * p
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

def _build_target(n, m, p):
    """Build the sparse MM tensor as a list of (a, b, c) indices where T=1."""
    nm, mp, pn = n * m, m * p, p * n
    indices = []
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                indices.append((a, b, c))
    return indices, nm, mp, pn

def _verify_decomp(U, V, W, indices, nm, mp, pn, tol=1e-9):
    """Check if sum_t U[a,t]*V[b,t]*W[c,t] == 1 for all (a,b,c) in indices."""
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    rank = U.shape[1]
    for (a, b, c) in indices:
        s = 0.0
        for t in range(rank):
            s += U[a, t] * V[b, t] * W[c, t]
        if abs(s - 1.0) > tol:
            return False
    return True

def _als_step(U, V, W, indices, nm, mp, pn, tol=1e-10):
    """
    One ALS iteration: given V and W, solve for U; given U and W, solve for V; given U and V, solve for W.
    For each column index a in U, we need: sum_t U[a,t] * V[b,t] * W[c,t] = 1 for all (b,c) such that (a,b,c) in indices.
    This is a linear system in U[a,:].
    """
    rank = U.shape[1]
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    
    # Build linear systems for U
    # For each a, collect all (b, c) pairs where (a, b, c) is in indices
    # The equation is: sum_t U[a,t] * (V[b,t] * W[c,t]) = 1
    # So for fixed a, we have a linear system: M_a @ U[a,:] = 1
    # where M_a has rows corresponding to (b,c) pairs and columns t, with M_a[row, t] = V[b,t]*W[c,t]
    
    U_new = np.zeros_like(U)
    V_new = np.zeros_like(V)
    W_new = np.zeros_like(W)
    
    # Solve for U
    for a in range(nm):
        rows = []
        for (aa, b, c) in indices:
            if aa == a:
                rows.append((b, c))
        if not rows:
            U_new[a, :] = U[a, :]
            continue
        M = np.zeros((len(rows), rank))
        for i, (b, c) in enumerate(rows):
            M[i, :] = V[b, :] * W[c, :]
        rhs = np.ones(len(rows))
        try:
            sol, residuals, rank_m, sv = np.linalg.lstsq(M, rhs, rcond=tol)
            U_new[a, :] = sol
        except:
            U_new[a, :] = U[a, :]
    
    # Solve for V
    for b in range(mp):
        rows = []
        for (a, bb, c) in indices:
            if bb == b:
                rows.append((a, c))
        if not rows:
            V_new[b, :] = V[b, :]
            continue
        M = np.zeros((len(rows), rank))
        for i, (a, c) in enumerate(rows):
            M[i, :] = U_new[a, :] * W[c, :]
        rhs = np.ones(len(rows))
        try:
            sol, residuals, rank_m, sv = np.linalg.lstsq(M, rhs, rcond=tol)
            V_new[b, :] = sol
        except:
            V_new[b, :] = V[b, :]
    
    # Solve for W
    for c in range(pn):
        rows = []
        for (a, b, cc) in indices:
            if cc == c:
                rows.append((a, b))
        if not rows:
            W_new[c, :] = W[c, :]
            continue
        M = np.zeros((len(rows), rank))
        for i, (a, b) in enumerate(rows):
            M[i, :] = U_new[a, :] * V_new[b, :]
        rhs = np.ones(len(rows))
        try:
            sol, residuals, rank_m, sv = np.linalg.lstsq(M, rhs, rcond=tol)
            W_new[c, :] = sol
        except:
            W_new[c, :] = W[c, :]
    
    return U_new, V_new, W_new

def _compute_error(U, V, W, indices, nm, mp, pn):
    """Compute max absolute error."""
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    rank = U.shape[1]
    max_err = 0.0
    for (a, b, c) in indices:
        s = 0.0
        for t in range(rank):
            s += U[a, t] * V[b, t] * W[c, t]
        err = abs(s - 1.0)
        if err > max_err:
            max_err = err
    return max_err

def _try_round_half_integer(U, V, W, indices, nm, mp, pn, tol=1e-7):
    """Try to round to half-integers and verify."""
    U_r = np.round(np.array(U, dtype=float) * 2) / 2
    V_r = np.round(np.array(V, dtype=float) * 2) / 2
    W_r = np.round(np.array(W, dtype=float) * 2) / 2
    
    if _verify_decomp(U_r, V_r, W_r, indices, nm, mp, pn, tol=1e-6):
        return U_r, V_r, W_r, True
    
    # Try rounding to integers
    U_r2 = np.round(np.array(U, dtype=float))
    V_r2 = np.round(np.array(V, dtype=float))
    W_r2 = np.round(np.array(W, dtype=float))
    
    if _verify_decomp(U_r2, V_r2, W_r2, indices, nm, mp, pn, tol=1e-6):
        return U_r2, V_r2, W_r2, True
    
    return None, None, None, False

def construct():
    n, m, p = DIMENSIONS
    start = time.time()
    
    # Fallback: schoolbook rank 80
    sb = _schoolbook()
    best = sb
    best_rank = 80
    
    indices, nm, mp, pn = _build_target(n, m, p)
    
    # Try ranks from 76 down to 64
    for trial_rank in [76, 72, 68, 64, 60, 56]:
        if time.time() - start > 90:
            break
        
        found = False
        for trial in range(20):
            if time.time() - start > 100:
                break
            
            rng = np.random.RandomState(1000 + trial * 7)
            U = rng.randn(nm, trial_rank) * 0.1
            V = rng.randn(mp, trial_rank) * 0.1
            W = rng.randn(pn, trial_rank) * 0.1
            
            # Run ALS iterations
            max_iter = 50
            prev_err = float('inf')
            for it in range(max_iter):
                U, V, W = _als_step(U, V, W, indices, nm, mp, pn)
                err = _compute_error(U, V, W, indices, nm, mp, pn)
                if abs(prev_err - err) < 1e-12:
                    break
                prev_err = err
            
            err = _compute_error(U, V, W, indices, nm, mp, pn)
            if err < 1e-8:
                Ur, Vr, Wr, ok = _try_round_half_integer(U, V, W, indices, nm, mp, pn)
                if ok:
                    best = {"U": Ur.tolist(), "V": Vr.tolist(), "W": Wr.tolist()}
                    best_rank = trial_rank
                    found = True
                    break
        
        if found:
            break
    
    if best_rank < 80:
        return best
    return sb
# EVOLVE-BLOCK-END
