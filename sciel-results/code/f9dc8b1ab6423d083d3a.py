import time
import numpy as np
from itertools import product
from scipy.optimize import minimize

DIMENSIONS = [3, 3, 7]

def _build_target(n, m, p):
    T = np.zeros((n * m, m * p, p * n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T[a, b, c] = 1.0
    return T

def _verify(U, V, W, T, tol=1e-8):
    if U.shape[1] != V.shape[1] or V.shape[1] != W.shape[1]:
        return False
    S = np.einsum('at,bt,ct->abc', U, V, W)
    return np.max(np.abs(S - T)) < tol

def _snap_half(x):
    return np.round(x * 2.0) / 2.0

def _snap_int(x):
    return np.round(x)

def _lattice_search(n, m, p, T, max_rank, deadline, grid, seed_base=0):
    """Search for a decomposition with entries in the given grid."""
    nm, mp, pn = n * m, m * p, p * n
    best = None
    best_rank = max_rank + 1
    for r in range(2, max_rank + 1):
        if time.time() > deadline:
            break
        # Number of entries: nm*r + mp*r + pn*r
        total_entries = (nm + mp + pn) * r
        if total_entries > 60:  # too large for random lattice search
            continue
        # Try a limited number of random lattice samples
        max_samples = 200
        for s in range(max_samples):
            if time.time() > deadline:
                break
            rng = np.random.RandomState(seed_base + r * 1000 + s)
            U = rng.choice(grid, size=(nm, r))
            V = rng.choice(grid, size=(mp, r))
            W = rng.choice(grid, size=(pn, r))
            if _verify(U, V, W, T):
                best = (U.copy(), V.copy(), W.copy())
                best_rank = r
                break
        if best is not None and best_rank == r:
            # Found at this rank; try to see if lower works? We iterate downward later.
            # Actually we iterate r from low to high, so first found is minimal in this loop.
            break
    return best, best_rank

def _greedy_merge(n, m, p, T, U, V, W, deadline, max_iter=50):
    """Try to merge pairs of terms into a single term."""
    nm, mp, pn = n * m, m * p, p * n
    rank = U.shape[1]
    U = U.copy()
    V = V.copy()
    W = W.copy()
    
    for _ in range(max_iter):
        if time.time() > deadline:
            break
        if rank < 2:
            break
        merged = False
        # Try all pairs
        for i in range(rank):
            if time.time() > deadline:
                break
            for j in range(i + 1, rank):
                # Try to replace terms i and j with a single term
                # We need: u_i v_i w_i + u_j v_j w_j = u' v' w'
                # This is generally not possible, but we can try:
                # Fix u' = u_i + u_j, then solve for v', w' such that
                # u'_a * v'_b * w'_c = (u_i v_i w_i + u_j v_j w_j)[a,b,c]
                # This is a rank-1 approximation problem.
                # Alternatively, try: u' = u_i, v' = v_i + v_j, w' = w_i (won't work generally)
                
                # Let's try a different approach: use the fact that
                # (u_i + u_j) * v_i * w_i = u_i v_i w_i + u_j v_i w_i
                # So if we set u' = u_i + u_j, v' = v_i, w' = w_i,
                # we get u_i v_i w_i + u_j v_i w_i, but we need u_j v_j w_j.
                # This only works if v_j = v_i and w_j = w_i.
                
                # Better: try to find u', v', w' such that the sum is rank-1.
                # The sum tensor is S = u_i v_i w_i + u_j v_j w_j.
                # Check if S is rank-1.
                S_ij = np.einsum('a,b,c->abc', U[:, i], V[:, i], W[:, i]) + \
                       np.einsum('a,b,c->abc', U[:, j], V[:, j], W[:, j])
                
                # Check rank-1: for all a,b,c,d: S[a,b,c]*S[a,d,e] == S[a,b,e]*S[a,d,c]
                # Simpler: try to find u' such that S = u' * M where M is rank-1.
                # Or: use SVD on a matricization.
                
                # Matricize: reshape S to (nm, mp*pn)
                S_mat = S_ij.reshape(nm, mp * pn)
                # Check if rank-1
                try:
                    svals = np.linalg.svd(S_mat, compute_uv=False)
                    if len(svals) > 0 and svals[0] > 1e-10:
                        rank_ratio = svals[1] / svals[0] if len(svals) > 1 else 0
                        if rank_ratio < 1e-8:
                            # It's rank-1!
                            U_s, S_s, Vt_s = np.linalg.svd(S_mat, full_matrices=False)
                            u_prime = U_s[:, 0] * np.sqrt(S_s[0])
                            M_prime = (np.sqrt(S_s[0]) * Vt_s[0, :]).reshape(mp, pn)
                            # Now M_prime should be v' * w'
                            M_mat = M_prime
                            U_m, S_m, Vt_m = np.linalg.svd(M_mat, compute_uv=False)
                            if len(S_m) > 0 and S_m[0] > 1e-10:
                                rank_ratio_m = S_m[1] / S_m[0] if len(S_m) > 1 else 0
                                if rank_ratio_m < 1e-8:
                                    v_prime = U_m[:, 0] * np.sqrt(S_m[0])
                                    w_prime = (np.sqrt(S_m[0]) * Vt_m[0, :])
                                    # Verify
                                    S_new = np.einsum('a,b,c->abc', u_prime, v_prime, w_prime)
                                    if np.max(np.abs(S_new - S_ij)) < 1e-8:
                                        # Merge: remove j, replace i with the new term
                                        U[:, i] = u_prime
                                        V[:, i] = v_prime
                                        W[:, i] = w_prime
                                        # Remove column j
                                        U = np.delete(U, j, axis=1)
                                        V = np.delete(V, j, axis=1)
                                        W = np.delete(W, j, axis=1)
                                        rank -= 1
                                        merged = True
                                        break
                except Exception:
                    pass
            if merged:
                break
        if not merged:
            break
    
    return U, V, W, rank

def _continuous_optimize(n, m, p, T, r, deadline, max_restarts=5):
    """Try to find a rank-r decomposition via continuous optimization."""
    nm, mp, pn = n * m, m * p, p * n
    
    def compute_error_and_grad(x):
        U = x[:nm*r].reshape(nm, r)
        V = x[nm*r:nm*r+mp*r].reshape(mp, r)
        W = x[nm*r+mp*r:].reshape(pn, r)
        S = np.einsum('at,bt,ct->abc', U, V, W)
        diff = S - T
        err = np.sum(diff**2)
        grad_U = 2.0 * np.einsum('abc,bt,ct->at', diff, V, W)
        grad_V = 2.0 * np.einsum('abc,at,ct->bt', diff, U, W)
        grad_W = 2.0 * np.einsum('abc,at,bt->ct', diff, U, V)
        grad = np.concatenate([grad_U.flatten(), grad_V.flatten(), grad_W.flatten()])
        return err, grad
    
    best = None
    for restart in range(max_restarts):
        if time.time() > deadline:
            break
        rng = np.random.RandomState(restart * 7919 + r)
        U_init = rng.randn(nm, r) * 0.5
        V_init = rng.randn(mp, r) * 0.5
        W_init = rng.randn(pn, r) * 0.5
        x0 = np.concatenate([U_init.flatten(), V_init.flatten(), W_init.flatten()])
        
        try:
            res = minimize(
                compute_error_and_grad,
                x0,
                method='L-BFGS-B',
                jac=True,
                options={'maxiter': 200, 'ftol': 1e-14, 'gtol': 1e-10, 'maxls': 30}
            )
            if res.fun < 1e-10:
                U_sol = res.x[:nm*r].reshape(nm, r)
                V_sol = res.x[nm*r:nm*r+mp*r].reshape(mp, r)
                W_sol = res.x[nm*r+mp*r:].reshape(pn, r)
                if _verify(U_sol, V_sol, W_sol, T):
                    # Try to snap to half-integers
                    U_snap = _snap_half(U_sol)
                    V_snap = _snap_half(V_sol)
                    W_snap = _snap_half(W_sol)
                    if _verify(U_snap, V_snap, W_snap, T):
                        best = (U_snap, V_snap, W_snap)
                    else:
                        U_snap2 = _snap_int(U_sol)
                        V_snap2 = _snap_int(V_sol)
                        W_snap2 = _snap_int(W_sol)
                        if _verify(U_snap2, V_snap2, W_snap2, T):
                            best = (U_snap2, V_snap2, W_snap2)
                        else:
                            best = (U_sol, V_sol, W_sol)
                    break
        except Exception:
            pass
    
    return best

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 115  # Safety margin
    
    nm, mp, pn = n * m, m * p, p * n
    T = _build_target(n, m, p)
    
    # Schoolbook solution (rank 63)
    rank_sb = n * m * p
    U_sb = np.zeros((nm, rank_sb))
    V_sb = np.zeros((mp, rank_sb))
    W_sb = np.zeros((pn, rank_sb))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i*m+j, col] = 1.0
                V_sb[j*p+k, col] = 1.0
                W_sb[k*n+i, col] = 1.0
                col += 1
    
    best_result = {
        "U": U_sb.tolist(),
        "V": V_sb.tolist(),
        "W": W_sb.tolist()
    }
    best_rank = rank_sb
    
    # Phase 1: Greedy merge from schoolbook
    if time.time() < deadline:
        U_m, V_m, W_m, rank_m = _greedy_merge(n, m, p, T, U_sb, V_sb, W_sb, deadline)
        if rank_m < best_rank and _verify(U_m, V_m, W_m, T):
            best_rank = rank_m
            best_result = {
                "U": U_m.tolist(),
                "V": V_m.tolist(),
                "W": W_m.tolist()
            }
    
    # Phase 2: Lattice search for small ranks
    if time.time() < deadline:
        grid = np.array([0.0, 0.5, -0.5, 1.0, -1.0, 2.0, -2.0])
        # Try ranks from 2 up to min(best_rank-1, 15)
        max_lattice_rank = min(best_rank - 1, 15)
        for r in range(2, max_lattice_rank + 1):
            if time.time() > deadline:
                break
            result, found_rank = _lattice_search(n, m, p, T, r, deadline, grid, seed_base=r)
            if result is not None:
                best_rank = r
                best_result = {
                    "U": result[0].tolist(),
                    "V": result[1].tolist(),
                    "W": result[2].tolist()
                }
                break  # Found at this rank, which is minimal in this loop
    
    # Phase 3: Continuous optimization for intermediate ranks
    if time.time() < deadline:
        # Try ranks from max(2, best_rank-5) to best_rank-1
        start_r = max(2, best_rank - 5)
        for r in range(start_r, best_rank):
            if time.time() > deadline:
                break
            result = _continuous_optimize(n, m, p, T, r, deadline, max_restarts=3)
            if result is not None:
                best_rank = r
                best_result = {
                    "U": result[0].tolist(),
                    "V": result[1].tolist(),
                    "W": result[2].tolist()
                }
                break
    
    return best_result
