"""Rewritten: smooth-surrogate + vectorized polish for 200-circle packing."""
import numpy as np
import time
from scipy.optimize import minimize

N = 200

# Precompute pairwise index pairs once (i<j)
_i, _j = np.triu_indices(N, 1)
_PAIRS = np.stack([_i, _j], axis=1)  # shape (M,2)
M = _i.shape[0]

def _score(x):
    """x: (N,2). Return the raw packing score (largest common radius)."""
    diff = x[:, None, :] - x[None, :, :]
    d = np.sqrt(np.sum(diff * diff, axis=2))
    np.fill_diagonal(d, np.inf)
    min_pair = d.min()
    wall = 0.5 - np.abs(x).max(axis=1)
    min_wall = wall.min()
    return min(min_pair / 2.0, min_wall)

def _valid(x):
    """Check a candidate is finite and inside the square."""
    if not np.all(np.isfinite(x)):
        return False
    if np.abs(x).max() > 0.5 + 1e-9:
        return False
    return True

def _softmin_obj(x, beta):
    """Negative soft-min of distances (pairwise and wall), for minimization.
    Larger beta = sharper approximation to the true min."""
    diff = x[:, None, :] - x[None, :, :]
    d = np.sqrt(np.sum(diff * diff, axis=2))
    np.fill_diagonal(d, np.inf)
    # pairwise distances
    d_flat = d.ravel()
    d_flat = d_flat[np.isfinite(d_flat)]
    # wall distances
    wall = 0.5 - np.abs(x)
    w_flat = wall.ravel()
    # combined soft-min
    all_d = np.concatenate([d_flat, w_flat])
    all_d = np.clip(all_d, 1e-12, None)
    # log-sum-exp for soft-min: -softmin = logsumexp(-d/beta)
    v = -all_d / beta
    v_max = v.max()
    lse = v_max + np.log(np.sum(np.exp(v - v_max)))
    return lse

def _softmin_grad(x, beta):
    """Gradient of the soft-min objective (for minimization)."""
    diff = x[:, None, :] - x[None, :, :]
    d = np.sqrt(np.sum(diff * diff, axis=2))
    np.fill_diagonal(d, np.inf)
    d_flat = d.ravel()
    d_flat = d_flat[np.isfinite(d_flat)]
    wall = 0.5 - np.abs(x)
    w_flat = wall.ravel()
    all_d = np.concatenate([d_flat, w_flat])
    all_d = np.clip(all_d, 1e-12, None)
    v = -all_d / beta
    v_max = v.max()
    exp_v = np.exp(v - v_max)
    s = np.sum(exp_v)
    # softmin = -beta * (v_max + log(s))
    # d(softmin)/d(all_d) = exp_v / s
    grad_all = exp_v / s  # shape (len(all_d),)
    # Now build gradient w.r.t. x
    grad = np.zeros_like(x)
    # pairwise part: for each pair (i,j), contribution to grad[i] and grad[j]
    # d = |x_i - x_j|, dd/dx_i = (x_i - x_j)/d
    n_pair = d_flat.shape[0]
    # Reconstruct per-pair gradient weights
    # grad_all[:n_pair] corresponds to flattened d (N*N with diagonal inf masked)
    # We need to map back. Use the full (N,N) structure.
    d_full = d.copy()
    np.fill_diagonal(d_full, np.inf)
    d_full_flat = d_full.ravel()
    finite_mask = np.isfinite(d_full_flat)
    # grad_all[:n_pair] is in the same order as d_flat (finite entries of d_full)
    g_pair = np.zeros_like(d_full_flat)
    g_pair[finite_mask] = grad_all[:n_pair]
    g_pair = g_pair.reshape(N, N)
    # For each (i,j): weight g_pair[i,j] * (x_i - x_j)/d[i,j]
    # grad[i] += sum_j g_pair[i,j] * (x_i - x_j)/d[i,j]
    # grad[j] -= sum_i g_pair[i,j] * (x_i - x_j)/d[i,j]
    # Vectorized:
    diff_full = x[:, None, :] - x[None, :, :]
    # d_full has inf on diagonal, set to 1 to avoid div by zero
    d_safe = np.where(np.isfinite(d_full), d_full, 1.0)
    unit = diff_full / d_safe[:, :, None]
    g = g_pair[:, :, None] * unit
    # grad[i] = sum_j g[i,j]
    grad += np.sum(g, axis=1)
    # wall part: wall_k = 0.5 - |x_k|, d(wall)/dx_k = -sign(x_k)
    n_wall = w_flat.shape[0]
    g_wall = grad_all[n_pair:n_pair + n_wall].reshape(N, 2)
    sign_x = np.sign(x)
    # d(wall)/dx = -sign(x), so d(obj)/dx = grad_all * (-sign(x))
    grad += -g_wall * sign_x
    return grad

def _slsqp_constraints(x, r):
    """Return list of constraint dicts for SLSQP: pairwise and wall."""
    # We use a vectorized approach: build all constraints at once
    # But SLSQP expects a list. For efficiency, we'll use a single
    # "sum" constraint trick is not valid; instead use the fact that
    # SLSQP handles many constraints. For 200 points, 19900+200 = 20100 constraints
    # is heavy but vectorized.
    pass

def _slsqp_fun_and_grad(x, r):
    """Objective for SLSQP: maximize r subject to constraints.
    We minimize -r (constant) subject to d_ij >= 2r and wall >= r.
    Actually, to polish, we fix r and check feasibility, or we
    maximize r. Let's do: minimize 0 subject to constraints, with r as
    the target. Better: use SLSQP to maximize r where r is a variable.
    But simpler: fix r to current best, and use SLSQP to find a feasible
    configuration (minimize 0). Then we can increase r.
    
    Alternative: use SLSQP with objective = -r, variable = [x_flat, r],
    constraints: d_ij - 2r >= 0, wall_i - r >= 0.
    """
    pass

def construct():
    t_start = time.time()
    DEADLINE = 145.0  # safety margin before 150s

    # --- Hexagonal initialization ---
    # Hexagonal lattice: rows offset by half spacing
    # For N points, use a hex grid that fits in the square
    # Spacing s, rows: y = -0.5 + s/2 + row*s*sqrt(3)/2
    # x = -0.5 + s/2 + col*s + (row%2)*s/2
    # Choose s so that ~N points fit
    # Approximate: area per point = s^2 * sqrt(3)/2
    # s ≈ sqrt(1 / (N * sqrt(3)/2)) = sqrt(2/(N*sqrt(3)))
    s = np.sqrt(2.0 / (N * np.sqrt(3.0)))
    # Generate hex grid points
    points = []
    # Number of rows: enough to cover the square
    n_rows = int(np.ceil(1.0 / (s * np.sqrt(3.0) / 2.0))) + 2
    n_cols = int(np.ceil(1.0 / s)) + 2
    for row in range(-1, n_rows + 1):
        y = -0.5 + s * np.sqrt(3.0) / 2.0 * (row + 0.5)
        if y < -0.5 or y > 0.5:
            continue
        offset = (row % 2) * s / 2.0
        for col in range(-1, n_cols + 1):
            x = -0.5 + s * (col + 0.5) + offset
            if x < -0.5 or x > 0.5:
                continue
            points.append([x, y])
    points = np.array(points)
    # If too many, take first N; if too few, pad with grid
    if len(points) < N:
        # Fallback: square grid
        m = int(np.ceil(np.sqrt(N)))
        ticks = (np.arange(m) + 0.5) / m - 0.5
        grid = np.array([(x, y) for y in ticks for x in ticks])
        points = np.vstack([points, grid])
    x = points[:N].copy()
    
    # Ensure inside square
    x = np.clip(x, -0.5 + 1e-6, 0.5 - 1e-6)
    
    best_x = x.copy()
    best_score = _score(best_x)

    # --- Smooth surrogate optimization with annealing ---
    # Schedule of beta values (decreasing = sharper)
    beta_schedule = [200.0, 100.0, 50.0, 25.0, 15.0, 10.0, 7.0, 5.0, 4.0, 3.0]
    
    for beta in beta_schedule:
        if time.time() - t_start > DEADLINE - 20.0:
            break
        try:
            res = minimize(
                _softmin_obj,
                x,
                jac=_softmin_grad,
                args=(beta,),
                method='L-BFGS-B',
                options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-8, 'maxls': 50}
            )
            if _valid(res.x):
                s = _score(res.x)
                if s > best_score:
                    best_score = s
                    best_x = res.x.copy()
                x = res.x.copy()
            else:
                break
        except Exception:
            break

    # --- SLSQP polish with exact min constraints ---
    # Use a variable r and maximize r subject to constraints
    # x_flat (2N) + r (1) = 2N+1 variables
    if time.time() - t_start < DEADLINE - 10.0:
        try:
            def _slsqp_objective(z):
                # z = [x_flat, r], minimize -r
                return -z[-1]

            def _slsqp_grad(z):
                g = np.zeros_like(z)
                g[-1] = -1.0
                return g

            def _slsqp_constraints(z):
                x_z = z[:2*N].reshape(N, 2)
                r_z = z[-1]
                # Pairwise: d_ij - 2r >= 0
                diff = x_z[:, None, :] - x_z[None, :, :]
                d = np.sqrt(np.sum(diff * diff, axis=2))
                np.fill_diagonal(d, np.inf)
                pair_c = d - 2.0 * r_z
                np.fill_diagonal(pair_c, 0.0)
                # Wall: 0.5 - |x_i| - r >= 0
                wall_c = 0.5 - np.abs(x_z) - r_z
                # Combine: return as a single array with 'ineq' (>= 0)
                # SLSQP: 'ineq' means fun >= 0
                # We'll return two constraint dicts
                return [
                    {'type': 'ineq', 'fun': lambda z, p=pair_c: p},
                    {'type': 'ineq', 'fun': lambda z, w=wall_c: w},
                ]

            # Better: use a vectorized constraint function
            def _slsqp_cons(z):
                x_z = z[:2*N].reshape(N, 2)
                r_z = z[-1]
                diff = x_z[:, None, :] - x_z[None, :, :]
                d = np.sqrt(np.sum(diff * diff, axis=2))
                np.fill_diagonal(d, np.inf)
                pair_c = (d - 2.0 * r_z).ravel()
                pair_c = pair_c[np.isfinite(pair_c)]
                wall_c = (0.5 - np.abs(x_z) - r_z).ravel()
                return np.concatenate([pair_c, wall_c])

            def _slsqp_jac(z):
                # Jacobian of constraints w.r.t. z
                x_z = z[:2*N].reshape(N, 2)
                r_z = z[-1]
                diff = x_z[:, None, :] - x_z[None, :, :]
                d = np.sqrt(np.sum(diff * diff, axis=2))
                np.fill_diagonal(d, np.inf)
                d_safe = np.where(np.isfinite(d), d, 1.0)
                unit = diff / d_safe[:, :, None]
                # Pairwise constraint: d_ij - 2r
                # d/dx_i: sum_j unit[i,j]  (for the i component)
                # Actually for constraint c_ij = d_ij - 2r:
                # dc/dx_i = unit[i,j], dc/dx_j = -unit[i,j], dc/dr = -2
                jac_pair = np.zeros((M, 2*N + 1))
                # For each pair (i,j):
                # jac_pair[k, 2*i:2*i+2] = unit[i,j]
                # jac_pair[k, 2*j:2*j+2] = -unit[i,j]
                # jac_pair[k, -1] = -2
                for k in range(M):
                    i, j = _i[k], _j[k]
                    jac_pair[k, 2*i:2*i+2] = unit[i, j]
                    jac_pair[k, 2*j:2*j+2] = -unit[i, j]
                    jac_pair[k, -1] = -2.0
                # Wall constraint: 0.5 - |x_ik| - r
                # d/dx_ik = -sign(x_ik), d/dr = -1
                jac_wall = np.zeros((N*2, 2*N + 1))
                sign_x = np.sign(x_z)
                jac_wall[:, :2*N] = -sign_x.ravel()[:, None]
                jac_wall[:, -1] = -1.0
                return np.vstack([jac_pair, jac_wall])

            z0 = np.concatenate([best_x.ravel(), [best_score]])
            bounds = [(-0.5 + 1e-9, 0.5 - 1e-9)] * (2*N) + [(0.0, 0.5)]
            
            res = minimize(
                _slsqp_objective,
                z0,
                jac=_slsqp_grad,
                method='SLSQP',
                bounds=bounds,
                constraints={'type': 'ineq', 'fun': _slsqp_cons, 'jac': _slsqp_jac},
                options={'maxiter': 50, 'ftol': 1e-12, 'disp': False}
            )
            if _valid(res.x[:2*N].reshape(N, 2)):
                s = _score(res.x[:2*N].reshape(N, 2))
                if s > best_score:
                    best_score = s
                    best_x = res.x[:2*N].reshape(N, 2).copy()
        except Exception:
            pass

    # Final validity check
    if not _valid(best_x):
        # Fallback to square grid
        m = int(np.ceil(np.sqrt(N)))
        ticks = (np.arange(m) + 0.5) / m - 0.5
        best_x = np.array([(x, y) for y in ticks for x in ticks])[:N]

    return best_x
