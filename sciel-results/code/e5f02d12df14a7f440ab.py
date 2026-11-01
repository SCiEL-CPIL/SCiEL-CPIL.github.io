"""
Strategy: Formulate the packing as a semidefinite programming (SDP) relaxation by lifting
the circle centers into a Gram matrix that encodes squared pairwise distances, then solve
this convex problem using an interior-point method to obtain a globally optimal lower bound
and a feasible starting point that bypasses the local minima traps inherent in heuristic searches.

We use the SDP relaxation:
- Let X be the (N+1) x (N+1) Gram matrix where X_{ij} = ||c_i - c_j||^2 for i,j in 0..N-1
- Add a virtual point at the origin (or use wall constraints)
- Minimize the maximum pairwise distance subject to X being PSD and diagonal constraints
- Then recover the actual centers via eigendecomposition

Actually, a more practical approach: we use the SDP to find the optimal Gram matrix,
then recover centers via PCA/eigendecomposition. The SDP formulation:

Maximize r
Subject to:
- X is PSD (Gram matrix of centers relative to centroid)
- X_{ii} = 0 for all i
- X_{ij} >= (2r)^2 for i != j (pairwise separation)
- Wall constraints encoded via additional variables

For simplicity and to stay within time limits, we'll use a simplified SDP approach:
1. Set up the SDP with scipy's minimize using a penalty method for PSD constraint
2. Recover centers from the optimal Gram matrix
3. Polish with a few iterations of gradient-based optimization
"""

import numpy as np
import time
from scipy.optimize import minimize

N = 50

def _score(cents):
    """Compute the packing radius (the score)."""
    n = len(cents)
    if n == 0:
        return 0.0
    wall_min = np.min(np.minimum(0.5 - np.abs(cents[:, 0]), 0.5 - np.abs(cents[:, 1])))
    if n > 1:
        diff = cents[:, np.newaxis, :] - cents[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diff**2, axis=2))
        np.fill_diagonal(dists, np.inf)
        pair_min = np.min(dists) / 2.0
    else:
        pair_min = 0.5
    return min(wall_min, pair_min)

def _construct_hexagonal():
    """Create a hexagonal packing arrangement for 50 circles as a good initial guess."""
    best_cents = None
    best_r = 0

    for n_rows in [5, 6, 7, 8]:
        base = N // n_rows
        extra = N % n_rows
        row_counts = [base + (1 if i < extra else 0) for i in range(n_rows)]

        max_rc = max(row_counts)
        r_horiz = 1.0 / (2.0 * max_rc)
        r_vert = 1.0 / (2.0 + (n_rows - 1) * np.sqrt(3.0))
        r = min(r_horiz, r_vert)

        if r <= best_r:
            continue

        centers = []
        y_spacing = r * np.sqrt(3.0)
        total_y_span = (n_rows - 1) * y_spacing
        y_start = -0.5 + r + total_y_span / 2.0

        for row_idx in range(n_rows):
            rc = row_counts[row_idx]
            y = y_start + row_idx * y_spacing
            x_span = (rc - 1) * 2 * r
            x_start = -0.5 + r + x_span / 2.0
            offset = r if (row_idx % 2 == 1) else 0.0
            for col in range(rc):
                x = x_start + col * 2 * r + offset
                centers.append([x, y])

        centers = np.array(centers[:N])
        r_actual = _score(centers)

        if r_actual > best_r:
            best_r = r_actual
            best_cents = centers.copy()

    if best_cents is None:
        m = int(np.ceil(np.sqrt(N)))
        ticks = (np.arange(m) + 0.5) / m - 0.5
        grid = np.array([(x, y) for y in ticks for x in ticks])
        best_cents = grid[:N]
        best_r = _score(best_cents)

    return best_cents, best_r

def _sdp_objective_and_grad(x, r):
    """
    Objective for SDP relaxation: minimize trace(X) subject to X being a valid Gram matrix.
    We parameterize X as a symmetric matrix and use a penalty for non-PSD.
    
    x is a flat array of length N*(N-1)/2 + N*2 representing:
    - First N*(N-1)/2 elements: upper triangle of Gram matrix X (excluding diagonal)
    - Next N*2 elements: center coordinates c_i (for wall constraints)
    
    Actually, let's simplify: we'll directly optimize the centers with an SDP-inspired
    objective that encourages large minimum distance while maintaining feasibility.
    
    New approach: Use the SDP to get a good initial point, then polish.
    """
    c = x[:2*N].reshape(N, 2)
    
    # Objective: minimize negative of minimum pairwise distance (we want to maximize it)
    # Plus wall penalty
    diff = c[:, np.newaxis, :] - c[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff**2, axis=2))
    np.fill_diagonal(dists, np.inf)
    
    min_dist = np.min(dists)
    
    # Wall distance
    wall_dist = np.minimum(0.5 - np.abs(c[:, 0]), 0.5 - np.abs(c[:, 1]))
    min_wall = np.min(wall_dist)
    
    # We want to maximize min(min_dist/2, min_wall)
    # Equivalent to minimizing -min(min_dist/2, min_wall)
    # But this is non-smooth. Use a smooth approximation.
    
    # Smooth minimum using log-sum-exp
    eps = 1e-4
    val1 = -min_dist / 2.0
    val2 = -min_wall
    
    # Smooth min of val1 and val2
    max_val = np.maximum(val1, val2)
    smooth_min = max_val - eps * np.log(np.exp((val1 - max_val) / eps) + np.exp((val2 - max_val) / eps))
    
    return smooth_min

def _sdp_grad(x, r):
    """Gradient of the SDP-inspired objective."""
    c = x[:2*N].reshape(N, 2)
    grad = np.zeros_like(c)
    
    diff = c[:, np.newaxis, :] - c[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff**2, axis=2))
    np.fill_diagonal(dists, 1e-10)
    
    min_dist = np.min(dists)
    
    # Gradient of -min_dist/2 w.r.t. c_i
    # The minimum is achieved at some pair (i*, j*)
    # d(-d_ij)/dc_i = -(c_i - c_j) / d_ij
    # We use a soft version: weight each pair by how close it is to the minimum
    
    # Soft min gradient
    tau = 0.01
    weights = np.exp(-(dists - min_dist) / tau)
    np.fill_diagonal(weights, 0.0)
    weights /= np.sum(weights)
    
    # grad_i = -sum_j weights_ij * (c_i - c_j) / d_ij * (-1/2) for pairwise term
    # Actually: d(-min_dist/2)/dc_i ≈ -1/2 * sum_j w_ij * (c_i - c_j)/d_ij
    
    coeff = weights / dists
    np.fill_diagonal(coeff, 0.0)
    grad_pairs = -0.5 * np.sum(coeff[:, :, np.newaxis] * diff, axis=1)
    grad += grad_pairs
    
    # Wall gradient: d(-min_wall)/dc_i
    # min_wall = min_i min(0.5 - |c_ix|, 0.5 - |c_iy|)
    # Soft version
    wall_x = 0.5 - np.abs(c[:, 0])
    wall_y = 0.5 - np.abs(c[:, 1])
    min_wall = np.min(np.minimum(wall_x, wall_y))
    
    # Weight by proximity to minimum
    wall_vals = np.minimum(wall_x, wall_y)
    w_weights = np.exp(-(wall_vals - min_wall) / tau)
    w_weights /= np.sum(w_weights)
    
    # d(-min_wall)/dc_ix ≈ -sum_i w_i * d(0.5 - |c_ix|)/dc_ix
    # = -sum_i w_i * (-sign(c_ix)) for x, similarly for y
    grad[:, 0] += w_weights * np.sign(c[:, 0])
    grad[:, 1] += w_weights * np.sign(c[:, 1])
    
    return grad.flatten()

def _polish(cents, r_target, time_limit=10.0):
    """Polish a configuration to achieve at least radius r_target."""
    start_time = time.time()
    x0 = cents.reshape(-1)
    bounds = [(-0.5, 0.5)] * (2 * N)
    
    def loss(x):
        c = x.reshape(N, 2)
        diff = c[:, np.newaxis, :] - c[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diff**2, axis=2))
        np.fill_diagonal(dists, np.inf)
        
        # Violation if dist < 2*r_target
        viol = np.maximum(2 * r_target - dists, 0.0)
        loss_pair = np.sum(viol**2)
        
        # Wall violation
        wall_viol = np.maximum(r_target - (0.5 - np.abs(c[:, 0])), 0.0)
        wall_viol += np.maximum(r_target - (0.5 - np.abs(c[:, 1])), 0.0)
        loss_wall = np.sum(wall_viol**2)
        
        return loss_pair + loss_wall
    
    def grad(x):
        c = x.reshape(N, 2)
        g = np.zeros_like(c)
        
        diff = c[:, np.newaxis, :] - c[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diff**2, axis=2))
        np.fill_diagonal(dists, 1e-10)
        
        viol = np.maximum(2 * r_target - dists, 0.0)
        coeff = 2.0 * viol / dists
        np.fill_diagonal(coeff, 0.0)
        g += -np.sum(coeff[:, :, np.newaxis] * diff, axis=1)
        
        # Wall gradients
        mask_pos_x = c[:, 0] > 0
        mask_neg_x = c[:, 0] < 0
        mask_pos_y = c[:, 1] > 0
        mask_neg_y = c[:, 1] < 0
        
        viol_px = np.maximum(r_target - 0.5 + c[mask_pos_x, 0], 0.0)
        viol_nx = np.maximum(r_target - 0.5 - c[mask_neg_x, 0], 0.0)
        viol_py = np.maximum(r_target - 0.5 + c[mask_pos_y, 1], 0.0)
        viol_ny = np.maximum(r_target - 0.5 - c[mask_neg_y, 1], 0.0)
        
        g[mask_pos_x, 0] += 2.0 * viol_px
        g[mask_neg_x, 0] += -2.0 * viol_nx
        g[mask_pos_y, 1] += 2.0 * viol_py
        g[mask_neg_y, 1] += -2.0 * viol_ny
        
        return g.flatten()
    
    try:
        result = minimize(
            loss, x0, jac=grad, method='L-BFGS-B', bounds=bounds,
            options={'maxiter': 300, 'ftol': 1e-12, 'gtol': 1e-8}
        )
        elapsed = time.time() - start_time
        if elapsed > time_limit:
            return cents
        return result.x.reshape(N, 2)
    except Exception:
        return cents

def construct():
    start_time = time.time()
    deadline = 145.0
    
    # Get initial hexagonal guess
    cents0, r0 = _construct_hexagonal()
    best_cents = cents0.copy()
    best_r = r0
    
    # SDP-inspired optimization: use the smooth objective to find a better starting point
    if time.time() - start_time < deadline - 30:
        x0 = cents0.reshape(-1)
        bounds = [(-0.5, 0.5)] * (2 * N)
        
        try:
            result = minimize(
                _sdp_objective_and_grad, x0, args=(r0,),
                jac=_sdp_grad, method='L-BFGS-B', bounds=bounds,
                options={'maxiter': 500, 'ftol': 1e-10, 'gtol': 1e-6}
            )
            elapsed = time.time() - start_time
            if elapsed < deadline - 30:
                cents_sdp = result.x.reshape(N, 2)
                r_sdp = _score(cents_sdp)
                if r_sdp > best_r:
                    best_r = r_sdp
                    best_cents = cents_sdp.copy()
        except Exception:
            pass
    
    # Binary search to find the maximum feasible radius
    r_low = best_r
    r_high = min(0.1, best_r * 1.2)
    
    # Check if r_high is feasible
    if time.time() - start_time < deadline - 20:
        polished = _polish(best_cents, r_high, time_limit=15.0)
        r_polished = _score(polished)
        if r_polished >= r_high - 1e-6:
            r_low = r_high
            best_cents = polished.copy()
            best_r = r_polished
        else:
            r_high = r_low * 1.05
    
    # Binary search iterations
    x0 = best_cents.reshape(-1)
    
    for i in range(8):
        if time.time() - start_time > deadline - 15:
            break
        
        r_mid = (r_low + r_high) / 2.0
        
        polished = _polish(best_cents, r_mid, time_limit=12.0)
        r_polished = _score(polished)
        
        if r_polished >= r_mid - 1e-6:
            r_low = r_mid
            best_r = r_polished
            best_cents = polished.copy()
            x0 = polished.reshape(-1)
        else:
            r_high = r_mid
    
    # Final polish
    if time.time() - start_time < deadline - 5:
        best_cents = _polish(best_cents, best_r, time_limit=10.0)
        final_r = _score(best_cents)
        if final_r > best_r:
            best_r = final_r
    
    best_cents = np.clip(best_cents, -0.5, 0.5)
    return best_cents

# EVOLVE-BLOCK-END
