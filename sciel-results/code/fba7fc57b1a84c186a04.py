"""Pack 100 equal spheres in a unit cube [-1/2, 1/2]^3.

Strategy:
  1. Build an initial packing from a 5x5x4 grid (100 points) with small random jitter.
  2. Run a differentiable smooth-min (log-sum-exp) surrogate objective that penalises
     both inter-sphere overlaps and wall violations.  Minimise the surrogate with
     L-BFGS-B (scipy) which gives us full gradient information over all 300 coordinates
     simultaneously.  This escapes the local minima of single-particle local search.
  3. After the smooth phase, run a short force-directed / local refinement pass that
     directly maximises the true minimum clearance (piecewise-linear objective) using
     a simple repulsion scheme, to polish the result.
  4. Return the best valid configuration found within the time budget.
"""

import time
import numpy as np
from scipy.optimize import minimize

N = 100
DEADLINE_MARGIN = 5.0  # seconds before the 120 s kill we must return


def _true_radius(centres: np.ndarray) -> float:
    """Largest common radius for the given centres (0 if invalid)."""
    if centres.shape != (N, 3):
        return 0.0
    if not np.all(np.isfinite(centres)):
        return 0.0
    # Wall clearance
    wall = 0.5 - np.max(np.abs(centres), axis=1)
    min_wall = float(np.min(wall))
    if min_wall <= 0:
        return 0.0
    # Pairwise distances
    # Use vectorised computation for speed
    diff = centres[:, None, :] - centres[None, :, :]
    dists = np.sqrt(np.sum(diff * diff, axis=2))
    np.fill_diagonal(dists, np.inf)
    min_pair = float(np.min(dists)) / 2.0
    return min(min_wall, min_pair)


def _smooth_objective_and_grad(x: np.ndarray, alpha: float):
    """
    x: (3N,) flattened centres.
    Returns (objective, gradient).
    Objective: log-sum-exp of pairwise violation terms + wall violation terms.
    We want to MINIMISE the maximum violation; the log-sum-exp is a smooth upper bound.
    """
    C = x.reshape(N, 3)

    # Pairwise: for i<j, distance d_ij. Violation if d_ij < 2r.
    # We use a surrogate: for each pair, penalty = max(0, 2r - d_ij)^2 is non-smooth.
    # Instead use: for each pair, t_ij = 2r - d_ij.  We want all t_ij <= 0.
    # Smooth max via log-sum-exp: LSE(t) = (1/alpha)*log(sum(exp(alpha*t))).
    # But we need it differentiable and we want to push t negative.
    # Actually, let's define the objective as the LSE of the positive parts.
    # Simpler: minimise the LSE of (2r - d_ij) for all pairs, plus wall terms.
    # When all 2r - d_ij < 0, LSE is negative and small; we want it as negative as possible?
    # No — we want to MAXIMISE r.  Let me reframe.

    # Better approach: fix a target radius r_target and check feasibility,
    # OR: use the objective = -min_clearance smoothed.
    # Let's use: obj = LSE( -(d_ij/2) for pairs, -(0.5-|c_ik|) for walls )
    # Minimising this pushes all clearances up.

    # Pairwise clearances: c_ij = d_ij / 2
    diff = C[:, None, :] - C[None, :, :]
    dists = np.sqrt(np.sum(diff * diff, axis=2) + 1e-12)
    np.fill_diagonal(dists, 1e12)  # avoid self
    pair_clear = dists / 2.0

    # Wall clearances: w_ik = 0.5 - |c_ik|
    wall_clear = 0.5 - np.abs(C)  # shape (N, 3)

    # Combine all clearances into one vector
    # For pairs, only take upper triangle to avoid double counting
    iu = np.triu_indices(N, k=1)
    pair_vals = pair_clear[iu]  # (N*(N-1)/2,)
    wall_vals = wall_clear.ravel()  # (3N,)

    all_clear = np.concatenate([pair_vals, wall_vals])  # (M,)

    # Smooth minimum: -LSE(-alpha * clear) / alpha  ≈  min(clear) when alpha large
    # min(clear) = -1/alpha * log(sum(exp(-alpha * clear)))
    # So objective = -1/alpha * log(sum(exp(-alpha * all_clear)))
    # Gradient wrt x: need d obj / d x

    # For numerical stability
    neg_a_c = -alpha * all_clear
    max_val = np.max(neg_a_c)
    exp_terms = np.exp(neg_a_c - max_val)
    sum_exp = np.sum(exp_terms)
    obj = -(max_val + np.log(sum_exp)) / alpha

    # Gradient: d obj / d clear_j = exp_terms[j] / sum_exp
    # Then chain rule through clearances to x.
    grad_clear = exp_terms / sum_exp  # (M,)

    # Build gradient for x (3N,)
    grad_x = np.zeros(3 * N)

    # Wall part: clear = 0.5 - |c_ik|, d clear / d c_ik = -sign(c_ik)
    wall_grad = grad_clear[-3 * N:]  # (3N,)
    # d wall_clear[i,k] / d x[3i+k] = -sign(C[i,k])
    sign_C = np.sign(C)
    grad_x += -wall_grad.reshape(N, 3) * sign_C  # wait: d obj/dx = sum_j (d obj/d clear_j)(d clear_j/dx)
    # d clear_j / d x for wall: -sign(C)
    # So contribution = grad_clear[j] * (-sign(C))
    # grad_x += wall_grad.reshape(N,3) * (-sign_C)

    # Pair part: clear_ij = ||C_i - C_j|| / 2
    # d clear_ij / d C_i = (C_i - C_j) / (2 * ||C_i - C_j||) = (C_i - C_j) / (2 * dist_ij)
    # d clear_ij / d C_j = (C_j - C_i) / (2 * dist_ij)
    # grad_x gets: for each pair (i,j), grad_clear[pair_idx] * [d clear/d C_i, d clear/d C_j]

    # This is expensive to do in a loop. Let's vectorise.
    # pair_vals has N*(N-1)/2 entries corresponding to iu indices.
    # We need to scatter gradients.

    # For efficiency, compute pairwise vectors
    # diff[i,j] = C_i - C_j, dists[i,j] = ||diff[i,j]||
    # d clear_ij / d C_i = diff[i,j] / (2 * dists[i,j])
    # d clear_ij / d C_j = -diff[i,j] / (2 * dists[i,j])

    # Only upper triangle pairs are in our list.
    # Let's build the gradient by accumulating.
    # For each pair (i,j) with i<j:
    #   g = grad_clear[pair_idx]
    #   grad_x[3i:3i+3] += g * diff[i,j] / (2*dists[i,j])
    #   grad_x[3j:3j+3] -= g * diff[i,j] / (2*dists[i,j])

    # Vectorised approach:
    # Create a (N, N, 3) tensor of diff, and (N,N) of dists.
    # We only need upper triangle.
    # Let's use a different strategy: accumulate using matrix operations.

    # Actually, for N=100, N*(N-1)/2 = 4950 pairs. A Python loop over 4950 is fine.
    pair_grad = grad_clear[:len(iu)]
    ii, jj = iu
    for idx in range(len(ii)):
        i = ii[idx]
        j = jj[idx]
        d = dists[i, j]
        if d < 1e-10:
            continue
        g = pair_grad[idx]
        vec = diff[i, j] / (2.0 * d)
        grad_x[3 * i:3 * i + 3] += g * vec
        grad_x[3 * j:3 * j + 3] -= g * vec

    return obj, grad_x


def _refine_force_directed(centres: np.ndarray, steps: int = 500, lr_init: float = 0.001):
    """
    Force-directed refinement: each sphere repels others and is pushed away from walls.
    Maximise minimum clearance iteratively.
    """
    C = centres.copy().astype(np.float64)
    for step in range(steps):
        # Compute forces
        diff = C[:, None, :] - C[None, :, :]
        dists = np.sqrt(np.sum(diff * diff, axis=2) + 1e-12)
        np.fill_diagonal(dists, 1.0)

        # Repulsion: force on i from j = (C_i - C_j) / dists[i,j]^3  (inverse-square-ish)
        # Scale so that at close range it's strong
        inv_dist3 = 1.0 / (dists ** 3)
        np.fill_diagonal(inv_dist3, 0.0)
        forces = np.sum(diff * inv_dist3[:, :, None], axis=1)

        # Wall repulsion: push away from walls
        wall_force = np.zeros_like(C)
        for k in range(3):
            # If C[i,k] > 0, push negative; if < 0, push positive.
            # Force = -sign(C[i,k]) * (1 / (0.5 - |C[i,k]|)^2) but clipped
            dist_wall = 0.5 - np.abs(C[:, k])
            dist_wall = np.maximum(dist_wall, 1e-4)
            wall_force[:, k] = -np.sign(C[:, k]) / (dist_wall ** 2)

        total_force = forces + wall_force

        # Adaptive learning rate
        lr = lr_init * (1.0 - step / steps)
        lr = max(lr, 1e-6)

        C += lr * total_force

        # Clamp to cube interior (with small margin)
        margin = 1e-6
        C = np.clip(C, -0.5 + margin, 0.5 - margin)

    return C


def construct():
    t_start = time.time()
    deadline = t_start + 120.0 - DEADLINE_MARGIN

    # Initial guess: 5x5x4 grid
    nx, ny, nz = 5, 5, 4
    ticks_x = (np.arange(nx) + 0.5) / nx - 0.5
    ticks_y = (np.arange(ny) + 0.5) / ny - 0.5
    ticks_z = (np.arange(nz) + 0.5) / nz - 0.5
    grid = np.array([(x, y, z) for z in ticks_z for y in ticks_y for x in ticks_x])
    # Add small jitter to break symmetry
    rng = np.random.default_rng(42)
    grid = grid + rng.normal(0, 0.01, grid.shape)
    # Clamp
    grid = np.clip(grid, -0.5 + 0.01, 0.5 - 0.01)

    best_centres = grid.copy()
    best_radius = _true_radius(best_centres)

    # Phase 1: Smooth minimisation with L-BFGS-B
    # Start with moderate alpha, increase over iterations
    alpha_schedule = [50, 100, 200, 400, 800]

    x0 = grid.ravel()
    current_x = x0.copy()

    for alpha in alpha_schedule:
        if time.time() > deadline - 20.0:
            break
        try:
            res = minimize(
                _smooth_objective_and_grad,
                current_x,
                args=(alpha,),
                method='L-BFGS-B',
                jac=True,
                options={'maxiter': 300, 'maxfun': 500, 'gtol': 1e-8, 'ftol': 1e-12},
            )
            if res.success or res.fun < 0:
                current_x = res.x
                C = current_x.reshape(N, 3)
                C = np.clip(C, -0.5 + 1e-8, 0.5 - 1e-8)
                r = _true_radius(C)
                if r > best_radius:
                    best_radius = r
                    best_centres = C.copy()
        except Exception:
            pass

    # Phase 2: Force-directed refinement
    if time.time() < deadline - 10.0:
        C_ref = _refine_force_directed(best_centres, steps=1000, lr_init=0.0005)
        C_ref = np.clip(C_ref, -0.5 + 1e-8, 0.5 - 1e-8)
        r_ref = _true_radius(C_ref)
        if r_ref > best_radius:
            best_radius = r_ref
            best_centres = C_ref.copy()

    # Phase 3: One more short L-BFGS polish with high alpha
    if time.time() < deadline - 5.0:
        try:
            x0 = best_centres.ravel()
            res = minimize(
                _smooth_objective_and_grad,
                x0,
                args=(1000,),
                method='L-BFGS-B',
                jac=True,
                options={'maxiter': 200, 'maxfun': 300, 'gtol': 1e-10, 'ftol': 1e-14},
            )
            if res.fun < 0:
                C = res.x.reshape(N, 3)
                C = np.clip(C, -0.5 + 1e-8, 0.5 - 1e-8)
                r = _true_radius(C)
                if r > best_radius:
                    best_radius = r
                    best_centres = C.copy()
        except Exception:
            pass

    # Final validation
    best_centres = np.clip(best_centres, -0.5 + 1e-10, 0.5 - 1e-10)
    if not np.all(np.isfinite(best_centres)):
        # Fallback to grid
        best_centres = grid.copy()

    return best_centres


# Keep the module-level N for compatibility
N = 100
