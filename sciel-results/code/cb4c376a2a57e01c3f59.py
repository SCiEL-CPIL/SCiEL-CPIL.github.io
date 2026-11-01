"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 50

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize


def _compute_score(points):
    """Compute the packing score: min of pairwise distances/2 and boundary distances."""
    if len(points) != N:
        return 0.0
    if not np.all(np.isfinite(points)):
        return 0.0
    # Check all points are inside the unit circle
    dists_to_origin = np.linalg.norm(points, axis=1)
    if np.any(dists_to_origin >= 1.0):
        return 0.0
    # Compute all pairwise distances
    diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
    dists = np.linalg.norm(diff, axis=2)
    # Get minimum pairwise distance (excluding diagonal)
    np.fill_diagonal(dists, np.inf)
    min_pairwise = np.min(dists) / 2.0
    # Minimum distance to boundary
    min_boundary = 1.0 - np.max(dists_to_origin)
    return min(min_pairwise, min_boundary)


def _hex_lattice_init(n_points, seed=0):
    """Generate initial positions using hexagonal lattice within unit disk."""
    rng = np.random.RandomState(seed)
    best_spacing = None
    best_points = None
    best_score = -1.0

    # Try different spacings
    for spacing in np.linspace(0.12, 0.25, 15):
        k = int(1.0 / spacing) + 2
        rows = [((i + 0.5 * (j % 2)) * spacing, j * spacing * np.sqrt(3.0) / 2.0)
                for i in range(-k, k + 1) for j in range(-k, k + 1)]
        points = np.array(rows)
        radius = np.hypot(points[:, 0], points[:, 1])
        mask = radius <= 1.0 - spacing / 2.0
        points = points[mask]

        if len(points) >= n_points:
            # Sort by distance from center, take closest n_points
            order = np.argsort(radius, kind="stable")
            selected = points[order[:n_points]]
            score = _compute_score(selected)
            if score > best_score:
                best_score = score
                best_points = selected.copy()
                best_spacing = spacing

    if best_points is not None:
        return best_points

    # Fallback: random positions
    theta = rng.uniform(0, 2 * np.pi, n_points)
    r = np.sqrt(rng.uniform(0, 1, n_points))
    return np.column_stack([r * np.cos(theta), r * np.sin(theta)])


def _softmin_objective_and_grad(x, beta):
    """
    Objective: -r_smooth where r_smooth = softmin of (pairwise_dists/2, boundary_dists)
    x: (2N,) vector of centers
    Returns (value, grad) where value is the objective (to minimize), grad is its gradient.
    """
    pts = x.reshape(-1, 2)
    # Pairwise distances
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dists = np.linalg.norm(diff, axis=2)
    np.fill_diagonal(dists, np.inf)
    # Boundary distances
    boundary_dists = 1.0 - np.linalg.norm(pts, axis=1)

    # All "constraints" values: we want to maximize the minimum of these
    # r = min(min_pairwise/2, min_boundary)
    # Let's define a = dists/2 (upper triangle) and b = boundary_dists
    # We want softmin(a, b)
    # Flatten unique pairwise distances (i<j)
    iu = np.triu_indices(N, 1)
    a = dists[iu] / 2.0
    b = boundary_dists
    vals = np.concatenate([a, b])

    # Softmin: -logsumexp(-beta * vals) / beta
    # For numerical stability
    m = np.min(vals)
    z = -beta * (vals - m)
    # z is <= 0, so exp(z) <= 1
    # logsumexp(z) = log(sum(exp(z)))
    # Since z <= 0, sum(exp(z)) <= N_vals
    # We can compute directly if not too large
    exp_z = np.exp(z)
    s = np.sum(exp_z)
    if s < 1e-300:
        return 0.0, np.zeros_like(x)
    log_s = np.log(s)
    softmin_val = -(log_s - beta * m) / beta
    # Objective to minimize is -softmin_val
    value = -softmin_val

    # Gradient
    # d(softmin)/d(vals) = -exp(z)/s
    # Actually: softmin = -1/beta * log(sum(exp(-beta*vals)))
    # d(softmin)/d(vals_k) = -1/beta * (1/sum) * exp(-beta*vals_k) * (-beta)
    #                       = exp(-beta*vals_k) / sum(exp(-beta*vals))
    # Let w_k = exp(z_k) / s  (these sum to 1, and are positive)
    # d(softmin)/d(vals_k) = w_k
    # We are minimizing -softmin, so d(obj)/d(vals_k) = -w_k
    w = exp_z / s
    d_obj_d_vals = -w

    # Now map back to x
    grad = np.zeros_like(x)
    # For pairwise terms (i, j) where i < j
    # val_k = dist(i,j)/2
    # d(dist)/d(pts_i) = (pts_i - pts_j) / dist
    # d(dist)/d(pts_j) = (pts_j - pts_i) / dist
    # d(val)/d(pts_i) = 0.5 * (pts_i - pts_j) / dist
    # d(val)/d(pts_j) = 0.5 * (pts_j - pts_i) / dist
    # d(obj)/d(pts_i) += d_obj_d_vals[k] * d(val)/d(pts_i)
    # d(obj)/d(pts_j) += d_obj_d_vals[k] * d(val)/d(pts_j)

    # Vectorized update for pairwise
    # We need to accumulate contributions from all pairs
    # Let's loop over pairs for gradient accumulation (N=50 is small enough)
    idx = 0
    for i in range(N):
        for j in range(i + 1, N):
            d = d_obj_d_vals[idx]
            if d == 0.0:
                idx += 1
                continue
            diff_ij = pts[i] - pts[j]
            dist_ij = np.linalg.norm(diff_ij)
            if dist_ij < 1e-12:
                idx += 1
                continue
            # d(val)/d(pts_i) = 0.5 * diff_ij / dist_ij
            # d(val)/d(pts_j) = -0.5 * diff_ij / dist_ij
            grad[2*i] += d * 0.5 * diff_ij[0] / dist_ij
            grad[2*i+1] += d * 0.5 * diff_ij[1] / dist_ij
            grad[2*j] -= d * 0.5 * diff_ij[0] / dist_ij
            grad[2*j+1] -= d * 0.5 * diff_ij[1] / dist_ij
            idx += 1

    # For boundary terms
    # val_k = 1 - norm(pts_i)
    # d(val)/d(pts_i) = -pts_i / norm(pts_i)
    for i in range(N):
        d = d_obj_d_vals[N*(N-1)//2 + i]
        if d == 0.0:
            continue
        norm_i = np.linalg.norm(pts[i])
        if norm_i < 1e-12:
            continue
        # d(obj)/d(pts_i) = d_obj_d_vals[k] * (-pts_i / norm_i)
        grad[2*i] += d * (-pts[i][0] / norm_i)
        grad[2*i+1] += d * (-pts[i][1] / norm_i)

    return value, grad


def _simulated_annealing(points, deadline, seed=0):
    """
    Simulated Annealing on the 100D vector.
    Uses the soft-min objective.
    """
    rng = np.random.RandomState(seed)
    x = points.flatten().copy()
    best_x = x.copy()
    best_score = _compute_score(points)
    current_score = best_score

    # Annealing parameters
    T_start = 0.05
    T_end = 0.0001
    beta_start = 10.0
    beta_end = 100.0

    # We will run until deadline
    # Estimate steps per second? Just run as many as we can.
    # Each step involves a gradient computation (O(N^2)) and a random step.
    # Let's aim for ~1000-5000 steps depending on speed.

    # To make it efficient, we'll batch steps or just run a loop.
    # Let's use a loop with a time check.

    n_steps_total = 0
    while time.time() < deadline:
        # Determine current temperature and beta based on progress
        # We don't know total steps in advance, so use a time-based schedule
        # Or just fix a number of steps and break on time.
        # Let's use a fixed number of steps per "epoch" and restart if stuck?
        # No, just simple SA.

        # Let's use a schedule based on n_steps_total assuming ~1000 steps/s
        # But better to just use a fixed T decay per step and cap steps.
        # Let's do 5000 steps max or until time up.
        if n_steps_total >= 5000:
            break

        # T and beta interpolation
        # Use a fraction of time elapsed? No, step-based is more stable for SA.
        # Let's assume 5000 steps is the full anneal.
        alpha = n_steps_total / 5000.0
        T = T_start * (T_end / T_start) ** alpha
        beta = beta_start + (beta_end - beta_start) * alpha

        # Compute current objective and grad
        obj, grad = _softmin_objective_and_grad(x, beta)

        # Gradient step with noise
        # Step size should be proportional to T
        step_size = 0.01 * T / T_start  # Scales down
        # Actually, let's use a fixed step size scaled by T
        # The gradient is of the smooth approximation.
        # Let's use a step size that allows movement.
        move = -grad * (0.05 * T) + rng.normal(0, 0.01 * T, size=x.shape)

        x_new = x + move

        # Clip to unit circle?
        # The objective penalizes going outside, but let's clip to be safe
        # Actually, the soft-min handles it, but clipping helps.
        pts_new = x_new.reshape(-1, 2)
        norms = np.linalg.norm(pts_new, axis=1)
        mask = norms > 1.0
        if np.any(mask):
            pts_new[mask] = pts_new[mask] / norms[mask]
        x_new = pts_new.flatten()

        # Evaluate new objective (smooth)
        obj_new, _ = _softmin_objective_and_grad(x_new, beta)

        # Acceptance
        if obj_new < obj:
            x = x_new
            obj = obj_new
        else:
            delta = obj_new - obj
            if delta < 0:
                accept = True
            else:
                accept = rng.random() < np.exp(-delta / T)

            if accept:
                x = x_new
                obj = obj_new

        # Check actual score periodically (every 10 steps)
        if n_steps_total % 10 == 0:
            pts = x.reshape(-1, 2)
            # Ensure inside
            norms = np.linalg.norm(pts, axis=1)
            if np.any(norms > 1.0):
                pts = pts / norms[:, np.newaxis]
            score = _compute_score(pts)
            if score > best_score:
                best_score = score
                best_x = x.copy()

        n_steps_total += 1

    return best_x.reshape(-1, 2), best_score


def construct():
    start_time = time.time()
    # Safety margin: return 10s before 150s limit
    deadline = start_time + 140.0

    best_points = None
    best_score = -1.0

    # Multi-start Simulated Annealing
    # We have ~140s.
    # Each SA run might take 10-20s depending on speed.
    # Let's try to run as many as we can.
    # First, get a good initial point from hex lattice.

    # Try a few seeds for initialization
    inits = []
    for s in range(3):
        pts = _hex_lattice_init(N, seed=s)
        inits.append(pts)

    # Also add a random init
    rng = np.random.RandomState(123)
    theta = rng.uniform(0, 2 * np.pi, N)
    r = np.sqrt(rng.uniform(0, 1, N))
    rand_pts = np.column_stack([r * np.cos(theta), r * np.sin(theta)])
    inits.append(rand_pts)

    for i, init_pts in enumerate(inits):
        if time.time() > deadline:
            break

        # Run SA
        # Give each SA a chunk of time, or just let it run until global deadline?
        # If we let it run until global deadline, only the first one runs.
        # We want diversity.
        # Let's allocate time.
        # Say 30s per SA run.
        sa_deadline = min(deadline, start_time + 35.0 * (i + 1))
        # Actually, let's just run until global deadline for the first,
        # and for subsequent ones, use remaining time.
        # But if the first one takes 140s, we do nothing else.
        # Better: limit each SA to a fixed time, e.g., 20s.
        # Then we can do ~7 restarts.

        local_deadline = min(deadline, start_time + 20.0)
        # Adjust for previous runs?
        # Simpler: just use a fixed 20s per run, and break if we exceed global deadline.
        # But we need to pass the *local* deadline to the SA function.

        # Let's refine:
        # Total time available: 140s.
        # We will do up to 5 restarts.
        # Each gets 25s.
        # If we finish early, we can do more or extend.

        # For now, let's just run one long SA and see?
        # No, multi-start is important.

        # Let's do this:
        # Run SA for a fixed number of steps or time.
        # The SA function currently takes a deadline.
        # Let's pass a local deadline.

        # Calculate local deadline:
        # If i == 0, use 30s.
        # If i > 0, use 20s.
        # And ensure we don't exceed global deadline.

        if i == 0:
            local_deadline = min(deadline, start_time + 30.0)
        else:
            local_deadline = min(deadline, start_time + 20.0)

        pts, score = _simulated_annealing(init_pts, local_deadline, seed=100 + i)

        if score > best_score:
            best_score = score
            best_points = pts.copy()

    # Final polish with Nelder-Mead on the best result?
    # Nelder-Mead on 100D is slow and often bad.
    # But a few iterations might help.
    if best_points is not None and time.time() < deadline - 5.0:
        # Quick Nelder-Mead polish
        def objective(x):
            pts = x.reshape(-1, 2)
            dists = np.linalg.norm(pts, axis=1)
            if np.any(dists > 1.0):
                return 10.0 + np.sum(dists[dists > 1.0] - 1.0)
            return -_compute_score(pts)

        try:
            result = minimize(
                objective,
                best_points.flatten(),
                method='Nelder-Mead',
                options={'maxiter': 500, 'xatol': 1e-6, 'fatol': 1e-8}
            )
            new_points = result.x.reshape(-1, 2)
            new_score = _compute_score(new_points)
            if new_score > best_score:
                best_score = new_score
                best_points = new_points
        except Exception:
            pass

    # Fallback
    if best_points is None:
        best_points = _hex_lattice_init(N, seed=0)

    return best_points
# EVOLVE-BLOCK-END
