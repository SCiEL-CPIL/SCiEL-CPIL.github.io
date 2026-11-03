"""Circle packing: 200 equal circles in unit square [-0.5, 0.5]^2.
Uses scipy L-BFGS-B with a smooth soft-min objective combining pairwise
repulsion and boundary repulsion, plus multi-start from hexagonal and random inits.
"""
import time
import numpy as np
from scipy.optimize import minimize
from scipy.spatial.distance import cdist

N = 200
DEADLINE = 145.0  # wall-clock safety margin (evaluator kills at 150s)


def _compute_radius(pts):
    """Compute the maximum common radius for the given centres."""
    # Distance to boundary
    boundary_min = np.min(0.5 - np.abs(pts))
    # Pairwise distances
    d = cdist(pts, pts)
    np.fill_diagonal(d, np.inf)
    min_pair = np.min(d)
    r = min(min_pair, 2.0 * boundary_min) / 2.0
    return r


def _soft_min(x, k):
    """Smooth approximation of min(x) using log-sum-exp trick."""
    # x is a 1D array
    return -np.log(np.sum(np.exp(-k * x))) / k


def _objective_and_grad(flat_pts, k):
    """
    Objective: maximize the minimum of (pairwise distances) and (2 * boundary distances).
    We minimize the negative of the soft-min of these values.
    flat_pts: shape (2*N,)
    k: softness parameter
    """
    pts = flat_pts.reshape(-1, 2)
    
    # Pairwise distances
    d = cdist(pts, pts)
    np.fill_diagonal(d, np.inf)
    
    # Boundary distances: 2*(0.5 - |coord|) for each coordinate
    # We want min over all coords of (0.5 - |coord|), then multiply by 2
    # Actually, the radius constraint is: r <= min_pair/2 and r <= min_boundary
    # So we want to maximize min(min_pair/2, min_boundary)
    # Equivalently, maximize min(min_pair, 2*min_boundary) / 2
    # Let's work with the unscaled values: maximize min(min_pair, 2*min_boundary)
    
    # 2 * boundary distances for each coordinate
    boundary_vals = 2.0 * (0.5 - np.abs(pts))  # shape (N, 2)
    
    # All values we care about
    # Pairwise: off-diagonal entries of d
    # Boundary: all entries of boundary_vals
    
    # For efficiency, we can compute soft-min separately
    # But for gradient, we need the full computation
    
    # Soft-min of pairwise distances
    # d has inf on diagonal, so we need to handle that
    # Actually, let's just use the finite entries
    d_flat = d[~np.eye(N, dtype=bool)]
    
    # Soft-min of boundary values
    b_flat = boundary_vals.flatten()
    
    # Combined: we want soft-min of all these values
    # But that's 19900 + 400 = 20300 values, which is fine
    
    all_vals = np.concatenate([d_flat, b_flat])
    
    # Soft-min
    sm = _soft_min(all_vals, k)
    
    # Gradient computation
    # d(sm)/d(x) for each coordinate
    # sm = -1/k * log(sum(exp(-k * v_i)))
    # d(sm)/d(v_i) = exp(-k * v_i) / sum(exp(-k * v_j))
    # By chain rule: d(sm)/d(x_j) = sum_i d(sm)/d(v_i) * d(v_i)/d(x_j)
    
    # Compute weights
    exp_vals = np.exp(-k * all_vals)
    weights = exp_vals / np.sum(exp_vals)
    
    # Now compute gradient with respect to flat_pts
    grad = np.zeros_like(flat_pts)
    
    # Pairwise distances gradient
    # v_i = ||p_a - p_b|| for pairs (a,b)
    # d(v_i)/d(p_a) = (p_a - p_b) / ||p_a - p_b||
    # d(v_i)/d(p_b) = (p_b - p_a) / ||p_a - p_b||
    
    # We need to accumulate contributions from all pairs
    # This is O(N^2) which is fine for N=200
    
    # For each pair (i,j) with i<j, weight w_ij
    # d(v_ij)/d(p_i) = (p_i - p_j) / v_ij
    # d(v_ij)/d(p_j) = (p_j - p_i) / v_ij
    
    # We have weights for the flat array, first N*(N-1) entries correspond to pairs
    # The ordering from d[~np.eye(N, dtype=bool)] is row-major
    # Row i has entries for j != i
    
    # Let's build the gradient more carefully
    # For row i of d, the entries are for j=0..i-1, i+1..N-1
    # In the flat array, row i starts at index sum_{r<i} (N-1)
    
    pair_weights = weights[:N*(N-1)]
    
    # Iterate over rows
    idx = 0
    for i in range(N):
        for j in range(N):
            if i == j:
                continue
            w = pair_weights[idx]
            idx += 1
            diff = pts[i] - pts[j]
            dist = np.linalg.norm(diff)
            if dist > 1e-12:
                grad[2*i:2*i+2] += w * diff / dist
                grad[2*j:2*j+2] -= w * diff / dist
    
    # Boundary distances gradient
    # v = 2*(0.5 - |x|) for each coordinate
    # d(v)/d(x) = -2 * sign(x)
    
    boundary_weights = weights[N*(N-1):]
    for i in range(N):
        for dim in range(2):
            w = boundary_weights[2*i + dim]
            x = flat_pts[2*i + dim]
            if x > 1e-12:
                grad[2*i + dim] += w * (-2.0)
            elif x < -1e-12:
                grad[2*i + dim] += w * (2.0)
            else:
                grad[2*i + dim] += w * (-2.0)  # arbitrary, but smooth enough
    
    # We want to maximize the soft-min, so we minimize -sm
    obj = -sm
    grad_obj = -grad
    
    return obj, grad_obj


def _generate_hex_initial(n, rng):
    """Initialize n points in a hexagonal lattice pattern inside the unit square."""
    s = np.sqrt(2.0 / (N * np.sqrt(3.0)))
    dx = s
    dy = s * np.sqrt(3.0) / 2.0
    rows = int(np.ceil(1.0 / dy)) + 2
    cols = int(np.ceil(1.0 / dx)) + 2
    pts = []
    for i in range(rows):
        y = -0.5 + i * dy
        offset = 0.0 if i % 2 == 0 else dx / 2.0
        for j in range(cols):
            x = -0.5 + j * dx + offset
            if -0.5 <= x <= 0.5 and -0.5 <= y <= 0.5:
                pts.append((x, y))
    pts = np.array(pts)
    if len(pts) >= n:
        idx = rng.choice(len(pts), size=n, replace=False)
        return pts[idx]
    else:
        result = np.empty((n, 2))
        result[:len(pts)] = pts
        result[len(pts):] = rng.uniform(-0.45, 0.45, size=(n - len(pts), 2))
        return result


def _generate_random_initial(n, rng):
    """Random initialization inside the unit square."""
    return rng.uniform(-0.45, 0.45, size=(n, 2))


def _run_l_bfgs_b(pts, time_limit, rng):
    """Run L-BFGS-B optimization with time limit."""
    flat_pts = pts.flatten()
    
    # Start with moderate k, increase over time
    k_start = 50.0
    k_end = 200.0
    
    t0 = time.time()
    best_flat = flat_pts.copy()
    best_obj = np.inf
    
    # We'll do multiple phases with increasing k
    n_phases = 3
    for phase in range(n_phases):
        if time.time() - t0 > time_limit:
            break
        
        k = k_start + (k_end - k_start) * phase / max(n_phases - 1, 1)
        
        # Run L-BFGS-B with time check via callback
        def callback(xk):
            if time.time() - t0 > time_limit - 2.0:
                return True  # Stop
            return False
        
        res = minimize(
            _objective_and_grad,
            flat_pts,
            args=(k,),
            jac=True,
            method='L-BFGS-B',
            options={
                'maxiter': 200,
                'ftol': 1e-10,
                'gtol': 1e-8,
                'maxls': 50,
                'callback': callback
            }
        )
        
        if res.x is not None:
            flat_pts = res.x
            obj, _ = _objective_and_grad(flat_pts, k)
            if obj < best_obj:
                best_obj = obj
                best_flat = flat_pts.copy()
        
        # Clip to stay inside square
        pts_tmp = best_flat.reshape(-1, 2)
        pts_tmp = np.clip(pts_tmp, -0.499, 0.499)
        best_flat = pts_tmp.flatten()
    
    return best_flat.reshape(-1, 2), best_obj


def _polish(pts, time_limit, rng):
    """Polish the solution with fine-tuned L-BFGS-B."""
    flat_pts = pts.flatten()
    k = 300.0
    
    t0 = time.time()
    
    def callback(xk):
        if time.time() - t0 > time_limit - 1.0:
            return True
        return False
    
    res = minimize(
        _objective_and_grad,
        flat_pts,
        args=(k,),
        jac=True,
        method='L-BFGS-B',
        options={
            'maxiter': 100,
            'ftol': 1e-12,
            'gtol': 1e-10,
            'maxls': 30,
            'callback': callback
        }
    )
    
    if res.x is not None:
        flat_pts = res.x
    
    pts_out = flat_pts.reshape(-1, 2)
    pts_out = np.clip(pts_out, -0.499, 0.499)
    r = _compute_radius(pts_out)
    return pts_out, r


def construct():
    """Construct 200 circle centres in the unit square maximizing common radius."""
    start_time = time.time()
    deadline = start_time + DEADLINE - 5.0  # 5 second safety margin

    rng = np.random.default_rng(42)

    best_pts = None
    best_r = -np.inf

    # Multi-start with different initializations
    n_starts = 6

    for start_idx in range(n_starts):
        if time.time() > deadline:
            break

        remaining = deadline - time.time()
        if remaining < 10.0:
            break

        # Alternate between hex and random initialization
        if start_idx % 2 == 0:
            pts = _generate_hex_initial(N, rng)
        else:
            pts = _generate_random_initial(N, rng)

        # L-BFGS-B optimization phase
        gd_time = min(15.0, remaining / 2.0)
        if gd_time <= 1.0:
            break

        pts, obj = _run_l_bfgs_b(pts, gd_time, rng)
        r = _compute_radius(pts)

        # Polish phase
        if time.time() < deadline - 3.0:
            polish_time = min(5.0, deadline - time.time() - 3.0)
            if polish_time > 2.0:
                pts, r = _polish(pts, polish_time, rng)

        if r > best_r:
            best_r = r
            best_pts = pts.copy()

    # If no valid result, return a simple grid
    if best_pts is None:
        m = int(np.ceil(np.sqrt(N)))
        ticks = (np.arange(m) + 0.5) / m - 0.5
        grid = np.array([(x, y) for y in ticks for x in ticks])
        return grid[:N]

    return best_pts
