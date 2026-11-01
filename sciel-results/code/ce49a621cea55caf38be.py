"""Packing 200 equal circles in a unit square using model-and-optimize approach."""
N = 200

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.spatial.distance import pdist, squareform
from scipy.optimize import minimize


def _compute_radius(c):
    """Compute the largest common radius for given centres."""
    n = c.shape[0]
    if n == 0:
        return 0.0
    boundary_dists = 0.5 - np.abs(c)
    min_boundary = np.min(boundary_dists)
    if n < 2:
        return min_boundary
    dists = pdist(c, metric='euclidean')
    min_pair_dist = np.min(dists)
    return min(min_pair_dist / 2.0, min_boundary)


def _energy(params, temperature=0.02):
    """Smooth energy with threshold-restricted soft-min for speed."""
    c = params.reshape(-1, 2)
    c = np.clip(c, -0.4999, 0.4999)
    boundary_dists = 0.5 - np.abs(c)
    min_boundary = np.min(boundary_dists)
    if N < 2:
        return -min_boundary
    dists = pdist(c, metric='euclidean')
    min_dist = np.min(dists)
    T = max(temperature, 1e-8)
    
    # Restrict to distances near minimum for speed
    threshold = min_dist * 1.3 + 1e-6
    mask = dists < threshold
    if np.any(mask):
        d_sub = dists[mask]
        d_min_sub = np.min(d_sub)
        soft_min_pair = d_min_sub - T * np.log(np.sum(np.exp(-(d_sub - d_min_sub) / T)))
    else:
        soft_min_pair = min_dist
        
    b_min = np.min(boundary_dists)
    b_threshold = b_min * 1.3 + 1e-6
    b_mask = boundary_dists < b_threshold
    if np.any(b_mask):
        b_sub = boundary_dists[b_mask]
        b_min_sub = np.min(b_sub)
        soft_min_bound = b_min_sub - T * np.log(np.sum(np.exp(-(b_sub - b_min_sub) / T)))
    else:
        soft_min_bound = b_min
        
    r_pair = soft_min_pair / 2.0
    r_bound = soft_min_bound
    r_vals = np.array([r_pair, r_bound])
    m = np.min(r_vals)
    soft_min_combined = m - T * np.log(np.sum(np.exp(-(r_vals - m) / T)))
    
    return -soft_min_combined


def _energy_focused(params, temperature=0.01):
    """Sharper energy focusing on the single worst constraint."""
    c = params.reshape(-1, 2)
    c = np.clip(c, -0.4999, 0.4999)
    boundary_dists = 0.5 - np.abs(c)
    min_boundary = np.min(boundary_dists)
    dists = pdist(c, metric='euclidean')
    min_dist = np.min(dists)
    T = max(temperature, 1e-8)
    
    # Very tight threshold - focus on the worst few
    threshold = min_dist * 1.1 + 1e-6
    mask = dists < threshold
    if np.any(mask):
        d_sub = dists[mask]
        d_min_sub = np.min(d_sub)
        soft_min_pair = d_min_sub - T * np.log(np.sum(np.exp(-(d_sub - d_min_sub) / T)))
    else:
        soft_min_pair = min_dist
        
    b_min = np.min(boundary_dists)
    b_threshold = b_min * 1.1 + 1e-6
    b_mask = boundary_dists < b_threshold
    if np.any(b_mask):
        b_sub = boundary_dists[b_mask]
        b_min_sub = np.min(b_sub)
        soft_min_bound = b_min_sub - T * np.log(np.sum(np.exp(-(b_sub - b_min_sub) / T)))
    else:
        soft_min_bound = b_min
        
    r_pair = soft_min_pair / 2.0
    r_bound = soft_min_bound
    r_vals = np.array([r_pair, r_bound])
    m = np.min(r_vals)
    soft_min_combined = m - T * np.log(np.sum(np.exp(-(r_vals - m) / T)))
    
    return -soft_min_combined


def _initialize_hexagonal():
    """Initialize with hexagonal pattern, trying multiple spacings."""
    best_c = None
    best_r = 0
    for d in np.linspace(0.055, 0.075, 25):
        points = []
        row_spacing = d * np.sqrt(3) / 2
        x_offset = d / 2
        y = -0.5 + d / 2
        row = 0
        while y <= 0.5 - d / 2 + 1e-9:
            x = -0.5 + d / 2
            if row % 2 == 1:
                x += x_offset
            while x <= 0.5 - d / 2 + 1e-9:
                if -0.5 <= x <= 0.5 and -0.5 <= y <= 0.5:
                    points.append([x, y])
                x += d
            y += row_spacing
            row += 1
        if len(points) >= N:
            c = np.array(points[:N])
            r = _compute_radius(c)
            if r > best_r:
                best_r = r
                best_c = c.copy()
    if best_c is None:
        m = int(np.ceil(np.sqrt(N)))
        ticks = (np.arange(m) + 0.5) / m - 0.5
        grid = np.array([(x, y) for y in ticks for x in ticks])
        best_c = grid[:N].copy()
    return best_c


def _push_apart(c, n_iters=50):
    """Directly push closest pairs apart and pull boundary circles inward."""
    c = c.copy()
    n = c.shape[0]
    for _ in range(n_iters):
        # Find closest pairs and push them apart
        dists = pdist(c, metric='euclidean')
        min_idx = np.argmin(dists)
        # Convert flat index to pair indices
        # scipy pdist index mapping: k = i*(2*n - i - 1)//2 + j - i - 1
        # Inverse is complex, so we use squareform for simplicity if n is small enough
        # For N=200, squareform creates a 200x200 matrix which is 40k elements, very fast.
        dist_mat = squareform(dists)
        np.fill_diagonal(dist_mat, np.inf)
        flat_min = np.argmin(dist_mat)
        i, j = np.unravel_index(flat_min, (n, n))
        d = c[i] - c[j]
        dist = np.linalg.norm(d)
        if dist > 1e-10:
            d_hat = d / dist
            push = 0.005
            c[i] += d_hat * push
            c[j] -= d_hat * push
        # Push boundary circles inward
        boundary_dists = 0.5 - np.abs(c)
        for axis in range(2):
            bd = 0.5 - np.abs(c[:, axis])
            min_bd = np.min(bd)
            if min_bd < 0.02:
                idx = np.argmin(bd)
                c[idx, axis] -= np.sign(c[idx, axis]) * 0.003
        c = np.clip(c, -0.4999, 0.4999)
    return c


def _local_optimize(c0, max_time=30.0):
    """Multi-temperature local optimization with L-BFGS-B."""
    start_time = time.time()
    best_c = c0.copy()
    best_r = _compute_radius(c0)
    current_c = c0.copy()
    
    # Apply push_apart initially to improve starting point
    if time.time() < start_time + max_time - 5:
        current_c = _push_apart(current_c, n_iters=20)
        r_check = _compute_radius(current_c)
        if r_check > best_r:
            best_r = r_check
            best_c = current_c.copy()

    configs = [
        (0.03, _energy),
        (0.02, _energy),
        (0.01, _energy_focused),
        (0.005, _energy_focused),
        (0.002, _energy_focused),
    ]
    for T, energy_fn in configs:
        if time.time() > start_time + max_time - 2:
            break
        def obj(params, _T=T, _efn=energy_fn):
            return _efn(params, temperature=_T)
        try:
            result = minimize(
                obj,
                current_c.flatten(),
                method='L-BFGS-B',
                bounds=[(-0.4999, 0.4999)] * (2 * N),
                options={'maxiter': 300, 'ftol': 1e-13, 'gtol': 1e-9, 'maxls': 50}
            )
            c_new = np.clip(result.x.reshape(-1, 2), -0.4999, 0.4999)
            r_new = _compute_radius(c_new)
            if r_new > best_r:
                best_r = r_new
                best_c = c_new.copy()
            current_c = c_new
        except Exception:
            continue
    return best_c, best_r


def _polish(c0, max_time=20.0):
    """High-precision final polish with very tight tolerances."""
    start_time = time.time()
    best_c = c0.copy()
    best_r = _compute_radius(c0)
    current_c = c0.copy()
    
    configs = [
        (0.001, _energy_focused),
        (0.0005, _energy_focused),
        (0.0002, _energy_focused),
    ]
    for T, energy_fn in configs:
        if time.time() > start_time + max_time - 1:
            break
        def obj(params, _T=T, _efn=energy_fn):
            return _efn(params, temperature=_T)
        try:
            result = minimize(
                obj,
                current_c.flatten(),
                method='L-BFGS-B',
                bounds=[(-0.4999, 0.4999)] * (2 * N),
                options={'maxiter': 500, 'ftol': 1e-15, 'gtol': 1e-12, 'maxls': 100}
            )
            c_new = np.clip(result.x.reshape(-1, 2), -0.4999, 0.4999)
            r_new = _compute_radius(c_new)
            if r_new > best_r:
                best_r = r_new
                best_c = c_new.copy()
            current_c = c_new
        except Exception:
            continue
    return best_c, best_r


def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin: 10s before 150s limit
    
    best_radius = 0.0
    best_c = None
    
    # Step 1: Initialization
    c_hex = _initialize_hexagonal()
    r_hex = _compute_radius(c_hex)
    best_radius = r_hex
    best_c = c_hex.copy()
    
    # Step 2: Local optimization from best initial
    if time.time() < deadline - 20:
        max_opt_time = min(15.0, deadline - time.time() - 5)
        c_opt, r_opt = _local_optimize(best_c, max_time=max_opt_time)
        if r_opt > best_radius:
            best_radius = r_opt
            best_c = c_opt.copy()
    
    # Step 3: A few random restarts
    np.random.seed(42)
    n_restarts = 3
    
    for trial in range(n_restarts):
        if time.time() > deadline - 15:
            break
        
        # Jittered grid initialization
        m = int(np.ceil(np.sqrt(N)))
        grid_x, grid_y = np.meshgrid(
            np.linspace(-0.45, 0.45, m),
            np.linspace(-0.45, 0.45, m)
        )
        grid = np.column_stack([grid_x.ravel(), grid_y.ravel()])
        np.random.shuffle(grid)
        c_init = grid[:N].copy()
        
        # Add random perturbation
        c_init += np.random.normal(0, 0.01, size=(N, 2))
        c_init = np.clip(c_init, -0.4999, 0.4999)
        
        r_init = _compute_radius(c_init)
        if r_init > best_radius:
            best_radius = r_init
            best_c = c_init.copy()
        
        # Local optimize
        remaining = deadline - time.time() - 5
        if remaining > 5:
            max_opt_time = min(10.0, remaining)
            c_opt, r_opt = _local_optimize(c_init, max_time=max_opt_time)
            if r_opt > best_radius:
                best_radius = r_opt
                best_c = c_opt.copy()
    
    # Step 4: Final high-precision polish
    if time.time() < deadline - 5:
        remaining = deadline - time.time() - 1
        if remaining > 3:
            c_pol, r_pol = _polish(best_c, max_time=min(10.0, remaining))
            if r_pol > best_radius:
                best_radius = r_pol
                best_c = c_pol.copy()
    
    # Final clip to ensure validity
    best_c = np.clip(best_c, -0.5 + 1e-12, 0.5 - 1e-12)
    
    # Verify the result is valid
    if best_c.shape != (N, 2) or not np.all(np.isfinite(best_c)):
        best_c = _initialize_hexagonal()
        best_c = np.clip(best_c, -0.5 + 1e-12, 0.5 - 1e-12)
    
    return best_c
# EVOLVE-BLOCK-END
