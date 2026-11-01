import numpy as np
import time
from scipy.optimize import minimize

N = 50
START_TIME = time.time()
TIME_LIMIT = 140.0

def _compute_radius(pts):
    """Compute the maximum radius for the given centers."""
    if np.any(np.abs(pts) > 0.5 + 1e-9):
        return -1e6
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff * diff, axis=2))
    np.fill_diagonal(dists, np.inf)
    min_pair_dist = np.min(dists)
    wall_dists = 0.5 - np.abs(pts)
    min_wall_dist = np.min(wall_dists)
    return min(min_pair_dist / 2.0, min_wall_dist)

def _softmin_radius_and_grad(pts, beta):
    """
    Compute smooth approximation of min radius and its gradient.
    Uses soft-min over pair distances/2 and wall distances.
    """
    pts = np.asarray(pts, dtype=np.float64)
    n = len(pts)
    
    # Pair distances
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff * diff, axis=2) + 1e-12)
    np.fill_diagonal(dists, np.inf)
    
    # Wall distances
    wall_dists = 0.5 - np.abs(pts)  # shape (N, 3)
    
    # Combine all constraints: pair distances / 2 and wall distances
    # We want softmin of all these values
    # Pair distances: N*(N-1)/2 values, each /2
    # Wall distances: N*3 values
    
    # Build array of all constraint values
    pair_vals = dists / 2.0  # (N, N)
    # Flatten upper triangle
    iu = np.triu_indices(n, k=1)
    pair_flat = pair_vals[iu]  # (M,)
    
    wall_flat = wall_dists.flatten()  # (3N,)
    
    all_vals = np.concatenate([pair_flat, wall_flat])
    
    # Soft min: -logsumexp(-x*beta)/beta
    # For stability, shift by min
    v_min = np.min(all_vals)
    exps = np.exp(-beta * (all_vals - v_min))
    logsumexp = np.log(np.sum(exps) + 1e-300)
    softmin_val = v_min - logsumexp / beta
    
    # Gradient w.r.t. pts
    # d(softmin)/d(all_vals) = -exps / sum(exps)
    weights = -exps / (np.sum(exps) + 1e-300)
    pair_w = weights[:len(pair_flat)]
    wall_w = weights[len(pair_flat):]
    
    # Gradient of pair distances
    grad_pts = np.zeros_like(pts)
    
    # For each pair (i,j), contribution to grad
    # d(dist/2)/d(pts_i) = (pts_i - pts_j) / dist
    # d(dist/2)/d(pts_j) = (pts_j - pts_i) / dist
    
    # Vectorized: for pair (i,j) with weight w
    # grad_pts[i] += w * (pts[i] - pts[j]) / dist[i,j]
    # grad_pts[j] += w * (pts[j] - pts[i]) / dist[i,j]
    
    # Build weight matrix
    wmat = np.zeros((n, n))
    wmat[iu] = pair_w
    wmat[iu[1], iu[0]] = pair_w
    
    # dists matrix (with inf on diagonal)
    # grad contribution from pair (i,j): wmat[i,j] * (pts[i]-pts[j]) / dists[i,j]
    # But we need to handle the inf on diagonal
    
    safe_dists = dists.copy()
    np.fill_diagonal(safe_dists, 1.0)
    
    # grad_pts[i] = sum_j wmat[i,j] * (pts[i]-pts[j]) / safe_dists[i,j]
    #             = sum_j wmat[i,j] * pts[i] / safe_dists[i,j] - sum_j wmat[i,j] * pts[j] / safe_dists[i,j]
    #             = pts[i] * sum_j wmat[i,j]/safe_dists[i,j] - sum_j wmat[i,j]*pts[j]/safe_dists[i,j]
    
    w_over_d = wmat / safe_dists
    np.fill_diagonal(w_over_d, 0.0)
    
    row_sum = np.sum(w_over_d, axis=1)  # (N,)
    # grad_pts[i] = row_sum[i] * pts[i] - sum_j w_over_d[i,j] * pts[j]
    grad_pts = row_sum[:, np.newaxis] * pts - w_over_d @ pts
    
    # Wall gradient
    # wall_dists = 0.5 - |pts|
    # d(wall_dists[k])/d(pts[k]) = -sign(pts[k])
    wall_grad = -np.sign(pts)  # (N, 3)
    # wall_w is (3N,), reshape to (N, 3)
    wall_w_reshaped = wall_w.reshape(n, 3)
    grad_pts += wall_w_reshaped * wall_grad
    
    return softmin_val, grad_pts

def _repulsion_anneal(pts, deadline, steps_per_iter=50, cooling_rate=0.97):
    """
    Simulated annealing with repulsive forces to spread points.
    """
    pts = pts.copy()
    current_r = _compute_radius(pts)
    best_r = current_r
    best_pts = pts.copy()
    
    step_size = 0.05
    temp = 0.02
    
    for _ in range(steps_per_iter):
        if time.time() > deadline:
            break
        
        # Generate random displacements
        deltas = np.random.normal(0, step_size, size=pts.shape)
        new_pts = pts + deltas
        new_pts = np.clip(new_pts, -0.499, 0.499)
        
        new_r = _compute_radius(new_pts)
        
        if new_r >= current_r:
            pts = new_pts
            current_r = new_r
            if current_r > best_r:
                best_r = current_r
                best_pts = pts.copy()
        else:
            # Accept with probability exp((new_r - current_r) / temp)
            prob = np.exp((new_r - current_r) / temp)
            if np.random.random() < prob:
                pts = new_pts
                current_r = new_r
        
        temp *= cooling_rate
        step_size *= 0.99
    
    return best_pts, best_r

def _lloyd_optimize(pts, deadline, max_iters=50):
    """
    Lloyd's algorithm for k-center / max-min distance.
    """
    pts = pts.copy()
    current_r = _compute_radius(pts)
    best_r = current_r
    best_pts = pts.copy()
    
    for it in range(max_iters):
        if time.time() > deadline:
            break
        
        # Compute Voronoi-like assignment: each point moves toward
        # the centroid of its "Voronoi cell" approximated by nearest neighbors
        # Simpler: move each point away from its closest neighbor
        
        # Find nearest neighbor for each point
        diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diff * diff, axis=2))
        np.fill_diagonal(dists, np.inf)
        
        nearest_idx = np.argmin(dists, axis=1)
        nearest_pts = pts[nearest_idx]
        
        # Move away from nearest neighbor
        move_dir = pts - nearest_pts
        move_norms = np.linalg.norm(move_dir, axis=1, keepdims=True)
        move_norms = np.maximum(move_norms, 1e-12)
        move_dir = move_dir / move_norms
        
        # Also consider wall proximity
        wall_dists = 0.5 - np.abs(pts)
        min_wall = np.min(wall_dists, axis=1, keepdims=True)
        
        # If close to wall, move inward
        wall_dir = np.sign(pts)
        wall_pressure = np.where(min_wall < 0.15, 1.0, 0.0)
        
        step = 0.01
        pts = pts + step * move_dir + step * 0.5 * wall_dir * wall_pressure
        pts = np.clip(pts, -0.499, 0.499)
        
        new_r = _compute_radius(pts)
        if new_r > best_r:
            best_r = new_r
            best_pts = pts.copy()
        
        current_r = new_r
    
    return best_pts, best_r

def _gradient_optimize(pts, deadline, beta_start=20.0, beta_end=200.0, max_iters=30):
    """
    Gradient-based optimization using smooth soft-min objective.
    """
    pts = pts.copy()
    current_r = _compute_radius(pts)
    best_r = current_r
    best_pts = pts.copy()
    
    n_betas = max_iters
    betas = np.linspace(beta_start, beta_end, n_betas)
    
    for beta in betas:
        if time.time() > deadline:
            break
        
        # Use L-BFGS-B on the negative soft-min
        def obj_and_grad(x):
            x_reshaped = x.reshape(-1, 3)
            val, grad = _softmin_radius_and_grad(x_reshaped, beta)
            return -val, -grad
        
        try:
            res = minimize(obj_and_grad, pts.flatten(), jac=True, method='L-BFGS-B',
                          options={'maxiter': 20, 'ftol': 1e-12, 'gtol': 1e-8})
            new_pts = res.x.reshape(-1, 3)
            new_pts = np.clip(new_pts, -0.499, 0.499)
            
            new_r = _compute_radius(new_pts)
            if new_r > best_r:
                best_r = new_r
                best_pts = new_pts.copy()
            pts = new_pts
        except Exception:
            pass
    
    return best_pts, best_r

def _metropolis_mc(pts, deadline, n_steps=5000):
    """
    Metropolis Monte Carlo to escape local minima.
    """
    pts = pts.copy()
    current_r = _compute_radius(pts)
    best_r = current_r
    best_pts = pts.copy()
    
    temp = 0.005
    step_size = 0.03
    
    for step in range(n_steps):
        if time.time() > deadline:
            break
        
        # Random displacement
        deltas = np.random.normal(0, step_size, size=pts.shape)
        new_pts = pts + deltas
        new_pts = np.clip(new_pts, -0.499, 0.499)
        
        new_r = _compute_radius(new_pts)
        
        if new_r >= current_r:
            pts = new_pts
            current_r = new_r
            if current_r > best_r:
                best_r = current_r
                best_pts = pts.copy()
        else:
            prob = np.exp((new_r - current_r) / temp)
            if np.random.random() < prob:
                pts = new_pts
                current_r = new_r
        
        # Anneal
        temp *= 0.9995
        step_size *= 0.9998
    
    return best_pts, best_r

def _seed_fcc():
    """Generate FCC lattice initialization."""
    fcc_pts = []
    cell_size = 1.0 / 3.0
    for i in range(3):
        for j in range(3):
            for k in range(3):
                x0, y0, z0 = i * cell_size, j * cell_size, k * cell_size
                fcc_pts.append([x0, y0, z0])
                fcc_pts.append([x0 + 0.5 * cell_size, y0 + 0.5 * cell_size, z0])
                fcc_pts.append([x0 + 0.5 * cell_size, y0, z0 + 0.5 * cell_size])
                fcc_pts.append([x0, y0 + 0.5 * cell_size, z0 + 0.5 * cell_size])
    fcc_pts = np.array(fcc_pts)
    fcc_pts -= 0.5
    mask = np.all(np.abs(fcc_pts) <= 0.5, axis=1)
    fcc_pts = fcc_pts[mask]
    
    if len(fcc_pts) > N:
        selected = []
        remaining = fcc_pts.copy()
        for _ in range(N):
            if not remaining.size:
                break
            pt = remaining[0]
            selected.append(pt)
            d = np.linalg.norm(remaining - pt, axis=1)
            remaining = remaining[d > 0.15]
        fcc_pts = np.array(selected)
    
    if len(fcc_pts) < N:
        n_pad = N - len(fcc_pts)
        pad = np.random.uniform(-0.45, 0.45, size=(n_pad, 3))
        fcc_pts = np.vstack([fcc_pts, pad])
        
    return fcc_pts

def _seed_grid():
    """Generate grid initialization."""
    m = 4
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid_pts = np.array([(x, y, z) for z in ticks for y in ticks for x in ticks])
    grid_init = grid_pts[:N].copy()
    grid_init += np.random.normal(0, 0.005, size=(N, 3))
    grid_init = np.clip(grid_init, -0.5, 0.5)
    return grid_init

def _seed_random():
    """Generate random initialization."""
    return np.random.uniform(-0.45, 0.45, size=(N, 3))

def _seed_bcc():
    """Generate BCC lattice initialization."""
    bcc_pts = []
    cell_size = 1.0 / 3.0
    for i in range(3):
        for j in range(3):
            for k in range(3):
                x0, y0, z0 = i * cell_size, j * cell_size, k * cell_size
                bcc_pts.append([x0, y0, z0])
                bcc_pts.append([x0 + 0.5 * cell_size, y0 + 0.5 * cell_size, z0 + 0.5 * cell_size])
    bcc_pts = np.array(bcc_pts)
    bcc_pts -= 0.5
    mask = np.all(np.abs(bcc_pts) <= 0.5, axis=1)
    bcc_pts = bcc_pts[mask]
    
    if len(bcc_pts) > N:
        selected = []
        remaining = bcc_pts.copy()
        for _ in range(N):
            if not remaining.size:
                break
            pt = remaining[0]
            selected.append(pt)
            d = np.linalg.norm(remaining - pt, axis=1)
            remaining = remaining[d > 0.15]
        bcc_pts = np.array(selected)
    
    if len(bcc_pts) < N:
        n_pad = N - len(bcc_pts)
        pad = np.random.uniform(-0.45, 0.45, size=(n_pad, 3))
        bcc_pts = np.vstack([bcc_pts, pad])
        
    return bcc_pts

def _seed_shell_core():
    """Generate shell-core initialization: some points on faces, some in core."""
    pts = []
    # Core: 2x2x2 grid
    core_ticks = np.array([-0.2, 0.2])
    for x in core_ticks:
        for y in core_ticks:
            for z in core_ticks:
                pts.append([x, y, z])
    
    # Shell: points on face centers and edges
    face_pts = [
        [0, 0, 0.45], [0, 0, -0.45],
        [0, 0.45, 0], [0, -0.45, 0],
        [0.45, 0, 0], [-0.45, 0, 0],
    ]
    pts.extend(face_pts)
    
    # Edge midpoints
    edge_pts = [
        [0.45, 0.45, 0], [0.45, -0.45, 0], [-0.45, 0.45, 0], [-0.45, -0.45, 0],
        [0.45, 0, 0.45], [0.45, 0, -0.45], [-0.45, 0, 0.45], [-0.45, 0, -0.45],
        [0, 0.45, 0.45], [0, -0.45, 0.45], [0, 0.45, -0.45], [0, -0.45, -0.45],
    ]
    pts.extend(edge_pts)
    
    # Corner regions
    corner_pts = [
        [0.4, 0.4, 0.4], [0.4, 0.4, -0.4], [0.4, -0.4, 0.4], [0.4, -0.4, -0.4],
        [-0.4, 0.4, 0.4], [-0.4, 0.4, -0.4], [-0.4, -0.4, 0.4], [-0.4, -0.4, -0.4],
    ]
    pts.extend(corner_pts)
    
    pts = np.array(pts)
    if len(pts) > N:
        pts = pts[:N]
    elif len(pts) < N:
        n_pad = N - len(pts)
        pad = np.random.uniform(-0.45, 0.45, size=(n_pad, 3))
        pts = np.vstack([pts, pad])
    
    pts += np.random.normal(0, 0.01, size=pts.shape)
    pts = np.clip(pts, -0.499, 0.499)
    return pts

def construct():
    best_pts = None
    best_r = -np.inf
    
    deadline = START_TIME + TIME_LIMIT
    
    # Generate diverse initializations
    seeds = [
        _seed_fcc(),
        _seed_grid(),
        _seed_random(),
        _seed_bcc(),
        _seed_shell_core(),
    ]
    
    # Add more random seeds if time allows
    while time.time() < deadline - 60.0:
        seeds.append(_seed_random())
    
    for init_pts in seeds:
        if time.time() > deadline - 30.0:
            break
        
        init_pts = np.clip(init_pts, -0.499, 0.499)
        
        # Phase 1: Repulsion annealing to spread points
        if time.time() < deadline - 25.0:
            pts, r = _repulsion_anneal(init_pts, deadline - 25.0, steps_per_iter=100)
            if r > best_r:
                best_r = r
                best_pts = pts.copy()
        else:
            pts, r = init_pts, _compute_radius(init_pts)
        
        # Phase 2: Lloyd optimization
        if time.time() < deadline - 20.0:
            pts, r = _lloyd_optimize(pts, deadline - 20.0, max_iters=30)
            if r > best_r:
                best_r = r
                best_pts = pts.copy()
        
        # Phase 3: Gradient-based optimization with increasing beta
        if time.time() < deadline - 15.0:
            pts, r = _gradient_optimize(pts, deadline - 15.0, beta_start=20.0, beta_end=150.0, max_iters=25)
            if r > best_r:
                best_r = r
                best_pts = pts.copy()
        
        # Phase 4: Metropolis MC to escape local minima
        if time.time() < deadline - 10.0:
            pts, r = _metropolis_mc(pts, deadline - 10.0, n_steps=3000)
            if r > best_r:
                best_r = r
                best_pts = pts.copy()
        
        # Phase 5: Final gradient polish
        if time.time() < deadline - 5.0:
            pts, r = _gradient_optimize(pts, deadline - 5.0, beta_start=100.0, beta_end=300.0, max_iters=20)
            if r > best_r:
                best_r = r
                best_pts = pts.copy()
    
    # If no valid points found, return grid
    if best_pts is None:
        m = int(np.ceil(N ** (1.0 / 3.0) - 1e-9))
        ticks = (np.arange(m) + 0.5) / m - 0.5
        grid = np.array([(x, y, z) for z in ticks for y in ticks for x in ticks])
        return grid[:N]
    
    # Final safety: ensure all points are strictly inside
    best_pts = np.clip(best_pts, -0.4999, 0.4999)
    
    return best_pts
