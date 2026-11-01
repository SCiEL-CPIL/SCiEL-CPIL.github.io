import numpy as np
import time

N = 300
SECTOR_ANGLE = np.pi / 3.0
SECTOR_ANGLES = np.arange(6) * SECTOR_ANGLE

# Precompute rotation matrices for 6-fold symmetry
_ROT_MATS = np.array([
    [[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]]
    for a in SECTOR_ANGLES
])

def _hex_grid_points(max_radius=1.0, spacing=0.12):
    """Generate hexagonal lattice points within a radius."""
    pts = []
    k_max = int(max_radius / spacing) + 1
    for j in range(-k_max, k_max + 1):
        for i in range(-k_max, k_max + 1):
            x = (i + 0.5 * (j % 2)) * spacing
            y = j * spacing * np.sqrt(3.0) / 2.0
            if x * x + y * y <= max_radius * max_radius:
                pts.append((x, y))
    return np.array(pts) if pts else np.array([[0.0, 0.0]])

def _sector_angle_sort(points):
    """Sort points by angle within a sector."""
    angles = np.arctan2(points[:, 1], points[:, 0])
    # Map to [0, pi/3)
    angles_mod = np.mod(angles, SECTOR_ANGLE)
    return np.argsort(angles_mod)

def _expand_sector(sector_pts, n_total):
    """Expand sector points to full 6-fold symmetric configuration."""
    n_sector = len(sector_pts)
    all_pts = []
    for k in range(6):
        R = _ROT_MATS[k]
        rotated = sector_pts @ R.T
        all_pts.append(rotated)
    all_pts = np.vstack(all_pts)
    # Remove duplicates (points on sector boundaries)
    if len(all_pts) > n_total:
        # Keep first n_total
        all_pts = all_pts[:n_total]
    return all_pts

def _compute_score(points):
    """Compute packing score: min of all pair distances/2 and boundary distances."""
    diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
    d = np.sqrt(np.sum(diff * diff, axis=2))
    np.fill_diagonal(d, np.inf)
    min_pair = np.min(d) / 2.0
    r = np.sqrt(np.sum(points * points, axis=1))
    min_boundary = np.min(1.0 - r)
    return min(min_pair, min_boundary)

def _init_sector_hex(n_total):
    """Initialize sector points using hexagonal lattice."""
    # Estimate number of points in sector
    n_sector_target = n_total // 6
    # Some points may be on boundaries, so aim slightly higher
    n_sector = n_sector_target + 2
    
    # Estimate spacing for hexagonal packing
    # Area per point ~ pi * r^2, total area ~ pi, so r ~ 1/sqrt(n)
    # Hexagonal spacing s ~ r * sqrt(2*sqrt(3)/pi) ~ r * 1.075
    r_est = 1.0 / np.sqrt(n_total / np.pi)
    s_est = r_est * 1.08
    
    # Generate hex lattice points
    hex_pts = _hex_grid_points(max_radius=1.0, spacing=s_est)
    
    # Keep only points in first sector [0, pi/3)
    angles = np.arctan2(hex_pts[:, 1], hex_pts[:, 0])
    angles_mod = np.mod(angles, SECTOR_ANGLE)
    
    # Include points very close to boundaries (within tolerance)
    sector_mask = (angles_mod < SECTOR_ANGLE - 1e-9) | (angles_mod > SECTOR_ANGLE - 0.01)
    sector_mask = angles_mod < SECTOR_ANGLE - 1e-6  # Strictly inside
    
    sector_pts = hex_pts[sector_mask]
    
    # Sort by angle
    if len(sector_pts) > 0:
        sort_idx = _sector_angle_sort(sector_pts)
        sector_pts = sector_pts[sort_idx]
    
    # Select best n_sector points
    if len(sector_pts) >= n_sector:
        # Prefer points closer to origin for better packing
        r = np.sqrt(np.sum(sector_pts * sector_pts, axis=1))
        sort_by_r = np.argsort(r)
        sector_pts = sector_pts[sort_by_r[:n_sector]]
        # Re-sort by angle
        sort_idx = _sector_angle_sort(sector_pts)
        sector_pts = sector_pts[sort_idx]
    else:
        # Pad with random points if needed
        while len(sector_pts) < n_sector:
            a = np.random.uniform(0, SECTOR_ANGLE)
            r = np.random.uniform(0, 1.0)
            sector_pts = np.vstack([sector_pts, [[r * np.cos(a), r * np.sin(a)]]])
        sort_idx = _sector_angle_sort(sector_pts)
        sector_pts = sector_pts[sort_idx]
    
    return sector_pts[:n_sector]

def _init_sector_ring(n_total):
    """Initialize sector points using concentric rings."""
    n_sector = n_total // 6
    
    # Create rings
    points = [[0.0, 0.0]]
    remaining = n_sector - 1
    ring_radius = 0.15
    
    while remaining > 0:
        # Number of points on this ring in sector
        circumference = 2 * np.pi * ring_radius
        # Approximate number of points that fit
        n_ring = max(1, int(circumference * SECTOR_ANGLE / (2 * ring_radius * 0.55)))
        n_ring = min(n_ring, remaining)
        
        for i in range(n_ring):
            angle = (i + 0.5) * SECTOR_ANGLE / n_ring
            x = ring_radius * np.cos(angle)
            y = ring_radius * np.sin(angle)
            points.append([x, y])
        
        remaining -= n_ring
        ring_radius += 0.15
    
    points = np.array(points[:n_sector])
    if len(points) < n_sector:
        # Pad
        while len(points) < n_sector:
            a = np.random.uniform(0, SECTOR_ANGLE)
            r = np.random.uniform(0, 1.0)
            points = np.vstack([points, [[r * np.cos(a), r * np.sin(a)]]])
    
    sort_idx = _sector_angle_sort(points)
    return points[sort_idx]

def _sector_objective(sector_pts, n_total):
    """Compute objective for sector: minimize max violation."""
    # Expand to full configuration
    full_pts = _expand_sector(sector_pts, n_total)
    
    if len(full_pts) < n_total:
        return 1.0
    
    # Compute all pair distances
    diff = full_pts[:, np.newaxis, :] - full_pts[np.newaxis, :, :]
    d = np.sqrt(np.sum(diff * diff, axis=2))
    np.fill_diagonal(d, np.inf)
    
    # Boundary
    r = np.sqrt(np.sum(full_pts * full_pts, axis=1))
    boundary_viol = np.maximum(0, r - 1.0)
    
    # Pair violations: we want d[i,j] >= 2*r for all pairs
    # Since we're maximizing r, let's compute the min distance
    min_pair_d = np.min(d) / 2.0
    min_boundary = np.min(1.0 - r)
    
    score = min(min_pair_d, min_boundary)
    return -score  # Negative because we minimize

def _optimize_sector(sector_pts, n_total, time_budget=30.0):
    """Optimize sector points using L-BFGS-B."""
    from scipy.optimize import minimize
    
    start = time.time()
    
    def obj_and_grad(x):
        sector = x.reshape(-1, 2)
        full_pts = _expand_sector(sector, n_total)
        
        if len(full_pts) < n_total:
            return 1.0, np.zeros_like(x)
        
        # Compute distances
        diff = full_pts[:, np.newaxis, :] - full_pts[np.newaxis, :, :]
        d = np.sqrt(np.sum(diff * diff, axis=2))
        np.fill_diagonal(d, np.inf)
        
        r = np.sqrt(np.sum(full_pts * full_pts, axis=1))
        min_pair_d = np.min(d) / 2.0
        min_boundary = np.min(1.0 - r)
        score = min(min_pair_d, min_boundary)
        
        # Gradient via finite differences (expensive but simple)
        grad = np.zeros_like(x)
        eps = 1e-6
        for i in range(len(x)):
            x_plus = x.copy()
            x_plus[i] += eps
            val_plus = -_sector_objective(x_plus.reshape(-1, 2), n_total)
            grad[i] = (val_plus + score) / eps
        
        return -score, grad
    
    try:
        res = minimize(
            obj_and_grad,
            sector_pts.flatten(),
            method='L-BFGS-B',
            jac=True,
            options={'maxiter': 200, 'gtol': 1e-8}
        )
        if time.time() - start > time_budget:
            return sector_pts
        if res.success and np.all(np.isfinite(res.x)):
            return res.x.reshape(-1, 2)
    except Exception:
        pass
    
    return sector_pts

def _local_polish_full(points, time_budget=20.0):
    """Local polish on full configuration."""
    start = time.time()
    best = points.copy()
    best_score = _compute_score(best)
    
    step = 0.003
    no_improve = 0
    
    while time.time() - start < time_budget:
        # Find worst constraint
        diff = best[:, np.newaxis, :] - best[np.newaxis, :, :]
        d = np.sqrt(np.sum(diff * diff, axis=2))
        np.fill_diagonal(d, np.inf)
        min_pair_d = np.min(d) / 2.0
        
        r = np.sqrt(np.sum(best * best, axis=1))
        min_boundary = np.min(1.0 - r)
        
        score = min(min_pair_d, min_boundary)
        
        if score > best_score + 1e-14:
            best_score = score
            no_improve = 0
        else:
            no_improve += 1
            if no_improve > 50:
                step *= 0.8
                if step < 1e-8:
                    break
                no_improve = 0
        
        # Try to improve worst constraint
        if min_pair_d < min_boundary:
            idx = np.unravel_index(np.argmin(d), d.shape)
            i, j = idx
            direction = best[i] - best[j]
            norm = np.linalg.norm(direction)
            if norm > 1e-12:
                direction /= norm
            else:
                direction = np.array([1.0, 0.0])
            
            improved = False
            for ss in [step, step * 0.5]:
                cand = best.copy()
                cand[i] += direction * ss
                cand[j] -= direction * ss
                if np.all(np.sqrt(np.sum(cand * cand, axis=1)) < 1.0):
                    new_score = _compute_score(cand)
                    if new_score > best_score:
                        best = cand
                        best_score = new_score
                        improved = True
                        break
        else:
            i = np.argmin(1.0 - r)
            direction = -best[i]
            norm = np.linalg.norm(direction)
            if norm > 1e-12:
                direction /= norm
            
            improved = False
            for ss in [step, step * 0.5]:
                cand = best.copy()
                cand[i] += direction * ss
                if np.all(np.sqrt(np.sum(cand * cand, axis=1)) < 1.0):
                    new_score = _compute_score(cand)
                    if new_score > best_score:
                        best = cand
                        best_score = new_score
                        improved = True
                        break
        
        if not improved:
            step *= 0.9
    
    return best, best_score

def construct():
    start_time = time.time()
    time_limit = 140.0  # Safety margin
    
    best_full = None
    best_score = -np.inf
    
    # Try multiple sector initializations
    n_sector = N // 6
    
    inits = [
        _init_sector_hex(N),
        _init_sector_ring(N)
    ]
    
    for init in inits:
        full = _expand_sector(init, N)
        if len(full) == N:
            score = _compute_score(full)
            if score > best_score:
                best_score = score
                best_full = full.copy()
                best_sector = init.copy()
    
    if best_full is None:
        # Fallback: random initialization
        best_full = np.random.uniform(-0.9, 0.9, (N, 2))
        mask = np.sqrt(np.sum(best_full * best_full, axis=1)) <= 1.0
        while np.sum(mask) < N:
            new_pts = np.random.uniform(-0.9, 0.9, (N - np.sum(mask), 2))
            new_mask = np.sqrt(np.sum(new_pts * new_pts, axis=1)) <= 1.0
            best_full = np.vstack([best_full[mask], new_pts[new_mask]])
            mask = np.ones(len(best_full), dtype=bool)
        best_full = best_full[:N]
        best_score = _compute_score(best_full)
        # Extract sector
        angles = np.arctan2(best_full[:, 1], best_full[:, 0])
        angles_mod = np.mod(angles, SECTOR_ANGLE)
        sector_mask = angles_mod < SECTOR_ANGLE - 1e-6
        best_sector = best_full[sector_mask]
        sort_idx = _sector_angle_sort(best_sector)
        best_sector = best_sector[sort_idx]
    
    # Optimize sector
    elapsed = time.time() - start_time
    if elapsed < time_limit - 30:
        time_budget = min(40, time_limit - elapsed - 20)
        optimized_sector = _optimize_sector(best_sector, N, time_budget)
        full_opt = _expand_sector(optimized_sector, N)
        if len(full_opt) == N:
            score = _compute_score(full_opt)
            if score > best_score:
                best_score = score
                best_full = full_opt
                best_sector = optimized_sector
    
    # Local polish on full configuration
    elapsed = time.time() - start_time
    if elapsed < time_limit - 10:
        time_budget = min(25, time_limit - elapsed - 5)
        if time_budget > 5:
            try:
                best_full, best_score = _local_polish_full(best_full, time_budget)
            except Exception:
                pass
    
    # Ensure valid output
    if best_full is None or len(best_full) != N:
        # Last resort
        best_full = np.zeros((N, 2))
        for i in range(N):
            a = 2 * np.pi * i / N
            r = 0.5
            best_full[i] = [r * np.cos(a), r * np.sin(a)]
    
    # Final validation
    if not np.all(np.isfinite(best_full)):
        best_full = np.zeros((N, 2))
    
    return best_full
