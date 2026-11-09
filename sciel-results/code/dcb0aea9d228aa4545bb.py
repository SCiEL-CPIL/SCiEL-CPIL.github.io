# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.spatial import ConvexHull
from scipy.optimize import minimize

def _all_triangle_indices(n):
    """Precompute all C(n,3) triangle index triples."""
    idx = []
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                idx.append((i, j, k))
    return np.array(idx, dtype=int)

def _triangle_areas(pts, tri_idx):
    """Compute all triangle areas vectorized."""
    a = pts[tri_idx[:, 0]]
    b = pts[tri_idx[:, 1]]
    c = pts[tri_idx[:, 2]]
    area = 0.5 * np.abs(
        a[:, 0] * (b[:, 1] - c[:, 1]) +
        b[:, 0] * (c[:, 1] - a[:, 1]) +
        c[:, 0] * (a[:, 1] - b[:, 1])
    )
    return area

def _signed_triangle_areas(pts, tri_idx):
    """Compute signed triangle areas (positive for CCW)."""
    a = pts[tri_idx[:, 0]]
    b = pts[tri_idx[:, 1]]
    c = pts[tri_idx[:, 2]]
    area = 0.5 * (
        a[:, 0] * (b[:, 1] - c[:, 1]) +
        b[:, 0] * (c[:, 1] - a[:, 1]) +
        c[:, 0] * (a[:, 1] - b[:, 1])
    )
    return area

def _hull_area(pts):
    """Compute convex hull area."""
    try:
        hull = ConvexHull(pts)
        return hull.volume
    except Exception:
        return None

def _score(pts):
    """Compute the score: min_triangle_area / hull_area."""
    try:
        n = len(pts)
        tri_idx = _all_triangle_indices(n)
        areas = _triangle_areas(pts, tri_idx)
        min_area = areas.min()
        hull_area = _hull_area(pts)
        if hull_area is None or hull_area <= 0:
            return 0.0
        return min_area / hull_area
    except Exception:
        return 0.0

def _soft_min(areas, beta):
    """Compute soft minimum using log-sum-exp."""
    shifted = -beta * areas
    max_shifted = shifted.max()
    log_sum = max_shifted + np.log(np.exp(shifted - max_shifted).sum())
    return -log_sum / beta

def _soft_min_grad(areas, beta):
    """Compute gradient of soft minimum w.r.t. areas."""
    shifted = -beta * areas
    max_shifted = shifted.max()
    exp_shifted = np.exp(shifted - max_shifted)
    weights = exp_shifted / exp_shifted.sum()
    return -weights

def _regular_polygon(n, r=1.0):
    """Generate n points on a regular polygon."""
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
    return r * np.stack([np.cos(angles), np.sin(angles)], axis=1)

def _fibonacci_sphere(n, r=1.0):
    """Generate n points using Fibonacci spiral on a disk."""
    pts = np.zeros((n, 2))
    golden_angle = np.pi * (3 - np.sqrt(5))
    for i in range(n):
        angle = i * golden_angle
        radius = r * np.sqrt((i + 0.5) / n)
        pts[i, 0] = radius * np.cos(angle)
        pts[i, 1] = radius * np.sin(angle)
    return pts

def _concentric_rings(n):
    """Generate points on concentric rings."""
    pts = []
    # Inner ring: 3 points
    for k in range(3):
        angle = 2 * np.pi * k / 3
        pts.append([0.35 * np.cos(angle), 0.35 * np.sin(angle)])
    # Middle ring: 4 points
    for k in range(4):
        angle = 2 * np.pi * k / 4 + np.pi / 4
        pts.append([0.65 * np.cos(angle), 0.65 * np.sin(angle)])
    # Outer ring: 6 points
    for k in range(6):
        angle = 2 * np.pi * k / 6 + np.pi / 6
        pts.append([1.0 * np.cos(angle), 1.0 * np.sin(angle)])
    return np.array(pts)

def _halton_sequence(n, dims=2):
    """Generate n Halton sequence points in [0,1]^dims."""
    def halton(index, base):
        f = 1.0
        r = 0.0
        while index > 0:
            f /= base
            r += f * (index % base)
            index //= base
        return r
    
    points = np.zeros((n, dims))
    for i in range(n):
        for j in range(dims):
            points[i, j] = halton(i + 1, j + 2)
    # Center and scale
    points = (points - 0.5) * 2
    return points

def _random_points(n, seed):
    """Generate random points in unit disk."""
    rng = np.random.default_rng(seed)
    angles = rng.uniform(0, 2 * np.pi, n)
    radii = np.sqrt(rng.uniform(0.1, 1.0, n))
    return np.stack([radii * np.cos(angles), radii * np.sin(angles)], axis=1)

def _perturbed_polygon(n, seed, perturbation=0.1):
    """Generate perturbed regular polygon."""
    rng = np.random.default_rng(seed)
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False) + rng.uniform(-perturbation, perturbation, n)
    radii = 1.0 + rng.uniform(-perturbation, perturbation, n)
    return np.stack([radii * np.cos(angles), radii * np.sin(angles)], axis=1)

def _optimize_lbfgsb(pts, tri_idx, n, time_limit=60.0, betas=None):
    """
    Optimize using L-BFGS-B with soft-min surrogate.
    """
    if betas is None:
        betas = [20.0, 40.0, 80.0, 160.0]
    
    start_time = time.time()
    x0 = pts.ravel()
    
    best_pts = pts.copy()
    best_score = _score(pts)
    
    for beta in betas:
        if time.time() - start_time > time_limit:
            break
        
        def obj(x):
            pts_x = x.reshape(n, 2)
            areas = _triangle_areas(pts_x, tri_idx)
            soft_min_val = _soft_min(areas, beta)
            hull_area = _hull_area(pts_x)
            if hull_area is None or hull_area <= 0:
                return 1e6
            # We want to maximize min_area / hull_area
            # Equivalent to minimizing -min_area + penalty for hull_area
            # Use: -soft_min + lambda * (hull_area - 1)^2
            hull_penalty = 0.05 * (hull_area - 1.0) ** 2
            return -soft_min_val + hull_penalty
        
        # Use numerical gradient
        result = minimize(
            obj, x0, method='L-BFGS-B',
            options={'maxiter': 500, 'maxfun': 3000, 'ftol': 1e-14, 'gtol': 1e-12}
        )
        
        if result.fun < 1e6:
            new_pts = result.x.reshape(n, 2)
            if np.all(np.isfinite(new_pts)):
                new_score = _score(new_pts)
                if new_score > best_score:
                    best_score = new_score
                    best_pts = new_pts.copy()
                    x0 = new_pts.ravel()
    
    return best_pts, best_score

def _subgradient_ascent(pts, tri_idx, n, time_limit=60.0, lr_init=0.01):
    """
    Subgradient ascent on the minimum triangle area with adaptive learning rate.
    """
    start_time = time.time()
    best_pts = pts.copy()
    best_score = _score(pts)
    
    lr = lr_init
    decay = 0.97
    stall_count = 0
    max_stall = 20
    
    while time.time() - start_time < time_limit:
        areas = _triangle_areas(best_pts, tri_idx)
        min_area = areas.min()
        # Use a tolerance based on the scale of areas
        tol = 1e-8 * max(1.0, np.median(areas))
        min_indices = np.where(np.abs(areas - min_area) < tol)[0]
        
        # Subgradient: average gradient of the minimum triangles
        signed_areas = _signed_triangle_areas(best_pts, tri_idx)
        signs = np.sign(signed_areas[min_indices])
        signs[signs == 0] = 1.0
        
        grad = np.zeros_like(best_pts)
        for idx in min_indices:
            i, j, k = tri_idx[idx]
            a = best_pts[i]
            b = best_pts[j]
            c = best_pts[k]
            s = signs[np.where(min_indices == idx)[0][0]]
            
            da = 0.5 * s * np.array([-(b[1] - c[1]), b[0] - c[0]])
            db = 0.5 * s * np.array([-(c[1] - a[1]), c[0] - a[0]])
            dc = 0.5 * s * np.array([-(a[1] - b[1]), a[0] - b[0]])
            
            grad[i] += da
            grad[j] += db
            grad[k] += dc
        
        grad /= len(min_indices)
        
        # Normalize gradient
        grad_norm = np.linalg.norm(grad)
        if grad_norm > 1e-12:
            grad /= grad_norm
        else:
            break
        
        # Update
        new_pts = best_pts + lr * grad
        
        # Project: ensure no points are collinear or too close
        new_score = _score(new_pts)
        if new_score > best_score:
            best_score = new_score
            best_pts = new_pts.copy()
            stall_count = 0
        else:
            stall_count += 1
            if stall_count >= max_stall:
                # Reduce learning rate more aggressively
                lr *= 0.5
                stall_count = 0
                if lr < 1e-6:
                    break
        
        lr *= decay
    
    return best_pts, best_score

def _local_refine(pts, tri_idx, n, time_limit=60.0):
    """
    Local refinement using coordinate descent with adaptive step sizes.
    """
    start_time = time.time()
    best_pts = pts.copy()
    best_score = _score(pts)
    
    # Normalize hull area to 1
    hull_area = _hull_area(best_pts)
    if hull_area and hull_area > 0:
        best_pts = best_pts / np.sqrt(hull_area)
        best_score = _score(best_pts)
    
    step_sizes = [0.02, 0.01, 0.005, 0.002, 0.001, 0.0005]
    
    for ss in step_sizes:
        if time.time() - start_time > time_limit:
            break
        
        improved = True
        while improved and time.time() - start_time < time_limit:
            improved = False
            for i in range(n):
                if time.time() - start_time > time_limit:
                    break
                # Try 12 directions
                angles = np.linspace(0, 2 * np.pi, 12, endpoint=False)
                for ang in angles:
                    dx = ss * np.cos(ang)
                    dy = ss * np.sin(ang)
                    trial = best_pts.copy()
                    trial[i, 0] += dx
                    trial[i, 1] += dy
                    trial_score = _score(trial)
                    if trial_score > best_score:
                        best_score = trial_score
                        best_pts = trial.copy()
                        improved = True
    
    return best_pts, best_score

def heilbronn_convex13() -> np.ndarray:
    """
    Construct an arrangement of 13 points to maximize the minimum triangle area
    relative to the convex hull area.
    """
    n = 13
    tri_idx = _all_triangle_indices(n)
    
    deadline = time.time() + 620  # Safety margin before 630s kill
    
    # Generate diverse starting configurations
    starts = []
    
    # 1. Regular polygon
    starts.append(_regular_polygon(n, r=1.0))
    
    # 2. Fibonacci disk
    starts.append(_fibonacci_sphere(n, r=1.0))
    
    # 3. Concentric rings
    starts.append(_concentric_rings(n))
    
    # 4. Halton sequence
    starts.append(_halton_sequence(n))
    
    # 5. Random starts with different seeds
    for seed in range(20):
        starts.append(_random_points(n, seed))
    
    # 6. Perturbed polygons
    for seed in range(10):
        starts.append(_perturbed_polygon(n, seed, perturbation=0.15))
    
    # 7. More structured patterns
    # Two concentric hexagons with offset
    for offset in [0.0, 0.1, 0.2]:
        pts = []
        for k in range(7):
            angle = 2 * np.pi * k / 7 + offset
            pts.append([0.5 * np.cos(angle), 0.5 * np.sin(angle)])
        for k in range(6):
            angle = 2 * np.pi * k / 6 + offset + np.pi / 6
            pts.append([1.0 * np.cos(angle), 1.0 * np.sin(angle)])
        starts.append(np.array(pts))
    
    # 8. Triangular lattice subset
    # Create a triangular lattice and pick 13 points
    lattice_pts = []
    for i in range(-3, 4):
        for j in range(-3, 4):
            x = i + 0.5 * j
            y = np.sqrt(3) / 2 * j
            lattice_pts.append([x, y])
    lattice_pts = np.array(lattice_pts)
    # Center the lattice
    lattice_pts -= lattice_pts.mean(axis=0)
    # Select 13 points that form a good configuration
    # Use a greedy approach to maximize minimum distance
    selected = [0]
    for _ in range(12):
        best_idx = -1
        best_min_dist = -1
        for idx in range(len(lattice_pts)):
            if idx in selected:
                continue
            min_dist = min(np.linalg.norm(lattice_pts[idx] - lattice_pts[s]) for s in selected)
            if min_dist > best_min_dist:
                best_min_dist = min_dist
                best_idx = idx
        if best_idx >= 0:
            selected.append(best_idx)
    if len(selected) == 13:
        starts.append(lattice_pts[selected])
    
    # 9. Hexagonal close packing subset
    hex_pts = []
    for ring in range(3):
        if ring == 0:
            hex_pts.append([0, 0])
        else:
            for k in range(6 * ring):
                angle = 2 * np.pi * k / (6 * ring)
                hex_pts.append([ring * np.cos(angle), ring * np.sin(angle)])
    hex_pts = np.array(hex_pts)
    if len(hex_pts) >= 13:
        starts.append(hex_pts[:13])
    
    # 10. Grid-like patterns
    for spacing in [0.4, 0.5, 0.6]:
        grid_pts = []
        for i in range(-2, 3):
            for j in range(-2, 3):
                grid_pts.append([i * spacing, j * spacing])
        grid_pts = np.array(grid_pts)
        if len(grid_pts) >= 13:
            starts.append(grid_pts[:13])
    
    best_pts = None
    best_score = -1.0
    
    # Optimize from each start
    for i, start_pts in enumerate(starts):
        if time.time() > deadline - 20:
            break
        
        # Scale start to have hull area ~1 for consistency
        hull_area = _hull_area(start_pts)
        if hull_area and hull_area > 0:
            start_pts = start_pts / np.sqrt(hull_area)
        
        # Time allocation
        remaining_starts = len(starts) - i
        time_per_opt = min(50.0, (deadline - time.time() - 20) / max(1, remaining_starts))
        if time_per_opt < 5.0:
            break
        
        # Phase 1: L-BFGS-B optimization
        pts, score = _optimize_lbfgsb(start_pts, tri_idx, n, time_limit=time_per_opt * 0.35)
        
        # Phase 2: Subgradient ascent
        if time.time() < deadline - 20:
            pts, score = _subgradient_ascent(pts, tri_idx, n, time_limit=time_per_opt * 0.35)
        
        # Phase 3: Local refinement
        if time.time() < deadline - 20:
            pts, score = _local_refine(pts, tri_idx, n, time_limit=time_per_opt * 0.30)
        
        if score > best_score:
            best_score = score
            best_pts = pts.copy()
    
    # Final polish on the best solution
    if best_pts is not None and time.time() < deadline - 30:
        time_left = deadline - time.time() - 15
        best_pts, best_score = _optimize_lbfgsb(best_pts, tri_idx, n, time_limit=time_left * 0.35, betas=[40.0, 80.0, 160.0, 320.0])
        if time.time() < deadline - 15:
            best_pts, best_score = _subgradient_ascent(best_pts, tri_idx, n, time_limit=time_left * 0.35)
        if time.time() < deadline - 15:
            best_pts, best_score = _local_refine(best_pts, tri_idx, n, time_limit=time_left * 0.30)
        
        # Extra polish: multiple rounds of local refinement
        if time.time() < deadline - 10:
            for _ in range(2):
                if time.time() < deadline - 10:
                    best_pts, best_score = _local_refine(best_pts, tri_idx, n, time_limit=(deadline - time.time() - 10) * 0.4)
    
    # Normalize: scale so hull area is exactly 1
    if best_pts is None:
        best_pts = _regular_polygon(n, r=1.0)
    
    hull_area = _hull_area(best_pts)
    if hull_area and hull_area > 0:
        best_pts = best_pts / np.sqrt(hull_area)
    
    # Final validation
    if not np.all(np.isfinite(best_pts)):
        best_pts = _regular_polygon(n, r=1.0)
    
    return best_pts

# EVOLVE-BLOCK-END
