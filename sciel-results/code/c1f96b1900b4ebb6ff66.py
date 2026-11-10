"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 50

# EVOLVE-BLOCK-START
import numpy as np
import time

def _compute_score(pts):
    """Compute the packing score: min(min pairwise dist/2, min (1 - |c_i|))."""
    if len(pts) < 2:
        return 0.0
    # Pairwise distances
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff**2, axis=-1))
    np.fill_diagonal(dists, np.inf)
    min_pair_dist = np.min(dists)
    # Distance from origin
    radii = np.sqrt(np.sum(pts**2, axis=1))
    min_container_dist = 1.0 - np.max(radii)
    return min(min_pair_dist / 2.0, min_container_dist)

def _hex_init(n, seed=42):
    """Initialize points on a hexagonal lattice inside the unit disk."""
    rng = np.random.RandomState(seed)
    # Generate hexagonal lattice points
    spacing = 0.15
    points = []
    for j in range(-8, 9):
        for i in range(-8, 9):
            x = (i + 0.5 * (j % 2)) * spacing
            y = j * spacing * np.sqrt(3.0) / 2.0
            if x*x + y*y < 0.95:
                points.append([x, y])
    points = np.array(points)
    if len(points) >= n:
        # Randomly select n points
        idx = rng.choice(len(points), size=n, replace=False)
        return points[idx]
    else:
        # If not enough, use random initialization
        angles = rng.uniform(0, 2*np.pi, n)
        radii = rng.uniform(0, 0.8, n)
        pts = np.column_stack([radii * np.cos(angles), radii * np.sin(angles)])
        return pts

def _repulsion_forces(pts, r_target, container_strength=5.0):
    """Compute repulsion forces between circles and attraction to center."""
    n = len(pts)
    forces = np.zeros_like(pts)
    
    # Pairwise repulsion
    for i in range(n):
        for j in range(i+1, n):
            diff = pts[i] - pts[j]
            dist = np.sqrt(np.sum(diff**2))
            if dist < 1e-10:
                # Perturb if coincident
                forces[i] += rng.uniform(-0.01, 0.01, 2)
                forces[j] += -forces[i]
                continue
            # Repulsion if closer than target distance
            target_dist = 2.0 * r_target
            if dist < target_dist:
                # Force proportional to overlap
                overlap = target_dist - dist
                force_mag = overlap * 10.0  # Spring constant
                direction = diff / dist
                forces[i] += direction * force_mag
                forces[j] -= direction * force_mag
    
    # Container constraint: push inward if too close to boundary
    for i in range(n):
        dist_origin = np.sqrt(np.sum(pts[i]**2))
        if dist_origin > 1.0 - r_target:
            # Push toward center
            if dist_origin > 1e-10:
                direction = -pts[i] / dist_origin
                excess = dist_origin - (1.0 - r_target)
                forces[i] += direction * excess * container_strength
    
    return forces

def _anneal(pts, r_target, max_iters=5000, lr_init=0.01, lr_decay=0.999, deadline=None):
    """Local search with annealing to maximize packing density."""
    best_pts = pts.copy()
    best_score = _compute_score(pts)
    lr = lr_init
    
    for it in range(max_iters):
        if deadline is not None and time.time() > deadline:
            break
        
        forces = _repulsion_forces(pts, r_target)
        # Add small random perturbation (annealing)
        noise = np.random.normal(0, lr * 0.1, size=pts.shape)
        pts_new = pts + lr * forces + noise
        
        # Check if valid (inside container)
        radii = np.sqrt(np.sum(pts_new**2, axis=1))
        if np.max(radii) > 1.0 - 1e-9:
            # Project back inside
            for i in range(len(pts_new)):
                if radii[i] > 1.0 - 1e-9:
                    scale = (1.0 - 1e-9) / radii[i]
                    pts_new[i] *= scale
        
        score_new = _compute_score(pts_new)
        if score_new > best_score:
            best_score = score_new
            best_pts = pts_new.copy()
        
        pts = pts_new
        lr *= lr_decay
        if lr < 1e-6:
            break
    
    return best_pts, best_score

def construct():
    start_time = time.time()
    deadline = start_time + 120  # 120s limit, leave margin
    
    # Try multiple initializations and target radii
    best_pts = None
    best_r = 0.0
    
    # Known best is ~0.1258, start from there and try to improve
    target_radii = [0.125, 0.126, 0.127, 0.128, 0.130]
    
    for r_target in target_radii:
        if time.time() > deadline - 10:
            break
        
        # Try multiple seeds
        for seed in [42, 123, 456, 789, 1000]:
            if time.time() > deadline - 5:
                break
            
            pts = _hex_init(N, seed)
            pts, score = _anneal(pts, r_target, max_iters=3000, deadline=deadline - 5)
            
            if score > best_r:
                best_r = score
                best_pts = pts.copy()
        
        # If we achieved the target, try to push higher
        if best_r >= r_target - 1e-6:
            continue
        else:
            break
    
    # Final polish with smaller steps
    if best_pts is not None and time.time() < deadline - 5:
        pts, score = _anneal(best_pts, best_r * 1.01, max_iters=2000, 
                            lr_init=0.005, lr_decay=0.9995, deadline=deadline - 2)
        if score > best_r:
            best_r = score
            best_pts = pts.copy()
    
    if best_pts is None:
        # Fallback: return hex lattice
        pts = _hex_init(N, 42)
        return pts
    
    return best_pts
# EVOLVE-BLOCK-END
