"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

N = 200
_DEADLINE = 120.0
_START = time.time()


def _remaining():
    return _DEADLINE - (time.time() - _START)


def _score(pts):
    """Compute the packing score for given centers in [-0.5, 0.5]^2."""
    if pts.shape[0] != N:
        return 0.0
    if not np.all(np.isfinite(pts)):
        return 0.0
    # Check bounds
    if np.any(pts < -0.5 - 1e-12) or np.any(pts > 0.5 + 1e-12):
        return 0.0
    # Wall distance
    wall = np.min(0.5 - np.abs(pts), axis=1)
    min_wall = np.min(wall)
    # Pairwise distances
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dist = np.sqrt(np.sum(diff ** 2, axis=2))
    np.fill_diagonal(dist, np.inf)
    min_dist = np.min(dist)
    return min(min_wall, min_dist / 2.0)


def _hex_init():
    """Hexagonal lattice initialization."""
    m = int(np.ceil(np.sqrt(N / (np.sqrt(3) / 2))))
    pts = []
    # Generate hex lattice points
    for row in range(m + 2):
        for col in range(m + 2):
            x = col * 1.0 / m + (0.5 / m if row % 2 else 0.0)
            y = row * np.sqrt(3) / 2 / m
            if -0.5 <= x <= 0.5 and -0.5 <= y <= 0.5:
                pts.append([x - 0.5, y - 0.5])
    pts = np.array(pts)
    # If we have too many, subsample; if too few, add random points
    if len(pts) >= N:
        # Use k-means-like selection: pick N points that maximize min distance
        # Simple approach: greedy farthest-point sampling
        remaining = list(range(len(pts)))
        selected = [remaining.pop(0)]
        while len(selected) < N and remaining:
            # Find point farthest from all selected
            best_idx = 0
            best_min = -1
            for i in remaining:
                dists = np.linalg.norm(pts[i] - pts[selected], axis=1)
                min_d = np.min(dists)
                if min_d > best_min:
                    best_min = min_d
                    best_idx = i
            selected.append(remaining.pop(best_idx))
        return pts[selected]
    else:
        # Add random points to fill
        while len(pts) < N:
            p = np.random.uniform(-0.5, 0.5, 2)
            pts = np.vstack([pts, p])
        return pts


def _force_relaxation(pts, n_iter=5000, step_init=0.01, step_decay=0.9995):
    """Physics-based relaxation to maximize minimum distance."""
    pts = pts.copy()
    step = step_init
    best_score = _score(pts)
    best_pts = pts.copy()
    
    for it in range(n_iter):
        if _remaining() < 5.0:
            break
        
        # Compute all pairwise distances
        diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
        dist = np.sqrt(np.sum(diff ** 2, axis=2) + 1e-12)
        np.fill_diagonal(dist, np.inf)
        
        # Forces: repulsion from all others, stronger for close pairs
        # Use inverse-square law with cutoff
        forces = np.zeros_like(pts)
        for i in range(N):
            # Wall forces
            for k in range(2):
                if pts[i, k] > 0:
                    forces[i, k] -= step * 10.0 * pts[i, k] / (0.5 - pts[i, k] + 1e-6)
                else:
                    forces[i, k] -= step * 10.0 * pts[i, k] / (0.5 + pts[i, k] + 1e-6)
            
            # Pairwise repulsion
            for j in range(N):
                if i == j:
                    continue
                d = dist[i, j]
                if d < 0.1:  # Only apply force for close pairs for efficiency
                    f = step / (d * d + 1e-6)
                    forces[i] += f * diff[i, j]
        
        pts += forces
        # Clamp to square
        pts = np.clip(pts, -0.5, 0.5)
        
        # Update step
        step *= step_decay
        
        # Track best
        if it % 50 == 0:
            s = _score(pts)
            if s > best_score:
                best_score = s
                best_pts = pts.copy()
    
    return best_pts, best_score


def _local_optimize(pts, n_iter=2000):
    """Local optimization using subgradient method on min distance."""
    pts = pts.copy()
    best_score = _score(pts)
    best_pts = pts.copy()
    step = 0.005
    
    for it in range(n_iter):
        if _remaining() < 5.0:
            break
        
        # Compute all pairwise distances
        diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
        dist2 = np.sum(diff ** 2, axis=2)
        dist = np.sqrt(dist2 + 1e-12)
        np.fill_diagonal(dist, np.inf)
        np.fill_diagonal(dist2, np.inf)
        
        # Find minimum distance pair
        min_dist_idx = np.unravel_index(np.argmin(dist), dist.shape)
        min_d = dist[min_dist_idx]
        
        # Also check wall distances
        wall_dist = np.min(0.5 - np.abs(pts), axis=1)
        min_wall_idx = np.argmin(wall_dist)
        min_wall = wall_dist[min_wall_idx]
        
        # Subgradient step
        grad = np.zeros_like(pts)
        
        # If min is a pairwise distance
        if min_d / 2.0 < min_wall:
            i, j = min_dist_idx
            if dist2[i, j] > 1e-12:
                grad[i] += step * diff[i, j] / dist2[i, j]
                grad[j] -= step * diff[i, j] / dist2[i, j]
        else:
            # If min is wall distance
            k = min_wall_idx
            for dim in range(2):
                if pts[k, dim] > 0:
                    grad[k, dim] -= step
                else:
                    grad[k, dim] += step
        
        pts += grad
        pts = np.clip(pts, -0.5, 0.5)
        
        step *= 0.999
        
        if it % 100 == 0:
            s = _score(pts)
            if s > best_score:
                best_score = s
                best_pts = pts.copy()
    
    return best_pts, best_score


def _perturb(pts, sigma=0.01):
    """Add random perturbation."""
    return pts + np.random.normal(0, sigma, pts.shape)


def _hill_climb(pts, n_restarts=5, n_iter=1000):
    """Simple hill climbing with restarts."""
    best_pts = pts.copy()
    best_score = _score(pts)
    
    for r in range(n_restarts):
        if _remaining() < 10.0:
            break
        
        current_pts = _perturb(best_pts, sigma=0.005)
        current_score = _score(current_pts)
        
        for it in range(n_iter):
            if _remaining() < 5.0:
                break
            
            # Try small moves
            move = np.random.normal(0, 0.002, current_pts.shape)
            new_pts = np.clip(current_pts + move, -0.5, 0.5)
            new_score = _score(new_pts)
            
            if new_score > current_score:
                current_pts = new_pts
                current_score = new_score
        
        if current_score > best_score:
            best_score = current_score
            best_pts = current_pts.copy()
    
    return best_pts, best_score


def construct():
    """Construct centers for N=200 equal circles in unit square."""
    # Start with hexagonal initialization
    pts = _hex_init()
    best_pts = pts.copy()
    best_score = _score(pts)
    
    # Phase 1: Force relaxation
    if _remaining() > 20.0:
        pts, score = _force_relaxation(pts, n_iter=3000, step_init=0.005)
        if score > best_score:
            best_score = score
            best_pts = pts.copy()
    
    # Phase 2: Local optimization
    if _remaining() > 15.0:
        pts, score = _local_optimize(best_pts, n_iter=1500)
        if score > best_score:
            best_score = score
            best_pts = pts.copy()
    
    # Phase 3: Hill climbing with perturbations
    if _remaining() > 10.0:
        pts, score = _hill_climb(best_pts, n_restarts=3, n_iter=500)
        if score > best_score:
            best_score = score
            best_pts = pts.copy()
    
    # Final polish: more local optimization
    if _remaining() > 5.0:
        pts, score = _local_optimize(best_pts, n_iter=1000)
        if score > best_score:
            best_score = score
            best_pts = pts.copy()
    
    return best_pts
