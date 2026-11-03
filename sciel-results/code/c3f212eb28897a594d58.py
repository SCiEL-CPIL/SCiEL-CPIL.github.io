import numpy as np
from scipy.optimize import minimize
import time

N = 50

def _build_ring_seed(n=N):
    """Build a concentric-ring seed layout for n circles in the unit disk."""
    if n == 1:
        return np.zeros((1, 2))
    # Decide ring structure: 1 center + rings
    # For 50 circles, a good structure is 1 + 6 + 12 + 18 + 13? 
    # Let's use a more systematic approach: choose number of rings and points per ring.
    # Common good packings for 50: center + rings of 6,12,18,14
    # But let's be adaptive: try to fill rings with even spacing.
    # We'll use rings with counts: 1, 6, 12, 18, 13 (total 50)
    # Actually, let's compute radii to maximize minimum spacing.
    
    # Simple approach: use rings with counts that are multiples of 6 for symmetry
    # 1 + 6 + 12 + 18 + 13 = 50
    ring_counts = [1, 6, 12, 18, 13]
    # Adjust if needed
    total = sum(ring_counts)
    while total < n:
        ring_counts[-1] += 1
        total += 1
    while total > n:
        if ring_counts[-1] > 1:
            ring_counts[-1] -= 1
            total -= 1
        else:
            ring_counts.pop()
            total -= 1
    
    points = [np.array([0.0, 0.0])]  # center
    for k, count in enumerate(ring_counts[1:], start=1):
        # Radius for this ring: approximate
        # For a ring with 'count' circles, the spacing along the circumference is 2*r*sin(pi/count)
        # We want this to be roughly equal to the radial spacing.
        # Let's use r_k = k * 0.28 as a starting guess, then refine.
        r = 0.28 * k
        # Clamp to stay inside
        r = min(r, 0.95)
        angles = 2.0 * np.pi * np.arange(count) / count
        # Add a small offset for odd rings to interleave
        if k % 2 == 0:
            angles += np.pi / count
        ring_pts = np.column_stack([r * np.cos(angles), r * np.sin(angles)])
        points.append(ring_pts)
    
    pts = np.vstack(points)
    # Trim or pad to n
    if len(pts) > n:
        pts = pts[:n]
    elif len(pts) < n:
        # Add extra points near the boundary
        extra = n - len(pts)
        r = 0.95
        angles = 2.0 * np.pi * np.arange(extra) / extra
        extra_pts = np.column_stack([r * np.cos(angles), r * np.sin(angles)])
        pts = np.vstack([pts, extra_pts])
    
    return pts[:n]


def _compute_score(pts):
    """Compute the packing radius for a set of points."""
    if len(pts) < 2:
        return 1.0 - np.linalg.norm(pts[0])
    # Pairwise distances
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff**2, axis=2))
    np.fill_diagonal(dists, np.inf)
    min_pair = np.min(dists) / 2.0
    # Boundary
    radii = np.linalg.norm(pts, axis=1)
    min_boundary = 1.0 - np.min(radii)
    return min(min_pair, min_boundary)


def _soft_min_score(pts, beta=100.0):
    """Soft-min surrogate for the packing radius. Lower is better."""
    n = len(pts)
    # Pairwise
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff**2, axis=2))
    np.fill_diagonal(dists, np.inf)
    # Soft min of pair distances / 2
    # Use log-sum-exp for stability
    pair_vals = dists / 2.0
    # Exclude diagonal
    mask = ~np.eye(n, dtype=bool)
    pair_vals = pair_vals[mask]
    # Soft min: -logsumexp(-x) / beta
    m = np.min(pair_vals)
    soft_min_pair = -np.log(np.sum(np.exp(-beta * (pair_vals - m)))) / beta + m
    
    # Boundary
    radii = np.linalg.norm(pts, axis=1)
    boundary_vals = 1.0 - radii
    mb = np.min(boundary_vals)
    soft_min_boundary = -np.log(np.sum(np.exp(-beta * (boundary_vals - mb)))) / beta + mb
    
    # Combine: soft min of the two
    combined = np.array([soft_min_pair, soft_min_boundary])
    mc = np.min(combined)
    result = -np.log(np.sum(np.exp(-beta * (combined - mc)))) / beta + mc
    return -result  # Negate so we minimize


def _is_feasible(pts, tol=1e-12):
    """Check if points are strictly feasible (inside, non-overlapping)."""
    n = len(pts)
    # Boundary
    radii = np.linalg.norm(pts, axis=1)
    if np.any(radii >= 1.0 - tol):
        return False
    # Pairwise
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff**2, axis=2))
    np.fill_diagonal(dists, np.inf)
    if np.min(dists) < 2.0 * tol:
        return False
    return True


def _force_directed_step(pts, beta=50.0, lr=0.01, steps=50):
    """Perform force-directed steps to improve the layout."""
    n = len(pts)
    for _ in range(steps):
        # Compute forces
        forces = np.zeros_like(pts)
        for i in range(n):
            for j in range(i + 1, n):
                diff = pts[i] - pts[j]
                dist = np.linalg.norm(diff)
                if dist < 1e-12:
                    # Random push
                    forces[i] += np.random.randn(2) * 0.01
                    forces[j] -= np.random.randn(2) * 0.01
                    continue
                # Repulsive force: stronger when close
                # Use soft-min gradient approximation
                weight = np.exp(-beta * dist)
                forces[i] += weight * diff / dist
                forces[j] -= weight * diff / dist
            # Boundary force: push inward if near boundary
            r = np.linalg.norm(pts[i])
            if r > 0.9:
                forces[i] -= (r - 0.9) * pts[i] / r
        # Update
        pts = pts + lr * forces
        # Project to stay inside disk
        radii = np.linalg.norm(pts, axis=1)
        for i in range(n):
            if radii[i] > 0.999:
                pts[i] = pts[i] * 0.999 / radii[i]
    return pts


def _polish(pts, max_iter=10000, tol=1e-14):
    """High-precision polish using coordinate descent."""
    n = len(pts)
    current_score = _compute_score(pts)
    
    for iteration in range(max_iter):
        improved = False
        for i in range(n):
            # Try small perturbations in x and y
            for dx in [0.001, -0.001, 0.0005, -0.0005, 0.0001, -0.0001]:
                for dy in [0.0, 0.001, -0.001, 0.0005, -0.0005]:
                    new_pts = pts.copy()
                    new_pts[i, 0] += dx
                    new_pts[i, 1] += dy
                    # Check if still inside
                    if np.linalg.norm(new_pts[i]) >= 1.0:
                        continue
                    new_score = _compute_score(new_pts)
                    if new_score > current_score + 1e-15:
                        pts = new_pts
                        current_score = new_score
                        improved = True
                        break
                if improved:
                    break
            if not improved:
                # Try gradient-based step
                # Compute gradient of soft-min
                beta = 200.0
                diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
                dists = np.sqrt(np.sum(diff**2, axis=2))
                np.fill_diagonal(dists, np.inf)
                grad = np.zeros_like(pts[i])
                for j in range(n):
                    if i == j:
                        continue
                    d = dists[i, j]
                    if d < 1e-12:
                        continue
                    weight = np.exp(-beta * d)
                    grad += weight * (pts[i] - pts[j]) / d
                r = np.linalg.norm(pts[i])
                if r > 0.85:
                    grad -= (r - 0.85) * pts[i] / r
                # Normalize and take small step
                norm_grad = np.linalg.norm(grad)
                if norm_grad > 1e-15:
                    step = 0.0001 * grad / norm_grad
                    new_pts = pts.copy()
                    new_pts[i] += step
                    if np.linalg.norm(new_pts[i]) < 1.0:
                        new_score = _compute_score(new_pts)
                        if new_score > current_score + 1e-15:
                            pts = new_pts
                            current_score = new_score
                            improved = True
        if not improved:
            break
    
    return pts, current_score


def construct():
    start_time = time.time()
    deadline = 140.0  # Leave 10s margin
    
    # Build initial seed
    pts = _build_ring_seed(N)
    
    # Initial score
    current_score = _compute_score(pts)
    best_pts = pts.copy()
    best_score = current_score
    
    # Phase 1: Force-directed optimization
    beta = 30.0
    lr = 0.05
    step_count = 0
    while time.time() - start_time < deadline - 20.0:
        # Force-directed step
        pts = _force_directed_step(pts, beta=beta, lr=lr, steps=20)
        score = _compute_score(pts)
        if score > best_score:
            best_pts = pts.copy()
            best_score = score
        
        # Occasionally do a local polish
        if step_count % 5 == 0:
            polished, pol_score = _polish(pts, max_iter=5000, tol=1e-13)
            if pol_score > best_score:
                best_pts = polished.copy()
                best_score = pol_score
                pts = polished.copy()
        
        # Anneal: decrease beta, increase lr slightly
        beta *= 0.95
        lr *= 0.98
        step_count += 1
    
    # Phase 2: Final high-precision polish
    if time.time() - start_time < deadline - 5.0:
        polished, pol_score = _polish(best_pts, max_iter=50000, tol=1e-15)
        if pol_score > best_score:
            best_pts = polished
            best_score = pol_score
    
    # Ensure feasibility: if any point is outside, pull it in
    radii = np.linalg.norm(best_pts, axis=1)
    for i in range(N):
        if radii[i] >= 1.0:
            best_pts[i] = best_pts[i] * 0.9999 / radii[i]
    
    return best_pts
