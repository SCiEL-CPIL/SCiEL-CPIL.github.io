"""Radial-ring packing strategy for 300 circles in unit disk."""
import numpy as np
import time

N = 300

def _ring_packing(ring_counts, time_budget):
    """
    Place circles in concentric rings.
    ring_counts: list of ints, number of circles per ring (innermost first).
    Returns (points, r) where points is (N,2) and r is the achieved radius.
    """
    n_rings = len(ring_counts)
    # Start with uniform radial spacing
    rads = np.linspace(0.0, 0.95, n_rings) if n_rings > 1 else np.array([0.0])
    # Angular offsets: stagger rings to maximize inter-ring separation
    angles = np.zeros(n_rings)
    for i in range(n_rings):
        if i > 0:
            angles[i] = np.pi / max(ring_counts[i], 1)  # half-step offset
    
    # Iteratively optimize radii and angles
    best_r = 0.0
    best_points = None
    t0 = time.time()
    
    for iteration in range(2000):
        if time.time() - t0 > time_budget:
            break
        
        # Build points from current parameters
        points = []
        for i, (cnt, rad, ang_off) in enumerate(zip(ring_counts, rads, angles)):
            if cnt == 0:
                continue
            if cnt == 1:
                pts = np.array([[0.0, 0.0]])
            else:
                thetas = ang_off + 2 * np.pi * np.arange(cnt) / cnt
                pts = np.column_stack([rad * np.cos(thetas), rad * np.sin(thetas)])
            points.append(pts)
        
        if not points:
            break
        all_points = np.vstack(points)
        
        # Compute current radius
        min_dist = np.inf
        for i in range(len(all_points)):
            for j in range(i + 1, len(all_points)):
                d = np.linalg.norm(all_points[i] - all_points[j])
                if d < min_dist:
                    min_dist = d
        min_cont = 1.0 - np.max(np.linalg.norm(all_points, axis=1))
        r = min(min_dist / 2.0, min_cont)
        
        if r > best_r:
            best_r = r
            best_points = all_points.copy()
        
        # Gradient-free perturbation: adjust radii and angles
        # Perturb radii
        for i in range(n_rings):
            if ring_counts[i] > 0:
                dr = 0.01 * (1.0 - rads[i])  # larger perturbation near edge
                rads[i] += dr * (2 * np.random.random() - 1)
                rads[i] = np.clip(rads[i], 0.0, 0.99)
        
        # Perturb angles
        for i in range(n_rings):
            if ring_counts[i] > 1:
                angles[i] += 0.05 * (2 * np.random.random() - 1)
        
        # Ensure monotonic radii
        rads = np.sort(rads)
    
    return best_points, best_r


def _repulsion_polish(points, r, time_budget):
    """Vectorized repulsion-based local optimization."""
    t0 = time.time()
    lr = 0.001
    n = len(points)
    best_r = r
    best_points = points.copy()
    
    for step in range(5000):
        if time.time() - t0 > time_budget:
            break
        
        # Compute forces
        forces = np.zeros_like(points)
        for i in range(n):
            # Container repulsion
            dist_to_center = np.linalg.norm(points[i])
            if dist_to_center > 0:
                forces[i] -= 2.0 * (dist_to_center - (1.0 - r)) * points[i] / dist_to_center
            # Pairwise repulsion
            for j in range(n):
                if i == j:
                    continue
                d = points[i] - points[j]
                dist = np.linalg.norm(d)
                if dist < 2 * r and dist > 1e-10:
                    forces[i] += 10.0 * (2 * r - dist) * d / dist
        
        # Update
        points = points + lr * forces
        # Clamp to container
        norms = np.linalg.norm(points, axis=1)
        for i in range(n):
            if norms[i] > 1.0 - r:
                points[i] *= (1.0 - r) / norms[i]
        
        # Compute current r
        min_dist = np.inf
        for i in range(n):
            for j in range(i + 1, n):
                d = np.linalg.norm(points[i] - points[j])
                if d < min_dist:
                    min_dist = d
        min_cont = 1.0 - np.max(np.linalg.norm(points, axis=1))
        r = min(min_dist / 2.0, min_cont)
        
        if r > best_r:
            best_r = r
            best_points = points.copy()
        
        lr *= 0.999
    
    return best_points, best_r


def _find_best_ring_config(time_budget):
    """Try multiple ring configurations and return the best."""
    t0 = time.time()
    best_points = None
    best_r = 0.0
    
    # Strategy: assign circles to rings based on circumference capacity
    # Try different numbers of rings
    for n_rings in range(15, 30):
        if time.time() - t0 > time_budget * 0.7:
            break
        
        # Distribute circles: more circles in outer rings
        # Use a power-law distribution
        weights = np.array([i ** 1.5 for i in range(1, n_rings + 1)])
        weights /= weights.sum()
        ring_counts = np.round(weights * N).astype(int)
        # Fix total
        diff = N - ring_counts.sum()
        # Add/remove from outermost rings
        idx = n_rings - 1
        while diff != 0 and idx >= 0:
            if diff > 0:
                ring_counts[idx] += 1
                diff -= 1
            elif diff < 0 and ring_counts[idx] > 0:
                ring_counts[idx] -= 1
                diff += 1
            idx -= 1
        
        if any(c < 0 for c in ring_counts):
            continue
        
        points, r = _ring_packing(ring_counts, time_budget * 0.3)
        if points is not None and r > best_r:
            best_r = r
            best_points = points
    
    return best_points, best_r


def construct():
    t_start = time.time()
    deadline = t_start + 140.0  # 10s safety margin
    
    # Phase 1: Find good ring configuration
    points, r = _find_best_ring_config(60.0)
    
    if points is None:
        # Fallback: simple hex lattice
        spacing = 2.0 / (np.sqrt(N) + 1.0)
        while True:
            k = int(1.0 / spacing) + 2
            rows = [((i + 0.5 * (j % 2)) * spacing, j * spacing * np.sqrt(3.0) / 2.0)
                    for i in range(-k, k + 1) for j in range(-k, k + 1)]
            pts = np.array(rows)
            radius = np.hypot(pts[:, 0], pts[:, 1])
            pts, radius = pts[radius <= 1.0 - spacing / 2.0], radius[radius <= 1.0 - spacing / 2.0]
            if len(pts) >= N:
                points = pts[np.argsort(radius, kind="stable")[:N]]
                r = min(np.min([np.linalg.norm(points[i] - points[j]) for i in range(N) for j in range(i+1, N)])/2,
                        1.0 - np.max(np.linalg.norm(points, axis=1)))
                break
            spacing *= 0.98
    
    # Phase 2: Polish with repulsion
    remaining = deadline - time.time()
    if remaining > 5.0 and points is not None:
        points, r = _repulsion_polish(points, r, remaining - 5.0)
    
    # Final validation
    if points is None or len(points) != N:
        # Emergency fallback
        points = np.zeros((N, 2))
        r = 0.01
    
    # Ensure finite and valid
    if not np.all(np.isfinite(points)):
        points = np.zeros((N, 2))
    
    return points
