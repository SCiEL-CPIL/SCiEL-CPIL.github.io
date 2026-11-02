import numpy as np
import time

N = 300

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # 15s margin
    
    # 1. Initialization: Hexagonal Lattice
    spacing = 2.0 / (np.sqrt(N) + 1.0)
    points = None
    while True:
        k = int(1.0 / spacing) + 2
        rows = [((i + 0.5 * (j % 2)) * spacing, j * spacing * np.sqrt(3.0) / 2.0)
                for i in range(-k, k + 1) for j in range(-k, k + 1)]
        pts = np.array(rows)
        radius = np.hypot(pts[:, 0], pts[:, 1])
        mask = radius <= 1.0 - spacing / 2.0
        pts = pts[mask]
        if len(pts) >= N:
            points = pts[np.argsort(radius, kind="stable")[:N]]
            break
        spacing *= 0.98
        
    # Add small noise to break symmetry
    np.random.seed(42)
    points += np.random.randn(N, 2) * 0.01
    
    # Ensure inside
    radii = np.linalg.norm(points, axis=1)
    for i in range(N):
        if radii[i] > 1.0:
            points[i] *= 1.0 / radii[i]

    def calc_score(pts):
        diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
        dists = np.linalg.norm(diff, axis=2)
        np.fill_diagonal(dists, np.inf)
        min_dist = np.min(dists)
        
        radii = np.linalg.norm(pts, axis=1)
        min_boundary = 1.0 - np.max(radii)
        
        return min(min_dist / 2.0, min_boundary)

    best_points = points.copy()
    best_score = calc_score(best_points)
    
    step_size = 0.02
    stuck_count = 0
    max_stuck = 2000
    temp = 1.0 # For simple SA-like acceptance
    
    while time.time() < deadline:
        idx = np.random.randint(N)
        theta = np.random.uniform(0, 2 * np.pi)
        delta = np.array([np.cos(theta), np.sin(theta)]) * step_size
        
        new_pt = best_points[idx].copy() + delta
        
        # Quick check: if it goes outside, skip
        if np.linalg.norm(new_pt) > 1.0:
            continue
            
        best_points[idx] = new_pt
        new_score = calc_score(best_points)
        
        if new_score > best_score:
            best_score = new_score
            stuck_count = 0
            step_size *= 0.99
        else:
            # Accept with probability exp(-(delta_E)/T) ? 
            # Or just revert for Hill Climbing.
            # Let's try Hill Climbing first, it's robust.
            best_points[idx] -= delta
            stuck_count += 1
            
            if stuck_count > max_stuck:
                # Perturb a few points to escape
                for _ in range(10):
                    j = np.random.randint(N)
                    best_points[j] += np.random.randn(2) * 0.05
                    # Ensure inside
                    r = np.linalg.norm(best_points[j])
                    if r > 1.0:
                        best_points[j] /= r
                best_score = calc_score(best_points)
                stuck_count = 0
                max_stuck = 5000
                step_size = 0.05

    return best_points
