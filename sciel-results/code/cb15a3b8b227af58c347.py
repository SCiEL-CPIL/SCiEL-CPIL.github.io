"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 300

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.spatial import cKDTree


def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin before 120s limit

    def compute_score(points):
        n = len(points)
        if n < 2:
            return 0.0
        # Compute minimum pairwise distance using cKDTree
        tree = cKDTree(points)
        dists, _ = tree.query(points, k=2)
        min_dist = np.min(dists[:, 1])
        # Compute minimum distance to boundary
        radii = np.hypot(points[:, 0], points[:, 1])
        min_boundary = 1.0 - np.max(radii)
        return min(min_dist / 2.0, min_boundary)

    def gen_hex_lattice(spacing):
        k = int(1.0 / spacing) + 3
        pts = []
        for j in range(-k, k + 1):
            for i in range(-k, k + 1):
                x = (i + 0.5 * (j % 2)) * spacing
                y = j * spacing * np.sqrt(3.0) / 2.0
                r = np.hypot(x, y)
                if r <= 1.0 - spacing * 0.5:
                    pts.append((x, y))
        return np.array(pts)

    # Start with hexagonal lattice - find good initial spacing
    spacing = 2.0 / (np.sqrt(N) + 1.0)
    while True:
        points = gen_hex_lattice(spacing)
        if len(points) >= N:
            # Sort by radius and take first N
            radii = np.hypot(points[:, 0], points[:, 1])
            points = points[np.argsort(radii, kind='stable')[:N]]
            break
        spacing *= 0.98

    best_score = compute_score(points)
    best_points = points.copy()

    # Local optimization with adaptive step size
    step_size = 0.01
    min_step = 1e-6
    no_improve_count = 0
    max_no_improve = 50

    while time.time() < deadline - 5.0:
        # Compute current state
        radii = np.hypot(points[:, 0], points[:, 1])
        tree = cKDTree(points)
        dists, _ = tree.query(points, k=2)
        min_dists = dists[:, 1]
        
        # Score components
        min_pair_dist = np.min(min_dists)
        min_boundary = 1.0 - np.max(radii)
        current_score = min(min_pair_dist / 2.0, min_boundary)
        
        # Try multiple random point moves
        improved = False
        for _ in range(20):  # Try 20 random moves per iteration
            # Pick a random point to move
            idx = np.random.randint(N)
            
            # Determine direction based on constraints
            if min_pair_dist / 2.0 < min_boundary:
                # Pairwise constraint is limiting - move away from nearest neighbor
                _, nn_idx = tree.query(points[idx], k=2)
                nn_idx = nn_idx[1]
                diff = points[idx] - points[nn_idx]
                norm = np.linalg.norm(diff)
                if norm > 1e-10:
                    direction = diff / norm
                else:
                    # Random direction if coincident
                    angle = np.random.uniform(0, 2 * np.pi)
                    direction = np.array([np.cos(angle), np.sin(angle)])
            else:
                # Boundary constraint is limiting - move inward
                r = np.linalg.norm(points[idx])
                if r > 1e-10:
                    direction = -points[idx] / r
                else:
                    # At center, move randomly
                    angle = np.random.uniform(0, 2 * np.pi)
                    direction = np.array([np.cos(angle), np.sin(angle)])
            
            # Add some randomness to direction
            noise_angle = np.random.uniform(-0.5, 0.5)
            cos_n, sin_n = np.cos(noise_angle), np.sin(noise_angle)
            direction = np.array([
                direction[0] * cos_n - direction[1] * sin_n,
                direction[0] * sin_n + direction[1] * cos_n
            ])
            
            # Try the move
            new_point = points[idx] + direction * step_size
            if np.hypot(new_point[0], new_point[1]) < 1.0:
                trial = points.copy()
                trial[idx] = new_point
                trial_score = compute_score(trial)
                if trial_score > current_score:
                    points = trial
                    current_score = trial_score
                    improved = True
                    if trial_score > best_score:
                        best_score = trial_score
                        best_points = points.copy()
        
        # Adaptive step size
        if improved:
            no_improve_count = 0
            step_size = min(step_size * 1.1, 0.05)  # Increase step if improving
        else:
            no_improve_count += 1
            step_size = max(step_size * 0.8, min_step)  # Decrease step if not improving
        
        # Reset if stuck for too long
        if no_improve_count >= max_no_improve:
            no_improve_count = 0
            step_size = 0.01  # Reset to default step

    # Final check
    final_score = compute_score(best_points)
    if final_score > best_score:
        best_score = final_score

    return best_points
# EVOLVE-BLOCK-END
