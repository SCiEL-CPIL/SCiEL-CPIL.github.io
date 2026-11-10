# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial.distance import pdist
import time


def min_max_dist_dim2_16() -> np.ndarray:
    """
    Creates 16 points in 2 dimensions in order to maximize the ratio of minimum to maximum distance.

    Returns
        points: np.ndarray of shape (16,2) containing the (x,y) coordinates of the 16 points.

    """
    start_time = time.time()
    deadline = start_time + 620.0  # Safety margin of 10 seconds

    def compute_score(points):
        if points.shape != (16, 2):
            return -np.inf
        d = pdist(points)
        if d.size == 0:
            return 0.0
        d_min = np.min(d)
        d_max = np.max(d)
        if d_max == 0:
            return 0.0
        ratio = d_min / d_max
        return ratio * ratio

    n = 16

    # Initialize with regular 16-gon on unit circle
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
    points = np.column_stack([np.cos(angles), np.sin(angles)])

    best_points = points.copy()
    best_score = compute_score(best_points)

    # Coordinate ascent with decreasing step sizes
    np.random.seed(42)
    step_sizes = [0.1, 0.05, 0.02, 0.01, 0.005, 0.002, 0.001]

    for step_size in step_sizes:
        if time.time() >= deadline:
            break
        improved = True
        while improved and time.time() < deadline:
            improved = False
            for i in range(n):
                if time.time() >= deadline:
                    break
                # Try moving point i in random direction
                for _ in range(20):
                    if time.time() >= deadline:
                        break
                    direction = np.random.randn(2)
                    norm = np.linalg.norm(direction)
                    if norm > 0:
                        direction /= norm
                    new_points = best_points.copy()
                    new_points[i] += direction * step_size
                    score = compute_score(new_points)
                    if score > best_score:
                        best_points = new_points
                        best_score = score
                        improved = True
                        break

    return best_points


# EVOLVE-BLOCK-END
