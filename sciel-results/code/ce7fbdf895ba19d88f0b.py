"""Pack 50 equal circles in a unit circle using simulated annealing + local refinement."""
import numpy as np
import time

N = 50


def _score(points):
    """Compute the minimum radius for a set of points."""
    if len(points) != N:
        return 0.0
    # Check all points are finite
    if not np.all(np.isfinite(points)):
        return 0.0
    # Distance from origin
    dists = np.sqrt(np.sum(points ** 2, axis=1))
    # Min distance to boundary
    r_boundary = 1.0 - dists
    min_boundary = np.min(r_boundary)
    if min_boundary <= 0:
        return 0.0
    # Pairwise distances
    from scipy.spatial.distance import cdist
    d = cdist(points, points)
    np.fill_diagonal(d, np.inf)
    min_pair = np.min(d) / 2.0
    return min(min_boundary, min_pair)


def _hex_init():
    """Generate initial hexagonal packing."""
    # Use a hexagonal grid with appropriate spacing
    best_points = None
    best_score = 0.0
    for spacing in np.arange(0.2, 0.05, -0.005):
        pts = []
        r_limit = 1.0 - spacing / 2.0
        rows = int(2.0 / spacing) + 2
        for j in range(-rows, rows + 1):
            for i in range(-rows, rows + 1):
                x = (i + 0.5 * (j % 2)) * spacing
                y = j * spacing * np.sqrt(3.0) / 2.0
                if x * x + y * y <= r_limit * r_limit:
                    pts.append([x, y])
        pts = np.array(pts)
        if len(pts) >= N:
            # Select N points closest to center
            dists = np.sqrt(np.sum(pts ** 2, axis=1))
            idx = np.argsort(dists)[:N]
            pts = pts[idx]
            s = _score(pts)
            if s > best_score:
                best_score = s
                best_points = pts.copy()
    return best_points if best_points is not None else np.random.randn(N, 2) * 0.5


def _anneal(points, deadline):
    """Simulated annealing to improve the packing."""
    points = points.astype(np.float64).copy()
    best_points = points.copy()
    best_score = _score(points)
    
    current_score = best_score
    T = 0.01  # Initial temperature
    T_min = 1e-8
    cooling = 0.999
    
    np.random.seed(42)
    
    while time.time() < deadline:
        if T < T_min:
            T = T_min
        # Random perturbation
        delta = np.random.randn(N, 2) * T
        new_points = points + delta
        
        new_score = _score(new_points)
        
        if new_score > current_score:
            points = new_points
            current_score = new_score
            if new_score > best_score:
                best_score = new_score
                best_points = new_points.copy()
        else:
            # Accept with probability exp((new_score - current_score) / T)
            if np.random.random() < np.exp((new_score - current_score) / max(T, 1e-10)):
                points = new_points
                current_score = new_score
        
        T *= cooling
    
    return best_points, best_score


def _refine(points, deadline):
    """Local refinement using gradient-based optimization."""
    points = points.astype(np.float64).copy()
    best_points = points.copy()
    best_score = _score(points)
    
    step_size = 0.001
    np.random.seed(123)
    
    while time.time() < deadline and step_size > 1e-6:
        improved = False
        for _ in range(200):
            if time.time() >= deadline:
                break
            # Try small random moves
            for i in range(N):
                for _ in range(5):
                    delta = np.random.randn(2) * step_size
                    old_x = points[i].copy()
                    points[i] = old_x + delta
                    s = _score(points)
                    if s > best_score:
                        best_score = s
                        best_points = points.copy()
                        improved = True
                    else:
                        points[i] = old_x
            if not improved:
                break
        step_size *= 0.8
    
    return best_points, best_score


def construct():
    start_time = time.time()
    deadline = start_time + 130.0  # Safety margin
    
    # Start with hexagonal initialization
    points = _hex_init()
    score = _score(points)
    
    # Simulated annealing
    points, score = _anneal(points, deadline - 20.0)
    
    # Local refinement
    points, score = _refine(points, deadline)
    
    # Final check
    final_score = _score(points)
    if final_score > score:
        score = final_score
    
    return points
