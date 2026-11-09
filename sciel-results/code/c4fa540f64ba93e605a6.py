"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 300

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.spatial.distance import pdist

def _compute_radius(points):
    """Compute the feasible radius for a set of points."""
    if points is None or len(points) == 0:
        return 0.0
    dist_boundary = 1.0 - np.linalg.norm(points, axis=1)
    min_boundary = np.min(dist_boundary)
    d = pdist(points)
    min_dist = np.min(d)
    min_pair = min_dist / 2.0
    return min(min_boundary, min_pair)

def _initial_hexagonal():
    """Generate a hexagonal lattice packing as starting point."""
    est_r = np.sqrt(0.9069 / N)
    est_s = 2 * est_r
    
    best_points = None
    best_r = 0.0
    
    for spacing in np.linspace(est_s * 0.95, est_s * 1.05, 10):
        a = spacing
        h = a * np.sqrt(3) / 2
        points = []
        max_range = int(np.ceil(1.0 / h)) + 2
        for j in range(-max_range, max_range + 1):
            for i in range(-max_range, max_range + 1):
                x = (i + 0.5 * (j % 2)) * a
                y = j * h
                if x*x + y*y <= 1.0:
                    points.append([x, y])
        points = np.array(points)
        if len(points) < N:
            continue
        dists = np.linalg.norm(points, axis=1)
        idx = np.argsort(dists)[:N]
        selected = points[idx]
        r = _compute_radius(selected)
        if r > best_r:
            best_r = r
            best_points = selected.copy()
    
    if best_points is None:
        best_points = np.zeros((N, 2))
        best_r = 0.01
    
    return best_points, best_r

def _initial_rings():
    """Generate concentric ring packing as starting point."""
    best_points = None
    best_r = 0.0
    
    # Try different numbers of rings
    for n_rings in range(5, 20):
        points = []
        # Center point
        points.append([0.0, 0.0])
        for ring in range(1, n_rings):
            # Distribute points evenly on ring
            # Radius of ring
            r_ring = ring * 2 * np.sqrt(0.9069 / N)
            if r_ring > 1.0:
                break
            # Number of points on this ring
            n_pts = int(2 * np.pi * r_ring / (2 * np.sqrt(0.9069 / N)))
            n_pts = max(n_pts, 1)
            angles = 2 * np.pi * np.arange(n_pts) / n_pts
            for angle in angles:
                x = r_ring * np.cos(angle)
                y = r_ring * np.sin(angle)
                if x*x + y*y <= 1.0:
                    points.append([x, y])
        
        points = np.array(points)
        if len(points) < N:
            continue
        # Select N points closest to center
        dists = np.linalg.norm(points, axis=1)
        idx = np.argsort(dists)[:N]
        selected = points[idx]
        r = _compute_radius(selected)
        if r > best_r:
            best_r = r
            best_points = selected.copy()
    
    if best_points is None:
        best_points = np.zeros((N, 2))
        best_r = 0.01
    
    return best_points, best_r

def _initial_random():
    """Generate random points inside unit circle."""
    best_points = None
    best_r = 0.0
    
    for _ in range(20):
        # Rejection sampling
        points = []
        while len(points) < N:
            x = np.random.uniform(-1, 1)
            y = np.random.uniform(-1, 1)
            if x*x + y*y <= 1.0:
                points.append([x, y])
        points = np.array(points[:N])
        r = _compute_radius(points)
        if r > best_r:
            best_r = r
            best_points = points.copy()
    
    if best_points is None:
        best_points = np.zeros((N, 2))
        best_r = 0.01
    
    return best_points, best_r

def _simulated_annealing(points, start_temp, end_temp, n_iterations, deadline):
    """Run simulated annealing to improve packing."""
    current_points = points.copy()
    current_r = _compute_radius(current_points)
    best_points = current_points.copy()
    best_r = current_r
    
    n_params = 2 * N
    # Temperature schedule
    temps = np.geomspace(start_temp, end_temp, n_iterations)
    
    for it in range(n_iterations):
        if time.time() >= deadline:
            break
        
        temp = temps[it]
        
        # Generate a random move: perturb one or a few points
        n_moves = np.random.choice([1, 2, 3, 5], p=[0.4, 0.3, 0.2, 0.1])
        indices = np.random.choice(N, size=n_moves, replace=False)
        
        new_points = current_points.copy()
        for idx in indices:
            # Random displacement
            angle = np.random.uniform(0, 2 * np.pi)
            magnitude = np.random.uniform(0, 0.02)
            new_points[idx] = current_points[idx] + magnitude * np.array([np.cos(angle), np.sin(angle)])
            # Ensure inside unit circle
            norm = np.linalg.norm(new_points[idx])
            if norm > 1.0:
                new_points[idx] = new_points[idx] / norm
        
        new_r = _compute_radius(new_points)
        
        # Accept if better or with probability exp((new_r - current_r) / temp)
        if new_r >= current_r:
            current_points = new_points
            current_r = new_r
        else:
            delta = new_r - current_r
            prob = np.exp(delta / temp)
            if np.random.random() < prob:
                current_points = new_points
                current_r = new_r
        
        if current_r > best_r:
            best_r = current_r
            best_points = current_points.copy()
    
    return best_points, best_r

def construct():
    """Construct the optimal packing."""
    start_time = time.time()
    deadline = start_time + 145.0  # 5 second safety margin
    
    # Generate diverse initial configurations
    hex_points, hex_r = _initial_hexagonal()
    ring_points, ring_r = _initial_rings()
    rand_points, rand_r = _initial_random()
    
    # Select the best initial configuration
    init_configs = [
        (hex_points, hex_r),
        (ring_points, ring_r),
        (rand_points, rand_r)
    ]
    init_configs.sort(key=lambda x: x[1], reverse=True)
    best_points, best_r = init_configs[0]
    
    # Run simulated annealing from multiple starting points
    start_points_list = [hex_points, ring_points, rand_points]
    
    for i, init_pts in enumerate(start_points_list):
        if time.time() >= deadline - 10:
            break
        
        # Run SA with different temperature schedules
        # Higher initial temp for global exploration
        start_temp = 0.01
        end_temp = 0.0001
        n_iterations = 5000
        
        sa_points, sa_r = _simulated_annealing(
            init_pts, start_temp, end_temp, n_iterations, deadline
        )
        
        if sa_r > best_r:
            best_r = sa_r
            best_points = sa_points.copy()
    
    # Final local refinement: try small perturbations
    if time.time() < deadline - 5:
        improved = True
        while improved and time.time() < deadline - 5:
            improved = False
            for i in range(N):
                if time.time() >= deadline - 5:
                    break
                for dx, dy in [(-0.001, 0), (0.001, 0), (0, -0.001), (0, 0.001)]:
                    new_points = best_points.copy()
                    new_points[i] = [best_points[i][0] + dx, best_points[i][1] + dy]
                    if np.linalg.norm(new_points[i]) > 1.0:
                        continue
                    r_new = _compute_radius(new_points)
                    if r_new > best_r + 1e-9:
                        best_r = r_new
                        best_points = new_points
                        improved = True
    
    return best_points
# EVOLVE-BLOCK-END
