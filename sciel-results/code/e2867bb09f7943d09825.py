"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time

N = 50

# EVOLVE-BLOCK-START
def construct():
    # Deadline guard
    start_time = time.time()
    time_limit = 115.0  # 5s buffer before 120s limit

    def calc_min_radius(centers):
        """
        Compute the largest common radius for a set of centers.
        r = min(min_pairwise_dist / 2, min_i (1 - |c_i|))
        """
        # Boundary constraint
        norms = np.linalg.norm(centers, axis=1)
        min_boundary = 1.0 - np.max(norms)

        # Pairwise distances
        # Use broadcasting for N=50 (50x50 matrix is small)
        diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
        dists = np.linalg.norm(diff, axis=2)
        np.fill_diagonal(dists, np.inf)
        min_pair_dist = np.min(dists)

        return min(min_pair_dist / 2.0, min_boundary)

    def get_hex_points(spacing):
        """Generate hexagonal lattice points inside the unit circle."""
        k = int(1.0 / spacing) + 2
        rows = [((i + 0.5 * (j % 2)) * spacing, j * spacing * np.sqrt(3.0) / 2.0)
                for i in range(-k, k + 1) for j in range(-k, k + 1)]
        points = np.array(rows)
        radius = np.hypot(points[:, 0], points[:, 1])
        # Filter points that fit in circle of radius 1 - spacing/2 (approx)
        # We will refine later, so just keep a generous set
        mask = radius <= 1.0 - spacing / 2.0
        points = points[mask]
        if len(points) >= N:
            # Sort by distance to center to get the most central ones
            idx = np.argsort(radius)
            return points[idx[:N]]
        return None

    # 1. Initialization: Hexagonal Grid
    # Start with a spacing that yields approximately N points.
    # Area of unit circle = pi. Area of hex cell ~ (2*r)^2 * sqrt(3)/2.
    # Let's just iterate spacing.
    spacing = 0.25
    best_init = None
    while spacing > 0.05:
        pts = get_hex_points(spacing)
        if pts is not None:
            best_init = pts
            break
        spacing *= 0.95

    if best_init is None:
        # Fallback to seed behavior
        spacing = 2.0 / (np.sqrt(N) + 1.0)
        while True:
            k = int(1.0 / spacing) + 2
            rows = [((i + 0.5 * (j % 2)) * spacing, j * spacing * np.sqrt(3.0) / 2.0)
                    for i in range(-k, k + 1) for j in range(-k, k + 1)]
            points = np.array(rows)
            radius = np.hypot(points[:, 0], points[:, 1])
            points, radius = points[radius <= 1.0 - spacing / 2.0], radius[radius <= 1.0 - spacing / 2.0]
            if len(points) >= N:
                best_init = points[np.argsort(radius, kind="stable")[:N]]
                break
            spacing *= 0.98
            if time.time() - start_time > 5.0:
                break

    current_centers = best_init.copy()
    current_r = calc_min_radius(current_centers)
    best_centers = current_centers.copy()
    best_r = current_r

    # 2. Simulated Annealing Optimization
    # We want to maximize r.
    # Move a point, calculate new r.
    # Accept if new_r > current_r or with probability exp((new_r - current_r) / T).
    
    # Initial temperature
    # Typical change in r is small.
    T = 0.005
    alpha = 0.99995  # Slow cooling
    T_min = 1e-6

    t_start = time.time()
    max_time = time_limit - 5.0  # 5s buffer

    # Pre-allocate arrays for speed?
    # For N=50, full recalculation is ~1-2ms in numpy.
    # 100,000 iterations * 1ms = 100s. This is tight.
    # Let's optimize the evaluation.
    
    # Instead of full N^2 distance matrix every time, we can update only the distances involving the moved point.
    # But implementing that with numpy broadcasting is tricky if we don't store the distance matrix.
    # Storing the distance matrix (50x50) and updating it is O(N) per step.
    # Let's try that.
    
    # Initialize distance matrix
    def calc_dist_matrix(centers):
        diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
        dists = np.linalg.norm(diff, axis=2)
        return dists

    dist_matrix = calc_dist_matrix(current_centers)
    np.fill_diagonal(dist_matrix, np.inf)
    
    # Helper to get min radius from dist matrix and centers
    def get_r_from_matrix(centers, dist_mat):
        norms = np.linalg.norm(centers, axis=1)
        min_boundary = 1.0 - np.max(norms)
        min_pair_dist = np.min(dist_mat)
        return min(min_pair_dist / 2.0, min_boundary)

    current_r = get_r_from_matrix(current_centers, dist_matrix)
    best_r = current_r
    best_centers = current_centers.copy()

    # Main SA Loop
    while time.time() - t_start < max_time:
        # Pick a random point to move
        i = np.random.randint(N)
        
        # Determine step size
        # Scale step size with temperature? Or fixed?
        # Fixed small step is safer.
        step_size = 0.01 * T  # Scale with T? No, T is related to energy difference.
        # Let's use a fixed step size that decreases slightly or stays constant.
        # Actually, for packing, larger steps help escape local minima.
        # Let's use step_size = 0.05 * T / 0.005 (normalized)
        # Or just a fixed small step like 0.01.
        
        # Let's try adaptive step size based on acceptance rate?
        # For simplicity, fixed step size 0.01.
        step = 0.01
        
        delta = np.random.normal(0, step, 2)
        new_centers = current_centers.copy()
        new_centers[i] += delta
        
        # Check boundary
        if np.linalg.norm(new_centers[i]) > 1.0:
            # Invalid move, reject immediately
            # Or accept with penalty? Reject is faster.
            continue
            
        # Update distance matrix for point i
        # dist_matrix[i, :] and dist_matrix[:, i]
        diff_i = new_centers[i] - current_centers
        # Wait, new_centers is a copy, so current_centers is still the old one.
        # We need to update dist_matrix based on new_centers[i] and ALL other centers.
        
        # dist_new[i, j] = || new_centers[i] - current_centers[j] || for j != i
        # dist_new[j, i] = same
        
        # Calculate new distances from i to all others
        diff_vec = new_centers[i] - current_centers
        new_dists_i = np.linalg.norm(diff_vec, axis=1)
        new_dists_i[i] = np.inf  # Self distance
        
        # Store old distances for rollback if needed
        old_dists_i = dist_matrix[i].copy()
        
        # Update dist_matrix
        dist_matrix[i] = new_dists_i
        dist_matrix[:, i] = new_dists_i
        
        # Calculate new radius
        new_r = get_r_from_matrix(new_centers, dist_matrix)
        
        # Acceptance criterion
        if new_r > current_r:
            # Always accept improvement
            current_centers = new_centers
            current_r = new_r
            if current_r > best_r:
                best_r = current_r
                best_centers = current_centers.copy()
        else:
            # Calculate probability of accepting worse solution
            # P = exp((new_r - current_r) / T)
            delta_r = new_r - current_r
            # Avoid overflow/underflow
            if delta_r > 0:
                prob = 1.0
            else:
                prob = np.exp(delta_r / T)
            
            if np.random.random() < prob:
                current_centers = new_centers
                current_r = new_r
            else:
                # Rollback distance matrix
                dist_matrix[i] = old_dists_i
                dist_matrix[:, i] = old_dists_i
        
        # Cooling
        T *= alpha
        if T < T_min:
            T = T_min

    return best_centers
# EVOLVE-BLOCK-END
