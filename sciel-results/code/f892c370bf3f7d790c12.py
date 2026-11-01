import numpy as np
import time
from scipy.spatial import cKDTree

N = 100

def construct():
    start_time = time.time()
    deadline = 145.0  # Safety margin (5s before 150s limit)
    
    def get_radius(centers):
        """Compute the maximum common radius for given centers."""
        if not np.all(np.isfinite(centers)):
            return 0.0
        if np.any(centers > 0.5) or np.any(centers < -0.5):
            return 0.0
        # Wall distance
        wall_dists = 0.5 - np.abs(centers)
        r_wall = np.min(wall_dists)
        # Pairwise distance using KDTree for efficiency
        tree = cKDTree(centers)
        # Get all pairwise distances (upper triangle)
        dists, _ = tree.query(centers, k=2)  # k=2 includes self, so take second column
        r_pair = np.min(dists[:, 1]) / 2.0
        return min(r_wall, r_pair)
    
    def lattice_annealing(centers, time_limit, initial_temp=0.05, cooling=0.999):
        """
        Discrete combinatorial annealing using Metropolis-Hastings on a lattice gas.
        Treats sphere centers as particles that can move to neighboring lattice sites.
        """
        centers = centers.copy()
        current_r = get_radius(centers)
        best_centers = centers.copy()
        best_r = current_r
        temp = initial_temp
        
        # Define a fine lattice for discrete moves
        # The cube is [-0.5, 0.5]^3, so we divide into cells
        n_cells = 20  # 20x20x20 = 8000 possible positions
        cell_size = 1.0 / n_cells
        lattice_points = []
        for i in range(n_cells):
            for j in range(n_cells):
                for k in range(n_cells):
                    x = -0.5 + (i + 0.5) * cell_size
                    y = -0.5 + (j + 0.5) * cell_size
                    z = -0.5 + (k + 0.5) * cell_size
                    lattice_points.append([x, y, z])
        lattice_points = np.array(lattice_points)
        
        # Map each center to its nearest lattice point
        def snap_to_lattice(c):
            """Snap a center to the nearest lattice point."""
            idx = np.argmin(np.sum((lattice_points - c)**2, axis=1))
            return lattice_points[idx]
        
        # Snap initial centers to lattice
        for i in range(N):
            centers[i] = snap_to_lattice(centers[i])
        
        current_r = get_radius(centers)
        best_r = current_r
        best_centers = centers.copy()
        
        # Precompute neighbor lists for each lattice point
        # For efficiency, we'll use a KDTree on lattice points
        lat_tree = cKDTree(lattice_points)
        
        while time.time() < time_limit and temp > 1e-4:
            # Choose which sphere to move
            idx = np.random.randint(N)
            
            # Choose a move type:
            # 1. Move to a random nearby lattice point (80% probability)
            # 2. Swap with another sphere (20% probability)
            
            if np.random.random() < 0.8:
                # Move to nearby lattice point
                # Find lattice points within a certain radius
                max_move_dist = cell_size * 3
                nearby = lat_tree.query_ball_point(centers[idx], r=max_move_dist)
                
                if len(nearby) > 1:
                    # Choose a different nearby point
                    choices = [p for p in nearby if p != np.argmin(np.sum((lattice_points - centers[idx])**2, axis=1))]
                    if choices:
                        new_pos_idx = np.random.choice(choices)
                        new_centers = centers.copy()
                        new_centers[idx] = lattice_points[new_pos_idx]
                        
                        new_r = get_radius(new_centers)
                        delta = new_r - current_r
                        
                        # Metropolis acceptance
                        if delta > 0 or np.random.random() < np.exp(delta / (temp + 1e-10)):
                            current_r = new_r
                            centers = new_centers
                            if current_r > best_r:
                                best_r = current_r
                                best_centers = centers.copy()
            else:
                # Swap with another sphere
                other_idx = np.random.randint(N)
                if other_idx != idx:
                    new_centers = centers.copy()
                    new_centers[idx], new_centers[other_idx] = centers[other_idx].copy(), centers[idx].copy()
                    
                    new_r = get_radius(new_centers)
                    delta = new_r - current_r
                    
                    # Metropolis acceptance
                    if delta > 0 or np.random.random() < np.exp(delta / (temp + 1e-10)):
                        current_r = new_r
                        centers = new_centers
                        if current_r > best_r:
                            best_r = current_r
                            best_centers = centers.copy()
            
            # Cool down
            temp *= cooling
        
        return best_centers, best_r
    
    def hybrid_optimize(centers, time_limit):
        """
        Hybrid approach: alternate between lattice annealing and continuous refinement.
        """
        centers = centers.copy()
        current_r = get_radius(centers)
        best_centers = centers.copy()
        best_r = current_r
        
        # Phase 1: Lattice annealing
        lat_time = time_limit * 0.6
        lat_limit = time.time() + lat_time
        lat_centers, lat_r = lattice_annealing(centers, lat_limit, initial_temp=0.04, cooling=0.9995)
        
        if lat_r > best_r:
            best_r = lat_r
            best_centers = lat_centers
        
        # Phase 2: Continuous local refinement
        cont_time = time_limit - lat_time
        cont_limit = time.time() + cont_time
        
        # Simple gradient-free refinement: perturb and accept if better
        centers = best_centers.copy()
        r = best_r
        
        while time.time() < cont_limit:
            # Random perturbation
            n_perturb = np.random.choice([1, 2, 3], p=[0.5, 0.3, 0.2])
            indices = np.random.choice(N, n_perturb, replace=False)
            
            step_size = 0.02
            new_centers = centers.copy()
            for idx in indices:
                new_centers[idx] += np.random.normal(0, step_size, 3)
            
            new_centers = np.clip(new_centers, -0.499, 0.499)
            
            new_r = get_radius(new_centers)
            
            if new_r > r:
                r = new_r
                centers = new_centers
                if r > best_r:
                    best_r = r
                    best_centers = centers.copy()
        
        return best_centers, best_r
    
    # Generate initial configurations
    np.random.seed(42)
    
    best_centers = None
    best_radius = 0.0
    
    # Configuration 1: Regular grid
    nx, ny, nz = 5, 5, 4
    xs = np.linspace(-0.42, 0.42, nx)
    ys = np.linspace(-0.42, 0.42, ny)
    zs = np.linspace(-0.42, 0.42, nz)
    X, Y, Z = np.meshgrid(xs, ys, zs, indexing='ij')
    base_centers = np.column_stack((X.ravel(), Y.ravel(), Z.ravel()))
    
    # Configuration 2: Face-centered cubic-like
    a = 0.7  # lattice constant
    fcc_points = []
    for i in range(-1, 2):
        for j in range(-1, 2):
            for k in range(-1, 2):
                fcc_points.append([i*a, j*a, k*a])
                fcc_points.append([i*a, j*a + a/2, k*a + a/2])
                fcc_points.append([i*a + a/2, j*a, k*a + a/2])
                fcc_points.append([i*a + a/2, j*a + a/2, k*a])
    
    fcc_array = np.array(fcc_points)
    mask = np.all(np.abs(fcc_array) <= 0.45, axis=1)
    fcc_array = fcc_array[mask]
    if len(fcc_array) >= N:
        idx = np.random.choice(len(fcc_array), N, replace=False)
        fcc_centers = fcc_array[idx]
    else:
        fcc_centers = base_centers.copy()
    
    # Try multiple initializations
    initializations = []
    
    # Grid with various jitter levels
    for jitter in [0.0, 0.02, 0.04, 0.06]:
        centers = base_centers + np.random.normal(0, jitter, base_centers.shape)
        centers = np.clip(centers, -0.49, 0.49)
        initializations.append(centers)
    
    # FCC with jitter
    for jitter in [0.0, 0.02, 0.04]:
        centers = fcc_centers + np.random.normal(0, jitter, fcc_centers.shape)
        centers = np.clip(centers, -0.49, 0.49)
        initializations.append(centers)
    
    # Random initialization
    for _ in range(3):
        centers = np.random.uniform(-0.45, 0.45, (N, 3))
        initializations.append(centers)
    
    # Phase 1: Hybrid optimization on multiple initializations
    for idx, init_centers in enumerate(initializations):
        if time.time() > deadline - 20:
            break
        
        remaining = deadline - 20 - time.time()
        if remaining < 10:
            break
        
        # Allocate time based on initialization quality
        r_init = get_radius(init_centers)
        
        if r_init > 0.08:
            time_budget = remaining * 0.5
        elif r_init > 0.06:
            time_budget = remaining * 0.4
        else:
            time_budget = remaining * 0.3
        
        hybrid_limit = time.time() + min(time_budget, 25)
        
        hybrid_centers, hybrid_r = hybrid_optimize(init_centers, min(time_budget, 25))
        
        if hybrid_r > best_radius:
            best_radius = hybrid_r
            best_centers = hybrid_centers
    
    # Ensure valid output
    if best_centers is None or not np.all(np.isfinite(best_centers)):
        return base_centers
    
    # Final clip to ensure validity
    best_centers = np.clip(best_centers, -0.5, 0.5)
    
    return best_centers
