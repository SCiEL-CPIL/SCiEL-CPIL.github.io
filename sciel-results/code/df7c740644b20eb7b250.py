import numpy as np
import time
import itertools

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 55  # 5 second buffer for safety
    
    n_points = 30
    # Pre-compute indices for all triples
    indices = list(itertools.combinations(range(n_points), 3))
    I = np.array([x[0] for x in indices])
    J = np.array([x[1] for x in indices])
    K = np.array([x[2] for x in indices])
    
    def calculate_score(points):
        X = points[:, 0]
        Y = points[:, 1]
        # Vectorized determinant calculation
        det = X[I] * (Y[J] - Y[K]) + X[J] * (Y[K] - Y[I]) + X[K] * (Y[I] - Y[J])
        abs_det = np.abs(det)
        
        # Exclude zero area
        mask = abs_det > 0
        if not np.any(mask):
            return 0
            
        non_zero_areas = abs_det[mask]
        
        # Find the most frequent value
        values, counts = np.unique(non_zero_areas, return_counts=True)
        return counts.max()

    # Initial Configuration: 6x5 Grid
    x_coords = np.linspace(-15, 15, 6)
    y_coords = np.linspace(-12, 12, 5)
    xx, yy = np.meshgrid(x_coords, y_coords)
    initial_points = np.column_stack((xx.flatten(), yy.flatten()))
    
    best_points = initial_points.copy()
    best_score = calculate_score(best_points)
    
    current_points = best_points.copy()
    current_score = best_score
    
    # Simulated Annealing
    T = 1.0 
    T_min = 0.01
    cooling = 0.9995
    
    # Random seed for reproducibility in this run, but usually we want randomness
    # np.random.seed(42) # Do not seed to allow exploration
    
    while time.time() < deadline:
        # Pick a random point to move
        idx = np.random.randint(n_points)
        
        # Generate new position
        new_x = np.random.randint(-30, 31)
        new_y = np.random.randint(-30, 31)
        
        # Check collision
        # Vectorized check against all other points
        # Create a mask of points that are NOT the one being moved
        mask = np.ones(n_points, dtype=bool)
        mask[idx] = False
        other_points = current_points[mask]
        
        # Check if new_x, new_y exists in other_points
        # (other_points[:, 0] == new_x) & (other_points[:, 1] == new_y)
        if np.any((other_points[:, 0] == new_x) & (other_points[:, 1] == new_y)):
            continue
            
        # Try the move
        old_pos = current_points[idx].copy()
        current_points[idx, 0] = new_x
        current_points[idx, 1] = new_y
        
        new_score = calculate_score(current_points)
        
        # Acceptance criterion
        delta = new_score - current_score
        if delta > 0:
            current_score = new_score
            if new_score > best_score:
                best_score = new_score
                best_points = current_points.copy()
        else:
            if T > T_min:
                prob = np.exp(delta / T)
                if np.random.rand() < prob:
                    current_score = new_score
                else:
                    current_points[idx, 0] = old_pos[0]
                    current_points[idx, 1] = old_pos[1]
            else:
                current_points[idx, 0] = old_pos[0]
                current_points[idx, 1] = old_pos[1]
                
        # Cool down
        T *= cooling
        
        # Optional: If stuck, randomize?
        # The cooling schedule is slow, so it should explore.
        
    return [tuple(p) for p in best_points]
# EVOLVE-BLOCK-END
