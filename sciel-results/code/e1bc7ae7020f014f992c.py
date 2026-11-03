"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 100

# EVOLVE-BLOCK-START
import numpy as np
import time

def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin
    
    # 1. Initialization: Random uniform distribution
    # We use a random start to avoid the specific local minima of grid structures
    # that deterministic or gradient-based methods might get stuck in.
    np.random.seed(42) 
    pos = np.random.uniform(-0.5, 0.5, (N, 3))
    
    # Initial score calculation
    def calc_score(p):
        # Pairwise distances
        # Using broadcasting for vectorization
        dists = np.linalg.norm(p[:, np.newaxis, :] - p[np.newaxis, :, :], axis=2)
        np.fill_diagonal(dists, np.inf)
        min_pair = np.min(dists)
        
        # Boundary distances
        # Distance to walls: 0.5 - abs(coord)
        boundary_dists = 0.5 - np.abs(p)
        min_boundary = np.min(boundary_dists)
        
        # Radius is half of the minimum clearance
        r = 0.5 * min(min_pair, min_boundary)
        return r

    current_score = calc_score(pos)
    best_score = current_score
    best_pos = pos.copy()
    
    # Simulated Annealing parameters
    T = 0.1  # Initial temperature
    T_min = 1e-6
    cooling = 0.9999
    max_iter = 500000
    
    # Pre-allocate memory for distance calculations if possible, 
    # but simple broadcasting is often fast enough for N=100 in NumPy.
    
    for i in range(max_iter):
        if time.time() > deadline:
            break
        if T < T_min:
            break
            
        # 1. Generate a move: Random perturbation of a random point
        idx = np.random.randint(0, N)
        # Step size should be adaptive or fixed relative to current T
        # A fixed step size scaled by T helps explore early and exploit later
        step_size = 0.1 * T + 0.001
        delta = np.random.normal(0, step_size, 3)
        
        new_pos = pos.copy()
        new_pos[idx] += delta
        
        # 2. Constraint Handling: Keep inside the cube
        # If a point goes out, reflect it or clamp it. Clamping is simpler and keeps it valid.
        # However, clamping might create a "sticky" boundary. 
        # For this task, being inside is a hard constraint.
        # If we clamp, the "move" is effectively smaller.
        new_pos[idx] = np.clip(new_pos[idx], -0.5, 0.5)
        
        # 3. Evaluate new score
        new_score = calc_score(new_pos)
        
        # 4. Acceptance Criterion
        delta_e = new_score - current_score
        if delta_e > 0:
            # Improvement
            pos = new_pos
            current_score = new_score
            if current_score > best_score:
                best_score = current_score
                best_pos = pos.copy()
        else:
            # Worse move: accept with probability exp(-delta_e / T)
            # Note: delta_e is negative, so -delta_e/T is positive
            # Since score is radius (positive), and we want to maximize, 
            # "energy" is -score. 
            # Standard SA minimizes energy. Here we maximize score.
            # Accept if exp((new_score - current_score) / T) > random
            # Wait, standard SA: accept worse (higher energy) with prob exp(-dE/T).
            # Here "worse" means lower score. dScore < 0.
            # We want to accept lower scores occasionally to escape minima.
            # Prob = exp( (new_score - current_score) / T )
            # If new_score < current_score, exponent is negative.
            if np.random.rand() < np.exp(delta_e / T):
                pos = new_pos
                current_score = new_score
                
        # 5. Cool down
        T *= cooling
        
    # Final check: ensure best_pos is valid and returned
    # The problem states: "A centre on or outside the cube... makes the candidate invalid."
    # Our clamping ensures |coord| <= 0.5.
    # We return best_pos.
    
    return best_pos
# EVOLVE-BLOCK-END
