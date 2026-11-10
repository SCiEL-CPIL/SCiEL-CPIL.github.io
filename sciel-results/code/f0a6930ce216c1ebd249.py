import numpy as np
import time
from scipy.optimize import minimize

def _compute_area(positions, n=64, n_heights=20000):
    """
    Compute the union area of the triangles.
    positions: array of shape (n,)
    """
    a = (np.arange(n) + 1) / n
    y = (np.arange(n_heights) + 0.5) / n_heights
    w = (1 - y) / n
    
    # L[i, m] = x_i + a_i * y_m
    L = positions[:, np.newaxis] + a[:, np.newaxis] * y[np.newaxis, :]
    
    # Sort L along axis 0 for each height m
    L_sorted = np.sort(L, axis=0)
    
    # Differences between consecutive sorted L values
    diffs = np.diff(L_sorted, axis=0)
    
    # Overlap contribution: min(diff, w)
    overlaps = np.minimum(diffs, w[np.newaxis, :])
    
    # Union length at each height: w + sum(overlaps)
    union_lengths = w + np.sum(overlaps, axis=0)
    
    # Area is the mean of union lengths
    area = np.mean(union_lengths)
    
    return area

def kakeya_needle_2d_positions64() -> np.ndarray:
    n = 64
    start_time = time.time()
    deadline = start_time + 590  # Safety margin before 600s limit

    a = (np.arange(n) + 1) / n
    
    # Initialization: Fan shape
    # x_i = -a_i + 0.5
    # This aligns the top vertices at (0.5, 1)
    best_positions = -a + 0.5
    best_area = _compute_area(best_positions, n, 20000)
    
    # Nelder-Mead Optimization
    # We use a high number of iterations and a tight tolerance
    # options: maxiter, fatol, xatol, disp
    # The function is non-smooth, so Nelder-Mead is suitable.
    
    # To speed up, we might use a coarser grid for optimization?
    # No, the score depends on the fine grid.
    # But Nelder-Mead might take many iterations.
    # Let's run it for a fixed time or iterations.
    
    current_pos = best_positions.copy()
    
    # Run Nelder-Mead
    # We need to be careful with the number of iterations.
    # Each eval takes ~5-10ms.
    # 600s / 0.01s = 60000 evals.
    # Nelder-Mead on 64 dims might need 1000-5000 evals to converge to a local min.
    
    try:
        result = minimize(
            _compute_area, 
            current_pos, 
            args=(n, 20000),
            method='Nelder-Mead',
            options={
                'maxiter': 5000,
                'fatol': 1e-8,
                'xatol': 1e-8,
                'disp': False,
                'adaptive': True
            }
        )
        
        if result.fun < best_area:
            best_area = result.fun
            best_positions = result.x.copy()
            
    except Exception as e:
        pass # If it fails, stick with current best
        
    # Check time
    if time.time() > deadline - 30:
        if not np.all(np.isfinite(best_positions)):
            return best_positions.astype(np.float64)
        return best_positions.astype(np.float64)
        
    # Polishing with Coordinate Descent
    # Simple hill climbing on each coordinate
    step_size = 0.01
    rng = np.random.RandomState(42)
    
    # We can do a few passes of coordinate descent
    # For each i, try small moves
    
    current_pos = best_positions.copy()
    current_area = best_area
    
    for _ in range(20): # 20 passes
        if time.time() > deadline - 10:
            break
        for i in range(n):
            # Try moving i
            # Binary search for best step?
            # Or just random small steps
            
            # Let's do a simple local search for x_i
            # Try +/- step_size
            best_local_pos = current_pos.copy()
            best_local_area = current_area
            
            # Try a few random deltas
            for _ in range(10):
                delta = rng.uniform(-step_size, step_size)
                trial_pos = current_pos.copy()
                trial_pos[i] += delta
                trial_area = _compute_area(trial_pos, n, 20000)
                
                if trial_area < best_local_area:
                    best_local_area = trial_area
                    best_local_pos = trial_pos
            
            if best_local_area < current_area:
                current_pos = best_local_pos
                current_area = best_local_area
                if current_area < best_area:
                    best_area = current_area
                    best_positions = current_pos.copy()
            
            # Decay step size slightly
            step_size *= 0.95
            if step_size < 1e-6:
                step_size = 1e-6

    # Final check
    if not np.all(np.isfinite(best_positions)):
        return best_positions.astype(np.float64)
    
    return best_positions.astype(np.float64)

if __name__ == "__main__":
    # Quick test
    pos = kakeya_needle_2d_positions64()
    print(f"Shape: {pos.shape}")
    print(f"Area: {_compute_area(pos)}")
