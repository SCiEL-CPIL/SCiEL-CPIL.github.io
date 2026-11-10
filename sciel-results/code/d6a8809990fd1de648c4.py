# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import differential_evolution

def _union_length(intervals):
    """
    Calculate the total length of the union of a list of intervals.
    intervals: list of (start, end) tuples.
    """
    if not intervals:
        return 0.0
    # Sort by start time
    sorted_intervals = sorted(intervals, key=lambda x: x[0])
    
    total_length = 0.0
    current_start, current_end = sorted_intervals[0]
    
    for start, end in sorted_intervals[1:]:
        if start <= current_end:
            # Overlapping or touching
            current_end = max(current_end, end)
        else:
            # Disjoint
            total_length += current_end - current_start
            current_start, current_end = start, end
            
    total_length += current_end - current_start
    return total_length

def _cost_volume(X, Y):
    """
    Estimate the volume integral \int_0^1 L_x(c) L_y(c) dc.
    X, Y: arrays of 8 base positions.
    """
    N = 8
    indices = np.arange(N)
    # Sample c values for integration
    c_samples = np.linspace(0, 1, 50)
    
    total_cost = 0.0
    for c in c_samples:
        # X intervals
        starts_x = X + c * indices / N
        ends_x = starts_x + 1.0 / N
        intervals_x = list(zip(starts_x, ends_x))
        len_x = _union_length(intervals_x)
        
        # Y intervals
        starts_y = Y + c * indices / N
        ends_y = starts_y + 1.0 / N
        intervals_y = list(zip(starts_y, ends_y))
        len_y = _union_length(intervals_y)
        
        total_cost += len_x * len_y
        
    return total_cost / len(c_samples)

def kakeya_needle_3d8() -> np.ndarray:
    N = 8
    
    # Initial guess: shift so that at c=1, all intervals overlap at [1/8, 2/8]
    # x_i + i/8 + 1/8 = 1/8 => x_i = -i/8
    X_init = -np.arange(N) / N
    Y_init = -np.arange(N) / N
    
    # Bounds for optimization
    # We want to keep the bounding box small.
    # x_i should be in a range such that the union doesn't spread too far.
    # Let's allow x_i to vary in [-1, 1]
    bounds = [(-1.0, 1.0)] * (2 * N)
    
    # Flatten X and Y into a single vector for differential_evolution
    def _cost_flat(params):
        X = params[:N]
        Y = params[N:]
        return _cost_volume(X, Y)
    
    try:
        res = differential_evolution(
            _cost_flat, 
            bounds, 
            seed=42, 
            maxiter=200, 
            tol=1e-8, 
            popsize=15,
            polish=True
        )
        if res.success:
            params_opt = res.x
        else:
            # Fallback to initial guess if DE fails
            params_opt = np.concatenate([X_init, Y_init])
    except Exception:
        params_opt = np.concatenate([X_init, Y_init])
        
    X_opt = params_opt[:N]
    Y_opt = params_opt[N:]
    
    # Construct the 64 positions
    # i = k // 8, j = k % 8
    i_indices = np.repeat(np.arange(N), N)
    j_indices = np.tile(np.arange(N), N)
    
    base_x = X_opt[i_indices]
    base_y = Y_opt[j_indices]
    
    positions = np.stack([base_x, base_y], axis=1)
    return positions
# EVOLVE-BLOCK-END
