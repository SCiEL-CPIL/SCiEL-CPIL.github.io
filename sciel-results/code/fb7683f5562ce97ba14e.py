"""Circle packing in unit square: pack 26 circles, maximize sum of radii."""
import numpy as np
import time
from scipy.optimize import minimize

N = 26
DEADLINE = 110.0  # seconds, leave margin before 120s limit


def _validate_and_compute(arr):
    """Check validity and compute sum of radii. Returns (valid, score)."""
    if arr is None or len(arr) != N:
        return False, 0.0
    x, y, r = arr[:, 0], arr[:, 1], arr[:, 2]
    # Check radii positive
    if np.any(r <= 0):
        return False, 0.0
    # Check inside unit square
    if np.any(x - r < -1e-9) or np.any(x + r > 1 + 1e-9):
        return False, 0.0
    if np.any(y - r < -1e-9) or np.any(y + r > 1 + 1e-9):
        return False, 0.0
    # Check pairwise non-overlap
    for i in range(N):
        for j in range(i + 1, N):
            dist = np.sqrt((x[i] - x[j])**2 + (y[i] - y[j])**2)
            if dist < r[i] + r[j] - 1e-9:
                return False, 0.0
    return True, float(np.sum(r))


def _grid_initial(n, cols):
    """Create a grid-based initial placement."""
    rows = int(np.ceil(n / cols))
    cells = []
    for i in range(cols):
        for j in range(rows):
            if len(cells) >= n:
                break
            x = (i + 0.5) / cols
            y = (j + 0.5) / rows
            r = 0.5 / max(cols, rows) * 0.95
            cells.append([x, y, r])
    return np.array(cells[:n])


def _random_initial(n, seed=0):
    """Random initial placement with small circles."""
    rng = np.random.default_rng(seed)
    x = rng.uniform(0.1, 0.9, n)
    y = rng.uniform(0.1, 0.9, n)
    r = rng.uniform(0.02, 0.06, n)
    return np.column_stack([x, y, r])


def _objective(params, n=N):
    """Penalty-based objective for scipy minimize (minimize negative sum of radii)."""
    x, y, r = params[:n], params[n:2*n], params[2*n:3*n]
    
    # Penalty for being outside unit square
    pen = 0.0
    pen += np.sum(np.maximum(0, -x + r)**2)
    pen += np.sum(np.maximum(0, x - r - 1)**2)
    pen += np.sum(np.maximum(0, -y + r)**2)
    pen += np.sum(np.maximum(0, y - r - 1)**2)
    
    # Penalty for overlapping
    for i in range(n):
        for j in range(i + 1, n):
            dist = np.sqrt((x[i] - x[j])**2 + (y[i] - y[j])**2)
            overlap = r[i] + r[j] - dist
            if overlap > 0:
                pen += overlap**2
    
    # Penalty for very small radii (encourage growth)
    pen += 0.01 * np.sum(np.maximum(0, 0.01 - r)**2)
    
    return -np.sum(r) + 10.0 * pen


def _local_optimize(arr0, max_iter=500, time_limit=20.0):
    """Run local optimization from initial placement."""
    start_time = time.time()
    n = len(arr0)
    x0 = arr0.flatten()
    
    best_arr = arr0.copy()
    best_valid, best_score = _validate_and_compute(best_arr)
    if not best_valid:
        # Try to fix by shrinking
        best_arr = arr0.copy()
        # Shrink all radii slightly
        best_arr[:, 2] *= 0.9
        best_valid, best_score = _validate_and_compute(best_arr)
    
    for attempt in range(3):
        if time.time() - start_time > time_limit:
            break
        
        # Add small noise for different starting points
        noise_scale = 0.005 * (attempt + 1)
        x_init = x0 + np.random.default_rng(attempt * 100 + 42).normal(0, noise_scale, len(x0))
        
        res = minimize(_objective, x_init, method='Nelder-Mead',
                      options={'maxiter': max_iter, 'xatol': 1e-6, 'fatol': 1e-6})
        
        if time.time() - start_time > time_limit:
            break
        
        arr = res.x.reshape(3, n).T
        # Ensure radii are positive
        arr[:, 2] = np.maximum(arr[:, 2], 0.001)
        
        valid, score = _validate_and_compute(arr)
        if valid and score > best_score:
            best_arr = arr.copy()
            best_score = score
    
    return best_arr, best_score


def _simulated_annealing(arr0, n_steps=5000, t_start=0.01, t_end=0.001, time_limit=30.0):
    """Simulated annealing to improve packing."""
    start_time = time.time()
    arr = arr0.copy()
    valid, current_score = _validate_and_compute(arr)
    if not valid:
        arr[:, 2] *= 0.9
        valid, current_score = _validate_and_compute(arr)
    
    best_arr = arr.copy()
    best_score = current_score
    rng = np.random.default_rng(123)
    
    n = len(arr)
    for step in range(n_steps):
        if time.time() - start_time > time_limit:
            break
        
        # Annealing schedule
        t = t_start * (t_end / t_start) ** (step / n_steps)
        
        # Random perturbation
        arr_new = arr.copy()
        # Move a random circle
        idx = rng.integers(0, n)
        arr_new[idx, 0] += rng.normal(0, 0.02)
        arr_new[idx, 1] += rng.normal(0, 0.02)
        # Try to grow a random circle
        grow_idx = rng.integers(0, n)
        arr_new[grow_idx, 2] *= (1 + rng.normal(0, 0.05))
        
        # Clamp to valid ranges
        arr_new[:, 0] = np.clip(arr_new[:, 0], 0.01, 0.99)
        arr_new[:, 1] = np.clip(arr_new[:, 1], 0.01, 0.99)
        arr_new[:, 2] = np.clip(arr_new[:, 2], 0.005, 0.5)
        
        new_valid, new_score = _validate_and_compute(arr_new)
        
        if new_valid:
            delta = new_score - current_score
            if delta > 0 or rng.random() < np.exp(delta / t):
                arr = arr_new
                current_score = new_score
                if current_score > best_score:
                    best_arr = arr.copy()
                    best_score = current_score
        else:
            # If invalid, reject (or could try to repair)
            pass
    
    return best_arr, best_score


def construct():
    """Construct 26 disjoint circles in unit square, maximizing sum of radii."""
    start_time = time.time()
    
    # Multiple strategies with different initializations
    best_arr = None
    best_score = 0.0
    
    # Strategy 1: Grid-based initialization
    for cols in [4, 5, 6]:
        if time.time() - start_time > DEADLINE - 5:
            break
        arr0 = _grid_initial(N, cols)
        arr, score = _local_optimize(arr0, max_iter=300, time_limit=15)
        if score > best_score:
            best_arr = arr
            best_score = score
    
    # Strategy 2: Random initialization with local optimization
    if time.time() - start_time < DEADLINE - 10:
        for seed in [42, 123, 456]:
            if time.time() - start_time > DEADLINE - 5:
                break
            arr0 = _random_initial(N, seed)
            arr, score = _local_optimize(arr0, max_iter=300, time_limit=15)
            if score > best_score:
                best_arr = arr
                best_score = score
    
    # Strategy 3: Simulated annealing from best found
    if best_arr is not None and time.time() - start_time < DEADLINE - 10:
        arr, score = _simulated_annealing(best_arr, n_steps=3000, time_limit=20)
        if score > best_score:
            best_arr = arr
            best_score = score
    
    # Final local polish
    if best_arr is not None and time.time() - start_time < DEADLINE - 3:
        arr, score = _local_optimize(best_arr, max_iter=200, time_limit=5)
        if score > best_score:
            best_arr = arr
            best_score = score
    
    if best_arr is None:
        # Fallback: simple grid
        best_arr = _grid_initial(N, 5)
    
    # Final validation and safety clamp
    valid, final_score = _validate_and_compute(best_arr)
    if not valid:
        # Shrink until valid
        for scale in [0.95, 0.9, 0.85, 0.8, 0.7]:
            test = best_arr.copy()
            test[:, 2] *= scale
            valid, final_score = _validate_and_compute(test)
            if valid:
                best_arr = test
                break
        else:
            # Last resort: tiny circles at grid positions
            side = int(np.ceil(np.sqrt(N)))
            cells = [((i + 0.5) / side, (j + 0.5) / side, 0.01) 
                    for i in range(side) for j in range(side)]
            best_arr = np.array(cells[:N])
    
    return best_arr
