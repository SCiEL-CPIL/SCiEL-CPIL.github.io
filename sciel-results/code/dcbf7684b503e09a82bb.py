"""Circle packing in perimeter-4 rectangle: pack 21 circles, maximize sum of radii."""
import numpy as np
import time
from scipy.optimize import minimize

N = 21
DEADLINE = 140.0  # seconds, safety margin before 150s kill

def _validate(arr):
    """Check that circles are valid: disjoint interiors, inside bounding box with perimeter <= 4."""
    if len(arr) != N:
        return False
    # Check radii positive
    if np.any(arr[:, 2] <= 0):
        return False
    # Check pairwise disjoint (allow tiny epsilon for numerical error)
    for i in range(N):
        for j in range(i + 1, N):
            dx = arr[i, 0] - arr[j, 0]
            dy = arr[i, 1] - arr[j, 1]
            dist = np.sqrt(dx * dx + dy * dy)
            if dist < arr[i, 2] + arr[j, 2] - 1e-9:
                return False
    # Bounding box
    xs = arr[:, 0]
    ys = arr[:, 1]
    rs = arr[:, 2]
    x_min = np.min(xs - rs)
    x_max = np.max(xs + rs)
    y_min = np.min(ys - rs)
    y_max = np.max(ys + rs)
    width = x_max - x_min
    height = y_max - y_min
    perimeter = 2 * (width + height)
    if perimeter > 4.0 + 1e-9:
        return False
    return True

def _score(arr):
    """Return sum of radii if valid, else -inf."""
    if not _validate(arr):
        return -np.inf
    return np.sum(arr[:, 2])

def _local_search(arr, max_iter=200, tol=1e-10):
    """
    Local search: alternate between optimizing positions (fixed radii) and optimizing radii (fixed positions).
    Use gradient-free optimization via Nelder-Mead or L-BFGS-B.
    """
    arr = arr.copy()
    best_score = _score(arr)
    
    for it in range(max_iter):
        improved = False
        
        # Step 1: Optimize positions with fixed radii
        radii = arr[:, 2].copy()
        n = len(arr)
        
        def obj_pos(x):
            pos = x.reshape(-1, 2)
            # Check if all circles fit in some box of perimeter <= 4
            # We need to find the minimum perimeter bounding box
            xs = pos[:, 0]
            ys = pos[:, 1]
            x_min = np.min(xs - radii)
            x_max = np.max(xs + radii)
            y_min = np.min(ys - radii)
            y_max = np.max(ys + radii)
            width = x_max - x_min
            height = y_max - y_min
            perim = 2 * (width + height)
            # Penalty if perimeter exceeds 4
            if perim > 4.0:
                return 1e6 * (perim - 4.0)
            # Also penalize overlapping circles
            penalty = 0.0
            for i in range(n):
                for j in range(i + 1, n):
                    dx = pos[i, 0] - pos[j, 0]
                    dy = pos[i, 1] - pos[j, 1]
                    dist = np.sqrt(dx * dx + dy * dy)
                    min_dist = radii[i] + radii[j]
                    if dist < min_dist:
                        penalty += 1e4 * (min_dist - dist) ** 2
            return penalty
        
        x0 = arr[:, :2].flatten()
        res = minimize(obj_pos, x0, method='Nelder-Mead', 
                       options={'maxiter': 500, 'xatol': 1e-8, 'fatol': 1e-10})
        if res.fun < obj_pos(x0) - 1e-12:
            arr[:, :2] = res.x.reshape(-1, 2)
            improved = True
        
        # Step 2: Optimize radii with fixed positions
        pos = arr[:, :2].copy()
        
        def obj_rad(r):
            # r is array of radii
            xs = pos[:, 0]
            ys = pos[:, 1]
            x_min = np.min(xs - r)
            x_max = np.max(xs + r)
            y_min = np.min(ys - r)
            y_max = np.max(ys + r)
            width = x_max - x_min
            height = y_max - y_min
            perim = 2 * (width + height)
            if perim > 4.0:
                return -1e6 * (perim - 4.0)
            # Penalize overlapping
            penalty = 0.0
            for i in range(n):
                for j in range(i + 1, n):
                    dx = pos[i, 0] - pos[j, 0]
                    dy = pos[i, 1] - pos[j, 1]
                    dist = np.sqrt(dx * dx + dy * dy)
                    min_dist = r[i] + r[j]
                    if dist < min_dist:
                        penalty += 1e4 * (min_dist - dist) ** 2
            return -np.sum(r) - penalty
        
        r0 = arr[:, 2].copy()
        bounds = [(0, 1.0) for _ in range(n)]
        res2 = minimize(obj_rad, r0, method='L-BFGS-B', bounds=bounds,
                        options={'maxiter': 300, 'ftol': 1e-12})
        if res2.fun < obj_rad(r0) - 1e-12:
            arr[:, 2] = res2.x
            improved = True
        
        if not improved:
            break
    
    return arr

def construct():
    start_time = time.time()
    
    # Initial configuration: grid layout
    # For 21 circles, try a 5x5 grid with 4 removed, or 3x7, etc.
    # Let's try multiple initial configurations
    
    best_arr = None
    best_score = -np.inf
    
    # Configuration 1: 5x5 grid (25 cells, use 21)
    side = 5
    cells = []
    for i in range(side):
        for j in range(side):
            cells.append([0.5 + i * 0.5, 0.5 + j * 0.5])
    # Take first 21
    init1 = np.array(cells[:N])
    # Scale to fit in unit square initially, radii small
    init1[:, 0] = (init1[:, 0] - 0.5) / (side - 1) * 3.9  # spread in ~3.9 width
    init1[:, 1] = (init1[:, 1] - 0.5) / (side - 1) * 3.9
    init1 = np.column_stack([init1, np.full(N, 0.1)])
    
    # Configuration 2: 3x7 grid
    init2 = np.zeros((N, 3))
    for idx in range(N):
        i = idx % 7
        j = idx // 7
        init2[idx, 0] = 0.5 + i * 0.5
        init2[idx, 1] = 0.5 + j * 0.5
    init2[:, 0] = (init2[:, 0] - 0.5) / 6 * 3.9
    init2[:, 1] = (init2[:, 1] - 0.5) / 2 * 3.9
    init2[:, 2] = 0.1
    
    # Configuration 3: Random
    np.random.seed(42)
    init3 = np.zeros((N, 3))
    init3[:, 0] = np.random.uniform(0, 4, N)
    init3[:, 1] = np.random.uniform(0, 4, N)
    init3[:, 2] = 0.1
    
    inits = [init1, init2, init3]
    
    for init in inits:
        if time.time() - start_time > DEADLINE - 5:
            break
        
        # Local search
        result = _local_search(init, max_iter=100)
        score = _score(result)
        
        if score > best_score:
            best_score = score
            best_arr = result.copy()
    
    # Try more iterations if time permits
    if best_arr is not None and time.time() - start_time < DEADLINE - 10:
        for _ in range(5):
            if time.time() - start_time > DEADLINE - 5:
                break
            # Perturb and re-optimize
            perturbed = best_arr.copy()
            perturbed[:, :2] += np.random.normal(0, 0.05, (N, 2))
            perturbed[:, 2] = np.maximum(perturbed[:, 2] + np.random.normal(0, 0.01, N), 0.01)
            
            result = _local_search(perturbed, max_iter=80)
            score = _score(result)
            
            if score > best_score:
                best_score = score
                best_arr = result.copy()
    
    if best_arr is None:
        # Fallback: return something valid
        best_arr = init1.copy()
    
    # Final validation and adjustment
    if not _validate(best_arr):
        # Try to fix by scaling down radii
        for scale in [0.9, 0.8, 0.7, 0.6, 0.5]:
            test = best_arr.copy()
            test[:, 2] *= scale
            if _validate(test):
                best_arr = test
                break
    
    return best_arr
