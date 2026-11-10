import numpy as np
import time
from scipy.optimize import minimize

def _compute_area(positions, n=64, n_heights=20000):
    """
    Compute the union area of the triangles.
    positions: array of shape (n,)
    """
    if positions.shape != (n,):
        positions = positions.reshape(n)
        
    a = (np.arange(n) + 1) / n
    y = (np.arange(n_heights) + 0.5) / n_heights
    w = (1.0 - y) / n
    
    L = positions[:, np.newaxis] + a[:, np.newaxis] * y[np.newaxis, :]
    L_sorted = np.sort(L, axis=0)
    diffs = np.diff(L_sorted, axis=0)
    
    overlaps = np.minimum(diffs, w[np.newaxis, :])
    union_lengths = w[np.newaxis, :] + np.sum(overlaps, axis=0)
    
    area = np.mean(union_lengths)
    return area

def _subgradient(positions, n=64, n_heights=20000):
    """
    Compute a subgradient of the area.
    """
    a = (np.arange(n) + 1) / n
    y = (np.arange(n_heights) + 0.5) / n_heights
    w = (1.0 - y) / n
    
    L = positions[:, np.newaxis] + a[:, np.newaxis] * y[np.newaxis, :]
    
    sort_idx = np.argsort(L, axis=0)
    L_sorted = np.take_along_axis(L, sort_idx, axis=0)
    diffs = np.diff(L_sorted, axis=0)
    
    sg = np.zeros((n, n_heights))
    
    left_idx = sort_idx[:-1, :]
    right_idx = sort_idx[1:, :]
    
    mask = diffs < w[np.newaxis, :]
    
    rows, cols = np.where(mask)
    
    if len(rows) > 0:
        right_orig = right_idx[rows, cols]
        left_orig = left_idx[rows, cols]
        
        np.add.at(sg, (right_orig, cols), 1.0)
        np.add.at(sg, (left_orig, cols), -1.0)
        
    gradient = np.mean(sg, axis=1)
    
    return gradient

def kakeya_needle_2d_positions64() -> np.ndarray:
    """
    Optimizes the positions of the 64 triangles to minimize the union area.
    """
    n = 64
    start_time = time.time()
    deadline = start_time + 590  # 10s safety margin
    
    a = (np.arange(n) + 1) / n
    
    # Initial guesses
    # 1. Theoretical optimum pivot at 0.5
    x_init_1 = -0.5 * a
    # 2. Pivot at 0
    x_init_2 = np.zeros(n)
    # 3. Pivot at 1
    x_init_3 = -a
    # 4. Random
    np.random.seed(42)
    x_init_4 = np.random.normal(0, 0.5, n)
    
    starts = [x_init_1, x_init_2, x_init_3, x_init_4]
    
    best_pos = None
    best_area = np.inf
    
    # Use a coarse grid for optimization to allow more iterations
    n_heights_coarse = 2000
    
    def objective_coarse(x):
        return _compute_area(x, n=n, n_heights=n_heights_coarse)
    
    def grad_coarse(x):
        return _subgradient(x, n=n, n_heights=n_heights_coarse)
    
    # Evaluate starts on coarse grid
    for pos in starts:
        if time.time() > deadline - 10:
            break
        area = objective_coarse(pos)
        if area < best_area:
            best_area = area
            best_pos = pos.copy()
            
    # Optimization Phase: L-BFGS-B
    # This is much more efficient than Nelder-Mead for 64D convex problems
    current_pos = best_pos.copy()
    
    try:
        # L-BFGS-B uses numerical gradients if jac is not provided, but we have an analytic subgradient.
        # However, the subgradient is not a true gradient (it's a subgradient of a non-smooth function).
        # L-BFGS-B might struggle with non-smoothness.
        # It's safer to let L-BFGS-B use numerical gradients on the coarse grid, which are effectively smoothed by the finite difference step.
        # Or we can use the analytic gradient and hope it works.
        # Let's try with the analytic gradient first. If it fails or gets stuck, we can switch.
        
        # Actually, for non-smooth convex functions, L-BFGS-B with numerical gradients often works well because the finite difference approximates the gradient of a smoothed version.
        # Let's use numerical gradients for robustness.
        
        res = minimize(objective_coarse, current_pos, method='L-BFGS-B',
                       options={'maxiter': 5000, 'ftol': 1e-10, 'gtol': 1e-8, 'maxfun': 20000})
        
        if time.time() < deadline - 5:
            if res.fun < best_area:
                best_area = res.fun
                best_pos = res.x.copy()
                
    except Exception:
        pass
        
    # Refine with a few steps of subgradient descent on the coarse grid
    # This can help escape local minima if L-BFGS-B got stuck, though the problem is convex.
    # Since it's convex, L-BFGS-B should find the global minimum.
    # But the subgradient is non-smooth. L-BFGS-B assumes smoothness.
    # A hybrid approach:
    # 1. L-BFGS-B
    # 2. Subgradient descent with momentum
    
    current_pos = best_pos.copy()
    momentum = np.zeros(n)
    beta = 0.9
    lr = 0.05 # Larger LR for coarse grid
    
    # Run subgradient descent for a fixed number of iterations or until time limit
    n_iter = 5000
    for i in range(n_iter):
        if time.time() > deadline - 10:
            break
            
        grad = grad_coarse(current_pos)
        
        # Momentum update
        momentum = beta * momentum - lr * grad
        current_pos = current_pos + momentum
        
        area = objective_coarse(current_pos)
        
        if area < best_area:
            best_area = area
            best_pos = current_pos.copy()
            
        # Decay LR
        lr *= 0.999
        
    # Final Evaluation with Full Grid
    # The coarse grid solution might be slightly off.
    # We can try to polish it with a few iterations of coordinate descent on the full grid.
    # Or just return the coarse grid solution if it's good enough.
    # Given the convexity, the coarse grid solution should be very close to the full grid optimum.
    # However, the coarse grid area is an approximation.
    # Let's do a quick check if we have time.
    
    if time.time() < deadline - 5:
        full_area = _compute_area(best_pos, n=n, n_heights=20000)
        # If we have time, do a few steps of subgradient descent on the full grid
        # This is slow, so we limit it.
        
        current_pos = best_pos.copy()
        momentum = np.zeros(n)
        beta = 0.9
        lr = 0.01
        
        n_iter_full = 1000
        for i in range(n_iter_full):
            if time.time() > deadline - 2:
                break
                
            grad = _subgradient(current_pos, n=n, n_heights=20000)
            momentum = beta * momentum - lr * grad
            current_pos = current_pos + momentum
            
            area = _compute_area(current_pos, n=n, n_heights=20000)
            
            if area < full_area:
                full_area = area
                best_pos = current_pos.copy()
                
            lr *= 0.999

    if not np.all(np.isfinite(best_pos)):
        return np.zeros(n, dtype=np.float64)
        
    return best_pos.astype(np.float64)
