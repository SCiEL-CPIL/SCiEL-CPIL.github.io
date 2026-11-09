import numpy as np
import time

def _compute_area(x):
    """
    Compute the area of the union of the 64 triangles for given positions x.
    x: np.ndarray of shape (64,)
    Returns: float area
    """
    n = 64
    # Precompute constants outside the loop for speed if possible, 
    # but here we just use them.
    a = np.arange(1, n + 1) / n
    # y_m = (m + 0.5) / 20000
    y = (np.arange(20000) + 0.5) / 20000.0
    
    # L_i(y) = x_i + a_i * y
    # Shape: (64, 20000)
    # x[:, None] + a[:, None] * y[None, :]
    L = x[:, None] + a[:, None] * y[None, :]
    
    # Sort L along axis 0 for each y
    # This is expensive: 20000 sorts of length 64
    L_sorted = np.sort(L, axis=0)
    
    # Width w(y) = (1 - y) / 64
    w = (1.0 - y) / n
    
    # Union length calculation:
    # w + sum over consecutive pairs of min(L_{j+1} - L_j, w)
    # L_sorted has shape (64, 20000)
    # Diffs: L_sorted[1:, :] - L_sorted[:-1, :] has shape (63, 20000)
    diffs = L_sorted[1:, :] - L_sorted[:-1, :]
    
    # min(diffs, w[None, :])
    # w has shape (20000,), so broadcast to (63, 20000)
    mins = np.minimum(diffs, w[None, :])
    
    # Sum over axis 0 (the 63 intervals)
    union_lengths = w + np.sum(mins, axis=0)
    
    # Area is mean of union_lengths
    return np.mean(union_lengths)

def kakeya_needle_2d_positions64() -> np.ndarray:
    n = 64
    start_time = time.time()
    deadline = start_time + 600.0 # 30s margin
    
    # Initial guess: Keich construction
    k = int(np.log2(n))
    j = np.arange(1, k + 1)
    x_keich = np.zeros(n)
    for i in range(n):
        eps = np.array(list(str(bin(i))[2:]), dtype=int)
        eps = np.pad(eps, (k - len(eps), 0), "constant")
        x_keich[i] = np.sum((1 - j) * eps * (2.0 ** -j) / k)
    
    # Also try a few random restarts if time permits? 
    # For now, start with Keich.
    x = x_keich.copy()
    best_x = x.copy()
    best_area = _compute_area(x)
    
    # Coordinate Descent
    # We will iterate until time runs out or convergence.
    # For each coordinate, do a ternary search to minimize area.
    
    # Ternary search parameters
    # The range of x_i doesn't matter much for the relative positions, 
    # but we should search in a reasonable window around current value.
    # Since the function is convex, we can start with a wide window and shrink.
    
    def line_search(x, i):
        # Minimize _compute_area(x) w.r.t x[i]
        # Use scipy.optimize.minimize_scalar for robustness
        from scipy.optimize import minimize_scalar
        
        # Define objective for scalar
        def obj(t):
            x_new = x.copy()
            x_new[i] = t
            return _compute_area(x_new)
        
        # We need a bracket. 
        # The function is convex. 
        # Let's try a bracket around x[i] +/- 1.0
        # If the minimum is outside, we might miss it, but Keich is a good start.
        # Actually, shifting all x_i by a constant doesn't change the area.
        # So we can fix x[0] = 0 to remove the gauge freedom.
        # Let's enforce x[0] = 0 always? 
        # The problem doesn't require it, but it helps optimization.
        # However, the evaluator just checks the area.
        # Let's just search in [x[i] - 2, x[i] + 2].
        
        res = minimize_scalar(obj, bounds=(x[i] - 2.0, x[i] + 2.0), method='bounded', options={'xatol': 1e-6})
        return res.x
    
    # To speed up, we can precompute parts of the area calculation?
    # No, _compute_area is fast enough.
    
    num_iterations = 0
    while time.time() < deadline:
        improved = False
        for i in range(n):
            if time.time() > deadline - 5.0:
                break
            new_val = line_search(x, i)
            # Check if it actually improved
            # To avoid redundant calls, we can just update and check area at end of pass
            x[i] = new_val
            improved = True
        
        if not improved:
            break
        
        current_area = _compute_area(x)
        if current_area < best_area:
            best_area = current_area
            best_x = x.copy()
        
        num_iterations += 1
        
        # Check convergence
        # If change in area is very small, stop
        if num_iterations > 10 and abs(best_area - current_area) < 1e-10:
            # Maybe do a few more passes with tighter tolerance?
            # For now, break if stuck
            pass
            
    return best_x
