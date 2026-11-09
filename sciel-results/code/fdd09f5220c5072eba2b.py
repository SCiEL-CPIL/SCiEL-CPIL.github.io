import numpy as np
import time

_START_TIME = None

def _elapsed():
    global _START_TIME
    if _START_TIME is None:
        _START_TIME = time.time()
    return time.time() - _START_TIME

def _norm(x):
    return float(np.linalg.norm(x))

def _safe_call(F, x):
    """Call F safely, returning None on failure."""
    try:
        y = F(np.asarray(x, dtype=float))
        y = np.asarray(y, dtype=float)
        if y.shape != (2,):
            return None
        if not np.all(np.isfinite(y)):
            return None
        return y
    except Exception:
        return None

def find_root_system(F, x0):
    global _START_TIME
    _START_TIME = time.time()
    deadline = _START_TIME + 115.0  # Safety margin from 120s limit
    
    x0 = np.asarray(x0, dtype=float)
    
    # 1. Global Search Phase
    # We will generate a large number of random points and evaluate them.
    # This is robust to multiple roots and bad initial guesses.
    # Budget: ~400 calls for global search, ~50 for local polish.
    
    best_x = x0.copy()
    best_norm = float('inf')
    
    # Evaluate initial point
    Fx0 = _safe_call(F, x0)
    if Fx0 is not None:
        n0 = _norm(Fx0)
        if n0 < best_norm:
            best_norm = n0
            best_x = x0.copy()
        if best_norm <= 1e-8:
            return best_x

    # Generate random candidates
    # We use a uniform distribution over a wide box. 
    # If the root is far away, we might miss it, but for "fixed systems" 
    # often roots are near origin or within reasonable bounds.
    # We'll try a few scales: 1, 10, 100, 1000.
    
    # To be efficient, we batch the random generation.
    # NumPy vectorization doesn't help with function calls (must be looped), 
    # but we can pre-generate the points.
    
    total_calls = 1
    target_calls = 400
    
    # Define scales to search
    scales = [1.0, 5.0, 20.0, 100.0, 500.0]
    
    # We will distribute the remaining budget across these scales.
    # More weight to smaller scales if we assume roots are closer.
    # But to be safe, we'll just iterate through them.
    
    # Let's just generate a big list of random points first, then shuffle?
    # No, let's do it in chunks to allow early termination if found.
    
    # We'll create a generator of random points.
    # For each scale, we'll try a few points.
    
    # Strategy: 
    # 1. Try x0 (done)
    # 2. Try points on a grid around origin? Or random?
    # Random is better for unknown landscapes.
    
    # Let's mix scales.
    # 50% of points in [-10, 10]
    # 30% in [-100, 100]
    # 20% in [-1000, 1000]
    
    # We'll generate points on the fly.
    
    while total_calls < target_calls and _elapsed() < deadline - 10.0:
        # Decide scale
        r = np.random.rand()
        if r < 0.5:
            scale = 10.0
        elif r < 0.8:
            scale = 100.0
        else:
            scale = 1000.0
            
        # Generate a random point
        # Avoid exact duplicates with x0? Not strictly necessary.
        x_candidate = np.random.uniform(-scale, scale, 2)
        
        # Check if we are too close to previous best? 
        # If we are within 1e-4 of best_x, skip to save calls?
        # Maybe not, the function might be noisy or have multiple roots nearby.
        # But if best_norm is already small, we should stop.
        
        Fx_candidate = _safe_call(F, x_candidate)
        total_calls += 1
        
        if Fx_candidate is None:
            continue
            
        n_candidate = _norm(Fx_candidate)
        
        if n_candidate < best_norm:
            best_norm = n_candidate
            best_x = x_candidate.copy()
            
            if best_norm <= 1e-8:
                return best_x

    # 2. Local Polish Phase
    # Use Broyden's method starting from best_x.
    # This should converge very quickly if the global search found a point
    # in the basin of attraction of the root.
    
    x = best_x
    Fx = _safe_call(F, x)
    if Fx is None:
        return best_x
        
    if _norm(Fx) <= 1e-8:
        return best_x

    # Broyden Setup
    h = 1e-6
    e1 = np.array([h, 0.0])
    e2 = np.array([0.0, h])
    
    # Check budget for polish
    # We have ~100 calls left.
    
    Fx1 = _safe_call(F, x + e1)
    total_calls += 1
    Fx2 = _safe_call(F, x + e2)
    total_calls += 1
    
    if Fx1 is None or Fx2 is None:
        # Fallback to larger h
        h = 1e-4
        e1 = np.array([h, 0.0])
        e2 = np.array([0.0, h])
        Fx1 = _safe_call(F, x + e1)
        total_calls += 1
        Fx2 = _safe_call(F, x + e2)
        total_calls += 1
        if Fx1 is None or Fx2 is None:
            return best_x

    J = np.zeros((2, 2))
    J[:, 0] = (Fx1 - Fx) / h
    J[:, 1] = (Fx2 - Fx) / h

    x_old = x
    Fx_old = Fx
    current_norm = _norm(Fx)
    
    max_polish_iter = 50 # We have ~100 calls, so 50 iterations is safe.
    
    for i in range(max_polish_iter):
        if total_calls >= 490: # Stop before hitting 500 hard limit? 
            # Actually, the limit is >500. So we can go up to 499.
            # But let's keep a buffer.
            if total_calls > 480:
                break
                
        if _elapsed() > deadline:
            break
            
        # Solve J * d = -Fx
        try:
            d = np.linalg.solve(J, -Fx)
        except np.linalg.LinAlgError:
            # Regularize
            try:
                d = np.linalg.solve(J + 1e-8 * np.eye(2), -Fx)
            except np.linalg.LinAlgError:
                break

        if not np.all(np.isfinite(d)):
            break
            
        # Line search
        alpha = 1.0
        found_step = False
        
        # We can afford a few line search steps here, but we are tighter on budget.
        # Let's limit line search to 5 steps max.
        for _ls in range(5):
            if total_calls > 480:
                break
            if _elapsed() > deadline:
                break
                
            x_new = x + alpha * d
            Fx_new = _safe_call(F, x_new)
            total_calls += 1
            
            if Fx_new is None:
                alpha *= 0.5
                continue
                
            norm_new = _norm(Fx_new)
            
            if norm_new < current_norm:
                # Accept
                x = x_new
                Fx = Fx_new
                current_norm = norm_new
                found_step = True
                break
            else:
                alpha *= 0.5
                
        if not found_step:
            # If we didn't find a step, maybe try a smaller step size blindly?
            # Or break.
            # If we are stuck, breaking is safer than wasting calls.
            break
            
        if current_norm <= 1e-8:
            break
            
        # Broyden Update
        s = x - x_old
        y = Fx - Fx_old
        s_dot_s = np.dot(s, s)
        if s_dot_s < 1e-16:
            break
        J = J + np.outer(y - J @ s, s) / s_dot_s
        
        x_old = x
        Fx_old = Fx

    return x
