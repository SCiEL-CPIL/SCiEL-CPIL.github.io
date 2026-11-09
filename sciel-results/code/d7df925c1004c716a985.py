"""Root finding for ill-conditioned equations: Global Rational Interpolation on Inverse Function."""
import time
import math
import numpy as np

# EVOLVE-BLOCK-START
def find_root(f, x0):
    start_time = time.time()
    DEADLINE = 118.0  # Safety margin below 120s limit
    MAX_EVALS = 490   # Safety margin below 500 limit
    
    eval_count = 0
    best_x = float(x0)
    best_abs_f = float('inf')
    
    # History of (f_val, x_val)
    history = []

    def safe_eval(x):
        nonlocal eval_count, best_x, best_abs_f
        if eval_count >= MAX_EVALS or (time.time() - start_time) > DEADLINE:
            return None
        try:
            val = f(x)
            if not math.isfinite(val):
                return None
            eval_count += 1
            abs_val = abs(val)
            if abs_val < best_abs_f:
                best_abs_f = abs_val
                best_x = float(x)
            # Store in history
            history.append((val, float(x)))
            return val
        except Exception:
            return None

    def check_budget():
        return eval_count >= MAX_EVALS or (time.time() - start_time) > DEADLINE

    # 1. Initial Check
    f0 = safe_eval(float(x0))
    if f0 is not None and abs(f0) <= 1e-8:
        return best_x

    # 2. Global Exploration & Interpolation Loop
    # We want to build a model of x = g(y) where y = f(x).
    # We use a rational interpolation model: g(y) = P(y) / Q(y)
    # To avoid instability, we use a weighted least squares fit on log-spaced or recent points,
    # or simply a polynomial fit if the range is small.
    # Given "ill-conditioned", the mapping y->x might be multi-valued or steep.
    # A robust approach: Fit a low-order polynomial to the most recent N points 
    # in the (y, x) space, but with safeguards.
    
    # Alternatively, use the "Inverse Interpolation" method directly:
    # If we have points (y_i, x_i), we can interpolate x as a function of y.
    # For small N, Newton-Gregory or Lagrange interpolation can be used.
    # However, if f is ill-conditioned, f(x) might be very flat or steep.
    
    # Strategy:
    # 1. Perform a local secant step to get a second point.
    # 2. Use a global "Rational" fit on the history to predict the next x.
    # 3. If the prediction is bad (large jump or out of bounds), fall back to local search.
    
    # Initial exploration to get a slope estimate
    x1 = float(x0)
    y1 = f0
    
    # Try a small perturbation to get a second point for secant
    scale = max(1.0, abs(float(x0)))
    delta = 1e-4 * scale
    x_trial = x1 + delta
    y_trial = safe_eval(x_trial)
    
    if y_trial is None:
        x_trial = x1 - delta
        y_trial = safe_eval(x_trial)
        
    if y_trial is None:
        return best_x
        
    # Now we have at least 2 points.
    # We will iterate:
    # 1. Build a model of x(y) using history.
    # 2. Predict x_next for y=0.
    # 3. Evaluate f(x_next).
    # 4. Add to history.
    # 5. Repeat.
    
    # To make the model robust, we use a weighted rational fit.
    # We focus on the most recent points to track local behavior, 
    # but keep global points to avoid local minima traps if possible.
    # Actually, for root finding, local behavior is usually more important 
    # unless we are far away. 
    # Let's use a "Global-Local" hybrid:
    # - If |y| is large, use global history to find a better region.
    # - If |y| is small, use local history for precision.
    
    # Simple robust model: 
    # Fit a quadratic x = a*y^2 + b*y + c to the last 3 points?
    # Or use a rational function x = (a*y + b) / (c*y + d) ?
    # Rational functions are better for inverse mappings of non-linear functions.
    
    # Let's implement a simple rational interpolator using 4 points (if available)
    # or polynomial if fewer.
    
    def predict_x_from_history(hist):
        """
        Uses the last few points in hist (list of (y, x)) to predict x for y=0.
        Uses a rational model x = (p0 + p1*y + p2*y^2) / (q0 + q1*y)
        We solve for coefficients using least squares if N > 4, or direct interpolation if N=4.
        For simplicity and stability, we use a weighted polynomial fit on the last 5 points
        with weights decaying for older points.
        """
        if len(hist) < 2:
            return None
        
        # Take last 5 points
        pts = hist[-5:]
        ys = np.array([p[0] for p in pts])
        xs = np.array([p[1] for p in pts])
        
        # Weights: recent points matter more
        n = len(ys)
        weights = np.array([1.0 / (i + 1) for i in range(n-1, -1, -1)])
        
        # Fit a polynomial x = w0 + w1*y + w2*y^2 + w3*y^3
        # We need to be careful with scaling of y.
        # If y is huge, y^3 will overflow or dominate.
        # Normalize y by the max abs y in the window
        y_max = np.max(np.abs(ys))
        if y_max < 1e-12:
            # All y are near 0, just average x
            return np.average(xs, weights=weights)
        
        ys_norm = ys / y_max
        
        # Design matrix for polynomial degree 3
        A = np.vander(ys_norm, 4, increasing=True)
        W = np.diag(weights)
        
        # Weighted Least Squares: (A^T W A) w = A^T W x
        ATA = A.T @ W @ A
        ATx = A.T @ W @ xs
        
        try:
            # Add small regularization for stability
            reg = 1e-12 * np.trace(ATA) / 4
            ATA += reg * np.eye(4)
            w = np.linalg.solve(ATA, ATx)
            
            # Predict x for y=0 (which is y_norm=0)
            # x_pred = w[0] + w[1]*0 + w[2]*0 + w[3]*0 = w[0]
            x_pred = w[0]
            
            # Sanity check: x_pred should be in the vicinity of the recent xs
            # If it's wildly off, it might be an extrapolation artifact.
            x_mean = np.average(xs, weights=weights)
            x_std = np.sqrt(np.average((xs - x_mean)**2, weights=weights))
            
            if abs(x_pred - x_mean) > 10 * x_std and x_std > 1e-12:
                # Extrapolation is risky, fall back to secant-like linear interpolation
                # Use last 2 points
                y1, x1 = pts[-2]
                y2, x2 = pts[-1]
                if abs(y1 - y2) > 1e-12:
                    # Linear interp for y=0
                    x_lin = x1 - y1 * (x1 - x2) / (y1 - y2)
                    return x_lin
                else:
                    return x_mean
            
            return x_pred
        except Exception:
            return None

    # Main Loop
    for i in range(400):
        if check_budget():
            break
            
        x_pred = predict_x_from_history(history)
        
        if x_pred is None:
            # Fallback: Secant step using last two points
            if len(history) >= 2:
                y1, x1 = history[-2]
                y2, x2 = history[-1]
                if abs(y1 - y2) > 1e-12:
                    x_pred = x1 - y1 * (x1 - x2) / (y1 - y2)
                else:
                    break
            else:
                break
        
        # Safeguard: Don't jump too far
        # Limit step size relative to current scale
        current_scale = max(1.0, abs(float(x0)))
        # Dynamic scale based on recent x's
        if history:
            recent_xs = [h[1] for h in history[-5:]]
            current_scale = max(current_scale, max(abs(x) for x in recent_xs))
        
        max_step = 1.0 * current_scale
        if abs(x_pred - best_x) > max_step:
            # Clip the step
            x_pred = best_x + math.copysign(max_step, x_pred - best_x)
        
        # Evaluate
        y_new = safe_eval(x_pred)
        
        if y_new is None:
            break
            
        if abs(y_new) <= 1e-8:
            return best_x
            
        # If we are stagnating (no improvement in best_abs_f for a while), 
        # we might be stuck in a local minimum of f(x)^2 or a flat region.
        # The prompt mentions "ill-conditioned", which often implies flat regions 
        # where f(x) is close to 0 but not quite, or steep regions.
        # If we are in a flat region, the polynomial fit in (y, x) space might 
        # struggle if y changes slowly.
        
        # Check for stagnation
        if i > 10 and abs(y_new) > best_abs_f * 1.1:
            # We are getting worse. Try a random perturbation to escape?
            # Or just continue, the history will update.
            pass
            
    return best_x

# EVOLVE-BLOCK-END
