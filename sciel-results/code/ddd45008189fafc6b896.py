"""Root finder for smooth polynomial equations with call-budget awareness."""
import time
import math

_DEADLINE_START = time.time()
_WALL_LIMIT = 115.0  # safety margin before 150s kill
_SAFETY_MARGIN = 5.0


def _time_left():
    return _WALL_LIMIT - _SAFETY_MARGIN - (time.time() - _DEADLINE_START)


def _is_finite(x):
    return x is not None and isinstance(x, (int, float)) and math.isfinite(x)


def find_root(f, x0):
    """
    Find x such that |f(x)| <= 1e-8 using a combination of:
    1. Newton's method with central finite-difference derivative (fastest for smooth polynomials)
    2. Secant method (robust fallback if derivative is flat or Newton diverges)
    3. Bisection fallback for guaranteed convergence if a bracket is found
    4. Final polish pass with tighter tolerance
    
    Returns the best x found (minimizing |f(x)|) within the call budget.
    """
    tol = 1e-8
    max_total_calls = 490  # stay well under 500
    
    def safe_eval(x):
        """Evaluate f(x) safely, returning (fx, ok)."""
        try:
            fx = f(x)
        except Exception:
            return None, False
        if not _is_finite(fx):
            return None, False
        return fx, True
    
    # Track all evaluations to count calls
    call_count = 0
    
    def counted_eval(x):
        nonlocal call_count
        call_count += 1
        return safe_eval(x)
    
    # First, evaluate at x0
    x = float(x0)
    fx, ok = counted_eval(x)
    if not ok:
        # Try nearby points
        for dx in [1e-4, -1e-4, 1e-3, -1e-3, 1e-2, -1e-2, 0.1, -0.1, 1.0, -1.0]:
            x_try = x + dx
            fx, ok = counted_eval(x_try)
            if ok:
                x = x_try
                break
        if not ok:
            return x  # return best effort
    
    if abs(fx) <= tol:
        return x
    
    best_x = x
    best_fx = fx
    
    # Strategy 1: Newton's method with central finite-difference derivative
    # This is the fastest converging method for smooth polynomials.
    x_newton = x
    for _ in range(20):
        if call_count >= max_total_calls:
            break
        if _time_left() < 0:
            break
        
        # We already have fx for x_newton from previous step or initial eval
        # But to be safe and simple, let's re-eval if we moved, or use cached.
        # In this loop, we maintain x_newton and its fx.
        
        if abs(fx) <= tol:
            return x_newton
            
        h = 1e-6 * max(1.0, abs(x_newton))
        
        # Central difference derivative
        fx_plus, ok1 = counted_eval(x_newton + h)
        if not ok1:
            break
        fx_minus, ok2 = counted_eval(x_newton - h)
        if not ok2:
            break
        
        d = (fx_plus - fx_minus) / (2.0 * h)
        
        if abs(d) < 1e-15:
            # Derivative is effectively zero. Newton is not applicable.
            # Break to try Secant or Bisection.
            break
        
        x_next = x_newton - fx / d
        
        if not _is_finite(x_next):
            break
            
        # Check if step is reasonable to avoid divergence
        # If step is huge, we might be far from root or derivative is tiny.
        # For smooth polynomials, Newton usually behaves well unless near a local extremum.
        
        fx_next, ok = counted_eval(x_next)
        if not ok:
            break
            
        if abs(fx_next) < abs(best_fx):
            best_fx = fx_next
            best_x = x_next
            
        # Update for next iteration
        x_newton = x_next
        fx = fx_next
        
    if abs(best_fx) <= tol:
        return best_x

    # Strategy 2: Secant method
    # Use the best point found so far and a neighbor to start secant.
    # If Newton failed due to flat derivative, Secant might also struggle,
    # but it uses two points so it's more robust to local flatness than 1-point Newton.
    
    # Initialize secant points. Use best_x and a slightly perturbed version.
    x_prev = best_x
    fx_prev = best_fx
    
    # Try to find a second point that is distinct.
    h_sec = 1e-4 * max(1.0, abs(x_prev))
    x_curr = x_prev + h_sec
    fx_curr, ok = counted_eval(x_curr)
    
    if not ok:
        x_curr = x_prev - h_sec
        fx_curr, ok = counted_eval(x_curr)
        
    if not ok:
        # If we can't even evaluate a neighbor, we are in trouble.
        # Try larger steps.
        for dx in [1e-2, -1e-2, 0.1, -0.1, 1.0, -1.0]:
            x_try = x_prev + dx
            fx_try, ok_try = counted_eval(x_try)
            if ok_try:
                x_curr = x_try
                fx_curr = fx_try
                break
        if not ok:
            return best_x
            
    if abs(fx_curr) <= tol:
        return x_curr
        
    secant_iters = 0
    max_secant_iters = 20
    while secant_iters < max_secant_iters and call_count < max_total_calls:
        if _time_left() < 0:
            break
        if fx_curr is None or fx_prev is None:
            break
        if abs(fx_curr) <= tol:
            return x_curr
        if abs(fx_prev) <= tol:
            return x_prev
        
        if abs(fx_curr - fx_prev) < 1e-15:
            break
        
        x_next = x_curr - fx_curr * (x_curr - x_prev) / (fx_curr - fx_prev)
        
        if not _is_finite(x_next):
            break
        
        fx_next, ok = counted_eval(x_next)
        if not ok:
            break
        
        if abs(fx_next) < abs(best_fx):
            best_fx = fx_next
            best_x = x_next
        
        # Update
        x_prev = x_curr
        fx_prev = fx_curr
        x_curr = x_next
        fx_curr = fx_next
        secant_iters += 1
    
    if abs(best_fx) <= tol:
        return best_x
    
    # Strategy 3: Bisection - need to find a bracket
    # Try to find a sign change around best_x
    x_lo = best_x
    x_hi = best_x
    f_lo = best_fx
    
    # Expand search range to find sign change
    step = 1e-2
    found_bracket = False
    for _ in range(50):
        if call_count >= max_total_calls:
            break
        if _time_left() < 0:
            break
        
        # Try expanding in both directions from best_x
        x_try_left = best_x - step
        f_try_left, ok_left = counted_eval(x_try_left)
        if ok_left:
            if best_fx * f_try_left < 0:
                x_lo = x_try_left
                x_hi = best_x
                f_lo = f_try_left
                f_hi = best_fx
                found_bracket = True
                break
            if abs(f_try_left) < abs(best_fx):
                best_x = x_try_left
                best_fx = f_try_left

        x_try_right = best_x + step
        f_try_right, ok_right = counted_eval(x_try_right)
        if ok_right:
            if best_fx * f_try_right < 0:
                x_lo = best_x
                x_hi = x_try_right
                f_lo = best_fx
                f_hi = f_try_right
                found_bracket = True
                break
            if abs(f_try_right) < abs(best_fx):
                best_x = x_try_right
                best_fx = f_try_right

        step *= 2.0
    
    if found_bracket:
        # Bisection
        for _ in range(60):
            if call_count >= max_total_calls:
                break
            if _time_left() < 0:
                break
            
            x_mid = (x_lo + x_hi) / 2.0
            f_mid, ok = counted_eval(x_mid)
            if not ok:
                break
            
            if abs(f_mid) <= tol:
                return x_mid
            
            if abs(f_mid) < abs(best_fx):
                best_fx = f_mid
                best_x = x_mid
            
            if f_lo * f_mid <= 0:
                x_hi = x_mid
                f_hi = f_mid
            else:
                x_lo = x_mid
                f_lo = f_mid
    
    # Strategy 4: Final polish pass with tighter tolerance
    # Use secant/Newton with tighter tol to squeeze out last bits of accuracy
    polish_tol = 1e-12
    x_polish = best_x
    for _ in range(10):
        if call_count >= max_total_calls:
            break
        if _time_left() < 0:
            break
        
        fx_p, ok = counted_eval(x_polish)
        if not ok:
            break
        
        if abs(fx_p) <= polish_tol:
            best_x = x_polish
            best_fx = fx_p
            break
        
        h = 1e-8 * max(1.0, abs(x_polish))
        fx_p_plus, ok1 = counted_eval(x_polish + h)
        if not ok1:
            break
        fx_p_minus, ok2 = counted_eval(x_polish - h)
        if not ok2:
            break
        
        d = (fx_p_plus - fx_p_minus) / (2.0 * h)
        if abs(d) < 1e-15:
            break
        
        x_next = x_polish - fx_p / d
        if not _is_finite(x_next):
            break
        
        x_polish = x_next
        
    return best_x
