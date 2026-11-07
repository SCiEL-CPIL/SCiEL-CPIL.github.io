"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import math

def find_root(f, x0):
    """
    Robust root finder using a combination of secant and damped Newton steps.
    """
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin before 120s limit
    tol = 1e-9
    max_calls = 490
    
    def safe_call(x_val):
        """Call f safely, returning (value, calls_made) or (None, calls_made) on error."""
        nonlocal call_count
        try:
            val = f(x_val)
            call_count += 1
            if not math.isfinite(val):
                return None, call_count
            return val, call_count
        except Exception:
            call_count += 1
            return None, call_count
    
    x = float(x0)
    if not math.isfinite(x):
        x = 0.0
    
    call_count = 0
    best_x = x
    best_f = None
    
    # Initial evaluation
    f_curr, call_count = safe_call(x)
    if f_curr is not None:
        best_f = f_curr
        best_x = x
        if abs(f_curr) <= tol:
            return x
    
    # Set up second point for secant
    h = 1e-4 * max(1.0, abs(x))
    x_next = x + h
    f_next, call_count = safe_call(x_next)
    
    if f_next is not None:
        if abs(f_next) <= tol:
            return x_next
        if best_f is None or abs(f_next) < abs(best_f):
            best_f = f_next
            best_x = x_next
    
    # If we don't have two valid points, try a different step
    if f_next is None or f_curr is None:
        # Try smaller step
        h2 = 1e-6 * max(1.0, abs(x))
        x_alt = x + h2
        f_alt, call_count = safe_call(x_alt)
        if f_alt is not None:
            if abs(f_alt) <= tol:
                return x_alt
            if best_f is None or abs(f_alt) < abs(best_f):
                best_f = f_alt
                best_x = x_alt
            # Use x and x_alt for secant
            if f_curr is not None:
                x_next = x_alt
                f_next = f_alt
            else:
                # Can't do secant, try Newton with FD derivative
                x = x_alt
                f_curr = f_alt
                h3 = 1e-6 * max(1.0, abs(x))
                d, _ = safe_call(x + h3)
                if d is not None and f_curr is not None:
                    deriv = (d - f_curr) / h3
                    if deriv != 0 and math.isfinite(deriv):
                        x_new = x - f_curr / deriv
                        if math.isfinite(x_new):
                            x = x_new
                            f_curr, call_count = safe_call(x)
                            if f_curr is not None and abs(f_curr) <= tol:
                                return x
                            if f_curr is not None and (best_f is None or abs(f_curr) < abs(best_f)):
                                best_f = f_curr
                                best_x = x
                return best_x
    
    # Now we have x, f_curr and x_next, f_next
    # Use secant method with damping
    prev_x = x
    prev_f = f_curr
    curr_x = x_next
    curr_f = f_next
    
    for i in range(400):
        if time.time() > deadline:
            return best_x
        if call_count >= max_calls:
            return best_x
        
        # Check if current is good enough
        if curr_f is not None and abs(curr_f) <= tol:
            return curr_x
        if prev_f is not None and abs(prev_f) <= tol:
            return prev_x
        
        # Update best
        if prev_f is not None and (best_f is None or abs(prev_f) < abs(best_f)):
            best_f = prev_f
            best_x = prev_x
        if curr_f is not None and (best_f is None or abs(curr_f) < abs(best_f)):
            best_f = curr_f
            best_x = curr_x
        
        # Secant step
        if prev_f is not None and curr_f is not None and prev_f != curr_f:
            denom = curr_f - prev_f
            if denom != 0:
                x_new = curr_x - curr_f * (curr_x - prev_x) / denom
                if math.isfinite(x_new):
                    # Damping: ensure step isn't too large
                    step = abs(x_new - curr_x)
                    if step > 10 * max(1.0, abs(curr_x)):
                        # Damp the step
                        x_new = curr_x + math.copysign(min(step, 10 * max(1.0, abs(curr_x))), x_new - curr_x)
                    
                    f_new, call_count = safe_call(x_new)
                    if f_new is not None:
                        if abs(f_new) <= tol:
                            return x_new
                        if abs(f_new) < abs(curr_f):
                            # Convergence
                            prev_x, prev_f = curr_x, curr_f
                            curr_x, curr_f = x_new, f_new
                        else:
                            # Divergence: try Newton step with FD derivative
                            h_fd = 1e-6 * max(1.0, abs(curr_x))
                            f_fd, call_count = safe_call(curr_x + h_fd)
                            if f_fd is not None and curr_f is not None:
                                deriv = (f_fd - curr_f) / h_fd
                                if deriv != 0 and math.isfinite(deriv):
                                    x_newton = curr_x - curr_f / deriv
                                    if math.isfinite(x_newton):
                                        step_n = abs(x_newton - curr_x)
                                        if step_n > 5 * max(1.0, abs(curr_x)):
                                            x_newton = curr_x + math.copysign(min(step_n, 5 * max(1.0, abs(curr_x))), x_newton - curr_x)
                                        f_newton, call_count = safe_call(x_newton)
                                        if f_newton is not None:
                                            if abs(f_newton) <= tol:
                                                return x_newton
                                            if abs(f_newton) < abs(curr_f):
                                                prev_x, prev_f = curr_x, curr_f
                                                curr_x, curr_f = x_newton, f_newton
                                            else:
                                                # Both failed, shrink step
                                                curr_x = curr_x + 0.5 * (prev_x - curr_x)
                                                curr_f, call_count = safe_call(curr_x)
                                                if curr_f is not None and abs(curr_f) <= tol:
                                                    return curr_x
                                        else:
                                            # Call failed, shrink step
                                            curr_x = curr_x + 0.5 * (prev_x - curr_x)
                                            curr_f, call_count = safe_call(curr_x)
                                            if curr_f is not None and abs(curr_f) <= tol:
                                                return curr_x
                                    else:
                                        curr_x = curr_x + 0.5 * (prev_x - curr_x)
                                        curr_f, call_count = safe_call(curr_x)
                                        if curr_f is not None and abs(curr_f) <= tol:
                                            return curr_x
                                else:
                                    curr_x = curr_x + 0.5 * (prev_x - curr_x)
                                    curr_f, call_count = safe_call(curr_x)
                                    if curr_f is not None and abs(curr_f) <= tol:
                                        return curr_x
                            else:
                                curr_x = curr_x + 0.5 * (prev_x - curr_x)
                                curr_f, call_count = safe_call(curr_x)
                                if curr_f is not None and abs(curr_f) <= tol:
                                    return curr_x
                    else:
                        # Call failed, shrink step
                        curr_x = curr_x + 0.5 * (prev_x - curr_x)
                        curr_f, call_count = safe_call(curr_x)
                        if curr_f is not None and abs(curr_f) <= tol:
                            return curr_x
                else:
                    # Non-finite new x, shrink step
                    curr_x = curr_x + 0.5 * (prev_x - curr_x)
                    curr_f, call_count = safe_call(curr_x)
                    if curr_f is not None and abs(curr_f) <= tol:
                        return curr_x
            else:
                # prev_f == curr_f, try Newton
                h_fd = 1e-6 * max(1.0, abs(curr_x))
                f_fd, call_count = safe_call(curr_x + h_fd)
                if f_fd is not None and curr_f is not None:
                    deriv = (f_fd - curr_f) / h_fd
                    if deriv != 0 and math.isfinite(deriv):
                        x_newton = curr_x - curr_f / deriv
                        if math.isfinite(x_newton):
                            f_newton, call_count = safe_call(x_newton)
                            if f_newton is not None:
                                if abs(f_newton) <= tol:
                                    return x_newton
                                prev_x, prev_f = curr_x, curr_f
                                curr_x, curr_f = x_newton, f_newton
                            else:
                                curr_x = curr_x + 0.5 * (prev_x - curr_x)
                                curr_f, call_count = safe_call(curr_x)
                        else:
                            curr_x = curr_x + 0.5 * (prev_x - curr_x)
                            curr_f, call_count = safe_call(curr_x)
                    else:
                        curr_x = curr_x + 0.5 * (prev_x - curr_x)
                        curr_f, call_count = safe_call(curr_x)
                else:
                    curr_x = curr_x + 0.5 * (prev_x - curr_x)
                    curr_f, call_count = safe_call(curr_x)
        else:
            # Can't do secant, try Newton
            h_fd = 1e-6 * max(1.0, abs(curr_x))
            f_fd, call_count = safe_call(curr_x + h_fd)
            if f_fd is not None and curr_f is not None:
                deriv = (f_fd - curr_f) / h_fd
                if deriv != 0 and math.isfinite(deriv):
                    x_newton = curr_x - curr_f / deriv
                    if math.isfinite(x_newton):
                        f_newton, call_count = safe_call(x_newton)
                        if f_newton is not None:
                            if abs(f_newton) <= tol:
                                return x_newton
                            prev_x, prev_f = curr_x, curr_f
                            curr_x, curr_f = x_newton, f_newton
                        else:
                            curr_x = curr_x + 0.5 * (prev_x - curr_x)
                            curr_f, call_count = safe_call(curr_x)
                    else:
                        curr_x = curr_x + 0.5 * (prev_x - curr_x)
                        curr_f, call_count = safe_call(curr_x)
                else:
                    curr_x = curr_x + 0.5 * (prev_x - curr_x)
                    curr_f, call_count = safe_call(curr_x)
            else:
                curr_x = curr_x + 0.5 * (prev_x - curr_x)
                curr_f, call_count = safe_call(curr_x)
        
        if curr_f is not None and abs(curr_f) <= tol:
            return curr_x
    
    return best_x
# EVOLVE-BLOCK-END
