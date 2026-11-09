"""Root finding for ill-conditioned equations: secant-first hybrid with bracket fallback."""
# EVOLVE-BLOCK-START
import time
import math

def find_root(f, x0):
    """
    Find x such that |f(x)| <= 1e-8 using at most 500 evaluations of f.
    Returns the best x found (minimizing |f(x)|) within the budget.
    """
    start_time = time.time()
    deadline = start_time + 118.0
    max_evals = 500
    evals_used = 0

    def check_time():
        return time.time() < deadline

    def safe_eval(x):
        nonlocal evals_used
        if not check_time():
            return None
        if evals_used >= max_evals:
            return None
        if not math.isfinite(x):
            return None
        try:
            val = f(x)
            evals_used += 1
            if not isinstance(val, (int, float)):
                return None
            if math.isnan(val) or math.isinf(val):
                return None
            return float(val)
        except Exception:
            evals_used += 1
            return None

    # Track best point seen so far
    best_x = float(x0)
    best_fx = safe_eval(best_x)

    if best_fx is None:
        # Try a few simple perturbations if initial eval fails
        for dx in [1e-8, -1e-8, 1e-4, -1e-4, 1.0, -1.0, 10.0, -10.0]:
            if not check_time():
                break
            trial_x = float(x0) + dx
            trial_fx = safe_eval(trial_x)
            if trial_fx is not None:
                if best_fx is None or abs(trial_fx) < abs(best_fx):
                    best_x = trial_x
                    best_fx = trial_fx
                if best_fx is not None and abs(best_fx) <= 1e-8:
                    return best_x
        if best_fx is None:
            return best_x

    if abs(best_fx) <= 1e-8:
        return best_x

    # ============================================================
    # Phase 1: Secant method (1 eval per iteration, no derivative needed)
    # ============================================================
    x_prev = best_x
    fx_prev = best_fx
    # Choose second point - try multiple offsets and pick the one with best residual
    x_curr = best_x + 1e-6 * max(1.0, abs(best_x))
    fx_curr = safe_eval(x_curr)
    if fx_curr is None:
        x_curr = best_x - 1e-6 * max(1.0, abs(best_x))
        fx_curr = safe_eval(x_curr)
    if fx_curr is None:
        # Fallback: try larger perturbations
        for dx in [1e-4, -1e-4, 1e-2, -1e-2, 0.1, -0.1, 1.0, -1.0]:
            if not check_time():
                break
            x_curr = best_x + dx
            fx_curr = safe_eval(x_curr)
            if fx_curr is not None:
                break

    if fx_curr is not None:
        if abs(fx_curr) < abs(best_fx):
            best_x = x_curr
            best_fx = fx_curr
        if abs(best_fx) <= 1e-8:
            return best_x

        # Secant iterations
        for _ in range(300):
            if not check_time() or evals_used >= max_evals - 2:
                break
            if abs(fx_curr) <= 1e-8:
                best_x = x_curr
                best_fx = fx_curr
                return x_curr
            if abs(fx_prev) <= 1e-8:
                best_x = x_prev
                best_fx = fx_prev
                return x_prev

            denom = fx_curr - fx_prev
            if abs(denom) < 1e-18:
                break

            x_new = x_curr - fx_curr * (x_curr - x_prev) / denom

            if not math.isfinite(x_new):
                break

            # Safeguard: limit step
            max_step = 10.0 * max(1.0, abs(x_curr))
            if abs(x_new - x_curr) > max_step:
                x_new = x_curr - math.copysign(max_step, x_new - x_curr)

            fx_new = safe_eval(x_new)
            if fx_new is None:
                break

            if abs(fx_new) < abs(best_fx):
                best_x = x_new
                best_fx = fx_new
                if abs(best_fx) <= 1e-8:
                    return x_new

            # Shift
            x_prev, fx_prev = x_curr, fx_curr
            x_curr, fx_curr = x_new, fx_new

            # Convergence check
            if abs(x_curr - x_prev) < 1e-14 * max(1.0, abs(x_curr)):
                break

    # ============================================================
    # Phase 2: Bracket search + bisection
    # ============================================================
    if abs(best_fx) > 1e-8 and check_time() and evals_used < max_evals - 10:
        bracket_found = False
        a, b = None, None
        fa, fb = None, None

        base_scale = max(1.0, abs(best_x))
        # Try both symmetric and asymmetric bracket searches
        for scale in [1e-8, 1e-6, 1e-4, 1e-2, 1.0, 10.0, 100.0, 1000.0, 10000.0, 100000.0]:
            if not check_time() or evals_used >= max_evals - 5:
                break
            delta = scale * base_scale
            xa = best_x - delta
            xb = best_x + delta

            fa = safe_eval(xa)
            if fa is None:
                continue
            fb = safe_eval(xb)
            if fb is None:
                continue

            if fa * fb < 0:
                a, b = xa, xb
                bracket_found = True
                break
            elif abs(fa) < abs(best_fx):
                best_x = xa
                best_fx = fa
                if abs(best_fx) <= 1e-8:
                    return best_x
            elif abs(fb) < abs(best_fx):
                best_x = xb
                best_fx = fb
                if abs(best_fx) <= 1e-8:
                    return best_x

        if not bracket_found and check_time() and evals_used < max_evals - 5:
            # Try asymmetric brackets: [best_x, best_x + scale] and [best_x - scale, best_x]
            for scale in [1e-6, 1e-4, 1e-2, 1.0, 10.0, 100.0]:
                if not check_time() or evals_used >= max_evals - 5:
                    break
                delta = scale * base_scale
                # Right bracket
                xa = best_x
                fa = best_fx
                xb = best_x + delta
                fb = safe_eval(xb)
                if fb is not None and fa * fb < 0:
                    a, b = xa, xb
                    bracket_found = True
                    break
                # Left bracket
                xa = best_x - delta
                fa = safe_eval(xa)
                if fa is not None and fa * best_fx < 0:
                    a, b = xa, best_x
                    bracket_found = True
                    break

        if bracket_found and check_time() and evals_used < max_evals - 5:
            # Bisection
            for _ in range(80):
                if not check_time() or evals_used >= max_evals - 2:
                    break
                mid = 0.5 * (a + b)
                fm = safe_eval(mid)
                if fm is None:
                    break
                if abs(fm) < abs(best_fx):
                    best_x = mid
                    best_fx = fm
                if abs(fm) <= 1e-8:
                    return mid
                if fa * fm < 0:
                    b = mid
                    fb = fm
                else:
                    a = mid
                    fa = fm
                if abs(b - a) < 1e-14 * max(1.0, abs(mid)):
                    break

    # ============================================================
    # Phase 3: Golden Section Search for min |f(x)| in a local window
    # ============================================================
    if abs(best_fx) > 1e-8 and check_time() and evals_used < max_evals - 20:
        # Define a local window around best_x
        # We search for the minimum of g(x) = |f(x)| in [best_x - W, best_x + W]
        # Golden section search on g(x)
        
        def g(x):
            val = safe_eval(x)
            if val is None:
                return float('inf')
            return abs(val)
        
        # We need to be careful: g(x) calls safe_eval which counts evals.
        # Golden section search requires evaluating at 2 new points initially,
        # then 1 per iteration.
        
        # Let's try a few window sizes
        found_improvement = False
        for w_scale in [1e-2, 1e-1, 1.0, 10.0]:
            if not check_time() or evals_used >= max_evals - 10:
                break
            W = w_scale * max(1.0, abs(best_x))
            lo = best_x - W
            hi = best_x + W
            
            # Initialize golden section
            gr = (math.sqrt(5.0) - 1.0) / 2.0
            inv_gr = 1.0 / gr
            d = hi - lo
            x1 = lo + d * (1.0 - gr)
            x2 = lo + d * (1.0 - inv_gr)
            
            f1 = g(x1)
            f2 = g(x2)
            
            if f1 is float('inf') and f2 is float('inf'):
                continue
                
            # Update best
            if f1 < abs(best_fx):
                best_x = x1
                best_fx = safe_eval(x1) # Re-eval to get signed value? No, we just need to track best_x.
                # Actually, safe_eval is called inside g. We should track best_x based on g.
                # But we need signed f for secant later. Let's just update best_x and best_fx carefully.
                # To avoid double eval, we can just use the value from g if we stored it.
                # But g returns abs. We need the signed value for best_fx.
                # Let's just re-evaluate best_x at the end of the phase if it improved.
                # Or simpler: just track the x with min |f(x)|.
                # We'll update best_x and then re-evaluate best_fx at the end of the loop if needed.
                # Actually, let's just keep the x.
                pass
                
            if f2 < abs(best_fx):
                best_x = x2
                pass

            # Iteration
            for _ in range(30):
                if not check_time() or evals_used >= max_evals - 5:
                    break
                    
                if f1 < f2:
                    hi = x2
                    x2 = x1
                    f2 = f1
                    d = hi - lo
                    x1 = lo + d * (1.0 - gr)
                    f1 = g(x1)
                else:
                    lo = x1
                    x1 = x2
                    f1 = f2
                    d = hi - lo
                    x2 = lo + d * (1.0 - inv_gr)
                    f2 = g(x2)
                    
                # Check improvement
                if f1 < abs(best_fx):
                    best_x = x1
                if f2 < abs(best_fx):
                    best_x = x2
                    
                if abs(hi - lo) < 1e-10 * max(1.0, abs(best_x)):
                    break
                    
            # Re-evaluate best_x to get signed f
            # Only if we might have changed it
            # Actually, we can just update best_fx here
            current_fx = safe_eval(best_x)
            if current_fx is not None:
                if abs(current_fx) < abs(best_fx):
                    best_fx = current_fx
                if abs(best_fx) <= 1e-8:
                    return best_x

    # ============================================================
    # Phase 4: Final secant polish if close
    # ============================================================
    if abs(best_fx) <= 1e-4 and check_time() and evals_used < max_evals - 5:
        x = best_x
        fx = best_fx
        for _ in range(50):
            if not check_time() or evals_used >= max_evals - 1:
                break
            if abs(fx) <= 1e-9:
                return x
            # Adaptive step size: smaller when close, larger when not
            h = 1e-8 * max(1.0, abs(x))
            if abs(fx) < 1e-6:
                h = 1e-10 * max(1.0, abs(x))
            x2 = x + h
            fx2 = safe_eval(x2)
            if fx2 is None:
                x2 = x - h
                fx2 = safe_eval(x2)
            if fx2 is None:
                break
            if abs(fx2 - fx) < 1e-18:
                break
            x_new = x - fx * h / (fx2 - fx)
            if not math.isfinite(x_new):
                break
            # Safeguard: don't wander too far
            max_step = 10.0 * max(1.0, abs(x))
            if abs(x_new - x) > max_step:
                x_new = x - math.copysign(max_step, x_new - x)
            fx_new = safe_eval(x_new)
            if fx_new is None:
                break
            if abs(fx_new) < abs(best_fx):
                best_x = x_new
                best_fx = fx_new
            if abs(fx_new) <= 1e-8:
                return x_new
            x = x_new
            fx = fx_new

    return best_x
# EVOLVE-BLOCK-END
