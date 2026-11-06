"""Root finding via robust multi-method approach for ill-conditioned problems."""
# EVOLVE-BLOCK-START
import time
import math

def find_root(f, x0):
    """
    Find a root of f starting near x0.
    Returns x with |f(x)| <= 1e-8 using at most 500 evaluations.
    Uses a combination of Brent's method, Newton with safeguards, and
    bisection for robustness on ill-conditioned functions.
    """
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin for 150s limit
    max_evals = 490  # Safety margin for 500 limit
    
    eval_count = [0]
    
    def safe_f(x):
        if eval_count[0] >= max_evals:
            return 0.0  # Return 0 to avoid exceptions when budget exhausted
        eval_count[0] += 1
        try:
            val = f(x)
            if not isinstance(val, (int, float)):
                val = float(val)
            if math.isnan(val) or math.isinf(val):
                return 1e100  # Large value to steer away
            return val
        except Exception:
            return 1e100
    
    x = float(x0)
    fx = safe_f(x)
    if abs(fx) <= 1e-8:
        return x
    
    # Try Brent's method first (most robust for 1D root finding)
    # We need to find a bracket [a, b] where f(a) and f(b) have opposite signs
    # Start with a small interval around x0 and expand
    
    best_x = x
    best_fx = abs(fx)
    
    # First, try to find a bracket
    a = x
    fa = fx
    step = max(1e-4, abs(x) * 1e-4)
    
    # Try expanding bracket in both directions
    bracket_found = False
    for direction in [1, -1]:
        b = x + direction * step
        fb = safe_f(b)
        if time.time() > deadline:
            break
        if abs(fb) < best_fx:
            best_fx = abs(fb)
            best_x = b
        if abs(fb) <= 1e-8:
            return b
        if fa * fb < 0:
            bracket_found = True
            break
        # Expand step
        step *= 2.0
        if step > 1e10:
            break
    
    if not bracket_found:
        # Try more aggressive expansion
        step = max(1e-2, abs(x) * 1e-2)
        for direction in [1, -1]:
            for _ in range(50):
                if time.time() > deadline:
                    break
                b = x + direction * step
                fb = safe_f(b)
                if abs(fb) < best_fx:
                    best_fx = abs(fb)
                    best_x = b
                if abs(fb) <= 1e-8:
                    return b
                if fa * fb < 0:
                    bracket_found = True
                    break
                step *= 1.5
            if bracket_found:
                break
    
    if bracket_found:
        # Use Brent's method on the bracket [a, b]
        # Implement Brent's method manually
        a_val, b_val = a, b
        fa_val, fb_val = fa, fb
        
        # Ensure |fa| <= |fb|
        if abs(fa_val) > abs(fb_val):
            a_val, b_val = b_val, a_val
            fa_val, fb_val = fb_val, fa_val
        
        c_val = a_val
        fc_val = fa_val
        d_val = b_val - a_val
        e_val = d_val
        
        for _ in range(100):
            if time.time() > deadline:
                break
            if eval_count[0] >= max_evals:
                break
            
            # Check if we've found a root
            if abs(fa_val) <= 1e-8:
                return a_val
            if abs(fb_val) <= 1e-8:
                return b_val
            
            # Invariant: fa and fb have opposite signs
            # f(a)*f(b) < 0
            
            # Bisection
            m_val = 0.5 * (a_val + b_val)
            e_val = 0.5 * (a_val - b_val)
            
            # Check for convergence
            if abs(m_val) < 1e-14 and abs(fa_val) < 1e-14:
                return a_val
            
            # Decide whether to do bisection or interpolation
            if abs(e_val) >= 1e-14:
                # Previous step was not bisection
                if abs(d_val) >= 1e-14:
                    # Previous step was not bisection and interval is not too small
                    # Try inverse quadratic interpolation
                    if abs(fa_val) != abs(fb_val):
                        # Inverse quadratic interpolation
                        ratio = fa_val / fb_val
                        if ratio != 0 and ratio != 1:
                            num = ratio * (fb_val - fa_val)
                            den = fb_val - fa_val
                            p_val = m_val + num / den
                            # Check if p is in the acceptable range
                            if (p_val > min(a_val, b_val) + 1e-14 and p_val < max(a_val, b_val) - 1e-14) or \
                               (p_val < min(a_val, b_val) - 1e-14 and p_val > max(a_val, b_val) + 1e-14):
                                fp_val = safe_f(p_val)
                                if abs(fp_val) < best_fx:
                                    best_fx = abs(fp_val)
                                    best_x = p_val
                                if abs(fp_val) <= 1e-8:
                                    return p_val
                                if fp_val * fa_val < 0:
                                    b_val, fb_val = p_val, fp_val
                                else:
                                    a_val, fa_val = p_val, fp_val
                                c_val, fc_val = b_val, fb_val
                                d_val = b_val - a_val
                                e_val = d_val
                                continue
                # Secant method
                if abs(d_val) >= 1e-14:
                    ratio = fa_val / fb_val
                    if ratio != 0:
                        p_val = m_val + (fa_val - fb_val) * (fa_val / (fb_val - fa_val)) if (fb_val - fa_val) != 0 else m_val
                        # Simpler secant: p = a - fa*(b-a)/(fb-fa)
                        if fb_val != fa_val:
                            p_val = a_val - fa_val * (b_val - a_val) / (fb_val - fa_val)
                            if (p_val > min(a_val, b_val) + 1e-14 and p_val < max(a_val, b_val) - 1e-14) or \
                               (p_val < min(a_val, b_val) - 1e-14 and p_val > max(a_val, b_val) + 1e-14):
                                fp_val = safe_f(p_val)
                                if abs(fp_val) < best_fx:
                                    best_fx = abs(fp_val)
                                    best_x = p_val
                                if abs(fp_val) <= 1e-8:
                                    return p_val
                                if fp_val * fa_val < 0:
                                    b_val, fb_val = p_val, fp_val
                                else:
                                    a_val, fa_val = p_val, fp_val
                                c_val, fc_val = b_val, fb_val
                                d_val = b_val - a_val
                                e_val = d_val
                                continue
            
            # Bisection
            fp_val = safe_f(m_val)
            if abs(fp_val) < best_fx:
                best_fx = abs(fp_val)
                best_x = m_val
            if abs(fp_val) <= 1e-8:
                return m_val
            if fp_val * fa_val < 0:
                b_val, fb_val = m_val, fp_val
            else:
                a_val, fa_val = m_val, fp_val
            c_val, fc_val = b_val, fb_val
            d_val = b_val - a_val
            e_val = d_val
        
        # Return the best found
        if abs(fa_val) <= 1e-8:
            return a_val
        if abs(fb_val) <= 1e-8:
            return b_val
        if abs(fa_val) < abs(fb_val):
            return a_val
        return b_val
    else:
        # No bracket found; use Newton's method with safeguards
        # or just return the best point found
        # Try Newton with finite differences
        x_new = x
        for _ in range(100):
            if time.time() > deadline:
                break
            if eval_count[0] >= max_evals:
                break
            
            fx_new = safe_f(x_new)
            if abs(fx_new) < best_fx:
                best_fx = abs(fx_new)
                best_x = x_new
            if abs(fx_new) <= 1e-8:
                return x_new
            
            h = 1e-6 * max(1.0, abs(x_new))
            fx_plus = safe_f(x_new + h)
            d = (fx_plus - fx_new) / h
            if abs(d) < 1e-14:
                break
            
            x_next = x_new - fx_new / d
            
            # Safeguard: keep step reasonable
            if abs(x_next - x_new) > 10 * max(1.0, abs(x_new)):
                x_next = x_new + 10 * max(1.0, abs(x_new)) * (1 if x_next > x_new else -1)
            
            if x_next == x_new:
                break
            
            x_new = x_new + 0.5 * (x_next - x_new)  # Damped Newton
        
        return best_x

    return best_x
# EVOLVE-BLOCK-END
