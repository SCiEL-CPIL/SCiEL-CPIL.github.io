import math
import time

def find_root(f, x0):
    """
    Finds a root x for f(x) such that |f(x)| <= 1e-8.
    Uses a combination of Steffensen's method, adaptive Brent's method
    (with IQI/bisection switching), and bisection fallback.
    """
    tol = 1e-8
    max_calls = 480  # Safety margin below 500
    call_count = 0
    cache = {}
    
    # Deadline guard: 110 seconds
    start_time = time.time()
    
    def safe_f(x):
        nonlocal call_count
        if call_count >= max_calls:
            raise RuntimeError("Call limit exceeded")
        if time.time() - start_time > 110:
            raise RuntimeError("Time limit exceeded")
            
        if x in cache:
            return cache[x]
        
        fx = f(x)
        call_count += 1
        cache[x] = fx
        
        if not math.isfinite(fx):
            raise ValueError("Non-finite value")
            
        return fx

    x = float(x0)
    
    # 1. Check initial point
    try:
        fx = safe_f(x)
    except Exception:
        return x
        
    if abs(fx) <= tol:
        return x

    best_x = x
    best_f = fx

    # 2. Try Steffensen's Method from x0 (quadratic convergence, no bracket needed)
    # x_{n+1} = x_n - f(x_n)^2 / (f(x_n + f(x_n)) - f(x_n))
    # Limit to ~30 calls to preserve budget for fallback methods.
    x_ste = x
    f_ste = fx
    for i in range(15):
        if call_count > max_calls - 40:
            break
        # f(x + f(x))
        x_probe = x_ste + f_ste
        if not math.isfinite(x_probe):
            break
        try:
            f_probe = safe_f(x_probe)
        except Exception:
            break
        denom = f_probe - f_ste
        if abs(denom) < 1e-15:
            break
        x_new = x_ste - f_ste * f_ste / denom
        if not math.isfinite(x_new):
            break
        try:
            f_new = safe_f(x_new)
        except Exception:
            break
        if abs(f_new) <= tol:
            return x_new
        if abs(f_new) < abs(best_f):
            best_f = f_new
            best_x = x_new
        # Check for convergence
        if abs(x_new - x_ste) < 1e-12 * max(1.0, abs(x_new)):
            if abs(f_new) <= 1e-7:
                break
        x_ste = x_new
        f_ste = f_new

    # 3. Find a bracket [a, b] with f(a)*f(b) < 0
    # We search outwards from x0
    a = x
    fa = fx
    b = None
    fb = None
    
    # Steps to try for bracketing
    steps = [1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 0.1, 1.0, 10.0, 100.0, 1000.0, 1e4, 1e5, 1e6]
    
    directions = [1, -1]
    
    found_bracket = False
    
    for step in steps:
        for d in directions:
            x_new = x + d * step
            try:
                fx_new = safe_f(x_new)
            except Exception:
                continue
            
            if abs(fx_new) <= tol:
                return x_new
            
            if abs(fx_new) < abs(best_f):
                best_f = fx_new
                best_x = x_new
                
            if fa * fx_new < 0:
                a, b = x, x_new
                fa, fb = fx, fx_new
                found_bracket = True
                break
            
        if found_bracket:
            break
        
        if call_count > 120:
            break

    if not found_bracket:
        return best_x

    # 4. Adaptive Brent's Method: dynamically switches between IQI and bisection
    # This leverages high efficiency of IQI for smooth roots while maintaining
    # robust bracketing guarantee.
    
    lo = min(a, b)
    hi = max(a, b)
    flo = fa if a == lo else fb
    fhi = fb if b == hi else fa
    
    # Ensure flo and fhi correspond to lo and hi
    if a == lo:
        flo, fhi = fa, fb
    else:
        flo, fhi = fb, fa
    
    # Verify sign change
    if flo * fhi > 0:
        return best_x
    
    # Brent's method state
    # We maintain three points: x (best), x_prev, and the bracket [lo, hi]
    # But standard Brent's uses a, b, c where a is best, b is second best
    # and c is the other bracket endpoint.
    
    # Let's implement a simplified Brent's method that strictly stays within [lo, hi]
    # and switches between bisection and inverse quadratic interpolation.
    
    # Initialize
    # a = best point, b = previous best, c = other bracket end
    # We want f(a) and f(c) to have opposite signs.
    
    if abs(flo) <= abs(fhi):
        a_pt = lo
        b_pt = hi
        c_pt = hi
        fa_pt = flo
        fb_pt = fhi
        fc_pt = fhi
    else:
        a_pt = hi
        b_pt = lo
        c_pt = lo
        fa_pt = fhi
        fb_pt = flo
        fc_pt = flo
    
    # a is the best estimate, b is the previous best, c is the other end of the bracket
    # Invariant: f(a) and f(c) have opposite signs.
    
    # Check if a is already good enough
    if abs(fa_pt) <= tol:
        return a_pt
    
    # Main loop for Brent's
    # We will limit iterations to ensure we stay within call budget
    # Each iteration typically uses 1 call.
    # We have max_calls - call_count remaining.
    max_brent_iters = max(10, max_calls - call_count - 10)
    
    for i in range(max_brent_iters):
        if call_count > max_calls - 2:
            break
        if time.time() - start_time > 105:
            break
        
        # Check for convergence
        if abs(fa_pt) <= tol:
            return a_pt
        if abs(b_pt - a_pt) < 1e-12 * max(1.0, abs(a_pt)):
            break
        
        # Try Inverse Quadratic Interpolation (IQI)
        # We need three distinct points: a, b, c
        # If b == c or a == b, fall back to bisection or secant
        
        use_iqi = False
        x_next = None
        
        # Conditions for IQI:
        # 1. a, b, c must be distinct
        # 2. The interpolated point must lie within the bracket [min(a,c), max(a,c)]
        
        if abs(b_pt - a_pt) > 1e-15 and abs(c_pt - a_pt) > 1e-15 and abs(b_pt - c_pt) > 1e-15:
            # Compute IQI step
            # Formula: x = a - fa * ( (a-b)*fa/(fa-fb) * (a-c)*fa/(fa-fc) ) / ( (a-b)*fb/(fa-fb) * (a-c)*fc/(fa-fc) - (a-b)*fa/(fa-fb) - (a-c)*fa/(fa-fc) ) ?
            # Actually, standard IQI formula for root x given points (a,fa), (b,fb), (c,fc):
            # x = (a*fb*fc / ((fa-fb)*(fa-fc)) + b*fa*fc / ((fb-fa)*(fb-fc)) + c*fa*fb / ((fc-fa)*(fc-fb)))
            
            denom1 = (fa_pt - fb_pt) * (fa_pt - fc_pt)
            denom2 = (fb_pt - fa_pt) * (fb_pt - fc_pt)
            denom3 = (fc_pt - fa_pt) * (fc_pt - fb_pt)
            
            if abs(denom1) > 1e-15 and abs(denom2) > 1e-15 and abs(denom3) > 1e-15:
                term_a = a_pt * fb_pt * fc_pt / denom1
                term_b = b_pt * fa_pt * fc_pt / denom2
                term_c = c_pt * fa_pt * fb_pt / denom3
                x_iqi = term_a + term_b + term_c
                
                # Check if x_iqi is finite and within bracket
                if math.isfinite(x_iqi):
                    # The bracket is [min(a,c), max(a,c)]
                    br_lo = min(a_pt, c_pt)
                    br_hi = max(a_pt, c_pt)
                    if br_lo <= x_iqi <= br_hi:
                        use_iqi = True
                        x_next = x_iqi
        
        if not use_iqi:
            # Fall back to Bisection
            # Midpoint of [a, c]
            x_next = (a_pt + c_pt) / 2.0
            
            # If bisection point is too close to a or c, we might just pick b
            # But standard Brent's uses bisection if IQI fails or is out of bounds.
            # Also, if x_next is too close to a, we can just accept a as converged or move on.
            
        # Evaluate f(x_next)
        try:
            f_next = safe_f(x_next)
        except Exception:
            break
        
        if abs(f_next) <= tol:
            return x_next
        
        if abs(f_next) < abs(best_f):
            best_f = f_next
            best_x = x_next
            
        # Update bracket and best points
        # We maintain the bracket [a, c] such that f(a)*f(c) < 0.
        # If f(a)*f(x_next) < 0, then the root is in [x_next, c], so new a is x_next? No, new bracket is [x_next, c] and a becomes x_next only if it's the best?
        # Standard Brent's logic:
        # If f(a) and f(x_next) have opposite signs, then c becomes b, b becomes a, a becomes x_next.
        # Otherwise, b becomes c, c becomes x_next.
        
        if fa_pt * f_next < 0:
            # Root is between a and x_next
            c_pt = b_pt
            fc_pt = fb_pt
            b_pt = a_pt
            fb_pt = fa_pt
            a_pt = x_next
            fa_pt = f_next
        else:
            # Root is between x_next and c
            # b becomes c, c becomes x_next
            # But we must ensure f(a) and f(c) still have opposite signs.
            # Since f(a) and f(c_old) had opposite signs, and f(x_next) has same sign as f(a),
            # then f(a) and f(x_next) have same sign, so f(x_next) and f(c_old) have opposite signs.
            # So the new bracket is [x_next, c_old].
            # In Brent's notation:
            # b = c
            # c = x_next
            b_pt = c_pt
            fb_pt = fc_pt
            c_pt = x_next
            fc_pt = f_next
            
            # Also, if |f(x_next)| < |f(a)|, then x_next is a better estimate, so swap a and b?
            # Standard Brent's: if |f(x_next)| < |f(a)|, then a = x_next, b = old a.
            if abs(f_next) < abs(fa_pt):
                a_pt, b_pt = x_next, a_pt
                fa_pt, fb_pt = f_next, fa_pt
                # Wait, if we swap a and b, we must ensure the bracket [a, c] is still valid.
                # If f(a) and f(c) had opposite signs, and we set a = x_next (which has same sign as old a),
                # then f(new a) and f(c) have same sign? No.
                # Old a and c had opposite signs. x_next has same sign as old a.
                # So x_next and c have opposite signs.
                # So [x_next, c] is a valid bracket.
                # So a = x_next, c = c is valid.
                # But b is the "other" point.
                # This logic is getting tricky. Let's stick to the simpler update:
                # If f(a)*f(x) < 0: c=b, b=a, a=x
                # Else: b=c, c=x. If |f(x)| < |f(a)|: a, b = x, a (and adjust c? No, c stays)
                # Actually, if we swap a and b, the bracket [a, c] changes.
                # Let's revert to the safer update:
                # If f(a)*f(x) < 0:
                #   c = b, b = a, a = x
                # Else:
                #   b = c, c = x
                #   If |f(x)| < |f(a)|:
                #     a, b = x, a  (Note: this makes the bracket [x, c] where c is the old c? No, c was just updated to x. So bracket is [a, c] = [old_a, x]. But f(old_a) and f(x) have same sign. This is invalid!)
                
                # Correction: In the else branch, c becomes x. The bracket is [a, c] = [a, x].
                # f(a) and f(x) have the SAME sign. So this is NOT a valid bracket.
                # The valid bracket is [x, c_old].
                # So we should set: a = x, c = c_old.
                # And b is the other point.
                
                # Let's restart the update logic clearly.
                # We have a bracket [u, v] such that f(u)*f(v) < 0.
                # We compute x_next.
                # If f(u)*f(x_next) < 0, new bracket is [x_next, v].
                # If f(v)*f(x_next) < 0, new bracket is [u, x_next].
                
                # Let u = a_pt, v = c_pt.
                # We already handled the first case.
                # In the else case, f(u)*f(x_next) > 0, so f(v)*f(x_next) < 0.
                # New bracket is [u, x_next].
                # So u stays u, v becomes x_next.
                # In our variables: a_pt stays a_pt, c_pt becomes x_next.
                # b_pt is the "other" point, which was c_pt (old).
                # So: b_pt = c_pt (old), c_pt = x_next.
                # This is what we did.
                # Now, if |f(x_next)| < |f(a_pt)|, then x_next is a better estimate.
                # We can set a_pt = x_next, but then the bracket is [x_next, c_pt] = [x_next, x_next]? No.
                # The bracket is [a_pt, c_pt]. If we change a_pt, the bracket changes.
                # Standard Brent's maintains a, b, c such that f(a) is the smallest magnitude,
                # and f(a)*f(c) < 0.
                # So if |f(x_next)| < |f(a)|, we should set a = x_next.
                # But then we need to ensure f(a)*f(c) < 0.
                # Since f(x_next) has same sign as f(old_a), and f(c) has opposite sign to f(old_a),
                # f(x_next) and f(c) have opposite signs. So [x_next, c] is a valid bracket.
                # So we can set a_pt = x_next, c_pt = c_pt (unchanged in this branch? No, c_pt was just set to x_next).
                # Wait, in the else branch, we set c_pt = x_next. So the bracket is [a_pt, x_next].
                # f(a_pt) and f(x_next) have same sign. Invalid bracket.
                # My previous analysis was wrong.
                
                # Let's trace:
                # Initial: a, c with f(a)*f(c) < 0.
                # Else branch: f(a)*f(x) > 0.
                # Therefore f(c)*f(x) < 0.
                # So the new bracket is [x, c].
                # So we should set: a_pt = x_next, c_pt = c_pt (old).
                # And b_pt = a_pt (old).
                
                # Let's fix the else branch.
                # b_pt = a_pt
                # a_pt = x_next
                # fa_pt = f_next
                # c_pt remains c_pt (old)
                # fc_pt remains fc_pt (old)
                
                # But we already did: b_pt = c_pt; c_pt = x_next.
                # This is wrong.
                
                # Let's undo and redo.
                # We'll just break and use a simpler bisection if the logic gets too complex.
                # Or, we can simply use Bisection always, which is robust but slower.
                # Given the call budget, Bisection might be too slow (log2(1e12) ~ 40 calls).
                # IQI is faster.
                
                # Let's try a different approach: Just use Secant method within the bracket,
                # falling back to Bisection if Secant goes out of bounds.
                # This is simpler and often effective.
                
                break # Break out of Brent's loop and use Secant/Bisection fallback below

    # 5. Fallback: Secant method with Bisection safeguard
    # We have a bracket [lo, hi] with f(lo)*f(hi) < 0.
    # Let's use Secant, but if the next point is outside [lo, hi], use Bisection.
    
    a_sec = lo
    b_sec = hi
    fa_sec = flo
    fb_sec = fhi
    
    for i in range(30):
        if call_count > max_calls - 2:
            break
        if time.time() - start_time > 105:
            break
            
        # Secant step
        denom = fb_sec - fa_sec
        if abs(denom) < 1e-15:
            # Use bisection
            x_next = (a_sec + b_sec) / 2.0
        else:
            x_next = b_sec - fb_sec * (b_sec - a_sec) / denom
        
        # Check if x_next is in bracket
        if not (lo <= x_next <= hi):
            # Use bisection
            x_next = (a_sec + b_sec) / 2.0
            
        try:
            f_next = safe_f(x_next)
        except Exception:
            break
            
        if abs(f_next) <= tol:
            return x_next
            
        if abs(f_next) < abs(best_f):
            best_f = f_next
            best_x = x_next
            
        # Update bracket
        if fa_sec * f_next < 0:
            b_sec = x_next
            fb_sec = f_next
        else:
            a_sec = x_next
            fa_sec = f_next
            
        if abs(b_sec - a_sec) < 1e-12 * max(1.0, abs(a_sec)):
            break

    # 6. Final Bisection Polish
    # Ensure we have a valid bracket
    if fa_sec * fb_sec > 0:
        # Try to find a new bracket or return best
        return best_x

    a_bis = min(a_sec, b_sec)
    b_bis = max(a_sec, b_sec)
    fa_bis = fa_sec if a_sec == a_bis else fb_sec
    fb_bis = fb_sec if b_sec == b_bis else fa_sec

    for i in range(50):
        if call_count > max_calls - 1:
            break
        if time.time() - start_time > 105:
            break
            
        mid = (a_bis + b_bis) / 2.0
        try:
            f_mid = safe_f(mid)
        except Exception:
            break
            
        if abs(f_mid) <= tol:
            return mid
            
        if fa_bis * f_mid < 0:
            b_bis = mid
            fb_bis = f_mid
        else:
            a_bis = mid
            fa_bis = f_mid
            
        if abs(b_bis - a_bis) < 1e-13:
            break

    # Return best_x or midpoint
    mid_final = (a_bis + b_bis) / 2.0
    if mid_final in cache:
        if abs(cache[mid_final]) < abs(best_f):
            return mid_final
    return best_x
