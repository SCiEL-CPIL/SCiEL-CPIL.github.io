import math
import time
from scipy.optimize import brentq

# EVOLVE-BLOCK-START
def find_root(f, x0):
    start_time = time.time()
    deadline = start_time + 105.0  # Safety margin before 120s limit

    x = float(x0)
    fx = f(x)
    
    # Check if x0 is already a root
    if abs(fx) <= 1e-8:
        return x

    # Strategy: Find a bracket [a, b] where f(a) and f(b) have opposite signs.
    # Then use Brent's method to converge quickly.
    # We will test points x0 + delta and x0 - delta
    # delta starts small and grows exponentially
    
    # Start with a finer grid to catch high-frequency oscillations, then expand.
    # We also track the best point (min |f|) in case we fail to bracket.
    
    best_x = x
    best_f = fx
    best_abs = abs(fx)
    
    a = x
    fa = fx
    
    # A set of deltas to try. We start very small to handle high frequency, 
    # then expand. We also include some "random-ish" or non-uniform steps 
    # to avoid missing sign changes that fall exactly between grid points 
    # if the function is very steep or has narrow roots.
    
    # Note: For oscillatory functions, the period T might be small.
    # If we step by d, we need d < T/2 to be safe, but we don't know T.
    # However, if we miss a sign change, we might still find a point with low |f|.
    # But the requirement is |f(x)| <= 1e-8.
    # If we find a bracket, Brent's method is very robust.
    
    # Use a denser logarithmic sweep with factor ~3 to catch narrow oscillations.
    # We'll generate deltas from 1e-8 to 1e6 in steps of ~3x.
    
    bracket_found = False
    lo = 0.0
    hi = 0.0
    flo = 0.0
    fhi = 0.0
    
    # Generate dense logarithmic deltas: 1e-8 * 3^k for k=0..~40
    # 3^40 ≈ 1.2e19, so 1e-8 * 3^40 ≈ 1.2e11, which is more than enough.
    # Let's cap at 1e6 to keep it reasonable.
    deltas = []
    d = 1e-8
    while d <= 1e6:
        deltas.append(d)
        d *= 3.0
    # Ensure we cover up to at least 1e6
    if deltas[-1] < 1e6:
        deltas.append(1e6)
    
    for d in deltas:
        if time.time() > deadline:
            break
            
        # Check right side
        x_right = x + d
        try:
            fr = f(x_right)
        except:
            fr = None
        
        if fr is not None and math.isfinite(fr):
            abs_fr = abs(fr)
            if abs_fr < best_abs:
                best_abs = abs_fr
                best_x = x_right
                best_f = fr
            if abs_fr <= 1e-8:
                return x_right
            if fa * fr < 0:
                lo, hi = x, x_right
                flo, fhi = fa, fr
                bracket_found = True
                break
        
        # Check left side
        x_left = x - d
        try:
            fl = f(x_left)
        except:
            fl = None
            
        if fl is not None and math.isfinite(fl):
            abs_fl = abs(fl)
            if abs_fl < best_abs:
                best_abs = abs_fl
                best_x = x_left
                best_f = fl
            if abs_fl <= 1e-8:
                return x_left
            if fa * fl < 0:
                lo, hi = x_left, x
                flo, fhi = fl, fa
                bracket_found = True
                break
        
        if time.time() > deadline:
            break

    if bracket_found:
        # Use Brent's method for fast, robust convergence
        try:
            root = brentq(f, lo, hi, xtol=1e-10, rtol=1e-10, maxiter=100)
            if time.time() < deadline:
                # Verify
                f_root = f(root)
                if math.isfinite(f_root) and abs(f_root) <= 1e-8:
                    return root
        except Exception:
            pass
        
        # If brentq failed or wasn't good enough, try bisection manually?
        # Or just return best_x?
        # Let's try a few bisection steps if we have time.
        # But we might have lost flo/fhi if brentq modified them? No, brentq takes f, lo, hi.
        # We still have lo, hi. We can re-eval or use stored.
        # Stored flo/fhi correspond to lo/hi at the start of brentq.
        # If brentq failed, the bracket might still be valid.
        
        # Let's do a simple bisection fallback
        a_b, b_b = min(lo, hi), max(lo, hi)
        fa_b, fb_b = f(a_b), f(b_b) # Re-eval to be safe
        if fa_b * fb_b < 0:
            for _ in range(20):
                if time.time() > deadline:
                    break
                m = (a_b + b_b) / 2.0
                fm = f(m)
                if abs(fm) <= 1e-8:
                    return m
                if fa_b * fm < 0:
                    b_b = m
                    fb_b = fm
                else:
                    a_b = m
                    fa_b = fm
            # Return midpoint
            m = (a_b + b_b) / 2.0
            return m

    # If no bracket found, we have the best_x found during the scan.
    # Try to refine best_x using Newton's method if possible.
    # This is useful if the function has a minimum close to 0 but no sign change nearby 
    # (e.g., touch root) or if we missed a narrow sign change.
    
    if best_abs > 1e-8:
        # Try to refine best_x by looking for a local sign change in a small window.
        # If we can find a bracket near best_x, use bisection to converge.
        try:
            # Try a few small windows around best_x
            for window in [1e-3, 1e-2, 1e-1, 1.0, 10.0]:
                if time.time() > deadline:
                    break
                a_w = best_x - window
                b_w = best_x + window
                try:
                    fa_w = f(a_w)
                    fb_w = f(b_w)
                except:
                    continue
                if not (math.isfinite(fa_w) and math.isfinite(fb_w)):
                    continue
                if fa_w * fb_w < 0:
                    # Found a bracket near best_x, use bisection
                    for _ in range(40):
                        if time.time() > deadline:
                            break
                        m_w = (a_w + b_w) / 2.0
                        fm_w = f(m_w)
                        if abs(fm_w) <= 1e-8:
                            return m_w
                        if fa_w * fm_w < 0:
                            b_w = m_w
                            fb_w = fm_w
                        else:
                            a_w = m_w
                            fa_w = fm_w
                    m_w = (a_w + b_w) / 2.0
                    fm_w = f(m_w)
                    if math.isfinite(fm_w) and abs(fm_w) < best_abs:
                        return m_w
                    break
        except:
            pass
        
        # As a last resort, try a simple Newton step with central difference
        h = 1e-5 * max(1.0, abs(best_x))
        try:
            fx_best = f(best_x)
            fx_plus = f(best_x + h)
            fx_minus = f(best_x - h)
            if math.isfinite(fx_plus) and math.isfinite(fx_minus):
                deriv = (fx_plus - fx_minus) / (2 * h)
                if deriv != 0 and math.isfinite(deriv):
                    x_new = best_x - fx_best / deriv
                    if math.isfinite(x_new):
                        f_new = f(x_new)
                        if math.isfinite(f_new) and abs(f_new) < best_abs:
                            best_x = x_new
                            best_abs = abs(f_new)
        except:
            pass

    # Ensure we return a finite value
    if not math.isfinite(best_x):
        return x0
    return best_x
# EVOLVE-BLOCK-END
