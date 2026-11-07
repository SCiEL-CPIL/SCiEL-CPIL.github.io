"""
Scalar root finder on 3 fixed oscillatory equations.

Strategy:
  1. Evaluate f at x0. If |f| <= 1e-8, return immediately.
  2. Quick bracket search with geometric expansion.
  3. If bracket found, use scipy.optimize.brentq for fast convergence.
  4. If no bracket, use scipy.optimize.minimize_scalar on f(x)^2 with
     dynamically expanding bounds to find a root even without sign change.

All f calls are counted by the evaluator. The final check is not counted.
"""
import time
import math
import numpy as np
from scipy.optimize import brentq, minimize_scalar

# EVOLVE-BLOCK-START
_START_TIME = None
_DEADLINE = 145.0  # 145s out of 150s wall-clock limit

def _elapsed():
    global _START_TIME
    if _START_TIME is None:
        _START_TIME = time.time()
    return time.time() - _START_TIME

def _is_finite(x):
    return math.isfinite(x)

def _update_best(best_x, best_f, x, fx):
    """Update best if fx is finite and has smaller absolute value."""
    if _is_finite(fx) and abs(fx) < abs(best_f):
        return x, fx
    return best_x, best_f

def find_root(f, x0):
    """
    Find a root x of f such that |f(x)| <= 1e-8.
    Strategy:
    1. Check x0.
    2. Quick bracket search with geometric expansion.
    3. If bracket found, use brentq for fast convergence.
    4. If no bracket, use minimize_scalar on f(x)^2 with expanding bounds.
    """
    x0 = float(x0)
    if not _is_finite(x0):
        x0 = 0.0

    best_x = x0
    best_f = float('inf')

    try:
        f0 = f(x0)
        if _is_finite(f0):
            best_f = f0
            if abs(f0) <= 1e-8:
                return x0
    except:
        pass

    # Phase 1: Quick bracket search
    a = x0
    fa = best_f
    if not _is_finite(fa):
        found_good = False
        for dx in [1.0, -1.0, 0.5, -0.5, 2.0, -2.0]:
            try:
                x_try = x0 + dx
                f_try = f(x_try)
                if _is_finite(f_try):
                    a = x_try
                    fa = f_try
                    found_good = True
                    if abs(f_try) <= 1e-8:
                        return x_try
                    break
            except:
                continue
        if not found_good:
            return x0

    b = None
    fb = None
    
    base_step = max(1.0, abs(x0) * 0.1)
    
    for i in range(15):
        if _elapsed() > _DEADLINE - 10:
            break
            
        step = base_step * (2 ** i)
        
        try:
            b_try = a + step
            fb_try = f(b_try)
            if _is_finite(fb_try):
                best_x, best_f = _update_best(best_x, best_f, b_try, fb_try)
                if abs(fb_try) <= 1e-8:
                    return b_try
                if fa * fb_try < 0:
                    b = b_try
                    fb = fb_try
                    break
        except:
            pass
        
        try:
            b_try = a - step
            fb_try = f(b_try)
            if _is_finite(fb_try):
                best_x, best_f = _update_best(best_x, best_f, b_try, fb_try)
                if abs(fb_try) <= 1e-8:
                    return b_try
                if fa * fb_try < 0:
                    b = b_try
                    fb = fb_try
                    break
        except:
            pass

    # Phase 2: If bracket found, use brentq
    if b is not None and fb is not None and fa * fb < 0:
        try:
            root = brentq(f, a, b, xtol=1e-12, rtol=1e-12, maxiter=100)
            if _is_finite(root):
                return root
        except Exception:
            pass

    # Phase 3: Fallback - Minimize f(x)^2 with expanding bounds
    def sq_f(x):
        try:
            val = f(x)
            if not _is_finite(val):
                return 1e100
            return val * val
        except:
            return 1e100

    best_candidate_x = best_x
    best_candidate_val = abs(best_f) if _is_finite(best_f) else 1e100

    # Try multiple starting centers with expanding widths
    centers = [x0, a, x0 + 1.0, x0 - 1.0]
    
    for center in centers:
        if _elapsed() > _DEADLINE - 5:
            break
        
        for width in [1.0, 5.0, 20.0, 100.0, 500.0]:
            if _elapsed() > _DEADLINE - 5:
                break
            
            lower = center - width
            upper = center + width
            
            try:
                res = minimize_scalar(sq_f, bounds=(lower, upper), method='bounded', options={'xatol': 1e-14})
                if res.success and _is_finite(res.x):
                    val = res.fun
                    if val < best_candidate_val:
                        best_candidate_val = val
                        best_candidate_x = res.x
                        if val <= (1e-8) ** 2:
                            return best_candidate_x
            except Exception:
                pass

    # Final polish: if we're close, try to tighten with a very narrow window
    if best_candidate_val < 1e-4 and best_candidate_val > (1e-8) ** 2:
        if _elapsed() < _DEADLINE - 5:
            try:
                narrow_width = max(1e-6, math.sqrt(best_candidate_val) * 10)
                lower = best_candidate_x - narrow_width
                upper = best_candidate_x + narrow_width
                res = minimize_scalar(sq_f, bounds=(lower, upper), method='bounded', options={'xatol': 1e-15})
                if res.success and _is_finite(res.x):
                    if res.fun < best_candidate_val:
                        best_candidate_val = res.fun
                        best_candidate_x = res.x
            except:
                pass

    if best_candidate_val <= (1e-8) ** 2:
        return best_candidate_x

    return best_candidate_x if best_candidate_val < abs(best_f) else x0

# EVOLVE-BLOCK-END
