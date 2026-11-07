"""
Scalar root finder on 3 fixed oscillatory equations.

Strategy:
  1. Evaluate f at x0. If |f| <= 1e-8, return immediately.
  2. Coarse scan to find a sign-change bracket or the region with smallest |f(x)|.
  3. If bracket found, use brentq for fast convergence.
  4. Use minimize_scalar on f(x)^2 with expanding windows around the best point.
  5. Final high-precision polish pass if close but not converged.

All f calls are counted by the evaluator. The final check is not counted.
"""
import time
import math
import numpy as np
from scipy.optimize import brentq, minimize_scalar

# EVOLVE-BLOCK-START
_START_TIME = None
_DEADLINE = 140.0  # Safety margin before 150s wall-clock limit
_CALL_BUDGET = 480  # Keep under 500 calls

def _elapsed():
    global _START_TIME
    if _START_TIME is None:
        _START_TIME = time.time()
    return time.time() - _START_TIME

def _is_finite(x):
    return math.isfinite(x)

def find_root(f, x0):
    """
    Find a root x of f such that |f(x)| <= 1e-8.
    """
    x0 = float(x0)
    if not _is_finite(x0):
        x0 = 0.0

    start_time = time.time()
    call_count = 0
    best_x = x0
    best_f2 = float('inf')

    def obj(x):
        nonlocal call_count, best_x, best_f2
        if time.time() - start_time > _DEADLINE:
            raise TimeoutError("Time limit exceeded")
        call_count += 1
        if call_count > _CALL_BUDGET:
            raise TimeoutError("Call limit exceeded")
        try:
            fx = f(x)
        except:
            return 1e100
        if not _is_finite(fx):
            return 1e100
        f2 = fx * fx
        if f2 < best_f2:
            best_f2 = f2
            best_x = x
        return f2

    # Check x0
    try:
        f0 = f(x0)
        call_count += 1
        if _is_finite(f0):
            best_f2 = f0 * f0
            if abs(f0) <= 1e-8:
                return x0
    except:
        pass

    # Phase 1: Coarse scan to find sign-change bracket or best point
    a = x0
    fa = None
    if best_f2 < 1e100:
        fa = math.sqrt(best_f2) if best_f2 > 0 else 0.0
        # Recover sign
        try:
            fa = f(x0)
            call_count += 1
        except:
            fa = 0.0

    if not _is_finite(fa) or abs(fa) >= 1e100:
        # Find a good starting point
        found_good = False
        for dx in [1.0, -1.0, 0.5, -0.5, 2.0, -2.0, 0.1, -0.1]:
            if call_count > _CALL_BUDGET - 50:
                break
            try:
                x_try = x0 + dx
                f_try = f(x_try)
                call_count += 1
                if _is_finite(f_try):
                    a = x_try
                    fa = f_try
                    found_good = True
                    if abs(f_try) <= 1e-8:
                        return x_try
                    if f_try * f_try < best_f2:
                        best_f2 = f_try * f_try
                        best_x = x_try
                    break
            except:
                continue
        if not found_good:
            return best_x

    b = None
    fb = None
    base_step = max(1.0, abs(x0) * 0.1)

    for i in range(10):
        if _elapsed() > _DEADLINE - 20 or call_count > _CALL_BUDGET - 50:
            break
        step = base_step * (2 ** i)
        for direction in [1, -1]:
            if call_count > _CALL_BUDGET - 50:
                break
            try:
                b_try = a + direction * step
                fb_try = f(b_try)
                call_count += 1
                if _is_finite(fb_try):
                    if fb_try * fb_try < best_f2:
                        best_f2 = fb_try * fb_try
                        best_x = b_try
                    if abs(fb_try) <= 1e-8:
                        return b_try
                    if fa * fb_try < 0:
                        b = b_try
                        fb = fb_try
                        break
            except:
                pass
        if b is not None:
            break

    # Phase 2: If bracket found, use brentq
    if b is not None and fb is not None and fa * fb < 0:
        try:
            root = brentq(f, a, b, xtol=1e-12, rtol=1e-12, maxiter=100)
            if _is_finite(root):
                return root
        except Exception:
            pass

    # Phase 3: Minimize f(x)^2 with expanding windows
    # Start from best_x found during scan
    center = best_x
    widths = [0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 50.0]

    for w in widths:
        if _elapsed() > _DEADLINE - 10 or call_count > _CALL_BUDGET - 30:
            break
        lo = center - w
        hi = center + w
        try:
            res = minimize_scalar(obj, bounds=(lo, hi), method='bounded',
                                  options={'maxiter': 40, 'xatol': 1e-13, 'fatol': 1e-18})
            if res.success and _is_finite(res.x):
                if math.sqrt(res.fun) <= 1e-8:
                    return res.x
                if res.fun < best_f2:
                    best_f2 = res.fun
                    best_x = res.x
                    center = res.x
        except TimeoutError:
            break
        except Exception:
            continue

    # Final polish: if close, try very narrow window
    if best_f2 < 1e-6 and best_f2 > 1e-16:
        if _elapsed() < _DEADLINE - 5 and call_count < _CALL_BUDGET - 10:
            narrow_w = max(1e-8, math.sqrt(best_f2) * 50)
            lo = best_x - narrow_w
            hi = best_x + narrow_w
            try:
                res = minimize_scalar(obj, bounds=(lo, hi), method='bounded',
                                      options={'maxiter': 50, 'xatol': 1e-15, 'fatol': 1e-20})
                if res.success and _is_finite(res.x):
                    if res.fun < best_f2:
                        best_f2 = res.fun
                        best_x = res.x
            except:
                pass

    if math.sqrt(best_f2) <= 1e-8:
        return best_x

    return best_x

# EVOLVE-BLOCK-END
