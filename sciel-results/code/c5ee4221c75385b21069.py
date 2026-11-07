"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import math
import time
from scipy.optimize import brentq

def find_root(f, x0):
    TOL = 1e-10  # tighter than required 1e-8 for safety
    MAX_CALLS = 490  # safe margin under 500
    DEADLINE = time.time() + 110  # safe margin under 120s

    calls = 0
    best_x = float(x0)
    best_abs = float('inf')

    def safe_f(x):
        nonlocal calls, best_x, best_abs
        if calls >= MAX_CALLS:
            return float('inf')
        if time.time() > DEADLINE:
            return float('inf')
        calls += 1
        try:
            val = f(x)
            if not math.isfinite(val):
                return float('inf')
            if abs(val) < best_abs:
                best_abs = abs(val)
                best_x = x
            return val
        except Exception:
            return float('inf')

    x = float(x0)

    # Check if x0 is already a root
    fx = safe_f(x)
    if abs(fx) <= TOL:
        return x

    # Try to bracket a root near x0 by expanding window
    a, b = None, None
    fa, fb = None, None

    # Start with small window and expand
    step = max(1e-6, abs(x) * 1e-6)
    for expansion in range(30):
        if calls >= MAX_CALLS or time.time() > DEADLINE:
            break
        left = x - step
        right = x + step
        fl = safe_f(left)
        fr = safe_f(right)
        if math.isfinite(fl) and math.isfinite(fr):
            if abs(fl) <= TOL:
                return left
            if abs(fr) <= TOL:
                return right
            if fl * fr < 0:
                a, b = left, right
                fa, fb = fl, fr
                break
        step *= 2.0

    # If we found a bracket, use brentq
    if a is not None and b is not None:
        try:
            if time.time() < DEADLINE and calls < MAX_CALLS - 10:
                root = brentq(safe_f, a, b, xtol=1e-14, rtol=1e-14, maxiter=100)
                if math.isfinite(root):
                    return root
        except Exception:
            pass

    # If no bracket found, try a broader search
    # Sample points around x0 to find a sign change
    if calls < MAX_CALLS and time.time() < DEADLINE:
        base_step = max(1e-3, abs(x) * 1e-3)
        n_samples = 40
        prev_x = None
        prev_f = None
        for i in range(n_samples):
            if calls >= MAX_CALLS or time.time() > DEADLINE:
                break
            # Sample in both directions
            for sign in (1, -1):
                if calls >= MAX_CALLS or time.time() > DEADLINE:
                    break
                sx = x + sign * base_step * (i + 1)
                sf = safe_f(sx)
                if not math.isfinite(sf):
                    continue
                if abs(sf) <= TOL:
                    return sx
                if prev_x is not None and prev_f is not None:
                    if prev_f * sf < 0:
                        # Found bracket
                        try:
                            root = brentq(safe_f, prev_x, sx, xtol=1e-14, rtol=1e-14, maxiter=100)
                            if math.isfinite(root):
                                return root
                        except Exception:
                            pass
                prev_x = sx
                prev_f = sf

    # Return best candidate found
    return best_x
# EVOLVE-BLOCK-END
