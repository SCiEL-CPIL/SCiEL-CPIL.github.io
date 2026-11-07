"""Root finder for 3 fixed oscillatory equations."""
import time
import math

# EVOLVE-BLOCK-START
def find_root(f, x0):
    start_time = time.time()
    deadline = start_time + 145.0  # 15s safety margin before 160s kill
    
    x0 = float(x0)
    
    # Counted function wrapper with time guard
    call_count = 0
    
    def _f(x):
        nonlocal call_count
        call_count += 1
        if time.time() > deadline:
            raise TimeoutError("deadline exceeded")
        return f(x)
    
    # First, check if x0 is already a root
    try:
        fx0 = _f(x0)
        if abs(fx0) <= 1e-8:
            return x0
    except Exception:
        fx0 = None
    
    # Strategy: Multi-scale sign-change search to find a bracket, then Illinois method
    # For oscillatory functions, we need to search in multiple directions and scales
    
    # Generate candidate points to search for sign changes
    # Use logarithmic spacing in both positive and negative directions from x0
    candidates = []
    
    # Scale factors: from very small to very large
    base_scale = max(1.0, abs(x0)) if x0 != 0 else 1.0
    if x0 == 0:
        base_scale = 1.0
    
    # Generate points at various scales
    for scale in [1e-12, 1e-10, 1e-8, 1e-6, 1e-4, 1e-2, 1.0, 10.0, 100.0, 1000.0, 1e5, 1e7, 1e9]:
        offset = base_scale * scale
        candidates.append(x0 + offset)
        candidates.append(x0 - offset)
    
    # Also add some geometrically spaced points around x0
    for i in range(1, 20):
        step = base_scale * (2 ** i)
        candidates.append(x0 + step)
        candidates.append(x0 - step)
    
    # Deduplicate and sort
    seen = set()
    unique_candidates = []
    for c in candidates:
        if math.isfinite(c) and c not in seen:
            seen.add(c)
            unique_candidates.append(c)
    
    # Sort by distance from x0 to prioritize close points
    unique_candidates.sort(key=lambda c: abs(c - x0))
    
    # Evaluate f at all candidate points to find sign changes
    eval_points = []  # (x, fx) pairs
    prev_x = None
    prev_fx = None
    
    for x in unique_candidates:
        if time.time() > deadline:
            break
        try:
            fx = _f(x)
            if not math.isfinite(fx):
                prev_x = None
                prev_fx = None
                continue
            eval_points.append((x, fx))
            
            if prev_fx is not None and prev_fx * fx < 0:
                # Found a sign change! Use Illinois method
                a, b = prev_x, x
                fa, fb = prev_fx, fx
                root = _illinois(_f, a, b, fa, fb, deadline)
                if root is not None and math.isfinite(root):
                    return root
            elif abs(fx) <= 1e-8:
                return x
            
            prev_x = x
            prev_fx = fx
        except TimeoutError:
            break
        except Exception:
            prev_x = None
            prev_fx = None
    
    # If no bracket found, try a broader search with more points
    # Use a denser sampling in a wider range
    if time.time() < deadline:
        # Try points on a wider grid
        search_radius = max(1.0, abs(x0)) * 1e6
        n_points = 500
        for i in range(n_points):
            if time.time() > deadline:
                break
            x = x0 - search_radius + 2 * search_radius * i / (n_points - 1) if n_points > 1 else x0
            try:
                fx = _f(x)
                if not math.isfinite(fx):
                    continue
                if abs(fx) <= 1e-8:
                    return x
                eval_points.append((x, fx))
            except TimeoutError:
                break
            except Exception:
                continue
        
        # Check for sign changes in the broader search
        for i in range(1, len(eval_points)):
            if time.time() > deadline:
                break
            x1, f1 = eval_points[i-1]
            x2, f2 = eval_points[i]
            if f1 * f2 < 0:
                root = _illinois(_f, x1, x2, f1, f2, deadline)
                if root is not None and math.isfinite(root):
                    return root
    
    # Fallback: return x0 if nothing else works (will likely fail, but must return something)
    return x0


def _illinois(f, a, b, fa, fb, deadline):
    """Illinois method for root finding within a bracket [a, b]."""
    # Ensure fa and fb have opposite signs
    if fa * fb > 0:
        return None
    
    max_iter = 200
    for _ in range(max_iter):
        if time.time() > deadline:
            return None
        
        # Illinois: weight the midpoint towards the side with smaller |f|
        if abs(fa) < abs(fb):
            # fa is smaller, so move b closer to a
            # New b: (a*fb - b*fa) / (fb - fa) ... wait, standard Illinois:
            # If |fa| < |fb|, replace b with c = (a*fb - b*fa)/(fb - fa)
            # Actually, the standard Illinois formula:
            # c = (a*fb - b*fa) / (fb - fa) when |fa| < |fb|
            # Let me use the more common formulation:
            # If |fa| < |fb|: b_new = (a*fb - b*fa) / (fb - fa)
            # If |fa| >= |fb|: a_new = (a*fb - b*fa) / (fb - fa)
            # Wait, that's not right either. Let me use:
            # Illinois: c = (a*fb - b*fa) / (fb - fa)
            # If |fa| < |fb|, set b = c
            # If |fa| >= |fb|, set a = c
            
            c = (a * fb - b * fa) / (fb - fa)
            if c <= a or c >= b:
                c = (a + b) / 2.0  # fallback to bisection
            b, fb = c, f(c)
        else:
            c = (a * fb - b * fa) / (fb - fa)
            if c <= a or c >= b:
                c = (a + b) / 2.0
            a, fa = c, f(c)
        
        # Check convergence
        if abs(fa) <= 1e-10:
            return a
        if abs(fb) <= 1e-10:
            return b
        if abs(b - a) < 1e-14 * max(1.0, abs(a), abs(b)):
            return (a + b) / 2.0
    
    # Return the best point
    if abs(fa) < abs(fb):
        return a
    else:
        return b

# EVOLVE-BLOCK-END
