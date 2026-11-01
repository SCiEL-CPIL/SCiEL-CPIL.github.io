"""
Circle packing in unit square n=32.
Uses physics-inspired smooth objective with L-BFGS-B and a final polish step.
"""
import numpy as np
from scipy.optimize import minimize
import time

N = 32
DEADLINE = 150.0
SAFETY_MARGIN = 5.0

def _elapsed(start_time):
    return time.time() - start_time

def _is_valid(circles, tol=1e-6):
    """Check if circles are valid (inside square, non-overlapping)."""
    n = len(circles)
    if n == 0:
        return False
    x = circles[:, 0]
    y = circles[:, 1]
    r = circles[:, 2]
    
    # Check inside square
    if np.any(x < -tol) or np.any(y < -tol) or np.any(x > 1 + tol) or np.any(y > 1 + tol):
        return False
    if np.any(r < -tol):
        return False
    # Check that circles don't extend outside
    if np.any(x + r > 1 + tol) or np.any(y + r > 1 + tol) or np.any(x - r < -tol) or np.any(y - r < -tol):
        return False
    
    # Check non-overlap
    for i in range(n):
        for j in range(i + 1, n):
            dx = x[i] - x[j]
            dy = y[i] - y[j]
            dist = np.sqrt(dx * dx + dy * dy)
            if dist < r[i] + r[j] - tol:
                return False
    return True

def _sum_r(circles):
    return np.sum(circles[:, 2])

def _objective_and_grad(params, epsilon=0.05):
    """
    Smooth objective: minimize -sum(r) + Penalty
    Penalty is sum of exponential repulsions for overlaps and boundary violations.
    """
    n = N
    x = params[0::3]
    y = params[1::3]
    r = params[2::3]
    
    # Objective: -sum(r)
    obj = -np.sum(r)
    grad = np.zeros(3 * n)
    grad[2::3] = -1.0
    
    # Repulsion between circles
    # We want to minimize overlaps.
    # Penalty term: exp(-(dist - r_i - r_j) / epsilon)
    # This is large when dist < r_i + r_j.
    mu = 10.0 # Scaling factor for repulsion strength
    
    for i in range(n):
        for j in range(i + 1, n):
            dx = x[i] - x[j]
            dy = y[i] - y[j]
            dist_sq = dx*dx + dy*dy
            min_dist = r[i] + r[j]
            
            # If dist is very close to 0, handle carefully
            if dist_sq < 1e-12:
                dist = 1e-6
                dx = 1e-6
                dy = 0.0
            else:
                dist = np.sqrt(dist_sq)
            
            # If they are overlapping or close
            diff = dist - min_dist
            # Use a soft plus function or just exponential
            # To keep it smooth and bounded, we use exp(-diff/epsilon)
            # But we only care about the penalty if diff < 0?
            # Actually, exp(-diff/epsilon) is always positive.
            # We want the gradient to push them apart.
            
            # Let's use a potential V = exp(-diff/epsilon)
            # dV/ddiff = -1/epsilon * exp(-diff/epsilon)
            # dV/dx_i = dV/ddiff * ddiff/dx_i = (-1/epsilon * V) * (dx/dist)
            # dV/dr_i = dV/ddiff * ddiff/dr_i = (-1/epsilon * V) * (-1) = 1/epsilon * V
            
            # Wait, if diff is large positive (far apart), V is small.
            # If diff is negative (overlap), V is large.
            # We add mu * V to objective.
            
            val = np.exp(-diff / epsilon)
            # To avoid overflow if diff is very negative
            if val > 1e10:
                val = 1e10
            
            obj += mu * val
            
            # Gradients
            # d/dx_i (mu * exp(-diff/eps)) = mu * exp(-diff/eps) * (-1/eps) * (dx/dist)
            # d/dr_i (mu * exp(-diff/eps)) = mu * exp(-diff/eps) * (-1/eps) * (-1) = mu/eps * exp(-diff/eps)
            
            factor = mu * val / epsilon
            if dist > 1e-10:
                g_x_i = -factor * dx / dist
                g_y_i = -factor * dy / dist
                g_x_j = factor * dx / dist
                g_y_j = factor * dy / dist
            else:
                # If they are at the same point, push in arbitrary direction
                g_x_i = -factor
                g_y_i = 0.0
                g_x_j = factor
                g_y_j = 0.0
            
            grad[3*i + 0] += g_x_i
            grad[3*i + 1] += g_y_i
            grad[3*j + 0] += g_x_j
            grad[3*j + 1] += g_y_j
            
            # Radius gradients
            # d(diff)/dr_i = -1, so dV/dr_i = (-1/eps)*V * (-1) = V/eps
            grad[3*i + 2] += factor
            grad[3*j + 2] += factor

    # Boundary penalties
    # We want x >= r, y >= r, x+r <= 1, y+r <= 1
    # Let's use exponential penalties for violations
    # Violation v = max(0, r - x). Penalty exp(-v/epsilon)? No, that's for distance.
    # Better: Penalty = exp((r-x)/epsilon) if r>x else 0? Discontinuous.
    # Use soft max: m = (r-x + softplus(r-x))/2? 
    # Simpler: Just add large penalty if violated, but keep it smooth.
    # Let's use the same exponential form but for distance to boundary.
    # Distance to left wall: d_left = x - r. We want d_left >= 0.
    # Penalty = exp(-d_left / epsilon)
    
    # Left: x - r
    d_left = x - r
    p_left = np.exp(-d_left / epsilon)
    obj += mu * np.sum(p_left)
    # d/dx: -1/eps * p_left
    # d/dr: -1/eps * p_left * (-1) = 1/eps * p_left
    grad[0::3] += -mu * p_left / epsilon
    grad[2::3] += mu * p_left / epsilon

    # Right: 1 - x - r
    d_right = 1.0 - x - r
    p_right = np.exp(-d_right / epsilon)
    obj += mu * np.sum(p_right)
    # d/dx: -1/eps * p_right * (-1) = 1/eps * p_right
    # d/dr: -1/eps * p_right * (-1) = 1/eps * p_right
    grad[0::3] += mu * p_right / epsilon
    grad[2::3] += mu * p_right / epsilon

    # Bottom: y - r
    d_bottom = y - r
    p_bottom = np.exp(-d_bottom / epsilon)
    obj += mu * np.sum(p_bottom)
    grad[1::3] += -mu * p_bottom / epsilon
    grad[2::3] += mu * p_bottom / epsilon

    # Top: 1 - y - r
    d_top = 1.0 - y - r
    p_top = np.exp(-d_top / epsilon)
    obj += mu * np.sum(p_top)
    grad[1::3] += mu * p_top / epsilon
    grad[2::3] += mu * p_top / epsilon
    
    return obj, grad

def _push_apart(params, iterations=50):
    """
    Simple iterative push-apart to fix any remaining overlaps or boundary violations.
    This is a non-smooth projection step.
    """
    n = N
    x = params[0::3].copy()
    y = params[1::3].copy()
    r = params[2::3].copy()
    
    for _ in range(iterations):
        # Fix boundaries
        # x >= r
        for i in range(n):
            if x[i] < r[i]:
                x[i] = r[i]
            if x[i] + r[i] > 1.0:
                x[i] = 1.0 - r[i]
            if y[i] < r[i]:
                y[i] = r[i]
            if y[i] + r[i] > 1.0:
                y[i] = 1.0 - r[i]
        
        # Fix overlaps
        for i in range(n):
            for j in range(i + 1, n):
                dx = x[i] - x[j]
                dy = y[i] - y[j]
                dist = np.sqrt(dx*dx + dy*dy)
                min_dist = r[i] + r[j]
                if dist < min_dist:
                    if dist < 1e-10:
                        dx = 1e-10
                        dy = 0.0
                        dist = 1e-10
                    # Push apart along vector
                    overlap = min_dist - dist
                    # Move each by half the overlap
                    nx = dx / dist
                    ny = dy / dist
                    x[i] += 0.5 * overlap * nx
                    y[i] += 0.5 * overlap * ny
                    x[j] -= 0.5 * overlap * nx
                    y[j] -= 0.5 * overlap * ny
                    
                    # Re-clamp to boundaries immediately to prevent pushing out
                    x[i] = np.clip(x[i], r[i], 1.0 - r[i])
                    y[i] = np.clip(y[i], r[i], 1.0 - r[i])
                    x[j] = np.clip(x[j], r[j], 1.0 - r[j])
                    y[j] = np.clip(y[j], r[j], 1.0 - r[j])
    
    params[0::3] = x
    params[1::3] = y
    params[2::3] = r
    return params

def _optimize(params, time_limit=10.0, epsilon=0.05):
    """
    Run L-BFGS-B on the smooth objective.
    """
    start_time = time.time()
    
    def obj_and_grad(p):
        if time.time() - start_time > time_limit:
            raise Exception("Time limit exceeded")
        return _objective_and_grad(p, epsilon)
    
    try:
        result = minimize(
            obj_and_grad,
            params,
            method='L-BFGS-B',
            options={'maxiter': 100, 'ftol': 1e-12, 'gtol': 1e-8}
        )
        return result.x
    except Exception:
        return params

def _make_valid(params):
    """Ensure the solution is strictly valid by adjusting radii down if necessary."""
    n = N
    x = params[0::3].copy()
    y = params[1::3].copy()
    r = params[2::3].copy()
    
    # First, ensure inside square
    for i in range(n):
        # Max radius allowed by position
        r_max = min(x[i], 1.0 - x[i], y[i], 1.0 - y[i])
        if r[i] > r_max:
            r[i] = r_max * 0.999
    
    # Then, resolve overlaps by reducing radii
    for _ in range(20):
        changed = False
        for i in range(n):
            for j in range(i + 1, n):
                dx = x[i] - x[j]
                dy = y[i] - y[j]
                dist = np.sqrt(dx*dx + dy*dy)
                current_sum = r[i] + r[j]
                if dist < current_sum:
                    # Reduce radii to fit
                    # We want r_i + r_j <= dist
                    # Scale them down proportionally
                    if current_sum > 1e-10:
                        scale = dist / current_sum * 0.999
                        r[i] *= scale
                        r[j] *= scale
                        changed = True
                    else:
                        r[i] = 0.0
                        r[j] = 0.0
                        changed = True
        if not changed:
            break
            
    params[0::3] = x
    params[1::3] = y
    params[2::3] = r
    return params

def _hex_init(seed=0):
    """Hexagonal grid initialization."""
    rng = np.random.RandomState(seed)
    n = N
    side = int(np.ceil(np.sqrt(n)))
    positions = []
    for i in range(side):
        for j in range(side):
            if len(positions) >= n:
                break
            offset = 0.5 / side if i % 2 == 1 else 0
            x = (i + 0.5 + offset) / side
            y = (j + 0.5) / side
            positions.append((x, y))
    
    params = np.zeros(3 * n)
    for k in range(n):
        params[3 * k + 0] = positions[k][0] + rng.randn() * 0.01
        params[3 * k + 1] = positions[k][1] + rng.randn() * 0.01
        params[3 * k + 2] = 0.5 / side * 0.8
    return params

def _random_init(seed=0):
    """Random initialization."""
    rng = np.random.RandomState(seed)
    n = N
    params = np.zeros(3 * n)
    for k in range(n):
        params[3 * k + 0] = rng.uniform(0.1, 0.9)
        params[3 * k + 1] = rng.uniform(0.1, 0.9)
        params[3 * k + 2] = rng.uniform(0.02, 0.08)
    return params

def _halton_init(seed=0):
    """Halton sequence initialization for better distribution."""
    n = N
    def halton(index, base):
        f = 1.0
        r = 0.0
        while index > 0:
            f /= base
            r += f * (index % base)
            index //= base
        return r

    params = np.zeros(3 * n)
    for k in range(n):
        # Use different bases for x and y
        params[3 * k + 0] = halton(k + 1, 2)
        params[3 * k + 1] = halton(k + 1, 3)
        params[3 * k + 2] = 0.05 # Small initial radius
    return params

def construct():
    start_time = time.time()
    deadline = start_time + DEADLINE - SAFETY_MARGIN
    
    best_params = None
    best_score = -np.inf
    
    num_restarts = 30
    
    for restart in range(num_restarts):
        if time.time() > deadline:
            break
        
        # Select initialization strategy
        if restart < 10:
            params = _hex_init(seed=restart)
        elif restart < 20:
            params = _random_init(seed=restart)
        else:
            params = _halton_init(seed=restart)
            
        # Clip initial radii to reasonable bounds
        params[2::3] = np.clip(params[2::3], 0.01, 0.2)
        params[0::3] = np.clip(params[0::3], 0.0, 1.0)
        params[1::3] = np.clip(params[1::3], 0.0, 1.0)
        
        # Run optimization
        # Allocate time based on remaining budget
        time_per_restart = max(1.0, (deadline - time.time()) / (num_restarts - restart))
        
        # Phase 1: Coarse optimization
        opt_params = _optimize(params, time_limit=time_per_restart * 0.5, epsilon=0.1)
        
        # Phase 2: Fine optimization
        opt_params = _optimize(opt_params, time_limit=time_per_restart * 0.5, epsilon=0.02)
        
        # Phase 3: Polish (Push apart and make valid)
        opt_params = _push_apart(opt_params, iterations=20)
        opt_params = _make_valid(opt_params)
        
        # Check validity
        circles = np.column_stack([opt_params[0::3], opt_params[1::3], opt_params[2::3]])
        if _is_valid(circles):
            score = _sum_r(circles)
            if score > best_score:
                best_score = score
                best_params = opt_params.copy()
        
        # If we have a good solution, try to perturb it
        if best_params is not None and restart > 10:
             rng = np.random.RandomState(restart * 1000)
             perturbed = best_params + rng.randn(3 * N) * 0.02
             # Quick optimize on perturbed
             perturbed = _optimize(perturbed, time_limit=1.0, epsilon=0.05)
             perturbed = _push_apart(perturbed, iterations=10)
             perturbed = _make_valid(perturbed)
             p_circles = np.column_stack([perturbed[0::3], perturbed[1::3], perturbed[2::3]])
             if _is_valid(p_circles):
                 p_score = _sum_r(p_circles)
                 if p_score > best_score:
                     best_score = p_score
                     best_params = perturbed.copy()

    if best_params is None:
        # Fallback: Simple grid
        side = int(np.ceil(np.sqrt(N)))
        cells = []
        for i in range(side):
            for j in range(side):
                if len(cells) < N:
                    cells.append(((i + 0.5) / side, (j + 0.5) / side, 0.5 / side - 1e-3))
        return np.array(cells)
    
    # Final check
    final_circles = np.column_stack([best_params[0::3], best_params[1::3], best_params[2::3]])
    if not _is_valid(final_circles):
         # Force validity
         best_params = _make_valid(best_params)
         final_circles = np.column_stack([best_params[0::3], best_params[1::3], best_params[2::3]])
         
    return final_circles

if __name__ == "__main__":
    result = construct()
    print(result)
