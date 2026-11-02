"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 21

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

def construct():
    start_time = time.time()
    deadline = 130.0  # safety margin before 150s limit
    
    # We'll optimize: positions (x,y) for N circles, radii r for N circles, and rectangle width w
    # height = 2 - w
    # Total variables: 2*N + N + 1 = 3*N + 1 = 64
    
    def evaluate_solution(xs, ys, rs, w):
        """Compute objective (sum of radii) and check validity."""
        h = 2.0 - w
        if w <= 0 or h <= 0:
            return -np.inf
        # Check containment
        for i in range(len(rs)):
            if xs[i] - rs[i] < -1e-9 or xs[i] + rs[i] > w + 1e-9:
                return -np.inf
            if ys[i] - rs[i] < -1e-9 or ys[i] + rs[i] > h + 1e-9:
                return -np.inf
        # Check non-overlap
        for i in range(len(rs)):
            for j in range(i+1, len(rs)):
                dx = xs[i] - xs[j]
                dy = ys[i] - ys[j]
                dist = np.sqrt(dx*dx + dy*dy)
                if dist < rs[i] + rs[j] - 1e-9:
                    return -np.inf
        return np.sum(rs)
    
    def penalty_objective(params, w_init, penalty_coeff):
        """Penalty-based objective for optimization."""
        # params: [x0, y0, r0, x1, y1, r1, ..., xN-1, yN-1, rN-1, w]
        xs = params[0::3]
        ys = params[1::3]
        rs = params[2::3]
        w = params[-1]
        h = 2.0 - w
        
        obj = np.sum(rs)
        penalty = 0.0
        
        # Containment penalties
        for i in range(N):
            if xs[i] - rs[i] < 0:
                penalty += penalty_coeff * (xs[i] - rs[i])**2
            if xs[i] + rs[i] > w:
                penalty += penalty_coeff * (xs[i] + rs[i] - w)**2
            if ys[i] - rs[i] < 0:
                penalty += penalty_coeff * (ys[i] - rs[i])**2
            if ys[i] + rs[i] > h:
                penalty += penalty_coeff * (ys[i] + rs[i] - h)**2
        
        # Non-overlap penalties
        for i in range(N):
            for j in range(i+1, N):
                dx = xs[i] - xs[j]
                dy = ys[i] - ys[j]
                dist = np.sqrt(dx*dx + dy*dy + 1e-12)
                min_dist = rs[i] + rs[j]
                if dist < min_dist:
                    penalty += penalty_coeff * (min_dist - dist)**2
        
        return -(obj - penalty)  # Minimize negative
    
    def gradient_penalty(params, w_init, penalty_coeff):
        """Compute gradient of penalty objective."""
        xs = params[0::3].copy()
        ys = params[1::3].copy()
        rs = params[2::3].copy()
        w = params[-1]
        h = 2.0 - w
        
        grad = np.zeros_like(params)
        
        # Gradient of sum(rs)
        grad[2::3] = 1.0  # d(sum r)/dr_i = 1
        
        # Containment gradients
        for i in range(N):
            xi, yi, ri = xs[i], ys[i], rs[i]
            # x_i - r_i < 0
            if xi - ri < 0:
                g = 2 * penalty_coeff * (xi - ri)
                grad[3*i] += g
                grad[3*i+2] -= g
            # x_i + r_i > w
            if xi + ri > w:
                g = 2 * penalty_coeff * (xi + ri - w)
                grad[3*i] += g
                grad[3*i+2] += g
                grad[-1] -= g  # d/dw of (xi+ri-w)^2 = -2(xi+ri-w)
            # y_i - r_i < 0
            if yi - ri < 0:
                g = 2 * penalty_coeff * (yi - ri)
                grad[3*i+1] += g
                grad[3*i+2] -= g
            # y_i + r_i > h = 2-w
            if yi + ri > h:
                g = 2 * penalty_coeff * (yi + ri - h)
                grad[3*i+1] += g
                grad[3*i+2] += g
                grad[-1] += g  # d/dw of (yi+ri-(2-w))^2 = 2(yi+ri-2+w) * 1 = 2(yi+ri-h)
        
        # Non-overlap gradients
        for i in range(N):
            for j in range(i+1, N):
                dx = xs[i] - xs[j]
                dy = ys[i] - ys[j]
                dist = np.sqrt(dx*dx + dy*dy + 1e-12)
                min_dist = rs[i] + rs[j]
                if dist < min_dist:
                    diff = min_dist - dist
                    g = 2 * penalty_coeff * diff / dist
                    # d/dx_i: g * dx/dist
                    grad[3*i] += g * dx
                    grad[3*i+1] += g * dy
                    # d/dx_j: -g * dx/dist
                    grad[3*j] -= g * dx
                    grad[3*j+1] -= g * dy
                    # d/dr_i: g * 1
                    grad[3*i+2] += g
                    # d/dr_j: g * 1
                    grad[3*j+2] += g
        
        # Gradient of -obj = -sum(rs), already accounted for
        # The objective is -(obj - penalty) = -obj + penalty
        # d(-obj)/dr_i = -1, d(penalty) computed above
        # So total grad[2::3] should be -1 + penalty_grad
        # Let me recompute: objective = -sum(rs) + penalty
        # d/dparams: -1 for each r_i (from -sum), plus penalty gradient
        
        # Actually let me redo this cleanly
        # f = -sum(rs) + penalty
        # df/dr_i = -1 + d(penalty)/dr_i
        # df/dx_i = d(penalty)/dx_i
        # df/dy_i = d(penalty)/dy_i
        # df/dw = d(penalty)/dw
        
        # Recompute properly
        grad = np.zeros_like(params)
        grad[2::3] = -1.0  # from -sum(rs)
        
        # Add penalty gradients
        for i in range(N):
            xi, yi, ri = xs[i], ys[i], rs[i]
            if xi - ri < 0:
                g = 2 * penalty_coeff * (xi - ri)
                grad[3*i] += g
                grad[3*i+2] += (-g)  # d/dr_i of (xi-ri)^2 = -2(xi-ri)
            if xi + ri > w:
                g = 2 * penalty_coeff * (xi + ri - w)
                grad[3*i] += g
                grad[3*i+2] += g
                grad[-1] += (-g)
            if yi - ri < 0:
                g = 2 * penalty_coeff * (yi - ri)
                grad[3*i+1] += g
                grad[3*i+2] += (-g)
            if yi + ri > h:
                g = 2 * penalty_coeff * (yi + ri - h)
                grad[3*i+1] += g
                grad[3*i+2] += g
                grad[-1] += g
        
        for i in range(N):
            for j in range(i+1, N):
                dx = xs[i] - xs[j]
                dy = ys[i] - ys[j]
                dist = np.sqrt(dx*dx + dy*dy + 1e-12)
                min_dist = rs[i] + rs[j]
                if dist < min_dist:
                    diff = min_dist - dist
                    g = 2 * penalty_coeff * diff / dist
                    grad[3*i] += g * dx
                    grad[3*i+1] += g * dy
                    grad[3*j] -= g * dx
                    grad[3*j+1] -= g * dy
                    grad[3*i+2] += g
                    grad[3*j+2] += g
        
        return grad
    
    # Phase 1: Coarse grid search over rectangle aspect ratio
    best_params = None
    best_obj = -np.inf
    
    # Try different aspect ratios
    w_values = np.linspace(0.3, 1.7, 15)
    
    for w0 in w_values:
        if time.time() > deadline - 60:
            break
        h0 = 2.0 - w0
        
        # Initial guess: grid placement
        side = int(np.ceil(np.sqrt(N)))
        xs_init = np.array([(i + 0.5) * w0 / side for i in range(N) % side])
        ys_init = np.array([(j + 0.5) * h0 / side for j in range(N) // side])
        r_init = 0.5 * min(w0, h0) / side * 0.9
        
        params_init = np.zeros(3*N + 1)
        params_init[0::3] = xs_init
        params_init[1::3] = ys_init
        params_init[2::3] = r_init
        params_init[-1] = w0
        
        # Run optimization with increasing penalty
        for penalty_coeff in [10, 100, 1000, 10000]:
            if time.time() > deadline - 30:
                break
            
            result = minimize(
                penalty_objective,
                params_init,
                jac=gradient_penalty,
                args=(w0, penalty_coeff),
                method='L-BFGS-B',
                options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-8}
            )
            
            params_init = result.x
            # Clip radii to be positive
            params_init[2::3] = np.maximum(params_init[2::3], 0.01)
        
        # Evaluate
        xs = params_init[0::3]
        ys = params_init[1::3]
        rs = params_init[2::3]
        w = params_init[-1]
        obj = evaluate_solution(xs, ys, rs, w)
        
        if obj > best_obj:
            best_obj = obj
            best_params = params_init.copy()
    
    if best_params is None:
        # Fallback to seed
        side = int(np.ceil(np.sqrt(N)))
        cells = [((i + 0.5) / side, (j + 0.5) / side, 0.5 / side - 1e-3) for i in range(side) for j in range(side)]
        return np.array(cells[:N])
    
    # Phase 2: High-precision polish
    if time.time() < deadline - 10:
        for penalty_coeff in [10000, 100000, 1000000]:
            if time.time() > deadline - 5:
                break
            
            result = minimize(
                penalty_objective,
                best_params,
                jac=gradient_penalty,
                args=(best_params[-1], penalty_coeff),
                method='L-BFGS-B',
                options={'maxiter': 500, 'ftol': 1e-15, 'gtol': 1e-10}
            )
            
            params_polish = result.x
            params_polish[2::3] = np.maximum(params_polish[2::3], 0.001)
            
            xs = params_polish[0::3]
            ys = params_polish[1::3]
            rs = params_polish[2::3]
            w = params_polish[-1]
            obj = evaluate_solution(xs, ys, rs, w)
            
            if obj > best_obj:
                best_obj = obj
                best_params = params_polish.copy()
    
    # Extract final solution
    xs = best_params[0::3]
    ys = best_params[1::3]
    rs = best_params[2::3]
    w = best_params[-1]
    h = 2.0 - w
    
    # Verify validity one more time
    final_obj = evaluate_solution(xs, ys, rs, w)
    if final_obj == -np.inf:
        # Fallback
        side = int(np.ceil(np.sqrt(N)))
        cells = [((i + 0.5) / side, (j + 0.5) / side, 0.5 / side - 1e-3) for i in range(side) for j in range(side)]
        return np.array(cells[:N])
    
    # Return in the expected format: array of (x, y, r)
    # The evaluator expects circles in a bounding rectangle with perimeter at most 4
    # Our rectangle has width w, height h=2-w, perimeter 2(w+h) = 4. Good.
    result = np.column_stack([xs, ys, rs])
    return result
# EVOLVE-BLOCK-END
