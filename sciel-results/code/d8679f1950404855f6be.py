import numpy as np
import time
import sys
from scipy.optimize import minimize

N = 26
DEADLINE = 110.0  # Safety margin for 120s limit

def construct():
    start_time = time.time()
    
    def get_state(x, y, r):
        return np.concatenate([x, y, r])

    def unpack_state(z):
        x = z[0:N]
        y = z[N:2*N]
        r = z[2*N:3*N]
        return x, y, r

    def is_valid(x, y, r, tol=1e-7):
        # Check boundaries
        if np.any(r > x + tol) or np.any(r > y + tol) or np.any(r > 1-x + tol) or np.any(r > 1-y + tol):
            return False
        # Check overlaps
        for i in range(N):
            for j in range(i+1, N):
                dx = x[i] - x[j]
                dy = y[i] - y[j]
                dist = np.sqrt(dx*dx + dy*dy)
                if dist < r[i] + r[j] - tol:
                    return False
        return True

    def sum_radii(x, y, r):
        return np.sum(r)

    def repair_and_validate(x, y, r, max_iter=20):
        """Iteratively shrink radii to ensure validity, returning valid arrays or None."""
        x = x.copy()
        y = y.copy()
        r = r.copy()
        for _ in range(max_iter):
            valid = True
            # Check boundaries
            for i in range(N):
                min_dist_b = min(x[i], y[i], 1-x[i], 1-y[i])
                if r[i] > min_dist_b:
                    r[i] = min_dist_b * (1 - 1e-6)
                    valid = False
            
            # Check overlaps
            for i in range(N):
                for j in range(i+1, N):
                    dx = x[i] - x[j]
                    dy = y[i] - y[j]
                    dist = np.sqrt(dx*dx + dy*dy)
                    req_dist = r[i] + r[j]
                    if dist < req_dist:
                        # Scale down both radii so they just touch (with margin)
                        if req_dist > 0:
                            scale = (dist * (1 - 1e-6)) / req_dist
                        else:
                            scale = 0.5
                        r[i] *= scale
                        r[j] *= scale
                        valid = False
            
            if valid:
                return x, y, r
        # If still invalid after max_iter, try a more aggressive global shrink
        max_viol = 0.0
        for i in range(N):
            for j in range(i+1, N):
                dx = x[i] - x[j]
                dy = y[i] - y[j]
                dist = np.sqrt(dx*dx + dy*dy)
                req_dist = r[i] + r[j]
                if dist < req_dist:
                    max_viol = max(max_viol, (req_dist - dist) / max(req_dist, 1e-9))
        if max_viol > 0:
            shrink = 1.0 / (1.0 + max_viol) * 0.99
            r = r * shrink
            # Re-check
            valid = True
            for i in range(N):
                min_dist_b = min(x[i], y[i], 1-x[i], 1-y[i])
                if r[i] > min_dist_b:
                    r[i] = min_dist_b * (1 - 1e-6)
                    valid = False
            for i in range(N):
                for j in range(i+1, N):
                    dx = x[i] - x[j]
                    dy = y[i] - y[j]
                    dist = np.sqrt(dx*dx + dy*dy)
                    if dist < r[i] + r[j] - 1e-7:
                        valid = False
            if valid:
                return x, y, r
        return None

    def objective(z, penalty_weight=500.0):
        x, y, r = unpack_state(z)
        
        # Base objective: maximize sum of radii -> minimize -sum
        obj = -np.sum(r)
        
        # Boundary violations
        viol_x = np.maximum(0, r - x)
        viol_y = np.maximum(0, r - y)
        viol_1x = np.maximum(0, r - (1 - x))
        viol_1y = np.maximum(0, r - (1 - y))
        
        boundary_violation = np.sum(viol_x**2) + np.sum(viol_y**2) + np.sum(viol_1x**2) + np.sum(viol_1y**2)
        
        # Pairwise overlap violations
        overlap_penalty = 0.0
        for i in range(N):
            for j in range(i+1, N):
                dx = x[i] - x[j]
                dy = y[i] - y[j]
                dist = np.sqrt(dx*dx + dy*dy)
                min_dist = r[i] + r[j]
                if dist < min_dist:
                    overlap_penalty += (min_dist - dist)**2
        
        return obj + penalty_weight * (boundary_violation + overlap_penalty)

    def gradient(z, penalty_weight=500.0):
        x, y, r = unpack_state(z)
        grad = np.zeros(3*N)
        
        # Gradient of -sum(r)
        grad[2*N:3*N] = -1.0
        
        # Boundary gradients
        viol_x = np.maximum(0, r - x)
        grad[0:N] -= 2 * viol_x
        grad[2*N:3*N] += 2 * viol_x
        
        viol_y = np.maximum(0, r - y)
        grad[N:2*N] -= 2 * viol_y
        grad[2*N:3*N] += 2 * viol_y
        
        viol_1x = np.maximum(0, r - (1 - x))
        grad[0:N] += 2 * viol_1x
        grad[2*N:3*N] -= 2 * viol_1x
        
        viol_1y = np.maximum(0, r - (1 - y))
        grad[N:2*N] += 2 * viol_1y
        grad[2*N:3*N] -= 2 * viol_1y
        
        # Overlap gradients
        for i in range(N):
            for j in range(i+1, N):
                dx = x[i] - x[j]
                dy = y[i] - y[j]
                dist_sq = dx*dx + dy*dy
                dist = np.sqrt(dist_sq) if dist_sq > 1e-12 else 1e-12
                min_dist = r[i] + r[j]
                
                if dist < min_dist:
                    diff = min_dist - dist
                    factor = 2 * diff
                    
                    grad[i] += factor * (-dx / dist)
                    grad[N+i] += factor * (-dy / dist)
                    
                    grad[j] += factor * (dx / dist)
                    grad[N+j] += factor * (dy / dist)
                    
                    grad[2*N+i] += factor
                    grad[2*N+j] += factor
        
        return penalty_weight * grad

    def local_optimize(x0, y0, r0, max_iter=500, penalty_weight=500.0):
        """Run local optimization from initial guess."""
        z0 = get_state(x0, y0, r0)
        
        bounds = []
        for i in range(N):
            bounds.append((0, 1)) # x
            bounds.append((0, 1)) # y
            bounds.append((0.001, 0.5)) # r
            
        try:
            res = minimize(
                objective, 
                z0, 
                method='L-BFGS-B', 
                jac=gradient, 
                bounds=bounds,
                args=(penalty_weight,),
                options={'maxiter': max_iter, 'ftol': 1e-10, 'gtol': 1e-8}
            )
            
            x_opt, y_opt, r_opt = unpack_state(res.x)
            
            # Ensure radii are positive
            r_opt = np.maximum(r_opt, 0.001)
            
            # Try to validate and repair
            repaired = repair_and_validate(x_opt, y_opt, r_opt)
            if repaired is not None:
                return repaired
            else:
                return None
        except Exception:
            return None

    def simulated_annealing(x0, y0, r0, max_iter=5000, T_start=0.05, T_end=0.001):
        """Force-directed / simulated annealing approach."""
        x = x0.copy()
        y = y0.copy()
        r = r0.copy()
        
        rng = np.random.default_rng(12345)
        
        best_x, best_y, best_r = x.copy(), y.copy(), r.copy()
        best_score = sum_radii(x, y, r)
        
        for it in range(max_iter):
            # Cooling schedule
            T = T_start * (T_end / T_start) ** (it / max_iter)
            
            # Compute forces
            fx = np.zeros(N)
            fy = np.zeros(N)
            fr = np.zeros(N)
            
            # Repulsion between overlapping circles
            for i in range(N):
                for j in range(i+1, N):
                    dx = x[i] - x[j]
                    dy = y[i] - y[j]
                    dist = np.sqrt(dx*dx + dy*dy)
                    if dist < 1e-9:
                        dist = 1e-9
                        dx = rng.uniform(-1e-6, 1e-6)
                        dy = rng.uniform(-1e-6, 1e-6)
                    min_dist = r[i] + r[j]
                    if dist < min_dist:
                        force = (min_dist - dist) / dist
                        fx[i] += force * dx
                        fy[i] += force * dy
                        fx[j] -= force * dx
                        fy[j] -= force * dy
                        # Push radii down slightly when overlapping
                        fr[i] -= 0.5
                        fr[j] -= 0.5
            
            # Boundary forces
            for i in range(N):
                if x[i] < r[i]:
                    fx[i] += (r[i] - x[i])
                    fr[i] -= 0.5
                elif x[i] > 1 - r[i]:
                    fx[i] -= (x[i] - (1 - r[i]))
                    fr[i] -= 0.5
                if y[i] < r[i]:
                    fy[i] += (r[i] - y[i])
                    fr[i] -= 0.5
                elif y[i] > 1 - r[i]:
                    fy[i] -= (y[i] - (1 - r[i]))
                    fr[i] -= 0.5
            
            # Growth force (try to grow radii)
            fr += 0.1
            
            # Propose move
            step_size = 0.01 * (1 + T * 10)
            x_prop = x + fx * step_size + rng.normal(0, T * 0.5, N)
            y_prop = y + fy * step_size + rng.normal(0, T * 0.5, N)
            r_prop = r + fr * step_size * 0.1 + rng.normal(0, T * 0.1, N)
            r_prop = np.clip(r_prop, 0.001, 0.5)
            
            # Check if proposal is valid (or nearly valid)
            valid = True
            for i in range(N):
                if r_prop[i] > x_prop[i] or r_prop[i] > y_prop[i] or r_prop[i] > 1-x_prop[i] or r_prop[i] > 1-y_prop[i]:
                    valid = False
                    break
            if valid:
                for i in range(N):
                    for j in range(i+1, N):
                        dx = x_prop[i] - x_prop[j]
                        dy = y_prop[i] - y_prop[j]
                        dist = np.sqrt(dx*dx + dy*dy)
                        if dist < r_prop[i] + r_prop[j] - 1e-6:
                            valid = False
                            break
                    if not valid:
                        break
            
            if valid:
                score = sum_radii(x_prop, y_prop, r_prop)
                if score > best_score:
                    best_score = score
                    best_x, best_y, best_r = x_prop.copy(), y_prop.copy(), r_prop.copy()
                x, y, r = x_prop, y_prop, r_prop
            else:
                # Accept with probability exp(-delta/T) if worse
                # For simplicity, just reject invalid moves
                pass
            
            # Time check
            if time.time() - start_time > DEADLINE - 10:
                break
        
        repaired = repair_and_validate(best_x, best_y, best_r)
        if repaired is not None:
            return repaired
        return None

    def hexagonal_init(n_circles, perturb=0.0):
        """Create hexagonal packing initialization with optional perturbation."""
        x = np.zeros(n_circles)
        y = np.zeros(n_circles)
        r = np.zeros(n_circles)
        
        # For 26 circles, use a hexagonal pattern
        # Rows of 5, 5, 5, 5, 5, 1 or similar
        # Let's use a more standard hex grid
        cols = 6
        rows = 5
        
        # Position in hex grid
        idx = 0
        for row in range(rows):
            for col in range(cols):
                if idx >= n_circles:
                    break
                # Hex offset
                x_offset = 0.5 if row % 2 == 1 else 0.0
                x[idx] = (col + 0.5 + x_offset) / (cols + 0.5)
                y[idx] = (row + 0.5) / (rows + 0.5)
                
                # Radius estimate based on spacing
                spacing_x = 1.0 / (cols + 0.5)
                spacing_y = 1.0 / (rows + 0.5)
                r[idx] = 0.45 * min(spacing_x, spacing_y)
                
                idx += 1
        
        # Add remaining circles if needed
        while idx < n_circles:
            x[idx] = 0.5
            y[idx] = 0.5
            r[idx] = 0.05
            idx += 1
        
        # Perturb
        if perturb > 0:
            rng = np.random.default_rng(999)
            x += rng.uniform(-perturb, perturb, n_circles)
            y += rng.uniform(-perturb, perturb, n_circles)
            r *= rng.uniform(0.9, 1.1, n_circles)
        
        # Clip to valid range
        for i in range(n_circles):
            x[i] = np.clip(x[i], 0, 1)
            y[i] = np.clip(y[i], 0, 1)
            r[i] = min(r[i], min(x[i], 1-x[i], y[i], 1-y[i]) * 0.99)
            r[i] = max(r[i], 0.01)
        
        return x, y, r

    def grid_init(n_circles, perturb=0.0):
        """Grid initialization."""
        side = int(np.ceil(np.sqrt(n_circles)))
        x = np.zeros(n_circles)
        y = np.zeros(n_circles)
        r = np.zeros(n_circles)
        
        for i in range(n_circles):
            row = i // side
            col = i % side
            x[i] = (col + 0.5) / side
            y[i] = (row + 0.5) / side
            r[i] = 0.45 / side
        
        if perturb > 0:
            rng = np.random.default_rng(888)
            x += rng.uniform(-perturb, perturb, n_circles)
            y += rng.uniform(-perturb, perturb, n_circles)
        
        for i in range(n_circles):
            x[i] = np.clip(x[i], 0, 1)
            y[i] = np.clip(y[i], 0, 1)
            r[i] = min(r[i], min(x[i], 1-x[i], y[i], 1-y[i]) * 0.99)
            r[i] = max(r[i], 0.01)
        
        return x, y, r

    def random_init(rng):
        """Random initialization."""
        x = rng.uniform(0.1, 0.9, N)
        y = rng.uniform(0.1, 0.9, N)
        r = rng.uniform(0.03, 0.1, N)
        return x, y, r

    best_score = 0.0
    best_state = None
    
    rng = np.random.default_rng(42)
    
    # Phase 1: Multi-start local optimization with various initializations
    # Hexagonal starts
    for k in range(8):
        if time.time() - start_time > DEADLINE - 20:
            break
        x0, y0, r0 = hexagonal_init(N, perturb=0.02 * k)
        result = local_optimize(x0, y0, r0, max_iter=400, penalty_weight=500.0)
        if result is not None:
            x_opt, y_opt, r_opt = result
            score = sum_radii(x_opt, y_opt, r_opt)
            if score > best_score:
                best_score = score
                best_state = (x_opt.copy(), y_opt.copy(), r_opt.copy())
    
    # Grid starts
    for k in range(5):
        if time.time() - start_time > DEADLINE - 20:
            break
        x0, y0, r0 = grid_init(N, perturb=0.01 * k)
        result = local_optimize(x0, y0, r0, max_iter=400, penalty_weight=500.0)
        if result is not None:
            x_opt, y_opt, r_opt = result
            score = sum_radii(x_opt, y_opt, r_opt)
            if score > best_score:
                best_score = score
                best_state = (x_opt.copy(), y_opt.copy(), r_opt.copy())
    
    # Random starts
    for k in range(10):
        if time.time() - start_time > DEADLINE - 20:
            break
        x0, y0, r0 = random_init(rng)
        result = local_optimize(x0, y0, r0, max_iter=300, penalty_weight=500.0)
        if result is not None:
            x_opt, y_opt, r_opt = result
            score = sum_radii(x_opt, y_opt, r_opt)
            if score > best_score:
                best_score = score
                best_state = (x_opt.copy(), y_opt.copy(), r_opt.copy())
    
    # Phase 2: Simulated annealing / force-directed refinement from best states
    if best_state is not None:
        x_best, y_best, r_best = best_state
        sa_result = simulated_annealing(x_best, y_best, r_best, max_iter=3000, T_start=0.03, T_end=0.001)
        if sa_result is not None:
            x_opt, y_opt, r_opt = sa_result
            score = sum_radii(x_opt, y_opt, r_opt)
            if score > best_score:
                best_score = score
                best_state = (x_opt.copy(), y_opt.copy(), r_opt.copy())
    
    # Phase 3: More local optimization from the best state found
    if best_state is not None:
        x_best, y_best, r_best = best_state
        # Try with different penalty weights
        for pw in [300.0, 800.0, 1000.0]:
            if time.time() - start_time > DEADLINE - 5:
                break
            result = local_optimize(x_best, y_best, r_best, max_iter=300, penalty_weight=pw)
            if result is not None:
                x_opt, y_opt, r_opt = result
                score = sum_radii(x_opt, y_opt, r_opt)
                if score > best_score:
                    best_score = score
                    best_state = (x_opt.copy(), y_opt.copy(), r_opt.copy())
    
    # Final fallback
    if best_state is None:
        side = int(np.ceil(np.sqrt(N)))
        cells = [((i + 0.5) / side, (j + 0.5) / side, 0.5 / side - 1e-3) for i in range(side) for j in range(side)]
        return np.array(cells[:N])
    
    x_final, y_final, r_final = best_state
    
    # Final repair to ensure strict validity
    repaired = repair_and_validate(x_final, y_final, r_final, max_iter=30)
    if repaired is not None:
        x_final, y_final, r_final = repaired
    
    result = np.column_stack([x_final, y_final, r_final])
    return result

if __name__ == "__main__":
    res = construct()
    print(np.sum(res[:, 2]))
