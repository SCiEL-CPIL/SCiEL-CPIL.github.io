"""Circle packing in perimeter-4 rectangle: pack 21 circles, maximize sum of radii."""
import time
import numpy as np
from scipy.optimize import minimize

N = 21
DEADLINE = time.time() + 140.0  # safety margin before 150s kill


def _validity_check(circles, tol=1e-7):
    """Check if circles are valid: disjoint interiors, inside bounding rect with perimeter <= 4."""
    n = len(circles)
    if n != N:
        return False
    xs = circles[:, 0]
    ys = circles[:, 1]
    rs = circles[:, 2]
    if np.any(rs <= 0):
        return False
    # Bounding rectangle
    xmin = np.min(xs - rs)
    xmax = np.max(xs + rs)
    ymin = np.min(ys - rs)
    ymax = np.max(ys + rs)
    W = xmax - xmin
    H = ymax - ymin
    perimeter = 2 * (W + H)
    if perimeter > 4.0 + tol:
        return False
    # Check pairwise non-overlap
    for i in range(n):
        for j in range(i + 1, n):
            dx = xs[i] - xs[j]
            dy = ys[i] - ys[j]
            dist = np.sqrt(dx * dx + dy * dy)
            if dist < rs[i] + rs[j] - tol:
                return False
    return True


def _sum_radii(circles):
    return float(np.sum(circles[:, 2]))


def _force_relax(x, y, r, W, H, max_iter=1000, tol=1e-6):
    """
    Fast force-directed relaxation to resolve overlaps and boundary violations.
    x, y, r are arrays of length N.
    W, H are the fixed rectangle dimensions (W+H=2).
    Returns relaxed x, y, r.
    """
    n = len(x)
    x = x.copy()
    y = y.copy()
    r = r.copy()
    
    # Initialize forces
    fx = np.zeros(n)
    fy = np.zeros(n)
    
    for it in range(max_iter):
        if time.time() > DEADLINE - 5:
            break
            
        fx = np.zeros(n)
        fy = np.zeros(n)
        
        # Boundary forces: push circles inside [0, W] x [0, H]
        # Left wall
        viol_l = np.maximum(r - x, 0)
        fx += viol_l
        # Right wall
        viol_r = np.maximum(x + r - W, 0)
        fx -= viol_r
        # Bottom wall
        viol_b = np.maximum(r - y, 0)
        fy += viol_b
        # Top wall
        viol_t = np.maximum(y + r - H, 0)
        fy -= viol_t
        
        # Pairwise repulsion
        for i in range(n):
            for j in range(i + 1, n):
                dx = x[i] - x[j]
                dy = y[i] - y[j]
                d_sq = dx*dx + dy*dy
                d = np.sqrt(d_sq)
                min_d = r[i] + r[j]
                
                if d < min_d:
                    if d < 1e-8:
                        # Random push if coincident
                        dx = np.random.uniform(-0.1, 0.1)
                        dy = np.random.uniform(-0.1, 0.1)
                        d = np.sqrt(dx*dx + dy*dy)
                        if d < 1e-8: continue
                    
                    # Spring force: F = k * (min_d - d)
                    # We use a simple displacement correction
                    overlap = min_d - d
                    # Distribute correction based on relative radii? 
                    # Simple equal split is often stable enough for relaxation
                    push_x = (dx / d) * overlap * 0.5
                    push_y = (dy / d) * overlap * 0.5
                    
                    fx[i] += push_x
                    fy[i] += push_y
                    fx[j] -= push_x
                    fy[j] -= push_y
        
        # Apply forces with damping
        # Adaptive step size
        max_force = np.max(np.abs(fx)) + np.max(np.abs(fy))
        if max_force < tol:
            break
            
        step = 0.5 # Damping factor
        x += step * fx
        y += step * fy
        
        # Hard clamp to prevent flying out (optional, helps stability)
        # x = np.clip(x, r, W - r)
        # y = np.clip(y, r, H - r)
        
    return x, y, r

def _slsqp_optimize(x0, y0, r0, W0, H0, max_iter=200):
    """
    Optimize circle positions and radii in a fixed W x H rectangle (W+H=2).
    Variables: x, y, r for each circle.
    Objective: maximize sum(r) -> minimize -sum(r)
    Constraints:
    1. r_i >= 0 (handled by bounds)
    2. r_i <= x_i <= W - r_i
    3. r_i <= y_i <= H - r_i
    4. dist(i,j) >= r_i + r_j
    """
    n = N
    
    def objective(v):
        # v = [x0, y0, r0, x1, y1, r1, ...]
        xs = v[0::3]
        ys = v[1::3]
        rs = v[2::3]
        
        # Small penalty for negative radii (should be handled by bounds, but just in case)
        pen = 0.0
        if np.any(rs < 0):
            pen += 1000.0 * np.sum(-rs)
            
        # Penalty for being outside box (SLSQP handles constraints, but helps convergence)
        # We rely on constraints for strictness, but a soft penalty helps if constraints are slightly violated during iterations
        # Actually, SLSQP is better with hard constraints. Let's stick to -sum(r)
        # But wait, if the initial guess is bad, SLSQP might struggle.
        # The relaxation step should ensure a feasible or near-feasible start.
        
        return -np.sum(rs) + pen

    def grad_obj(v):
        g = np.zeros_like(v)
        g[2::3] = -1.0
        return g

    # Constraints
    constraints = []
    
    # Boundary constraints:
    # x_i - r_i >= 0
    # W - x_i - r_i >= 0
    # y_i - r_i >= 0
    # H - y_i - r_i >= 0
    
    # It's more efficient to define these as a single function returning an array of inequalities
    # But SLSQP expects a dict with 'fun' returning array.
    
    def boundary_constraints(v):
        xs = v[0::3]
        ys = v[1::3]
        rs = v[2::3]
        c1 = xs - rs
        c2 = W0 - xs - rs
        c3 = ys - rs
        c4 = H0 - ys - rs
        return np.concatenate([c1, c2, c3, c4])
    
    constraints.append({'type': 'ineq', 'fun': boundary_constraints})
    
    # Overlap constraints:
    # dist(i,j)^2 - (r_i + r_j)^2 >= 0
    # Using squared distance avoids sqrt in the constraint function, making it faster and smoother.
    # However, the gradient of sqrt is problematic at 0. Squared is better.
    
    def overlap_constraints(v):
        xs = v[0::3]
        ys = v[1::3]
        rs = v[2::3]
        
        cons = []
        for i in range(n):
            for j in range(i + 1, n):
                dx = xs[i] - xs[j]
                dy = ys[i] - ys[j]
                d_sq = dx*dx + dy*dy
                r_sum = rs[i] + rs[j]
                cons.append(d_sq - r_sum*r_sum)
        return np.array(cons)
    
    constraints.append({'type': 'ineq', 'fun': overlap_constraints})
    
    # Bounds
    bounds = []
    for i in range(n):
        bounds.append((0.0, W0)) # x
        bounds.append((0.0, H0)) # y
        bounds.append((1e-6, min(W0, H0)/2.0)) # r (max radius is half the smaller dimension)
    
    x0_flat = np.column_stack([x0, y0, r0]).flatten()
    
    try:
        res = minimize(objective, x0_flat, method='SLSQP', 
                       bounds=bounds, 
                       constraints=constraints,
                       options={'maxiter': max_iter, 'ftol': 1e-9, 'maxls': 100})
        
        if res.success or res.fun < -np.sum(r0) + 1e-6:
            x_opt = res.x[0::3]
            y_opt = res.x[1::3]
            r_opt = res.x[2::3]
            return x_opt, y_opt, r_opt
    except Exception:
        pass
        
    return None


def construct():
    t_start = time.time()
    best_val = -1
    best_circles = None
    
    # List of aspect ratios to try (W, where H = 2-W)
    # Perimeter is 4, so W+H=2.
    # Square is W=1, H=1.
    # Elongated: W=1.5, H=0.5 or W=0.5, H=1.5
    aspect_ratios = [1.0, 0.8, 1.25, 0.7, 1.4, 0.6, 1.6, 0.5, 1.8, 0.45]
    
    # Add some random aspect ratios
    rng = np.random.RandomState(42)
    for _ in range(10):
        w = rng.uniform(0.5, 1.5)
        aspect_ratios.append(w)
        
    # Remove duplicates roughly
    seen = set()
    unique_ratios = []
    for w in aspect_ratios:
        key = round(w, 2)
        if key not in seen:
            seen.add(key)
            unique_ratios.append(w)
            
    for W in unique_ratios:
        if time.time() > DEADLINE - 20:
            break
            
        H = 2.0 - W
        if H < 0.2 or W < 0.2:
            continue
            
        # Generate initial guess
        # Grid initialization
        side = int(np.ceil(np.sqrt(N)))
        x_init = np.zeros(N)
        y_init = np.zeros(N)
        r_init = np.zeros(N)
        
        # Fill grid
        idx = 0
        for i in range(side):
            for j in range(side):
                if idx >= N:
                    break
                # Center of cell
                x_init[idx] = (j + 0.5) * (W / side)
                y_init[idx] = (i + 0.5) * (H / side)
                # Initial radius: half the cell size minus a small margin
                cell_w = W / side
                cell_h = H / side
                r_init[idx] = 0.4 * min(cell_w, cell_h)
                idx += 1
        
        # Relaxation
        x_rel, y_rel, r_rel = _force_relax(x_init, y_init, r_init, W, H, max_iter=200)
        
        # SLSQP Optimization
        if time.time() < DEADLINE - 30:
            res = _slsqp_optimize(x_rel, y_rel, r_rel, W, H, max_iter=150)
            
            if res is not None:
                x_opt, y_opt, r_opt = res
                circles = np.column_stack([x_opt, y_opt, r_opt])
                
                # Check validity (SLSQP might leave slight violations)
                if _validity_check(circles):
                    val = _sum_radii(circles)
                    if val > best_val:
                        best_val = val
                        best_circles = circles.copy()
                        
                else:
                    # Try to repair by shrinking radii slightly
                    circles_repair = circles.copy()
                    circles_repair[:, 2] *= 0.98
                    if _validity_check(circles_repair):
                        val = _sum_radii(circles_repair)
                        if val > best_val:
                            best_val = val
                            best_circles = circles_repair.copy()
                            
        # Also try the relaxed result directly if SLSQP failed or wasn't run
        if time.time() < DEADLINE - 10:
            circles_rel = np.column_stack([x_rel, y_rel, r_rel])
            if _validity_check(circles_rel):
                val = _sum_radii(circles_rel)
                if val > best_val:
                    best_val = val
                    best_circles = circles_rel.copy()
    
    # Final Polish: Try to increase radii slightly if valid
    if best_circles is not None and time.time() < DEADLINE - 5:
        X = best_circles.copy()
        improved = True
        while improved and time.time() < DEADLINE - 2:
            improved = False
            for i in range(N):
                old_r = X[i, 2]
                # Try increasing radius by 0.1%
                X[i, 2] = old_r * 1.001
                if _validity_check(X):
                    improved = True
                else:
                    X[i, 2] = old_r
        if _validity_check(X):
            best_circles = X
            best_val = _sum_radii(X)

    if best_circles is None:
        # Fallback: simple grid in 1x1 square
        side = int(np.ceil(np.sqrt(N)))
        cells = []
        for i in range(side):
            for j in range(side):
                if len(cells) < N:
                    cells.append([(j + 0.5) / side, (i + 0.5) / side, 0.4 / side])
        return np.array(cells, dtype=float)
        
    return best_circles

# Quick self-test
if __name__ == "__main__":
    result = construct()
    print(f"Sum of radii: {_sum_radii(result)}")
    print(f"Valid: {_validity_check(result)}")
    xs, ys, rs = result[:,0], result[:,1], result[:,2]
    print(f"Perimeter: {2*(np.max(xs+rs)-np.min(xs-rs)+np.max(ys+rs)-np.min(ys-rs))}")

