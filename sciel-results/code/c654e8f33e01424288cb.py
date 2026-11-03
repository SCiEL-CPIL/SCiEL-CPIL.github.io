import numpy as np
from scipy.optimize import differential_evolution, minimize

N = 21

def construct():
    # Variables: x[0:N], y[N:2N], r[2N:3N], w[3N]
    # Total size: 3*N + 1
    
    # Bounds for Differential Evolution
    # x, y can be anywhere in [0, 2] initially, constraints will push them in.
    # r should be positive. Max possible r in a 2x0 or 1x1 rect is 0.5.
    # w is width, h = 2 - w.
    
    bounds = []
    for i in range(N):
        bounds.append((0.0, 2.0)) # x_i
    for i in range(N):
        bounds.append((0.0, 2.0)) # y_i
    for i in range(N):
        bounds.append((0.01, 0.5)) # r_i
    bounds.append((0.1, 1.9)) # w
    
    def objective(v):
        xs = v[0:N]
        ys = v[N:2*N]
        rs = v[2*N:3*N]
        w = v[3*N]
        h = 2.0 - w
        
        # We want to maximize sum(r). 
        # DE minimizes, so return -sum(r).
        # We don't add penalties here because DE is global and robust, 
        # but we might get infeasible points. 
        # However, standard DE doesn't handle constraints.
        # To guide DE towards feasible regions, we can add a heavy penalty 
        # for constraint violations in the objective function for the DE stage.
        
        penalty = 0.0
        # Boundary constraints
        for i in range(N):
            if xs[i] < rs[i]: penalty += (rs[i] - xs[i])**2
            if xs[i] > w - rs[i]: penalty += (xs[i] - (w - rs[i]))**2
            if ys[i] < rs[i]: penalty += (rs[i] - ys[i])**2
            if ys[i] > h - rs[i]: penalty += (ys[i] - (h - rs[i]))**2
            
        # Overlap constraints
        for i in range(N):
            for j in range(i+1, N):
                dx = xs[i] - xs[j]
                dy = ys[i] - ys[j]
                dist_sq = dx*dx + dy*dy
                min_dist = rs[i] + rs[j]
                if dist_sq < min_dist*min_dist:
                    # Penalty based on violation
                    diff = min_dist - np.sqrt(dist_sq)
                    penalty += diff*diff
        
        return -np.sum(rs) + 100.0 * penalty

    # Run Differential Evolution
    # tol=1e-6, maxiter=200, popsize=15
    # N=21, dim=64. This might be slow. 
    # 120s limit. DE with 64 dims and popsize 15 is 15*64*200 evals ~ 192,000 evals.
    # Each eval is O(N^2) = 441 ops. 192k * 441 ~ 85M ops. Should be fine in Python/NumPy if vectorized, 
    # but the loop in objective is slow.
    
    # Let's vectorize the penalty calculation in objective for speed.
    
    def objective_fast(v):
        xs = v[0:N]
        ys = v[N:2*N]
        rs = v[2*N:3*N]
        w = v[3*N]
        h = 2.0 - w
        
        # Vectorized boundary penalty
        # x_low = rs, x_high = w - rs
        # y_low = rs, y_high = h - rs
        
        pen_x_low = np.maximum(0, rs - xs)**2
        pen_x_high = np.maximum(0, xs - (w - rs))**2
        pen_y_low = np.maximum(0, rs - ys)**2
        pen_y_high = np.maximum(0, ys - (h - rs))**2
        
        boundary_pen = np.sum(pen_x_low + pen_x_high + pen_y_low + pen_y_high)
        
        # Vectorized overlap penalty
        # Create difference matrices
        dx = xs[:, None] - xs[None, :]
        dy = ys[:, None] - ys[None, :]
        dist_sq = dx**2 + dy**2
        
        min_dist = rs[:, None] + rs[None, :]
        
        # We only care about i < j to avoid double counting and self overlap
        # But it's easier to compute full matrix and mask diagonal
        # Violation is max(0, min_dist - sqrt(dist_sq))
        # To avoid sqrt, we can use squared distance? 
        # Penalty (ri+rj - dij)^2 is not smooth at contact.
        # Using (ri+rj)^2 - dij^2 is a proxy? No, standard is distance based.
        # Let's stick to sqrt for accuracy, N=21 is small.
        
        dist = np.sqrt(dist_sq)
        # Mask diagonal
        mask = np.ones((N, N), dtype=bool)
        np.fill_diagonal(mask, False)
        
        violations = np.maximum(0, min_dist - dist)
        overlap_pen = np.sum(violations[mask]**2)
        
        return -np.sum(rs) + 100.0 * (boundary_pen + overlap_pen)

    # DE settings
    # To save time, reduce popsize or maxiter?
    # 120s is generous. Let's try popsize=10, maxiter=100.
    # If it fails, SLSQP will fix it.
    
    try:
        de_result = differential_evolution(objective_fast, bounds, 
                                           maxiter=100, 
                                           popsize=10, 
                                           tol=1e-5,
                                           seed=42)
        x0 = de_result.x
    except Exception:
        # Fallback to random or grid if DE fails/crashes
        xs0 = np.random.uniform(0.1, 1.9, N)
        ys0 = np.random.uniform(0.1, 1.9, N)
        rs0 = np.full(N, 0.1)
        w0 = 1.0
        x0 = np.concatenate([xs0, ys0, rs0, [w0]])

    # Local refinement with SLSQP and explicit constraints
    def objective_local(v):
        xs = v[0:N]
        ys = v[N:2*N]
        rs = v[2*N:3*N]
        return -np.sum(rs)

    constraints = []
    # Boundary constraints
    for i in range(N):
        # x_i >= r_i  => x_i - r_i >= 0
        constraints.append({'type': 'ineq', 'fun': lambda v, i=i: v[i] - v[2*N+i]})
        # x_i <= w - r_i => w - r_i - x_i >= 0
        constraints.append({'type': 'ineq', 'fun': lambda v, i=i: v[3*N] - v[2*N+i] - v[i]})
        # y_i >= r_i
        constraints.append({'type': 'ineq', 'fun': lambda v, i=i: v[N+i] - v[2*N+i]})
        # y_i <= h - r_i = (2-w) - r_i => 2 - w - r_i - y_i >= 0
        constraints.append({'type': 'ineq', 'fun': lambda v, i=i: 2.0 - v[3*N] - v[2*N+i] - v[N+i]})
        
    # Overlap constraints
    for i in range(N):
        for j in range(i+1, N):
            # (xi-xj)^2 + (yi-yj)^2 >= (ri+rj)^2
            # (xi-xj)^2 + (yi-yj)^2 - (ri+rj)^2 >= 0
            constraints.append({
                'type': 'ineq', 
                'fun': lambda v, i=i, j=j: 
                    (v[i]-v[j])**2 + (v[N+i]-v[N+j])**2 - (v[2*N+i]+v[2*N+j])**2
            })
            
    # Bounds for SLSQP (tighter than DE)
    slsqp_bounds = []
    for i in range(N):
        slsqp_bounds.append((0.0, 2.0)) # x
    for i in range(N):
        slsqp_bounds.append((0.0, 2.0)) # y
    for i in range(N):
        slsqp_bounds.append((0.01, 0.5)) # r
    slsqp_bounds.append((0.1, 1.9)) # w

    # Run SLSQP
    # SLSQP can be slow with many constraints (N=21 -> ~200 constraints).
    # But it's local.
    
    res = minimize(objective_local, x0, method='SLSQP', 
                   bounds=slsqp_bounds, 
                   constraints=constraints,
                   options={'maxiter': 500, 'ftol': 1e-8})
    
    if res.success:
        x_sol = res.x
    else:
        # If SLSQP fails, try to repair or use DE result if valid
        x_sol = x0
        
    # Extract
    xs = x_sol[0:N]
    ys = x_sol[N:2*N]
    rs = x_sol[2*N:3*N]
    w = x_sol[3*N]
    h = 2.0 - w
    
    # Final validity check and repair
    # Check bounds
    for i in range(N):
        if xs[i] < rs[i]:
            xs[i] = rs[i]
        if xs[i] > w - rs[i]:
            xs[i] = w - rs[i]
        if ys[i] < rs[i]:
            ys[i] = rs[i]
        if ys[i] > h - rs[i]:
            ys[i] = h - rs[i]
            
    # Check overlaps and shrink if necessary
    # Simple iterative shrink
    valid = False
    for _ in range(10):
        valid = True
        for i in range(N):
            for j in range(i+1, N):
                dx = xs[i] - xs[j]
                dy = ys[i] - ys[j]
                dist = np.sqrt(dx*dx + dy*dy)
                min_d = rs[i] + rs[j]
                if dist < min_d - 1e-6:
                    valid = False
                    # Shrink both
                    factor = 0.95
                    rs[i] *= factor
                    rs[j] *= factor
                    break
            if not valid:
                break
        if valid:
            break
            
    if not valid:
        # Fallback to a known good grid
        side = 5
        xs_f = []
        ys_f = []
        rs_f = []
        count = 0
        for i in range(side):
            for j in range(side):
                if count >= N:
                    break
                x = (i + 0.5) / side
                y = (j + 0.5) / side
                xs_f.append(x)
                ys_f.append(y)
                rs_f.append(0.5 / side * 0.9)
                count += 1
        return np.column_stack([xs_f, ys_f, rs_f])
        
    out = np.zeros((N, 3))
    out[:, 0] = xs
    out[:, 1] = ys
    out[:, 2] = rs
    
    return out
