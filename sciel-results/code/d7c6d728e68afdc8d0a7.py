import numpy as np
import time
from scipy.optimize import minimize

N = 50

def _compute_radius(pts):
    """Compute the packing radius for a set of points."""
    pts = np.asarray(pts, dtype=float)
    if pts.shape != (N, 2):
        return 0.0
    if not np.all(np.isfinite(pts)):
        return 0.0
    # Boundary constraint
    dists_center = np.linalg.norm(pts, axis=1)
    boundary_min = np.min(1.0 - dists_center)
    if boundary_min <= 0:
        return 0.0
    # Pairwise distances
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dist = np.linalg.norm(diff, axis=2)
    # Only consider i < j
    iu = np.triu_indices(N, k=1)
    pair_min = np.min(dist[iu])
    r = min(boundary_min, pair_min / 2.0)
    return float(r)

def _objective_and_gradient(x):
    """Objective: negative of the minimum radius (to maximize).
    Returns (value, gradient)."""
    pts = x.reshape(N, 2)
    # Compute pairwise distances
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dist = np.linalg.norm(diff, axis=2)
    # Boundary distances
    dist_center = np.linalg.norm(pts, axis=1)
    
    # We want to maximize min( min_pair_dist/2, min_boundary_dist )
    # Equivalent to minimizing -min(...)
    # But min is non-differentiable. We can use a soft-min or just compute the gradient
    # of the active constraint.
    
    # For gradient descent on the "max-min" problem, we can use the fact that
    # the gradient of the minimum is the gradient of the active term.
    # However, a more robust approach for the optimizer is to minimize the sum of
    # squared violations if we frame it as a feasibility problem, or use a
    # barrier method.
    
    # Let's stick to a simple approach: minimize the negative of the radius.
    # To make it differentiable, we can use a soft minimum.
    # soft_min(a, b, k) = -1/k * log(exp(-k*a) + exp(-k*b))
    # But for N=50, there are 1225 pairs. Summing exp is expensive.
    
    # Alternative: Use the "max-min" formulation directly with a sub-gradient.
    # Or, simply minimize the sum of squared gaps where gap = 2r - d_ij for r > d_ij/2.
    # But we don't know r.
    
    # Let's use a standard technique: minimize the maximum of the "deficits".
    # Deficit_ij = 2*r - d_ij. We want all deficits <= 0.
    # This is a linear programming problem in r and positions, but positions are nonlinear.
    
    # Let's try a different objective: Minimize the sum of squared overlaps + boundary violations.
    # Overlap_ij = max(0, 2r - d_ij)^2. But r is a variable too.
    
    # Actually, the most effective way to solve this with scipy is to treat it as:
    # Minimize: -r
    # Subject to: d_ij >= 2r for all i<j
    #             |p_i| + r <= 1 for all i
    # Variables: p_1..p_N, r
    # This is a Non-Linear Programming problem.
    
    # Let's implement the objective and constraints for SLSQP.
    # Objective: -r
    # Constraints:
    #   d_ij - 2r >= 0
    #   1 - |p_i| - r >= 0
    
    # Since we are using a custom objective function for the helper, let's just return
    # the negative radius and let the caller handle constraints if needed, or
    # incorporate them into the objective with a penalty.
    
    # Penalty method is often more robust for this type of problem with SLSQP.
    # Objective: -r + lambda * (sum(max(0, 2r - d_ij)^2) + sum(max(0, |p_i| + r - 1)^2))
    
    # Let's compute the radius first.
    pair_min = np.min(dist[iu])
    boundary_min = np.min(1.0 - dist_center)
    r = min(boundary_min, pair_min / 2.0)
    
    # Compute penalty
    # We need to know which constraints are violated.
    # This requires knowing r.
    # If we use r from the current point, the penalty is 0 if feasible.
    # To guide the optimizer, we need a target r.
    # Let's use a fixed target r_target that we want to achieve, or just use the current r.
    # Actually, for a pure "maximize r" problem, the penalty method needs a target.
    
    # Let's switch strategy: Use the "max-min" objective directly.
    # Value = -min(boundary_min, pair_min/2)
    # Gradient:
    # If boundary is active: grad = -sum_{i active} (p_i / |p_i|) / N_active ? No.
    # The gradient of min(f_i) is the gradient of the active f_i.
    # If multiple are active, it's a convex combination.
    
    # Let's just return the value and a numerical gradient or zero, and use a method
    # that doesn't require gradients, or compute gradients carefully.
    # SLSQP can work without gradients (finite difference), but it's slow.
    # N=50 -> 100 variables. Finite difference takes 101 evaluations.
    # Each evaluation is fast. 101 * 500 iterations = 50k evals. Should be okay.
    
    return -r, None

def _constraints(x):
    """Constraints for SLSQP.
    Returns a list of constraint dicts.
    """
    pts = x[:2*N].reshape(N, 2)
    r = x[2*N]
    
    cons = []
    
    # Pairwise constraints: d_ij >= 2r  =>  d_ij - 2r >= 0
    # SLSQP constraints are of the form: fun(x) >= 0
    # We can vectorize this.
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dist = np.linalg.norm(diff, axis=2)
    iu = np.triu_indices(N, k=1)
    pair_constr = dist[iu] - 2*r
    
    # Boundary constraints: 1 - |p_i| >= r  =>  1 - |p_i| - r >= 0
    dist_center = np.linalg.norm(pts, axis=1)
    boundary_constr = 1.0 - dist_center - r
    
    # Combine into a single vector constraint function
    # SLSQP accepts a single function returning an array.
    # We'll create a wrapper that returns the concatenated array.
    
    return np.concatenate([pair_constr, boundary_constr])

def _constraint_func(x):
    """Combined constraint function for SLSQP."""
    return _constraints(x)

def _initial_hex_grid():
    """Generate a hexagonal grid initial guess."""
    # Target radius approx 0.125
    r = 0.125
    pts = []
    # Hex grid with spacing 2r
    # Rows
    row_h = r * np.sqrt(3)
    max_y = 1.0
    num_rows = int(max_y / row_h) + 2
    for j in range(-num_rows, num_rows + 1):
        y = j * row_h
        offset = r if j % 2 != 0 else 0
        max_x = np.sqrt(max(0, 1.0 - y*y))
        num_cols = int(max_x / (2*r)) + 2
        for i in range(-num_cols, num_cols + 1):
            x = i * 2*r + offset
            if x*x + y*y <= 1.0:
                pts.append([x, y])
    pts = np.array(pts)
    if len(pts) < N:
        # Fill with random points if not enough
        while len(pts) < N:
            p = np.random.randn(2) * 0.5
            pts = np.vstack([pts, p])
    else:
        # Select N closest to center
        d = np.linalg.norm(pts, axis=1)
        idx = np.argsort(d)
        pts = pts[idx[:N]]
    
    # Add r as the last variable
    x0 = np.concatenate([pts.flatten(), [r]])
    return x0

def construct():
    start_time = time.time()
    time_limit = 140.0  # Safety margin for 150s limit
    
    # 1. Generate initial guess
    x0 = _initial_hex_grid()
    
    # 2. Define constraints
    # SLSQP constraints: type='ineq' means fun(x) >= 0
    # We have 1225 pair constraints and 50 boundary constraints.
    # Total 1275 constraints. This might be heavy for SLSQP.
    
    # Alternative: Use a penalty method with a simpler optimizer or fewer constraints.
    # Or, use the "max-min" objective without explicit constraints, relying on the
    # optimizer to navigate the landscape.
    
    # Let's try SLSQP with the constraints first.
    # If it's too slow or fails, we'll fall back to a penalty method.
    
    # To reduce the number of constraints, we can only enforce the "most violated" ones
    # or use a penalty method.
    
    # Let's try a penalty method approach with BFGS or L-BFGS-B.
    # Objective: -r + mu * (sum(max(0, 2r - d_ij)^2) + sum(max(0, |p_i| + r - 1)^2))
    # This is smooth if we use a smooth max approximation, or we can use the exact max
    # and rely on the optimizer's robustness (though non-smooth).
    
    # Let's use a smooth approximation for the penalty:
    # max(0, x) approx x / (1 + exp(-k*x)) or similar.
    # Or simply: 0.5 * (x + sqrt(x^2 + eps))
    
    def penalty_objective(x):
        pts = x[:2*N].reshape(N, 2)
        r = x[2*N]
        
        # Radius term
        obj = -r
        
        # Pairwise penalty
        diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
        dist = np.linalg.norm(diff, axis=2)
        iu = np.triu_indices(N, k=1)
        pair_dist = dist[iu]
        violations = 2*r - pair_dist
        # Smooth penalty: 0.5 * (v + sqrt(v^2 + eps))
        eps = 1e-6
        pen_pair = 0.5 * (violations + np.sqrt(violations**2 + eps))
        obj += 10.0 * np.sum(np.maximum(0, pen_pair)) # Only penalize violations
        
        # Boundary penalty
        dist_center = np.linalg.norm(pts, axis=1)
        b_violations = dist_center + r - 1.0
        pen_b = 0.5 * (b_violations + np.sqrt(b_violations**2 + eps))
        obj += 10.0 * np.sum(np.maximum(0, pen_b))
        
        return obj
    
    # Initial r
    r0 = x0[2*N]
    
    # Bounds for r: [0, 1]
    # Bounds for positions: [-1, 1]
    bounds = [(-1.0, 1.0)] * (2*N) + [(0.0, 1.0)]
    
    # Optimize
    best_r = 0.0
    best_x = x0
    
    # Try SLSQP with constraints first
    try:
        res = minimize(
            lambda x: -x[2*N],  # Maximize r
            x0,
            method='SLSQP',
            bounds=bounds,
            constraints={'type': 'ineq', 'fun': _constraint_func},
            options={'maxiter': 200, 'ftol': 1e-6}
        )
        if time.time() - start_time < time_limit:
            r_res = res.x[2*N]
            if r_res > best_r:
                best_r = r_res
                best_x = res.x
    except Exception as e:
        pass
    
    # If SLSQP didn't work well or timed out, try penalty method
    if time.time() - start_time < time_limit - 10:
        try:
            res2 = minimize(
                penalty_objective,
                best_x,
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 500, 'ftol': 1e-8}
            )
            r_res2 = res2.x[2*N]
            # Check validity
            pts_res2 = res2.x[:2*N].reshape(N, 2)
            valid_r = _compute_radius(pts_res2)
            if valid_r > best_r:
                best_r = valid_r
                best_x = res2.x
        except Exception as e:
            pass
    
    # Final local refinement using the best point found
    if time.time() - start_time < time_limit - 5:
        # Do a few more iterations with SLSQP starting from the best point
        try:
            res3 = minimize(
                lambda x: -x[2*N],
                best_x,
                method='SLSQP',
                bounds=bounds,
                constraints={'type': 'ineq', 'fun': _constraint_func},
                options={'maxiter': 100, 'ftol': 1e-8}
            )
            r_res3 = res3.x[2*N]
            pts_res3 = res3.x[:2*N].reshape(N, 2)
            valid_r3 = _compute_radius(pts_res3)
            if valid_r3 > best_r:
                best_r = valid_r3
                best_x = res3.x
        except Exception as e:
            pass
    
    # Extract final points
    pts = best_x[:2*N].reshape(N, 2)
    
    # Ensure points are within the circle
    # If any point is outside, project it back
    for i in range(N):
        d = np.linalg.norm(pts[i])
        if d > 1.0 - best_r:
            if d > 1e-9:
                pts[i] = pts[i] * ((1.0 - best_r) / d)
    
    return pts

if __name__ == "__main__":
    pts = construct()
    r = _compute_radius(pts)
    print(f"Radius: {r}")
