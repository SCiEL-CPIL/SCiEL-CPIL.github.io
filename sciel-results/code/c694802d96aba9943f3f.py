"""Circle packing in unit square: Voronoi-based radii with multi-stage NLP optimization."""
import numpy as np
import time
from scipy.optimize import minimize
from scipy.spatial import Voronoi, Delaunay
import itertools

N = 26
DEADLINE = 145.0  # safety margin before 150s kill


def _voronoi_radii(points, n_iter=30):
    """Compute radii for points using Voronoi cells, iteratively."""
    n = len(points)
    if n < 3:
        return np.full(n, 0.1)
    
    # Pad with boundary points for Voronoi
    pad = 0.5
    extended = np.vstack([
        points,
        np.array([[-pad]*n, [1+pad]*n, [-pad, -pad]*n//2, [1+pad, 1+pad]*n//2])
    ])
    
    radii = np.full(n, 0.05)
    
    for _ in range(n_iter):
        try:
            vor = Voronoi(extended)
            for i in range(n):
                # Find the Voronoi region for point i
                region = vor.regions[vor.point_region[i]]
                if -1 in region:
                    continue
                vertices = vor.vertices[region]
                if len(vertices) < 3:
                    continue
                # Compute inradius: max distance to edges
                max_r = 0
                for j in range(len(vertices)):
                    v1 = vertices[j]
                    v2 = vertices[(j+1) % len(vertices)]
                    # Distance from point to line segment
                    dx = v2[0] - v1[0]
                    dy = v2[1] - v1[1]
                    if dx == 0 and dy == 0:
                        continue
                    t = ((points[i][0]-v1[0])*dx + (points[i][1]-v1[1])*dy) / (dx*dx + dy*dy)
                    t = max(0, min(1, t))
                    px = v1[0] + t*dx
                    py = v1[1] + t*dy
                    dist = np.sqrt((points[i][0]-px)**2 + (points[i][1]-py)**2)
                    max_r = max(max_r, dist)
                # Also check distance to boundaries
                max_r = min(max_r, points[i][0], 1-points[i][0], points[i][1], 1-points[i][1])
                radii[i] = max_r * 0.98  # slight shrink for safety
        except Exception:
            pass
    
    # Ensure minimum radius
    radii = np.maximum(radii, 0.02)
    return radii


def _make_grid_initial():
    """Grid-based initial guess with corner emphasis."""
    positions = []
    
    # 4 corner circles
    cr = 0.16
    corners = [(cr, cr), (1-cr, cr), (cr, 1-cr), (1-cr, 1-cr)]
    positions.extend(corners)
    
    # 8 edge circles (2 per edge)
    er = 0.11
    # Top
    positions.extend([(0.32, 1-er), (0.68, 1-er)])
    # Bottom
    positions.extend([(0.32, er), (0.68, er)])
    # Left
    positions.extend([(er, 0.32), (er, 0.68)])
    # Right
    positions.extend([(1-er, 0.32), (1-er, 0.68)])
    
    # Interior: 14 circles in a 4x4 grid pattern (minus corners already used)
    interior_positions = []
    for i in range(4):
        for j in range(4):
            x = 0.25 + i * 0.5 / 3
            y = 0.25 + j * 0.5 / 3
            # Skip if too close to corners
            if min(x, 1-x) < 0.2 and min(y, 1-y) < 0.2:
                continue
            interior_positions.append((x, y))
    
    # Take first 14
    positions.extend(interior_positions[:14 - (14 - len(interior_positions[:14]))])
    
    # Pad if needed
    while len(positions) < N:
        positions.append((0.5, 0.5))
    
    positions = positions[:N]
    points = np.array(positions)
    radii = _voronoi_radii(points, n_iter=15)
    
    x = points[:, 0]
    y = points[:, 1]
    return np.concatenate([x, y, radii])


def _make_random_initial(seed=42):
    """Random initial guess with Voronoi radii."""
    rng = np.random.default_rng(seed)
    
    # Place points with some structure
    points = rng.uniform(0.1, 0.9, size=(N, 2))
    
    # Force some to corners
    cr = 0.15
    corner_targets = [(cr, cr), (1-cr, cr), (cr, 1-cr), (1-cr, 1-cr)]
    for i, (cx, cy) in enumerate(corner_targets):
        # Find nearest point and move it toward corner
        dists = np.sqrt((points[:,0]-cx)**2 + (points[:,1]-cy)**2)
        idx = np.argmin(dists)
        points[idx] = [cx + rng.normal(0, 0.03), cy + rng.normal(0, 0.03)]
    
    points = np.clip(points, 0.05, 0.95)
    radii = _voronoi_radii(points, n_iter=15)
    
    x = points[:, 0]
    y = points[:, 1]
    return np.concatenate([x, y, radii])


def _objective_soft(x, beta=50.0):
    """Smooth objective: negative sum of radii with soft penalty for violations."""
    x_c = x[:N]
    y_c = x[N:2*N]
    r = x[2*N:]
    
    # Base objective
    obj = -np.sum(r)
    
    # Soft penalty for pairwise violations
    penalty = 0.0
    for i in range(N):
        for j in range(i+1, N):
            dx = x_c[i] - x_c[j]
            dy = y_c[i] - y_c[j]
            dist_sq = dx*dx + dy*dy
            min_dist = r[i] + r[j]
            # Soft penalty: use log-sum-exp style
            violation = min_dist*min_dist - dist_sq
            if violation > 0:
                penalty += violation * violation
    obj += 0.5 * beta * penalty
    
    # Boundary penalty
    for i in range(N):
        v1 = r[i] - x_c[i]
        if v1 > 0:
            obj += 0.5 * beta * v1 * v1
        v2 = x_c[i] + r[i] - 1
        if v2 > 0:
            obj += 0.5 * beta * v2 * v2
        v3 = r[i] - y_c[i]
        if v3 > 0:
            obj += 0.5 * beta * v3 * v3
        v4 = y_c[i] + r[i] - 1
        if v4 > 0:
            obj += 0.5 * beta * v4 * v4
    
    return obj


def _objective_exact(x):
    """Exact objective: negative sum of radii."""
    return -np.sum(x[2*N:])


def _grad_soft(x, beta=50.0):
    """Gradient of soft objective."""
    x_c = x[:N]
    y_c = x[N:2*N]
    r = x[2*N:]
    
    grad = np.zeros(3*N)
    grad[2*N:] = -1.0
    
    # Pairwise gradients
    for i in range(N):
        for j in range(i+1, N):
            dx = x_c[i] - x_c[j]
            dy = y_c[i] - y_c[j]
            dist_sq = dx*dx + dy*dy
            min_dist = r[i] + r[j]
            violation = min_dist*min_dist - dist_sq
            if violation > 0:
                # d/dx_i: -2*dx * 2*violation
                grad[i] += beta * (-2*dx) * violation
                grad[j] += beta * (2*dx) * violation
                grad[N+i] += beta * (-2*dy) * violation
                grad[N+j] += beta * (2*dy) * violation
                # d/dr_i: 2*min_dist * 2*violation
                grad[2*N+i] += beta * (2*min_dist) * violation
                grad[2*N+j] += beta * (2*min_dist) * violation
    
    # Boundary gradients
    for i in range(N):
        v1 = r[i] - x_c[i]
        if v1 > 0:
            grad[i] += beta * (-1) * v1
            grad[2*N+i] += beta * (1) * v1
        v2 = x_c[i] + r[i] - 1
        if v2 > 0:
            grad[i] += beta * (1) * v2
            grad[2*N+i] += beta * (1) * v2
        v3 = r[i] - y_c[i]
        if v3 > 0:
            grad[N+i] += beta * (-1) * v3
            grad[2*N+i] += beta * (1) * v3
        v4 = y_c[i] + r[i] - 1
        if v4 > 0:
            grad[N+i] += beta * (1) * v4
            grad[2*N+i] += beta * (1) * v4
    
    return grad


def _check_feasibility(x, tol=1e-6):
    """Check if solution is feasible."""
    x_c = x[:N]
    y_c = x[N:2*N]
    r = x[2*N:]
    
    # Boundary check
    for i in range(N):
        if x_c[i] < r[i] - tol or x_c[i] > 1 - r[i] + tol:
            return False
        if y_c[i] < r[i] - tol or y_c[i] > 1 - r[i] + tol:
            return False
        if r[i] < -tol:
            return False
    
    # Pairwise check
    for i in range(N):
        for j in range(i+1, N):
            dx = x_c[i] - x_c[j]
            dy = y_c[i] - y_c[j]
            dist = np.sqrt(dx*dx + dy*dy)
            if dist < r[i] + r[j] - tol:
                return False
    
    return True


def _polish_slsqp(x0, time_budget=20.0):
    """High-precision SLSQP polish."""
    start = time.time()
    
    def obj(x):
        return -np.sum(x[2*N:])
    
    constraints = []
    # Boundary constraints
    for i in range(N):
        constraints.append({'type': 'ineq', 'fun': lambda x, i=i: x[i] - x[2*N+i]})
        constraints.append({'type': 'ineq', 'fun': lambda x, i=i: 1 - x[i] - x[2*N+i]})
        constraints.append({'type': 'ineq', 'fun': lambda x, i=i: x[N+i] - x[2*N+i]})
        constraints.append({'type': 'ineq', 'fun': lambda x, i=i: 1 - x[N+i] - x[2*N+i]})
    
    # Pairwise constraints
    for i in range(N):
        for j in range(i+1, N):
            constraints.append({
                'type': 'ineq',
                'fun': lambda x, i=i, j=j: (x[i]-x[j])**2 + (x[N+i]-x[N+j])**2 - (x[2*N+i]+x[2*N+j])**2
            })
    
    bounds = [(0, 1)]*N + [(0, 1)]*N + [(0.01, 0.5)]*N
    
    try:
        res = minimize(
            obj, x0, method='SLSQP',
            bounds=bounds, constraints=constraints,
            options={'maxiter': 500, 'ftol': 1e-15}
        )
        if _check_feasibility(res.x, tol=1e-7):
            return res.x
    except Exception:
        pass
    
    return x0


def _polish_trust_constr(x0, time_budget=30.0):
    """Trust-constr polish with exact constraints."""
    start = time.time()
    
    def obj(x):
        return -np.sum(x[2*N:])
    
    def grad(x):
        g = np.zeros(3*N)
        g[2*N:] = -1.0
        return g
    
    def constraints_func(x):
        x_c = x[:N]
        y_c = x[N:2*N]
        r = x[2*N:]
        
        cons = []
        # Boundary
        for i in range(N):
            cons.append(x_c[i] - r[i])
            cons.append(1 - x_c[i] - r[i])
            cons.append(y_c[i] - r[i])
            cons.append(1 - y_c[i] - r[i])
            cons.append(r[i])
        
        # Pairwise
        for i in range(N):
            for j in range(i+1, N):
                dx = x_c[i] - x_c[j]
                dy = y_c[i] - y_c[j]
                cons.append(dx*dx + dy*dy - (r[i]+r[j])**2)
        
        return np.array(cons)
    
    def jac(x):
        x_c = x[:N]
        y_c = x[N:2*N]
        r = x[2*N:]
        
        n_cons = 5*N + N*(N-1)//2
        J = np.zeros((n_cons, 3*N))
        
        idx = 0
        for i in range(N):
            J[idx, i] = 1.0
            J[idx, 2*N+i] = -1.0
            idx += 1
            J[idx, i] = -1.0
            J[idx, 2*N+i] = -1.0
            idx += 1
            J[idx, N+i] = 1.0
            J[idx, 2*N+i] = -1.0
            idx += 1
            J[idx, N+i] = -1.0
            J[idx, 2*N+i] = -1.0
            idx += 1
            J[idx, 2*N+i] = 1.0
            idx += 1
        
        for i in range(N):
            for j in range(i+1, N):
                dx = x_c[i] - x_c[j]
                dy = y_c[i] - y_c[j]
                md = r[i] + r[j]
                J[idx, i] = 2*dx
                J[idx, j] = -2*dx
                J[idx, N+i] = 2*dy
                J[idx, N+j] = -2*dy
                J[idx, 2*N+i] = -2*md
                J[idx, 2*N+j] = -2*md
                idx += 1
        
        return J
    
    from scipy.optimize import NonlinearConstraint
    n_cons = 5*N + N*(N-1)//2
    constraint = NonlinearConstraint(constraints_func, 0, np.inf, jac=jac)
    
    bounds = [(0, 1)]*N + [(0, 1)]*N + [(0.01, 0.5)]*N
    
    try:
        res = minimize(
            obj, x0, method='trust-constr',
            jac=grad, bounds=bounds,
            constraints=[constraint],
            options={'maxiter': 800, 'gtol': 1e-10, 'xtol': 1e-12}
        )
        if _check_feasibility(res.x, tol=1e-7):
            return res.x
    except Exception:
        pass
    
    return x0


def construct():
    start_time = time.time()
    
    # Generate multiple initial guesses
    inits = []
    inits.append(_make_grid_initial())
    inits.append(_make_random_initial(42))
    inits.append(_make_random_initial(123))
    
    best_x = None
    best_obj = np.inf
    
    for init in inits:
        if time.time() - start_time > DEADLINE - 60:
            break
        
        # Stage 1: Soft objective with L-BFGS-B for quick improvement
        try:
            res1 = minimize(
                _objective_soft, init, method='L-BFGS-B',
                jac=_grad_soft,
                bounds=[(0,1)]*N + [(0,1)]*N + [(0.01, 0.5)]*N,
                options={'maxiter': 200, 'ftol': 1e-10}
            )
            if _check_feasibility(res1.x, tol=1e-5):
                obj_val = -np.sum(res1.x[2*N:])
                if obj_val < best_obj:
                    best_obj = obj_val
                    best_x = res1.x.copy()
        except Exception:
            pass
        
        # Stage 2: Trust-constr with exact constraints
        if time.time() - start_time > DEADLINE - 40:
            break
        
        current = res1.x if 'res1' in locals() and _check_feasibility(res1.x, tol=1e-5) else init
        try:
            res2 = _polish_trust_constr(current, time_budget=25)
            if _check_feasibility(res2, tol=1e-7):
                obj_val = -np.sum(res2[2*N:])
                if obj_val < best_obj:
                    best_obj = obj_val
                    best_x = res2.copy()
        except Exception:
            pass
        
        # Stage 3: SLSQP high-precision polish
        if time.time() - start_time > DEADLINE - 20:
            break
        
        if best_x is not None:
            try:
                res3 = _polish_slsqp(best_x, time_budget=15)
                if _check_feasibility(res3, tol=1e-8):
                    obj_val = -np.sum(res3[2*N:])
                    if obj_val < best_obj:
                        best_obj = obj_val
                        best_x = res3.copy()
            except Exception:
                pass
    
    # Final polish if time remains
    if best_x is not None and time.time() - start_time < DEADLINE - 10:
        try:
            res_final = _polish_slsqp(best_x, time_budget=10)
            if _check_feasibility(res_final, tol=1e-8):
                obj_val = -np.sum(res_final[2*N:])
                if obj_val < best_obj:
                    best_obj = obj_val
                    best_x = res_final.copy()
        except Exception:
            pass
    
    if best_x is None:
        # Fallback: return initial grid
        best_x = inits[0]
    
    # Format output
    x_c = best_x[:N]
    y_c = best_x[N:2*N]
    r = best_x[2*N:]
    
    # Ensure non-negative radii
    r = np.maximum(r, 0.01)
    
    result = np.column_stack([x_c, y_c, r])
    return result
