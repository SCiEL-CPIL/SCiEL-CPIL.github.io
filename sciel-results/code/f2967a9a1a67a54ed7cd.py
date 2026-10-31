"""Circle packing in unit square - 26 circles maximizing sum of radii.
Strategy: Voronoi Tessellation Duality.
1. Represent the packing by 2D points (centroids).
2. Compute the Voronoi diagram of these points.
3. For each Voronoi cell, compute the inradius (distance to the nearest edge/corner).
4. The radius of the circle is the inradius of its cell.
5. Optimize the positions of the points to maximize the sum of inradii.
   Since the Voronoi diagram is piecewise smooth, we can use subgradient ascent
   or a smoothed approximation (soft-min) for gradient-based optimization.
6. Use L-BFGS-B on the smoothed objective.
7. Polish with SLSQP hard constraints on the final positions and radii.
"""
import numpy as np
import time
from scipy.optimize import minimize
from scipy.spatial import Voronoi

N = 26

# EVOLVE-BLOCK-START

def _is_valid(pos, radii, tol=1e-7):
    """Strict validity check."""
    if np.any(pos[:, 0] - radii < -tol) or np.any(pos[:, 0] + radii > 1.0 + tol):
        return False
    if np.any(pos[:, 1] - radii < -tol) or np.any(pos[:, 1] + radii > 1.0 + tol):
        return False
    dx = pos[:, 0, np.newaxis] - pos[np.newaxis, :, 0]
    dy = pos[:, 1, np.newaxis] - pos[np.newaxis, :, 1]
    dist_sq = dx * dx + dy * dy
    min_dist = radii[:, np.newaxis] + radii[np.newaxis, :]
    mask = np.triu(np.ones((N, N), dtype=bool), k=1)
    if np.any(dist_sq[mask] < (min_dist[mask] - tol) ** 2):
        return False
    return True

def _post_process(pos, radii):
    """Shrink radii to ensure strict validity."""
    pos = pos.copy()
    radii = radii.copy()
    for _ in range(100):
        valid = True
        # Boundary check
        for i in range(N):
            r = radii[i]
            x, y = pos[i]
            max_r = min(x, 1.0 - x, y, 1.0 - y)
            if r > max_r:
                radii[i] = max(max_r * 0.9999, 1e-5)
                valid = False
        # Overlap check
        for i in range(N):
            for j in range(i + 1, N):
                dx = pos[i, 0] - pos[j, 0]
                dy = pos[i, 1] - pos[j, 1]
                dist = np.sqrt(dx * dx + dy * dy)
                sum_r = radii[i] + radii[j]
                if dist < sum_r:
                    # Shrink both proportionally to fix overlap
                    k = dist / (sum_r + 1e-12) * 0.9999
                    if k < 1.0:
                        radii[i] = max(radii[i] * k, 1e-5)
                        radii[j] = max(radii[j] * k, 1e-5)
                        valid = False
        if valid:
            break
    return pos, radii

def _compute_inradii(pos, eps=1e-5):
    """
    Compute the inradius for each point based on Voronoi cells.
    We add boundary points to ensure the Voronoi diagram covers the unit square.
    The inradius of a cell is the minimum distance from the point to the cell boundary.
    For a convex polygon, this is the minimum distance to any edge.
    """
    # We need to handle the boundary.
    # Instead of relying on Voronoi's unbounded cells which are tricky,
    # we can compute the distance to the nearest other point AND the boundary.
    # For point i, r_i = min( dist_to_boundary(i), min_{j!=i} dist(i,j)/2 )
    # This is the exact definition of the inradius of the Voronoi cell 
    # intersected with the domain, assuming the Voronoi cell is defined by 
    # bisectors with other points and the domain boundary.
    
    radii = np.zeros(N)
    for i in range(N):
        x, y = pos[i]
        # Distance to boundary
        r_bound = min(x, 1.0 - x, y, 1.0 - y)
        
        # Distance to other points
        min_dist = np.inf
        for j in range(N):
            if i == j:
                continue
            dx = x - pos[j, 0]
            dy = y - pos[j, 1]
            d = np.sqrt(dx*dx + dy*dy)
            if d < min_dist:
                min_dist = d
        
        r_voronoi = min_dist / 2.0
        radii[i] = min(r_bound, r_voronoi)
        
    return radii

def _objective_smooth(pos, beta=50.0):
    """
    Smoothed objective: -sum(softmin(radii))
    We use a soft-min to make the objective differentiable.
    softmin(x) = -1/beta * log(sum(exp(-beta*x_i)))
    Here we want to maximize sum(r_i).
    So we minimize -sum(r_i).
    To handle the non-differentiability of min(), we use a soft-min approximation
    for the radius calculation? 
    Actually, r_i = min(r_bound_i, d_ij/2).
    This is a min of linear functions (locally).
    We can use a smooth minimum function:
    m(a, b) = -1/beta * log(exp(-beta*a) + exp(-beta*b))
    
    So r_i_smooth = softmin(r_bound_i, min_j d_ij/2)
    Wait, min_j is also a min.
    r_i = min( r_bound_i, 0.5 * min_j d_ij )
    
    Let's compute the gradient of the smoothed radius.
    """
    pos = np.asarray(pos).reshape(N, 2)
    total_obj = 0.0
    grad = np.zeros_like(pos)
    
    for i in range(N):
        x, y = pos[i]
        r_bound = min(x, 1.0 - x, y, 1.0 - y)
        
        # Compute distances to all other points
        dists = np.zeros(N)
        for j in range(N):
            if i == j:
                dists[j] = np.inf
                continue
            dx = x - pos[j, 0]
            dy = y - pos[j, 1]
            dists[j] = np.sqrt(dx*dx + dy*dy)
        
        # r_voronoi = 0.5 * min(dists)
        # We want softmin of r_bound and 0.5*min(dists)
        # Let's just use a large beta for the min operations
        
        # Find the closest neighbor
        j_min = np.argmin(dists)
        d_min = dists[j_min]
        r_voronoi = 0.5 * d_min
        
        # Smooth min of r_bound and r_voronoi
        # m = -1/beta * log(exp(-beta*r_bound) + exp(-beta*r_voronoi))
        # grad m / grad r_bound = exp(-beta*r_bound) / (exp(-beta*r_bound) + exp(-beta*r_voronoi))
        # grad m / grad r_voronoi = exp(-beta*r_voronoi) / (exp(-beta*r_bound) + exp(-beta*r_voronoi))
        
        # To avoid overflow, shift by max
        a = -beta * r_bound
        b = -beta * r_voronoi
        m = max(a, b)
        sum_exp = np.exp(a - m) + np.exp(b - m)
        
        w_bound = np.exp(a - m) / sum_exp
        w_voronoi = np.exp(b - m) / sum_exp
        
        # Gradient of r_bound wrt pos
        grad_r_bound = np.zeros(2)
        if x < 0.5:
            grad_r_bound[0] = 1.0
        else:
            grad_r_bound[0] = -1.0
        if y < 0.5:
            grad_r_bound[1] = 1.0
        else:
            grad_r_bound[1] = -1.0
            
        # Gradient of r_voronoi wrt pos
        # r_voronoi = 0.5 * ||pos_i - pos_jmin||
        # grad = 0.5 * (pos_i - pos_jmin) / ||pos_i - pos_jmin||
        dx = x - pos[j_min, 0]
        dy = y - pos[j_min, 1]
        d = d_min
        if d > 1e-8:
            grad_r_voronoi = np.array([0.5 * dx / d, 0.5 * dy / d])
        else:
            grad_r_voronoi = np.zeros(2)
            
        # Total gradient of r_i
        grad_r_i = w_bound * grad_r_bound + w_voronoi * grad_r_voronoi
        
        # We want to maximize sum(r_i), so minimize -sum(r_i)
        # grad_obj = -grad_r_i
        grad[i] = -grad_r_i
        
        total_obj += -1.0 / beta * (m + np.log(sum_exp)) # This is -softmin
        # Actually, we just need the value for monitoring, gradient is what matters for L-BFGS
        # Value: -r_i_smooth = -(-1/beta * log(sum_exp) + m) = 1/beta * log(sum_exp) - m
        # Let's just compute the hard value for checking, but use smooth grad for optimization
        r_hard = min(r_bound, r_voronoi)
        total_obj += -r_hard # Use hard value for the objective function returned to optimizer?
        # L-BFGS uses the gradient provided. If we return hard value but smooth grad, it might be inconsistent.
        # Better to return the smooth value.
        
    return total_obj, grad

def _objective_and_grad(pos, beta=50.0):
    """
    Returns objective value and gradient for L-BFGS-B.
    Objective: -sum(r_i)
    """
    pos = np.asarray(pos).reshape(N, 2)
    total_obj = 0.0
    grad = np.zeros_like(pos)
    
    for i in range(N):
        x, y = pos[i]
        # Distances to boundaries
        r_bound = min(x, 1.0 - x, y, 1.0 - y)
        
        # Distances to other points
        # Vectorized for speed
        dx = x - pos[:, 0]
        dy = y - pos[:, 1]
        dists = np.sqrt(dx*dx + dy*dy)
        dists[i] = np.inf
        j_min = np.argmin(dists)
        d_min = dists[j_min]
        r_voronoi = 0.5 * d_min
        
        # Smooth min
        a = -beta * r_bound
        b = -beta * r_voronoi
        m = max(a, b)
        sum_exp = np.exp(a - m) + np.exp(b - m)
        
        w_bound = np.exp(a - m) / sum_exp
        w_voronoi = np.exp(b - m) / sum_exp
        
        # Gradient of r_bound
        grad_r_bound = np.zeros(2)
        if x < 0.5: grad_r_bound[0] = 1.0
        else: grad_r_bound[0] = -1.0
        if y < 0.5: grad_r_bound[1] = 1.0
        else: grad_r_bound[1] = -1.0
        
        # Gradient of r_voronoi
        dx_min = x - pos[j_min, 0]
        dy_min = y - pos[j_min, 1]
        if d_min > 1e-8:
            grad_r_voronoi = np.array([0.5 * dx_min / d_min, 0.5 * dy_min / d_min])
        else:
            grad_r_voronoi = np.zeros(2)
            
        grad_r_i = w_bound * grad_r_bound + w_voronoi * grad_r_voronoi
        grad[i] = -grad_r_i
        
        # Smooth value
        r_smooth = -1.0 / beta * (m + np.log(sum_exp))
        total_obj += -r_smooth
        
    return total_obj, grad

def _init_hexagonal():
    """Hexagonal lattice initialization."""
    pos = np.zeros((N, 2))
    # r ~ 0.1
    # 5x5 grid
    r = 0.10
    dx = r * np.sqrt(3)
    dy = r * 1.5
    k = 0
    for row in range(6):
        offset = dx / 2 if row % 2 == 1 else 0
        start_x = r + offset
        y = row * dy + r
        for col in range(5):
            if k >= N:
                break
            x = start_x + col * dx
            if x + r <= 1.0 and x - r >= 0.0 and y + r <= 1.0 and y - r >= 0.0:
                pos[k] = [x, y]
                k += 1
        if k >= N:
            break
    
    # Fill remaining randomly
    rng = np.random.RandomState(42)
    while k < N:
        x = rng.uniform(0.1, 0.9)
        y = rng.uniform(0.1, 0.9)
        pos[k] = [x, y]
        k += 1
    return pos

def _slsqp_polish(pos, radii, deadline):
    """
    SLSQP with hard inequality constraints.
    """
    x0 = np.concatenate([pos.flatten(), radii])
    
    bounds = []
    for i in range(N):
        bounds.append((0.001, 0.999))
        bounds.append((0.001, 0.999))
    for i in range(N):
        bounds.append((0.001, 0.4))
    
    constraints = []
    
    # Boundary constraints
    for i in range(N):
        constraints.append({'type': 'ineq',
                           'fun': lambda x, i=i: x[2*i] - x[2*N+i]})
        constraints.append({'type': 'ineq',
                           'fun': lambda x, i=i: 1.0 - x[2*i] - x[2*N+i]})
        constraints.append({'type': 'ineq',
                           'fun': lambda x, i=i: x[2*i+1] - x[2*N+i]})
        constraints.append({'type': 'ineq',
                           'fun': lambda x, i=i: 1.0 - x[2*i+1] - x[2*N+i]})
    
    # Overlap constraints
    for i in range(N):
        for j in range(i + 1, N):
            constraints.append({'type': 'ineq',
                                'fun': lambda x, i=i, j=j:
                                (x[2*i] - x[2*j])**2 + (x[2*i+1] - x[2*j+1])**2 -
                                (x[2*N+i] + x[2*N+j])**2})
    
    def objective(x):
        return -np.sum(x[2*N:])
    
    try:
        result = minimize(objective, x0, method='SLSQP', bounds=bounds,
                          constraints=constraints,
                          options={'maxiter': 200, 'ftol': 1e-10})
        if result.success or result.fun < objective(x0):
            x = result.x
            pos_out = x[:2*N].reshape(N, 2)
            radii_out = np.maximum(x[2*N:], 1e-4)
            pos_out, radii_out = _post_process(pos_out, radii_out)
            if _is_valid(pos_out, radii_out):
                return pos_out, radii_out
    except Exception:
        pass
    
    return pos, radii

def construct():
    start_time = time.time()
    deadline = start_time + 145.0
    
    # Initialization
    pos = _init_hexagonal()
    
    # Phase 1: L-BFGS-B on Voronoi inradius objective
    # We optimize positions only. Radii are derived.
    x0 = pos.flatten()
    
    def obj_wrapper(x):
        val, _ = _objective_and_grad(x, beta=50.0)
        return val
    
    def grad_wrapper(x):
        _, g = _objective_and_grad(x, beta=50.0)
        return g
        
    try:
        result = minimize(obj_wrapper, x0, jac=grad_wrapper, method='L-BFGS-B',
                          options={'maxiter': 1000, 'ftol': 1e-10})
        if result.success or result.fun < obj_wrapper(x0):
            pos = result.x.reshape(N, 2)
    except Exception:
        pass
    
    # Compute radii from optimized positions
    radii = _compute_inradii(pos)
    
    # Post-process to ensure validity
    pos, radii = _post_process(pos, radii)
    
    current_score = np.sum(radii)
    best_pos = pos.copy()
    best_radii = radii.copy()
    best_score = current_score
    
    # Phase 2: SLSQP Polish
    if time.time() < deadline - 20:
        pos_sl, rad_sl = _slsqp_polish(pos, radii, deadline)
        pos_pp, rad_pp = _post_process(pos_sl, rad_sl)
        score = np.sum(rad_pp)
        if _is_valid(pos_pp, rad_pp) and score > best_score:
            best_score = score
            best_pos = pos_pp.copy()
            best_radii = rad_pp.copy()
            
    # Phase 3: Local Search / Restart
    rng = np.random.RandomState(123)
    time_left = deadline - time.time()
    if time_left > 30:
        for _ in range(5):
            if time.time() > deadline - 10:
                break
            
            # Perturb best solution
            pos_p = best_pos + rng.normal(0, 0.01, (N, 2))
            # Re-optimize positions from perturbed start
            x0_p = pos_p.flatten()
            try:
                res_p = minimize(obj_wrapper, x0_p, jac=grad_wrapper, method='L-BFGS-B',
                                 options={'maxiter': 200, 'ftol': 1e-8})
                if res_p.fun < obj_wrapper(x0_p):
                    pos_p = res_p.x.reshape(N, 2)
            except Exception:
                pass
            
            rad_p = _compute_inradii(pos_p)
            pos_pp, rad_pp = _post_process(pos_p, rad_p)
            
            if _is_valid(pos_pp, rad_pp):
                score = np.sum(rad_pp)
                if score > best_score:
                    best_score = score
                    best_pos = pos_pp.copy()
                    best_radii = rad_pp.copy()
                    
                    # Polish the new best
                    if time.time() < deadline - 15:
                        pos_sl, rad_sl = _slsqp_polish(pos_pp, rad_pp, deadline)
                        pos_pp2, rad_pp2 = _post_process(pos_sl, rad_sl)
                        score2 = np.sum(rad_pp2)
                        if _is_valid(pos_pp2, rad_pp2) and score2 > best_score:
                            best_score = score2
                            best_pos = pos_pp2.copy()
                            best_radii = rad_pp2.copy()
    
    # Final Polish
    if time.time() < deadline - 5:
        pos_final, rad_final = _slsqp_polish(best_pos, best_radii, deadline)
        pos_pp, rad_pp = _post_process(pos_final, rad_final)
        score = np.sum(rad_pp)
        if _is_valid(pos_pp, rad_pp) and score > best_score:
            best_score = score
            best_pos = pos_pp.copy()
            best_radii = rad_pp.copy()
            
    # Final Validation
    best_pos, best_radii = _post_process(best_pos, best_radii)
    if not _is_valid(best_pos, best_radii):
        # Fallback
        pos = _init_hexagonal()
        radii = _compute_inradii(pos)
        pos, radii = _post_process(pos, radii)
        best_pos, best_radii = pos, radii
        
    result = np.column_stack([best_pos, best_radii])
    return result

# EVOLVE-BLOCK-END
