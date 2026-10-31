"""Circle packing in unit square - 26 circles maximizing sum of radii."""
import numpy as np
import time
from scipy.optimize import minimize

N = 26

# EVOLVE-BLOCK-START

def _is_valid(pos, radii, tol=1e-9):
    """Check if circles are valid (inside square, disjoint interiors)."""
    n = len(radii)
    for i in range(n):
        if pos[i, 0] - radii[i] < -tol or pos[i, 0] + radii[i] > 1.0 + tol:
            return False
        if pos[i, 1] - radii[i] < -tol or pos[i, 1] + radii[i] > 1.0 + tol:
            return False
    for i in range(n):
        for j in range(i + 1, n):
            dx = pos[i, 0] - pos[j, 0]
            dy = pos[i, 1] - pos[j, 1]
            dist_sq = dx * dx + dy * dy
            min_dist_sq = (radii[i] + radii[j]) ** 2
            if dist_sq < min_dist_sq - tol:
                return False
    return True

def _make_initial_config(strategy, seed=42):
    """Create an initial configuration based on strategy."""
    rng = np.random.RandomState(seed)
    pos = np.zeros((N, 2))
    radii = np.zeros(N)
    
    if strategy == 'grid':
        side = 5
        for i in range(5):
            for j in range(5):
                idx = i * 5 + j
                if idx < N:
                    pos[idx] = [(i + 0.5) / side, (j + 0.5) / side]
                    radii[idx] = 0.5 / side - 0.01
        # Adjust last one if needed
        if N == 26:
             pos[25] = [0.5, 0.05]
             radii[25] = 0.04
             
    elif strategy == 'hexagonal':
        # Hexagonal packing is dense. 
        # 5 rows. 
        # Row 0: 5 circles
        # Row 1: 5 circles
        # Row 2: 5 circles
        # Row 3: 5 circles
        # Row 4: 5 circles
        # Total 25. Add 1 in center? No, 26.
        # Let's try a 5x5 grid distorted to hex.
        
        # Standard hex packing of 25 circles in a square fits well.
        # r ~ 1 / (5 + 4*sqrt(3)/2) approx.
        # Let's just place them in a grid and let the optimizer fix it.
        side = 5
        for i in range(5):
            for j in range(5):
                idx = i * 5 + j
                if idx < N:
                    pos[idx] = [(i + 0.5) / side, (j + 0.5) / side]
                    radii[idx] = 0.09
        if N == 26:
            # Add a small circle in a gap
            pos[25] = [0.5, 0.5]
            radii[25] = 0.05

    elif strategy == 'random':
        pos = rng.uniform(0.1, 0.9, size=(N, 2))
        radii = rng.uniform(0.02, 0.08, size=N)
        
    else:
        pos, radii = _make_initial_config('grid', seed)
        pos += rng.normal(0, 0.02, size=pos.shape)
        pos = np.clip(pos, 0.05, 0.95)

    # Ensure initial validity by shrinking if necessary
    pos, radii = _post_process(pos, radii)
    return pos, radii

def _post_process(pos, radii):
    """Iteratively shrink circles to ensure validity."""
    pos = pos.copy()
    radii = radii.copy()
    
    for iteration in range(100):
        valid = True
        # Boundary check
        for i in range(N):
            max_r = min(pos[i, 0], 1.0 - pos[i, 0], pos[i, 1], 1.0 - pos[i, 1])
            if radii[i] > max_r * 0.98:
                radii[i] = max_r * 0.98
                valid = False
        
        # Overlap check
        for i in range(N):
            for j in range(i + 1, N):
                dx = pos[i, 0] - pos[j, 0]
                dy = pos[i, 1] - pos[j, 1]
                dist = np.sqrt(dx * dx + dy * dy)
                min_dist = radii[i] + radii[j]
                if dist < min_dist:
                    # Scale down both
                    if min_dist > 1e-9:
                        scale = (dist * 0.98) / min_dist
                    else:
                        scale = 0.5
                    radii[i] *= scale
                    radii[j] *= scale
                    valid = False
        
        if valid:
            break
    return pos, radii

def _objective(x):
    """Minimize negative sum of radii."""
    n = N
    radii = x[2*n:]
    return -np.sum(radii)

def _objective_grad(x):
    """Gradient of objective."""
    n = N
    grad = np.zeros_like(x)
    grad[2*n:] = -1.0
    return grad

def _constraints(x):
    """
    Returns list of constraint dicts.
    We use inequality constraints: g(x) >= 0
    """
    n = N
    pos = x[:2*n].reshape(n, 2)
    radii = x[2*n:]
    
    cons = []
    
    # Boundary constraints
    # x_i - r_i >= 0
    # 1 - x_i - r_i >= 0
    # y_i - r_i >= 0
    # 1 - y_i - r_i >= 0
    
    # We can combine these into a single function that returns an array, 
    # but SLSQP expects a list of dicts or a single dict with 'fun' and 'jac'.
    # For efficiency, we define one function for all boundary constraints.
    
    def bound_fun(x):
        n = N
        pos = x[:2*n].reshape(n, 2)
        radii = x[2*n:]
        vals = np.zeros(4*n)
        vals[0::4] = pos[:, 0] - radii
        vals[1::4] = 1.0 - pos[:, 0] - radii
        vals[2::4] = pos[:, 1] - radii
        vals[3::4] = 1.0 - pos[:, 1] - radii
        return vals

    def bound_jac(x):
        n = N
        jac = np.zeros((4*n, 3*n))
        for i in range(n):
            # d(x_i - r_i) / dx_i = 1, dr_i = -1
            jac[4*i, 2*i] = 1.0
            jac[4*i, 2*n+i] = -1.0
            
            # d(1 - x_i - r_i) / dx_i = -1, dr_i = -1
            jac[4*i+1, 2*i] = -1.0
            jac[4*i+1, 2*n+i] = -1.0
            
            # d(y_i - r_i) / dy_i = 1, dr_i = -1
            jac[4*i+2, 2*i+1] = 1.0
            jac[4*i+2, 2*n+i] = -1.0
            
            # d(1 - y_i - r_i) / dy_i = -1, dr_i = -1
            jac[4*i+3, 2*i+1] = -1.0
            jac[4*i+3, 2*n+i] = -1.0
        return jac

    cons.append({'type': 'ineq', 'fun': bound_fun, 'jac': bound_jac})
    
    # Overlap constraints
    # (x_i - x_j)^2 + (y_i - y_j)^2 - (r_i + r_j)^2 >= 0
    
    def overlap_fun(x):
        n = N
        pos = x[:2*n].reshape(n, 2)
        radii = x[2*n:]
        vals = np.zeros((n*(n-1))//2)
        idx = 0
        for i in range(n):
            for j in range(i+1, n):
                dx = pos[i, 0] - pos[j, 0]
                dy = pos[i, 1] - pos[j, 1]
                dist_sq = dx*dx + dy*dy
                min_dist_sq = (radii[i] + radii[j])**2
                vals[idx] = dist_sq - min_dist_sq
                idx += 1
        return vals

    def overlap_jac(x):
        n = N
        pos = x[:2*n].reshape(n, 2)
        radii = x[2*n:]
        m = (n*(n-1))//2
        jac = np.zeros((m, 3*n))
        idx = 0
        for i in range(n):
            for j in range(i+1, n):
                dx = pos[i, 0] - pos[j, 0]
                dy = pos[i, 1] - pos[j, 1]
                dist_sq = dx*dx + dy*dy
                if dist_sq < 1e-12:
                    dist_sq = 1e-12
                dist = np.sqrt(dist_sq)
                
                # d/dx_i (dist_sq - (r_i+r_j)^2) = 2*dx
                # d/dy_i (dist_sq - (r_i+r_j)^2) = 2*dy
                # d/dr_i = -2*(r_i+r_j)
                # d/dx_j = -2*dx
                # d/dy_j = -2*dy
                # d/dr_j = -2*(r_i+r_j)
                
                jac[idx, 2*i] = 2.0 * dx
                jac[idx, 2*i+1] = 2.0 * dy
                jac[idx, 2*n+i] = -2.0 * (radii[i] + radii[j])
                
                jac[idx, 2*j] = -2.0 * dx
                jac[idx, 2*j+1] = -2.0 * dy
                jac[idx, 2*n+j] = -2.0 * (radii[i] + radii[j])
                
                idx += 1
        return jac

    cons.append({'type': 'ineq', 'fun': overlap_fun, 'jac': overlap_jac})
    
    return cons

def _run_slsqp(pos, radii, deadline):
    """Run SLSQP optimization."""
    x0 = np.concatenate([pos.flatten(), radii])
    
    bounds = []
    for i in range(N):
        bounds.append((0.0, 1.0))
        bounds.append((0.0, 1.0))
        bounds.append((0.01, 0.4)) # Min radius 0.01, Max 0.4
    
    try:
        result = minimize(
            _objective,
            x0,
            jac=_objective_grad,
            method='SLSQP',
            bounds=bounds,
            constraints=_constraints(x0),
            options={'maxiter': 1000, 'ftol': 1e-12}
        )
        
        if result.success or result.fun < 0: # Check if it found something
            x_opt = result.x
            pos_opt = x_opt[:2*N].reshape(N, 2)
            radii_opt = x_opt[2*N:]
            
            # Post-process to ensure strict validity due to numerical errors
            pos_pp, radii_pp = _post_process(pos_opt, radii_opt)
            
            if _is_valid(pos_pp, radii_pp):
                return pos_pp, radii_pp
    except Exception as e:
        pass
    
    return None, None

def _force_directed_anneal(pos, radii, n_iter=500, deadline=None):
    """Force-directed annealing to explore topology and improve packing."""
    pos = pos.copy()
    radii = radii.copy()
    best_pos = pos.copy()
    best_radii = radii.copy()
    best_score = np.sum(radii)
    
    vel = np.zeros_like(pos)
    rng = np.random.RandomState(123)
    
    for it in range(n_iter):
        if deadline is not None and time.time() > deadline:
            break
        
        temp = 0.01 * np.exp(-0.005 * it)
        forces = np.zeros_like(pos)
        
        # Boundary forces
        for i in range(N):
            margin = 0.005
            if pos[i, 0] < radii[i] + margin:
                forces[i, 0] += 10.0 * (radii[i] + margin - pos[i, 0])
            if pos[i, 0] > 1.0 - radii[i] - margin:
                forces[i, 0] -= 10.0 * (pos[i, 0] - (1.0 - radii[i] - margin))
            if pos[i, 1] < radii[i] + margin:
                forces[i, 1] += 10.0 * (radii[i] + margin - pos[i, 1])
            if pos[i, 1] > 1.0 - radii[i] - margin:
                forces[i, 1] -= 10.0 * (pos[i, 1] - (1.0 - radii[i] - margin))
        
        # Overlap forces
        for i in range(N):
            for j in range(i + 1, N):
                dx = pos[i, 0] - pos[j, 0]
                dy = pos[i, 1] - pos[j, 1]
                dist_sq = dx * dx + dy * dy
                dist = np.sqrt(dist_sq) if dist_sq > 1e-12 else 0.0
                min_dist = radii[i] + radii[j]
                
                if dist < min_dist and dist > 1e-6:
                    overlap = min_dist - dist
                    force_mag = 15.0 * overlap
                    fx = force_mag * dx / dist
                    fy = force_mag * dy / dist
                    forces[i, 0] += fx
                    forces[i, 1] += fy
                    forces[j, 0] -= fx
                    forces[j, 1] -= fy
                elif dist < min_dist * 1.2 and dist > 1e-6:
                    force_mag = 2.0 * (min_dist * 1.2 - dist)
                    fx = force_mag * dx / dist
                    fy = force_mag * dy / dist
                    forces[i, 0] += fx
                    forces[i, 1] += fy
                    forces[j, 0] -= fx
                    forces[j, 1] -= fy
        
        vel = 0.85 * vel + 0.15 * forces
        pos += vel * 0.008 + rng.normal(0, temp, size=pos.shape)
        
        # Clip positions
        for i in range(N):
            pos[i, 0] = np.clip(pos[i, 0], 0.01, 0.99)
            pos[i, 1] = np.clip(pos[i, 1], 0.01, 0.99)
        
        # Grow radii where possible
        for i in range(N):
            max_grow = min(pos[i, 0] - radii[i], 1.0 - pos[i, 0] - radii[i],
                          pos[i, 1] - radii[i], 1.0 - pos[i, 1] - radii[i])
            for j in range(N):
                if i == j:
                    continue
                dx = pos[i, 0] - pos[j, 0]
                dy = pos[i, 1] - pos[j, 1]
                dist = np.sqrt(dx * dx + dy * dy)
                available = dist - radii[j]
                if available < max_grow:
                    max_grow = available
            
            if max_grow > 0.0005:
                radii[i] += max_grow * 0.6
        
        # Ensure minimum radius
        radii = np.maximum(radii, 0.002)
        
        pos_pp, radii_pp = _post_process(pos, radii)
        if _is_valid(pos_pp, radii_pp):
            score = np.sum(radii_pp)
            if score > best_score:
                best_score = score
                best_pos = pos_pp.copy()
                best_radii = radii_pp.copy()
    
    return best_pos, best_radii

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin before 150s limit
    
    best_pos = None
    best_radii = None
    best_score = -np.inf
    
    strategies = ['hexagonal', 'grid', 'random']
    
    # Phase 1: Multi-start with SLSQP
    for strategy in strategies:
        if time.time() > deadline - 40:
            break
        
        # Try 3 seeds for each strategy
        for seed in range(3):
            if time.time() > deadline - 30:
                break
            
            pos, radii = _make_initial_config(strategy, seed)
            
            # Run SLSQP
            pos_opt, radii_opt = _run_slsqp(pos, radii, deadline)
            
            if pos_opt is not None:
                score = np.sum(radii_opt)
                if score > best_score:
                    best_score = score
                    best_pos = pos_opt.copy()
                    best_radii = radii_opt.copy()
                    
                    # If we have a very good score, we can break early or refine
                    if best_score > 1.25:
                         break
        
        if best_score > 1.25:
            break

    # Phase 2: Force-directed annealing from best result
    if best_pos is not None and time.time() < deadline - 20:
        try:
            pos_fd, radii_fd = _force_directed_anneal(best_pos, best_radii, n_iter=500, deadline=deadline - 15)
            score_fd = np.sum(radii_fd)
            if _is_valid(pos_fd, radii_fd) and score_fd > best_score:
                best_score = score_fd
                best_pos = pos_fd.copy()
                best_radii = radii_fd.copy()
        except Exception:
            pass
    
    # Phase 3: Final SLSQP polish
    if best_pos is not None and time.time() < deadline - 5:
        pos_pol, radii_pol = _run_slsqp(best_pos, best_radii, deadline)
        if pos_pol is not None:
            score_pol = np.sum(radii_pol)
            if _is_valid(pos_pol, radii_pol) and score_pol > best_score:
                best_score = score_pol
                best_pos = pos_pol.copy()
                best_radii = radii_pol.copy()
    
    # Fallback
    if best_pos is None:
        side = 5
        cells = [((i + 0.5) / side, (j + 0.5) / side, 0.5 / side - 0.005) for i in range(side) for j in range(side)]
        result = np.array(cells[:N])
        return result
    
    result = np.column_stack([best_pos, best_radii])
    return result

# EVOLVE-BLOCK-END
