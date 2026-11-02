"""Circle packing in unit square: maximize sum of radii for 26 disjoint circles.
Strategy: Direct Nonlinear Constrained Optimization (SLSQP).
We formulate the problem as:
Maximize: sum(r_i)
Subject to:
1. x_i - r_i >= 0 (left wall)
2. 1 - x_i - r_i >= 0 (right wall)
3. y_i - r_i >= 0 (bottom wall)
4. 1 - y_i - r_i >= 0 (top wall)
5. r_i > 0 (positive radius, handled via bounds)
6. dist(i, j) - r_i - r_j >= 0 (non-overlap)

We use multiple diverse initializations (grid, hex, random) and run SLSQP on each,
keeping the best valid solution. SLSQP is chosen for its ability to handle
nonlinear inequality constraints effectively for this problem size (N=26).
"""
import time
import numpy as np
from scipy.optimize import minimize

N = 26
EPS = 1e-9
DEADLINE = 145.0  # Safety margin for 150s limit

def _validate(cands):
    """Check if candidates are valid and return sum of radii."""
    if not cands or len(cands) < N:
        return 0.0
    arr = np.array(cands[:N])
    # Check bounds
    for i in range(N):
        x, y, r = arr[i]
        if r <= EPS:
            return 0.0
        if x < r - 1e-8 or x > 1 - r + 1e-8 or y < r - 1e-8 or y > 1 - r + 1e-8:
            return 0.0
    # Check overlaps
    for i in range(N):
        for j in range(i + 1, N):
            dx = arr[i, 0] - arr[j, 0]
            dy = arr[i, 1] - arr[j, 1]
            dist = np.sqrt(dx * dx + dy * dy)
            if dist < arr[i, 2] + arr[j, 2] - 1e-8:
                return 0.0
    return float(arr[:, 2].sum())

def _objective_and_constraints(z, N):
    """
    Objective: minimize -sum(r)
    Constraints:
    - Wall constraints (4 per circle)
    - Pair constraints (N*(N-1)/2)
    """
    pos = z[:2 * N].reshape(N, 2)
    r = z[2 * N:]
    
    # Objective: -sum(r)
    obj = -np.sum(r)
    
    cons = []
    
    # Wall constraints
    # x - r >= 0
    cons.append({'type': 'ineq', 'fun': lambda z, i=0: z[2*i] - z[2*N+i], 'jac': lambda z: _jac_wall(z, 'x', 0, N)})
    # 1 - x - r >= 0
    cons.append({'type': 'ineq', 'fun': lambda z, i=0: 1 - z[2*i] - z[2*N+i], 'jac': lambda z: _jac_wall(z, 'x', 1, N)})
    # y - r >= 0
    cons.append({'type': 'ineq', 'fun': lambda z, i=0: z[2*i+1] - z[2*N+i], 'jac': lambda z: _jac_wall(z, 'y', 0, N)})
    # 1 - y - r >= 0
    cons.append({'type': 'ineq', 'fun': lambda z, i=0: 1 - z[2*i+1] - z[2*N+i], 'jac': lambda z: _jac_wall(z, 'y', 1, N)})
    
    # Wait, defining constraints this way creates N*4 separate constraint functions which is slow for SLSQP.
    # It is much better to vectorize the constraints if possible, or define them as arrays.
    # SLSQP accepts 'fun' returning an array for 'ineq' if 'jac' is a matrix.
    
    # Let's redefine to use vectorized constraints for efficiency.
    
    return obj, cons

def _jac_wall(z, axis, side, N):
    # This helper is not used in the final vectorized version, but kept for thought process.
    pass

def _optimize_from_start(z0, N, timeout=10):
    """Run SLSQP optimization from a starting point."""
    start_time = time.time()
    
    # Vectorized constraints
    # Variables: z[0:N] = x, z[N:2N] = y, z[2N:3N] = r
    
    def objective(z):
        return -np.sum(z[2*N:])
        
    def obj_grad(z):
        g = np.zeros_like(z)
        g[2*N:] = -1.0
        return g

    # Constraints
    # 1. Left wall: x_i - r_i >= 0
    # 2. Right wall: 1 - x_i - r_i >= 0
    # 3. Bottom wall: y_i - r_i >= 0
    # 4. Top wall: 1 - y_i - r_i >= 0
    # 5. Non-overlap: ||p_i - p_j|| - r_i - r_j >= 0
    
    cons = []
    
    # Wall constraints (vectorized)
    # Left
    cons.append({
        'type': 'ineq',
        'fun': lambda z: z[:N] - z[2*N:],
        'jac': lambda z: _jac_left(z, N)
    })
    # Right
    cons.append({
        'type': 'ineq',
        'fun': lambda z: 1 - z[:N] - z[2*N:],
        'jac': lambda z: _jac_right(z, N)
    })
    # Bottom
    cons.append({
        'type': 'ineq',
        'fun': lambda z: z[N:2*N] - z[2*N:],
        'jac': lambda z: _jac_bottom(z, N)
    })
    # Top
    cons.append({
        'type': 'ineq',
        'fun': lambda z: 1 - z[N:2*N] - z[2*N:],
        'jac': lambda z: _jac_top(z, N)
    })
    
    # Non-overlap constraints (vectorized over pairs)
    # We need to generate all pairs i < j
    indices = np.triu_indices(N, k=1)
    i_idx = indices[0]
    j_idx = indices[1]
    n_pairs = len(i_idx)
    
    def overlap_fun(z):
        xi = z[:N][i_idx]
        yi = z[N:2*N][i_idx]
        xj = z[:N][j_idx]
        yj = z[N:2*N][j_idx]
        ri = z[2*N:][i_idx]
        rj = z[2*N:][j_idx]
        dist = np.sqrt((xi - xj)**2 + (yi - yj)**2)
        return dist - ri - rj
        
    def overlap_jac(z):
        J = np.zeros((n_pairs, 3 * N))
        xi = z[:N][i_idx]
        yi = z[N:2*N][i_idx]
        xj = z[:N][j_idx]
        yj = z[N:2*N][j_idx]
        ri = z[2*N:][i_idx]
        rj = z[2*N:][j_idx]
        dx = xi - xj
        dy = yi - yj
        dist = np.sqrt(dx**2 + dy**2)
        
        # Avoid division by zero
        dist_safe = np.where(dist < 1e-9, 1e-9, dist)
        
        # Gradient w.r.t x_i
        J[:, i_idx * 1] = 0 # Placeholder, need to map correctly
        # Actually, J is (n_pairs, 3N).
        # Row k corresponds to pair (i_k, j_k).
        # Col x_i_k is i_k. Col x_j_k is j_k.
        # Col y_i_k is N + i_k. Col y_j_k is N + j_k.
        # Col r_i_k is 2N + i_k. Col r_j_k is 2N + j_k.
        
        # d(dist - ri - rj)/dx_i = dx/dist
        J[np.arange(n_pairs), i_idx] = dx / dist_safe
        # d(dist - ri - rj)/dx_j = -dx/dist
        J[np.arange(n_pairs), j_idx] = -dx / dist_safe
        # d(dist - ri - rj)/dy_i = dy/dist
        J[np.arange(n_pairs), N + i_idx] = dy / dist_safe
        # d(dist - ri - rj)/dy_j = -dy/dist
        J[np.arange(n_pairs), N + j_idx] = -dy / dist_safe
        # d(dist - ri - rj)/dr_i = -1
        J[np.arange(n_pairs), 2*N + i_idx] = -1.0
        # d(dist - ri - rj)/dr_j = -1
        J[np.arange(n_pairs), 2*N + j_idx] = -1.0
        
        return J
        
    cons.append({'type': 'ineq', 'fun': overlap_fun, 'jac': overlap_jac})
    
    # Bounds: r_i > 0, x,y in [0,1]
    bounds = []
    for i in range(N):
        bounds.append((0.0, 1.0)) # x
    for i in range(N):
        bounds.append((0.0, 1.0)) # y
    for i in range(N):
        bounds.append((1e-6, 0.5)) # r (max radius is 0.5)
        
    res = None
    try:
        res = minimize(
            objective, 
            z0, 
            method='SLSQP', 
            jac=obj_grad, 
            bounds=bounds, 
            constraints=cons, 
            options={'maxiter': 100, 'ftol': 1e-9, 'disp': False}
        )
    except Exception:
        pass
        
    if res is not None and res.success:
        return res.x
    elif res is not None:
        # Sometimes SLSQP fails but returns a good point
        return res.x
    else:
        return z0

def _jac_left(z, N):
    J = np.zeros((N, 3*N))
    idx = np.arange(N)
    J[idx, idx] = 1.0
    J[idx, 2*N + idx] = -1.0
    return J

def _jac_right(z, N):
    J = np.zeros((N, 3*N))
    idx = np.arange(N)
    J[idx, idx] = -1.0
    J[idx, 2*N + idx] = -1.0
    return J

def _jac_bottom(z, N):
    J = np.zeros((N, 3*N))
    idx = np.arange(N)
    J[idx, N + idx] = 1.0
    J[idx, 2*N + idx] = -1.0
    return J

def _jac_top(z, N):
    J = np.zeros((N, 3*N))
    idx = np.arange(N)
    J[idx, N + idx] = -1.0
    J[idx, 2*N + idx] = -1.0
    return J

def _generate_starts():
    """Generate diverse initial configurations."""
    starts = []
    
    # 1. Grid starts
    for side in [5, 6]:
        if side * side < N:
            continue
        pos = []
        r = []
        for i in range(side):
            for j in range(side):
                if len(pos) >= N:
                    break
                x = (j + 0.5) / side
                y = (i + 0.5) / side
                pos.append([x, y])
                r.append(0.4 / side)
        pos = np.array(pos[:N])
        r = np.array(r[:N])
        z0 = np.concatenate([pos[:, 0], pos[:, 1], r])
        starts.append(z0)
        
    # 2. Hexagonal-like
    pos = []
    r = []
    side = int(np.ceil(np.sqrt(N)))
    for i in range(side):
        for j in range(side):
            if len(pos) >= N:
                break
            x = (j + 0.5 * (i % 2)) / side
            y = (i + 0.5) / side
            pos.append([x, y])
            r.append(0.3 / side)
    pos = np.array(pos[:N])
    r = np.array(r[:N])
    z0 = np.concatenate([pos[:, 0], pos[:, 1], r])
    starts.append(z0)
    
    # 3. Random
    rng = np.random.default_rng(42)
    for _ in range(5):
        pos = rng.random((N, 2)) * 0.8 + 0.1
        r = np.full(N, 0.02)
        z0 = np.concatenate([pos[:, 0], pos[:, 1], r])
        starts.append(z0)
        
    return starts

def construct():
    t_start = time.time()
    best_score = 0.0
    best_z = None
    
    starts = _generate_starts()
    
    for z0 in starts:
        if time.time() > DEADLINE - 5:
            break
            
        z_opt = _optimize_from_start(z0, N)
        
        # Extract and validate
        x = z_opt[:N]
        y = z_opt[N:2*N]
        r = z_opt[2*N:]
        
        cands = list(zip(x, y, r))
        score = _validate(cands)
        
        if score > best_score:
            best_score = score
            best_z = z_opt.copy()
            
    # Final polish: iterate max radius
    if best_z is not None:
        x = best_z[:N]
        y = best_z[N:2*N]
        r = best_z[2*N:]
        
        # Iterate: fix pos, max rad
        for _ in range(20):
            if time.time() > DEADLINE - 1:
                break
            new_r = np.zeros(N)
            for i in range(N):
                dmin = min(x[i], 1 - x[i], y[i], 1 - y[i])
                for j in range(N):
                    if i != j:
                        d = np.hypot(x[i] - x[j], y[i] - y[j])
                        dmin = min(dmin, d - r[j])
                new_r[i] = max(0.0, dmin)
            if np.allclose(new_r, r, atol=1e-10):
                r = new_r
                break
            r = new_r
            
        cands = list(zip(x, y, r))
        final_score = _validate(cands)
        if final_score > best_score:
            best_score = final_score
            best_z = np.concatenate([x, y, r])
            
    if best_z is None:
        # Fallback
        side = int(np.ceil(np.sqrt(N)))
        cells = []
        for i in range(side):
            for j in range(side):
                if len(cells) >= N:
                    break
                cells.append(((j + 0.5) / side, (i + 0.5) / side, 0.4 / side))
        return np.array(cells[:N])
        
    return np.column_stack([best_z[:N], best_z[N:2*N], best_z[2*N:]])

if __name__ == '__main__':
    out = construct()
    print("sum of radii:", out[:, 2].sum())
