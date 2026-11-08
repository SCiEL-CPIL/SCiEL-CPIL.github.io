import numpy as np
import time
import math
from scipy.optimize import minimize

N = 21

# EVOLVE-BLOCK-START

def _voronoi_inradii(x, y, W, H, margin=0.0):
    """
    Compute the maximum radius each circle can have given its Voronoi cell
    within the rectangle [0,W]x[0,H].
    
    For each point i, find the minimum distance to:
    - The rectangle boundary
    - The perpendicular bisector between i and every other point j
    
    This gives the inradius of the Voronoi cell clipped to the rectangle.
    """
    n = len(x)
    r = np.zeros(n)
    
    for i in range(n):
        # Distance to rectangle boundaries
        d_boundary = min(x[i], W - x[i], y[i], H - y[i]) - margin
        
        # Distance to perpendicular bisectors with all other points
        d_voronoi = np.inf
        for j in range(n):
            if i == j:
                continue
            dx = x[j] - x[i]
            dy = y[j] - y[i]
            d_sq = dx*dx + dy*dy
            if d_sq < 1e-15:
                continue
            d = math.sqrt(d_sq)
            # The perpendicular bisector is at distance d/2 from each point
            # The closest point on the bisector to i is at distance d/2
            # But we need to check if the closest point on the bisector
            # is within the segment or beyond it
            # Actually for the inradius, we want the minimum distance from i
            # to the bisector line, which is d/2
            d_bisect = d / 2.0
            if d_bisect < d_voronoi:
                d_voronoi = d_bisect
        
        r[i] = max(0.0, min(d_boundary, d_voronoi))
    
    return r

def _compute_value(x, y, r, W):
    """
    Returns sum of radii if valid (all circles disjoint and within bounds), else 0.0
    """
    H = 2.0 - W
    if W <= 1e-9 or H <= 1e-9:
        return 0.0
        
    for i in range(N):
        if x[i] < r[i] - 1e-8 or x[i] > W - r[i] + 1e-8:
            return 0.0
        if y[i] < r[i] - 1e-8 or y[i] > H - r[i] + 1e-8:
            return 0.0
        if r[i] < 1e-8:
            return 0.0
            
    for i in range(N):
        for j in range(i+1, N):
            dx = x[i] - x[j]
            dy = y[i] - y[j]
            dist_sq = dx*dx + dy*dy
            min_dist = r[i] + r[j]
            if dist_sq < min_dist*min_dist - 1e-7:
                return 0.0
                
    return float(np.sum(r))

def _objective_smooth(v, W, beta=30.0):
    """
    Smooth objective: -sum(r) where r_i = soft_min over all constraints on circle i.
    Constraints for circle i:
      - boundary: x_i - r_i >= 0, W - x_i - r_i >= 0, y_i - r_i >= 0, H - y_i - r_i >= 0
      - pairwise: dist(i,j) - r_i - r_j >= 0 for all j != i
    
    We use: r_i = min of all these, approximated by soft-min.
    """
    H = 2.0 - W
    x = v[0:N]
    y = v[N:2*N]
    
    # For each circle, compute the soft-min of all its constraints
    # To avoid O(N^2) per objective call being too slow, we compute per-circle
    r = np.zeros(N)
    for i in range(N):
        # Boundary constraints
        c0 = x[i]  # x_i - r_i >= 0 -> r_i <= x_i
        c1 = W - x[i]
        c2 = y[i]
        c3 = H - y[i]
        
        # Pairwise constraints: r_i <= dist(i,j) - r_j for all j
        # But r_j is also variable... this is circular.
        # Alternative: use the Voronoi inradius as a smooth surrogate.
        # Actually, let's use a different formulation.
        
        # Let's define: for circle i, the maximum radius is limited by:
        #   r_i <= x_i, r_i <= W-x_i, r_i <= y_i, r_i <= H-y_i
        #   r_i + r_j <= dist(i,j) for all j
        # 
        # A smooth objective: maximize sum r_i subject to these.
        # Penalty form: -sum(r_i) + lambda * sum of violations^2
        
        # Actually, let me use the Voronoi approach:
        # r_i = inradius of Voronoi cell for point i in rectangle
        # This is smooth-ish (piecewise smooth) and automatically satisfies
        # all pairwise and boundary constraints.
        pass
    
    # Use Voronoi inradii approach
    r = _voronoi_inradii(x, y, W, H)
    return -np.sum(r)

def _objective_and_grad(v, W, lam=50.0):
    """
    Penalty-based objective with analytical gradient.
    v = [x1..xN, y1..yN, r1..rN]
    Objective: -sum(r) + lam * (overlap_penalty + boundary_penalty)
    """
    H = 2.0 - W
    x = v[0:N]
    y = v[N:2*N]
    r = v[2*N:3*N]
    
    val = -np.sum(r)
    grad = np.zeros(3*N)
    grad[2*N:3*N] = -1.0
    
    # Pairwise overlap penalty: max(0, r_i + r_j - d_ij)^2
    for i in range(N):
        for j in range(i+1, N):
            dx = x[i] - x[j]
            dy = y[i] - y[j]
            d_sq = dx*dx + dy*dy
            d = math.sqrt(d_sq + 1e-12)
            g = r[i] + r[j] - d
            if g > 0:
                pen = g * g
                val += lam * pen
                coeff = 2.0 * g / d
                grad[i] += lam * (-coeff * dx)
                grad[j] += lam * (coeff * dx)
                grad[N+i] += lam * (-coeff * dy)
                grad[N+j] += lam * (coeff * dy)
                grad[2*N+i] += lam * 2.0 * g
                grad[2*N+j] += lam * 2.0 * g
    
    # Boundary penalty
    for i in range(N):
        # x bounds
        if x[i] < r[i]:
            v_ = r[i] - x[i]
            val += lam * v_*v_
            grad[i] += 2.0 * lam * v_
            grad[2*N+i] -= 2.0 * lam * v_
        elif x[i] > W - r[i]:
            v_ = x[i] - (W - r[i])
            val += lam * v_*v_
            grad[i] -= 2.0 * lam * v_
            grad[2*N+i] += 2.0 * lam * v_
        
        # y bounds
        if y[i] < r[i]:
            v_ = r[i] - y[i]
            val += lam * v_*v_
            grad[N+i] += 2.0 * lam * v_
            grad[2*N+i] -= 2.0 * lam * v_
        elif y[i] > H - r[i]:
            v_ = y[i] - (H - r[i])
            val += lam * v_*v_
            grad[N+i] -= 2.0 * lam * v_
            grad[2*N+i] += 2.0 * lam * v_
        
        # radius lower bound
        if r[i] < 1e-5:
            v_ = 1e-5 - r[i]
            val += lam * v_*v_
            grad[2*N+i] -= 2.0 * lam * v_
    
    return val, grad

def _grid_init(W, H, n_cols, n_rows, seed=None):
    """Initialize circle centers in a grid pattern."""
    if seed is not None:
        np.random.seed(seed)
    
    x = np.zeros(N)
    y = np.zeros(N)
    
    idx = 0
    for row in range(n_rows):
        for col in range(n_cols):
            if idx >= N:
                break
            x[idx] = (col + 0.5) * (W / n_cols)
            y[idx] = (row + 0.5) * (H / n_rows)
            idx += 1
        if idx >= N:
            break
    
    # Add small random perturbation
    x += np.random.uniform(-0.005, 0.005, N)
    y += np.random.uniform(-0.005, 0.005, N)
    
    # Clamp to valid range
    for i in range(N):
        x[i] = np.clip(x[i], 0.01, W - 0.01)
        y[i] = np.clip(y[i], 0.01, H - 0.01)
    
    return x, y

def _run_lbfgsb(v0, W, deadline, maxiter=300, lam=100.0):
    """Run L-BFGS-B optimization."""
    H = 2.0 - W
    
    bounds = []
    for i in range(N):
        bounds.append((1e-4, W - 1e-4))
    for i in range(N):
        bounds.append((1e-4, H - 1e-4))
    for i in range(N):
        bounds.append((1e-5, 0.5))
    
    options = {'maxiter': maxiter, 'ftol': 1e-12, 'gtol': 1e-10, 'disp': False}
    
    try:
        res = minimize(
            lambda v: _objective_and_grad(v, W, lam)[0],
            v0,
            jac=lambda v: _objective_and_grad(v, W, lam)[1],
            method='L-BFGS-B',
            bounds=bounds,
            options=options
        )
        return res.x
    except Exception:
        return v0

def _simulated_annealing(x, y, W, deadline, n_iter=5000, T0=0.05, T_end=0.001):
    """
    Simulated annealing on circle centers.
    At each step, propose a random move for one center,
    accept if it improves the Voronoi-based sum of radii,
    or with probability exp(-delta/T) otherwise.
    """
    H = 2.0 - W
    x = x.copy()
    y = y.copy()
    
    # Compute initial energy (negative of sum of Voronoi inradii)
    r_init = _voronoi_inradii(x, y, W, H)
    E0 = -np.sum(r_init)
    
    best_x = x.copy()
    best_y = y.copy()
    best_E = E0
    
    rng = np.random.RandomState(42)
    
    for it in range(n_iter):
        if time.time() > deadline:
            break
        
        # Adaptive temperature
        T = T0 * (T_end / T0) ** (it / n_iter)
        
        # Choose a random circle to move
        i = rng.randint(N)
        
        # Propose a random displacement
        dx = rng.normal(0, 0.02)
        dy = rng.normal(0, 0.02)
        
        old_xi = x[i]
        old_yi = y[i]
        x[i] = np.clip(x[i] + dx, 0.01, W - 0.01)
        y[i] = np.clip(y[i] + dy, 0.01, H - 0.01)
        
        # Compute new energy
        r_new = _voronoi_inradii(x, y, W, H)
        E_new = -np.sum(r_new)
        
        delta = E_new - E0
        if delta < 0 or rng.random() < math.exp(-delta / max(T, 1e-10)):
            E0 = E_new
            if E0 < best_E:
                best_E = E0
                best_x = x.copy()
                best_y = y.copy()
        else:
            x[i] = old_xi
            y[i] = old_yi
    
    return best_x, best_y

def _polish(x, y, W, deadline):
    """
    High-precision polish: use L-BFGS-B with tight bounds and higher lambda.
    """
    H = 2.0 - W
    r = _voronoi_inradii(x, y, W, H)
    
    v0 = np.concatenate([x, y, r])
    
    bounds = []
    for i in range(N):
        bounds.append((max(1e-4, x[i]-0.002), min(W-1e-4, x[i]+0.002)))
    for i in range(N):
        bounds.append((max(1e-4, y[i]-0.002), min(H-1e-4, y[i]+0.002)))
    for i in range(N):
        bounds.append((max(1e-5, r[i]-0.002), min(0.5, r[i]+0.002)))
    
    options = {'maxiter': 500, 'ftol': 1e-14, 'gtol': 1e-12, 'disp': False}
    
    try:
        res = minimize(
            lambda v: _objective_and_grad(v, W, 500.0)[0],
            v0,
            jac=lambda v: _objective_and_grad(v, W, 500.0)[1],
            method='L-BFGS-B',
            bounds=bounds,
            options=options
        )
        v_opt = res.x
        x_opt = v_opt[0:N]
        y_opt = v_opt[N:2*N]
        r_opt = v_opt[2*N:3*N]
        
        val = _compute_value(x_opt, y_opt, r_opt, W)
        if val > 0:
            return x_opt, y_opt, r_opt, val
    except Exception:
        pass
    
    # Fallback: use Voronoi inradii
    r_vor = _voronoi_inradii(x, y, W, H)
    val = _compute_value(x, y, r_vor, W)
    return x, y, r_vor, val

def construct():
    deadline = time.time() + 145.0  # Safety margin before 150s limit
    
    best_val = 0.0
    best_x = np.zeros(N)
    best_y = np.zeros(N)
    best_r = np.zeros(N)
    best_W = 1.0
    
    # Aspect ratios to try (W, where H = 2-W)
    aspect_ratios = [0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1.0, 1.05, 1.1, 1.15, 1.2, 1.3]
    
    # Grid configurations
    grid_configs = [(4, 6), (5, 5), (6, 4), (3, 7), (7, 3), (4, 5), (5, 4)]
    
    np.random.seed(12345)
    
    # Phase 1: Multi-start L-BFGS-B optimization
    for W0 in aspect_ratios:
        if time.time() > deadline - 30:
            break
        H0 = 2.0 - W0
        
        for n_cols, n_rows in grid_configs:
            if time.time() > deadline - 30:
                break
            
            x0, y0 = _grid_init(W0, H0, n_cols, n_rows)
            r0 = _voronoi_inradii(x0, y0, W0, H0)
            
            v0 = np.concatenate([x0, y0, r0])
            
            # Run L-BFGS-B
            v_opt = _run_lbfgsb(v0, W0, deadline, maxiter=200, lam=100.0)
            
            x_opt = v_opt[0:N]
            y_opt = v_opt[N:2*N]
            r_opt = v_opt[2*N:3*N]
            
            val = _compute_value(x_opt, y_opt, r_opt, W0)
            
            if val > best_val:
                best_val = val
                best_x = x_opt.copy()
                best_y = y_opt.copy()
                best_r = r_opt.copy()
                best_W = W0
    
    # Phase 2: Simulated annealing exploration on best aspect ratio
    if time.time() < deadline - 20 and best_val > 0:
        W_best = best_W
        H_best = 2.0 - W_best
        
        # SA with multiple restarts
        for sa_restart in range(5):
            if time.time() > deadline - 20:
                break
            
            # Start from current best or a perturbed version
            if sa_restart == 0:
                x_sa = best_x.copy()
                y_sa = best_y.copy()
            else:
                x_sa = best_x + np.random.uniform(-0.05, 0.05, N)
                y_sa = best_y + np.random.uniform(-0.05, 0.05, N)
                for i in range(N):
                    x_sa[i] = np.clip(x_sa[i], 0.01, W_best - 0.01)
                    y_sa[i] = np.clip(y_sa[i], 0.01, H_best - 0.01)
            
            x_sa, y_sa = _simulated_annealing(
                x_sa, y_sa, W_best, deadline,
                n_iter=3000, T0=0.04, T_end=0.0005
            )
            
            # Polish the SA result
            x_pol, y_pol, r_pol, val_pol = _polish(x_sa, y_sa, W_best, deadline)
            
            if val_pol > best_val:
                best_val = val_pol
                best_x = x_pol.copy()
                best_y = y_pol.copy()
                best_r = r_pol.copy()
                best_W = W_best
    
    # Phase 3: Final high-precision polish
    if time.time() < deadline - 5 and best_val > 0:
        x_pol, y_pol, r_pol, val_pol = _polish(best_x, best_y, best_W, deadline)
        if val_pol > best_val:
            best_x = x_pol
            best_y = y_pol
            best_r = r_pol
            best_val = val_pol
    
    # Fallback: simple grid
    if best_val == 0.0:
        W = 1.0
        H = 1.0
        n_cols, n_rows = 5, 5
        for i in range(N):
            row = i // n_cols
            col = i % n_cols
            best_x[i] = (col + 0.5) / n_cols
            best_y[i] = (row + 0.5) / n_rows
            best_r[i] = 0.5 / n_cols - 1e-4
        best_W = W
    
    return np.column_stack([best_x, best_y, best_r])

# EVOLVE-BLOCK-END
