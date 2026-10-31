"""Heilbronn triangles: maximize minimum triangle area for 11 points in unit equilateral triangle."""
import numpy as np
import time
from scipy.optimize import minimize
from itertools import combinations

N = 11
H = np.sqrt(3) / 2

# Triangle vertices in Cartesian
V0 = np.array([0.0, 0.0])
V1 = np.array([1.0, 0.0])
V2 = np.array([0.5, H])

# Pre-compute triangle indices
tri_indices = list(combinations(range(N), 3))
tri_idx = np.array(tri_indices)  # (165, 3)

def calc_areas_bary(bary):
    """
    Calculate areas of all triangles formed by N points given in barycentric coords (u, v).
    Returns array of shape (165,) containing the area ratios.
    """
    u = bary[:, 0]
    v = bary[:, 1]
    w = 1.0 - u - v
    
    ui = u[tri_idx[:, 0]]
    vi = v[tri_idx[:, 0]]
    wi = w[tri_idx[:, 0]]
    
    uj = u[tri_idx[:, 1]]
    vj = v[tri_idx[:, 1]]
    wj = w[tri_idx[:, 1]]
    
    uk = u[tri_idx[:, 2]]
    vk = v[tri_idx[:, 2]]
    wk = w[tri_idx[:, 2]]
    
    det = ui * (vj * wk - vk * wj) - vi * (uj * wk - uk * wj) + wi * (uj * vk - uk * vj)
    
    return np.abs(det)

def clamp_bary(bary, eps=1e-9):
    """
    Clamp barycentric coordinates to the strictly valid triangle region.
    u >= eps, v >= eps, u+v <= 1 - eps
    """
    u = bary[:, 0].copy()
    v = bary[:, 1].copy()
    
    u = np.clip(u, eps, 1.0)
    v = np.clip(v, eps, 1.0)
    
    sums = u + v
    invalid = sums > (1.0 - eps)
    if np.any(invalid):
        scale = (1.0 - eps) / np.maximum(sums, eps)
        u[invalid] *= scale[invalid]
        v[invalid] *= scale[invalid]
        
    u = np.clip(u, eps, 1.0)
    v = np.clip(v, eps, 1.0)
    
    return np.column_stack([u, v])

def bary_to_cart(bary):
    """
    Convert barycentric coords (u, v) to Cartesian (x, y).
    Point = u*A + v*B + w*C where A=(0,0), B=(1,0), C=(0.5, H)
    x = v + 0.5*w
    y = w*H
    """
    u = bary[:, 0]
    v = bary[:, 1]
    w = 1.0 - u - v
    x = v + 0.5 * w
    y = w * H
    return np.column_stack([x, y])

def hard_min_obj(bary_flat):
    """Objective: negative of minimum triangle area."""
    bary = bary_flat.reshape(N, 2)
    bary = clamp_bary(bary)
    areas = calc_areas_bary(bary)
    return -np.min(areas)

def soft_min_obj(bary_flat, beta=100.0):
    """
    Objective: negative soft-min (log-sum-exp) of triangle areas.
    """
    bary = bary_flat.reshape(N, 2)
    bary = clamp_bary(bary)
    areas = calc_areas_bary(bary)
    
    scaled = -beta * areas
    max_scaled = np.max(scaled)
    if max_scaled > 700:
        return -np.min(areas)
    log_sum = max_scaled + np.log(np.sum(np.exp(scaled - max_scaled)))
    soft_min = -log_sum / beta
    
    return -soft_min

def generate_hexagonal_init():
    """
    Generate a hexagonal lattice-like initial configuration.
    """
    best_bary = None
    best_val = -np.inf
    
    for s in np.linspace(0.08, 0.5, 30):
        points = []
        for i in range(6):
            for j in range(6):
                u = i * s
                v = j * s * 0.5773502691896258  # sqrt(3)/2
                if u + v <= 0.98 and len(points) < N:
                    points.append([u, v])
        
        if len(points) == N:
            bary = np.array(points)
            val = -hard_min_obj(bary.flatten())
            if val > best_val:
                best_val = val
                best_bary = bary.copy()
    
    if best_bary is None:
        rng = np.random.default_rng(42)
        u = rng.random(N)
        v = rng.random(N) * (1 - u)
        best_bary = np.column_stack([u, v])
    
    return best_bary

def generate_random_init(seed=None):
    """Generate random valid barycentric coordinates."""
    rng = np.random.default_rng(seed)
    u = rng.random(N)
    v = rng.random(N) * (1 - u)
    return np.column_stack([u, v])

def generate_perturbed_init(base, noise_scale=0.03, seed=None):
    """Generate perturbed version of a base configuration."""
    rng = np.random.default_rng(seed)
    bary = base.copy()
    bary += rng.normal(0, noise_scale, bary.shape)
    return clamp_bary(bary)

def _sa_global(bary_init, deadline, seed=0):
    """
    Simulated annealing in barycentric space.
    Uses a Gaussian perturbation and accepts/rejects based on the hard min.
    """
    rng = np.random.default_rng(seed)
    bary = bary_init.copy()
    bary = clamp_bary(bary)
    
    current_val = -hard_min_obj(bary.flatten())
    best_bary = bary.copy()
    best_val = current_val
    
    T_start = 0.01
    T_end = 1e-6
    n_steps = 5000
    
    for step in range(n_steps):
        if time.time() > deadline - 2:
            break
        
        # Temperature schedule (geometric cooling)
        progress = step / n_steps
        T = T_start * (T_end / T_start) ** progress
        
        # Generate perturbation
        delta = rng.normal(0, 0.05, bary.shape)
        new_bary = clamp_bary(bary + delta)
        new_val = -hard_min_obj(new_bary.flatten())
        
        delta_val = new_val - current_val
        if delta_val > 0:
            bary = new_bary
            current_val = new_val
            if current_val > best_val:
                best_val = current_val
                best_bary = new_bary.copy()
        else:
            # Accept with probability exp(delta_val / T)
            if T > 0:
                prob = np.exp(delta_val / T)
                if rng.random() < prob:
                    bary = new_bary
                    current_val = new_val
    
    return best_bary, best_val

def _lbfgs_refine(bary_init, deadline, beta=200.0, maxiter=300):
    """
    L-BFGS-B refinement using soft-min objective.
    """
    if time.time() > deadline - 3:
        return bary_init, -hard_min_obj(bary_init.flatten())
    
    try:
        bounds = [(0, 1)] * (N * 2)
        
        def obj(x):
            return soft_min_obj(x, beta=beta)
        
        result = minimize(
            obj,
            bary_init.flatten(),
            method='L-BFGS-B',
            bounds=bounds,
            options={'maxiter': maxiter, 'ftol': 1e-15, 'gtol': 1e-12}
        )
        
        new_bary = clamp_bary(result.x.reshape(N, 2))
        new_val = -hard_min_obj(new_bary.flatten())
        
        return new_bary, new_val
    except Exception:
        return bary_init, -hard_min_obj(bary_init.flatten())

def _slsqp_polish(bary_init, deadline):
    """
    SLSQP polish for the constrained problem.
    """
    if time.time() > deadline - 2:
        return bary_init, -hard_min_obj(bary_init.flatten())
    
    try:
        # Convert to flat
        x0 = bary_init.flatten()
        
        # Bounds: 0 <= u_i <= 1, 0 <= v_i <= 1, and u_i + v_i <= 1
        bounds = []
        for i in range(N):
            bounds.append((0, 1))
            bounds.append((0, 1))
        
        # Constraints: u_i + v_i <= 1 for each i
        cons = []
        for i in range(N):
            def constraint(x, i=i):
                return 1.0 - x[2*i] - x[2*i+1]
            cons.append({'type': 'ineq', 'fun': constraint})
        
        def obj(x):
            return soft_min_obj(x, beta=500.0)
        
        result = minimize(
            obj,
            x0,
            method='SLSQP',
            bounds=bounds,
            constraints=cons,
            options={'maxiter': 200, 'ftol': 1e-14}
        )
        
        new_bary = clamp_bary(result.x.reshape(N, 2))
        new_val = -hard_min_obj(new_bary.flatten())
        
        return new_bary, new_val
    except Exception:
        return bary_init, -hard_min_obj(bary_init.flatten())

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin before 150s kill
    
    best_bary = None
    best_val = -np.inf
    
    def update_best(bary):
        """Update best if improved."""
        nonlocal best_bary, best_val
        bary = clamp_bary(bary)
        val = -hard_min_obj(bary.flatten())
        if val > best_val:
            best_val = val
            best_bary = bary.copy()
            return True
        return False

    def try_optimize(bary_init, use_sa=True, sa_steps=3000, sa_seed=0, lbfgs_beta=200, lbfgs_iter=300, slsqp_polish=True):
        """
        Multi-stage optimization:
        1. Simulated annealing for global exploration.
        2. L-BFGS-B with soft-min for local refinement.
        3. SLSQP polish for constrained precision.
        """
        if time.time() > deadline - 10:
            return
        
        bary = bary_init.copy()
        bary = clamp_bary(bary)
        
        # Stage 1: Simulated annealing
        if use_sa:
            bary, val = _sa_global(bary, deadline - 15, seed=sa_seed)
            update_best(bary)
        
        # Stage 2: L-BFGS-B refinement
        bary, val = _lbfgs_refine(bary, deadline - 8, beta=lbfgs_beta, maxiter=lbfgs_iter)
        update_best(bary)
        
        # Stage 3: SLSQP polish
        if slsqp_polish:
            bary, val = _slsqp_polish(bary, deadline)
            update_best(bary)

    # 1. Start with hexagonal lattice initialization
    hex_init = generate_hexagonal_init()
    try_optimize(hex_init, use_sa=True, sa_steps=4000, sa_seed=1, lbfgs_beta=200, lbfgs_iter=400, slsqp_polish=True)
    
    # 2. Try multiple random initializations with SA
    num_random_starts = 10
    for i in range(num_random_starts):
        if time.time() > deadline - 40:
            break
        rng_seed = 100 + i
        rand_init = generate_random_init(seed=rng_seed)
        try_optimize(rand_init, use_sa=True, sa_steps=3000, sa_seed=1000 + i, lbfgs_beta=150, lbfgs_iter=250, slsqp_polish=True)
    
    # 3. Perturb the best found and re-optimize
    if best_bary is not None:
        for i in range(8):
            if time.time() > deadline - 25:
                break
            perturbed = generate_perturbed_init(best_bary, noise_scale=0.02, seed=200 + i)
            try_optimize(perturbed, use_sa=True, sa_steps=2000, sa_seed=2000 + i, lbfgs_beta=250, lbfgs_iter=300, slsqp_polish=True)
    
    # 4. Final high-precision polish
    if best_bary is not None and time.time() < deadline - 10:
        try:
            bary, val = _lbfgs_refine(best_bary, deadline - 5, beta=500, maxiter=500)
            update_best(bary)
        except Exception:
            pass
        
        if time.time() < deadline - 5:
            try:
                bary, val = _slsqp_polish(best_bary, deadline)
                update_best(bary)
            except Exception:
                pass
    
    # 5. Convert best barycentric to Cartesian
    if best_bary is not None:
        best_cart = bary_to_cart(best_bary)
        return best_cart
    else:
        # Fallback: return hexagonal init converted to Cartesian
        return bary_to_cart(hex_init)
