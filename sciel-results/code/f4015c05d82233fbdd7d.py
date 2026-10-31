import numpy as np
from itertools import combinations
import time

N = 11
TIME_LIMIT = 145.0  # Safety margin for 150s limit

# Pre-compute triangle indices
tri_indices = list(combinations(range(N), 3))
tri_idx = np.array(tri_indices)  # (165, 3)

# Triangle vertices in Cartesian
V0 = np.array([0.0, 0.0])
V1 = np.array([1.0, 0.0])
V2 = np.array([0.5, np.sqrt(3) / 2])
H = np.sqrt(3) / 2

def bary_to_cart(bary):
    """Convert barycentric coords (u, v) to Cartesian (x, y)."""
    u = bary[:, 0]
    v = bary[:, 1]
    w = 1.0 - u - v
    x = v + 0.5 * w
    y = w * H
    return np.column_stack([x, y])

def calc_areas_bary(bary):
    """Calculate areas of all triangles formed by N points in barycentric coords."""
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

def clamp_bary(bary, eps=1e-10):
    """Clamp barycentric coordinates to the strictly valid triangle region."""
    u = bary[:, 0].copy()
    v = bary[:, 1].copy()
    
    u = np.clip(u, eps, 1 - eps)
    v = np.clip(v, eps, 1 - eps)
    
    sums = u + v
    invalid = sums > (1 - eps)
    if np.any(invalid):
        scale = (1 - eps) / np.where(sums > 0, sums, 1.0)
        u[invalid] *= scale[invalid]
        v[invalid] *= scale[invalid]
        
    u = np.clip(u, eps, 1 - eps)
    v = np.clip(v, eps, 1 - eps)
    
    return np.column_stack([u, v])

def min_area(bary):
    """Compute the minimum triangle area for a configuration."""
    areas = calc_areas_bary(bary)
    return np.min(areas)

def generate_hexagonal_init():
    """Generate a hexagonal lattice-like initial configuration."""
    best_bary = None
    best_val = -np.inf
    
    for scale in [0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5]:
        pts = []
        max_k = int(1.0 / scale) + 1
        for k in range(max_k + 1):
            w_val = k * scale
            if w_val > 1.0:
                break
            num_pts_in_row = int((1 - w_val) / scale) + 1
            for j in range(num_pts_in_row):
                u_val = j * scale
                v_val = (1 - w_val) - u_val
                if u_val >= -1e-9 and v_val >= -1e-9 and u_val + v_val <= 1 + 1e-9:
                    pts.append([u_val, v_val])
        
        if len(pts) >= N:
            rng = np.random.default_rng(42)
            for _ in range(10):
                if len(pts) == N:
                    bary = np.array(pts)
                else:
                    idx = rng.choice(len(pts), N, replace=False)
                    bary = np.array(pts)[idx]
                val = min_area(bary)
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

def generate_perturbed_init(base_bary, sigma=0.05, seed=None):
    """Generate a perturbed version of a base configuration."""
    rng = np.random.default_rng(seed)
    noise = rng.normal(0, sigma, base_bary.shape)
    return clamp_bary(base_bary + noise)

def hill_climb(bary_init, rng, max_iter=200, step_size=0.02):
    """
    Simple hill climbing: try random perturbations and keep improvements.
    """
    current = bary_init.copy()
    current_val = min_area(current)
    
    for i in range(max_iter):
        # Try a few perturbations
        best_candidate = None
        best_candidate_val = current_val
        
        for _ in range(10):
            # Random perturbation
            noise = rng.normal(0, step_size, current.shape)
            candidate = clamp_bary(current + noise)
            val = min_area(candidate)
            
            if val > best_candidate_val:
                best_candidate_val = val
                best_candidate = candidate
        
        if best_candidate is not None:
            current = best_candidate
            current_val = best_candidate_val
        else:
            # No improvement, break early
            break
            
    return current, current_val

def simulated_annealing(bary_init, rng, time_limit=5.0):
    """
    Simulated annealing: accept worse solutions with decreasing probability.
    """
    start_time = time.time()
    current = bary_init.copy()
    current_val = min_area(current)
    best = current.copy()
    best_val = current_val
    
    T = 0.01  # Initial temperature
    cooling = 0.995
    
    while time.time() - start_time < time_limit:
        # Generate candidate
        noise = rng.normal(0, 0.01, current.shape)
        candidate = clamp_bary(current + noise)
        candidate_val = min_area(candidate)
        
        # Accept if better, or with probability exp(-delta/T)
        delta = candidate_val - current_val
        if delta > 0 or rng.random() < np.exp(delta / T):
            current = candidate
            current_val = candidate_val
            
            if current_val > best_val:
                best_val = current_val
                best = current.copy()
        
        T *= cooling
        if T < 1e-8:
            break
            
    return best, best_val

def refine_local(bary_init, time_limit=5.0):
    """
    Refine a configuration using multiple optimization stages.
    """
    start_time = time.time()
    bounds = [(0, 1)] * (N * 2)
    
    bary_current = bary_init.copy()
    current_val = min_area(bary_current)
    
    # Stage 1: Hill climbing
    try:
        from scipy.optimize import minimize
        result1 = minimize(
            lambda x: -min_area(x.reshape(N, 2)),
            bary_current.flatten(),
            method='Nelder-Mead',
            options={'maxiter': 500, 'xatol': 1e-12, 'fatol': 1e-16, 'adaptive': True}
        )
        candidate = clamp_bary(result1.x.reshape(N, 2))
        val = min_area(candidate)
        if val > current_val:
            bary_current = candidate
            current_val = val
    except Exception:
        pass
    
    if time.time() - start_time > time_limit * 0.5:
        return bary_current, current_val
        
    # Stage 2: SLSQP polish
    try:
        from scipy.optimize import minimize
        result2 = minimize(
            lambda x: -min_area(x.reshape(N, 2)),
            bary_current.flatten(),
            method='SLSQP',
            bounds=bounds,
            options={'maxiter': 300, 'ftol': 1e-14}
        )
        candidate = clamp_bary(result2.x.reshape(N, 2))
        val = min_area(candidate)
        if val > current_val:
            bary_current = candidate
            current_val = val
    except Exception:
        pass
        
    return bary_current, current_val

def construct():
    start_time = time.time()
    deadline = start_time + TIME_LIMIT
    
    best_bary = None
    best_val = -np.inf
    
    def update_best(bary):
        nonlocal best_bary, best_val
        bary = clamp_bary(bary)
        val = min_area(bary)
        if val > best_val:
            best_val = val
            best_bary = bary.copy()
            return True
        return False

    # 1. Try Hexagonal Init
    hex_init = generate_hexagonal_init()
    refined_hex, val_hex = refine_local(hex_init, time_limit=8.0)
    update_best(refined_hex)
    
    # 2. Population-based global search
    rng = np.random.default_rng(42)
    
    # Initialize population with diverse starts
    population = [hex_init.copy()]
    
    # Add random starts
    for i in range(10):
        pop_member = generate_random_init(seed=1000 + i)
        population.append(pop_member)
    
    # Add perturbed versions of hex_init
    for i in range(5):
        pop_member = generate_perturbed_init(hex_init, sigma=0.05, seed=2000 + i)
        population.append(pop_member)
    
    # Evaluate initial population
    pop_vals = [min_area(p) for p in population]
    
    # Main search loop: alternate between global moves and local refinement
    while time.time() < deadline - 15:
        # Pick a member from population (weighted by fitness)
        # Use tournament selection
        idx1 = rng.integers(0, len(population))
        idx2 = rng.integers(0, len(population))
        
        if pop_vals[idx1] >= pop_vals[idx2]:
            parent_idx = idx1
        else:
            parent_idx = idx2
        
        parent = population[parent_idx].copy()
        
        # Generate child via mutation
        mutation_type = rng.random()
        
        if mutation_type < 0.5:
            # Gaussian perturbation
            noise = rng.normal(0, 0.03, parent.shape)
            child = clamp_bary(parent + noise)
        elif mutation_type < 0.8:
            # Crossover with another population member
            other_idx = rng.integers(0, len(population))
            if other_idx != parent_idx:
                other = population[other_idx]
                # Blend
                alpha = rng.uniform(0.3, 0.7)
                child = clamp_bary(alpha * parent + (1 - alpha) * other)
            else:
                noise = rng.normal(0, 0.03, parent.shape)
                child = clamp_bary(parent + noise)
        else:
            # Large jump (restart-like)
            child = generate_random_init(seed=int(time.time() * 1000) % 100000)
        
        child_val = min_area(child)
        
        # Local refinement
        if time.time() < deadline - 20:
            refined_child, refined_val = refine_local(child, time_limit=3.0)
            child = refined_child
            child_val = refined_val
        
        # Update population: replace worst member if child is better
        worst_idx = np.argmin(pop_vals)
        if child_val > pop_vals[worst_idx]:
            population[worst_idx] = child
            pop_vals[worst_idx] = child_val
        
        # Update best
        if child_val > best_val:
            best_val = child_val
            best_bary = child.copy()
    
    # 3. Final high-precision polish
    if best_bary is not None and time.time() < deadline - 5:
        try:
            from scipy.optimize import minimize
            bounds = [(0, 1)] * (N * 2)
            result_slsqp = minimize(
                lambda x: -min_area(x.reshape(N, 2)),
                best_bary.flatten(),
                method='SLSQP',
                bounds=bounds,
                options={'maxiter': 1000, 'ftol': 1e-15}
            )
            update_best(clamp_bary(result_slsqp.x.reshape(N, 2)))
        except Exception:
            pass
        
        if best_bary is not None and time.time() < deadline - 2:
            try:
                from scipy.optimize import minimize
                result_nm = minimize(
                    lambda x: -min_area(x.reshape(N, 2)),
                    best_bary.flatten(),
                    method='Nelder-Mead',
                    options={'maxiter': 3000, 'xatol': 1e-14, 'fatol': 1e-18, 'adaptive': True}
                )
                update_best(clamp_bary(result_nm.x.reshape(N, 2)))
            except Exception:
                pass
        
        # One more SLSQP pass
        if best_bary is not None and time.time() < deadline - 0.5:
            try:
                from scipy.optimize import minimize
                bounds = [(0, 1)] * (N * 2)
                result_final = minimize(
                    lambda x: -min_area(x.reshape(N, 2)),
                    best_bary.flatten(),
                    method='SLSQP',
                    bounds=bounds,
                    options={'maxiter': 2000, 'ftol': 1e-16}
                )
                update_best(clamp_bary(result_final.x.reshape(N, 2)))
            except Exception:
                pass

    # Convert to cartesian
    cart = bary_to_cart(best_bary)
    
    return cart

# EVOLVE-BLOCK-END
