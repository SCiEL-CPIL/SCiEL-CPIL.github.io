import numpy as np
from scipy.optimize import minimize, differential_evolution
import time
from itertools import combinations

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
AREA_TOTAL = np.sqrt(3) / 4.0

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

def clamp_bary(bary, eps=1e-9):
    """Clamp barycentric coordinates to the valid triangle region."""
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

def get_min_area(bary):
    """Return the minimum triangle area for a configuration."""
    areas = calc_areas_bary(bary)
    return np.min(areas)

def penalty_objective(x_flat, t, penalty_factor=100.0):
    """
    Objective for penalty method:
    Minimize -t + penalty * sum(max(0, t - area_i)^2)
    This encourages t to increase while keeping all areas >= t.
    """
    bary = x_flat.reshape(N, 2)
    bary = clamp_bary(bary)
    areas = calc_areas_bary(bary)
    
    # Violations: where area < t
    violations = np.maximum(0, t - areas)
    penalty = np.sum(violations ** 2)
    
    return -t + penalty_factor * penalty

def solve_epigraph(bary_init, t_init, time_limit=5.0):
    """
    Solve the epigraph problem using penalty method with L-BFGS-B.
    We optimize over (bary, t) jointly.
    """
    start_time = time.time()
    
    # State: [u1, v1, ..., u11, v11, t]
    x0 = np.append(bary_init.flatten(), t_init)
    bounds = [(0, 1)] * (N * 2) + [(0, 1.0)] # t is bounded by max possible area (which is < 1)
    
    current_penalty = 10.0
    best_bary = bary_init.copy()
    best_t = t_init
    best_score = -np.inf
    
    for iter_num in range(20):
        if time.time() - start_time > time_limit:
            break
            
        # Define objective for current penalty
        def obj(x):
            bary_part = x[:-1]
            t_val = x[-1]
            return penalty_objective(bary_part, t_val, current_penalty)
            
        try:
            res = minimize(
                obj,
                x0,
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 50, 'ftol': 1e-12, 'gtol': 1e-8}
            )
            
            new_bary = clamp_bary(res.x[:-1].reshape(N, 2))
            new_t = res.x[-1]
            
            # Check feasibility of the new t with the new bary
            actual_min_area = get_min_area(new_bary)
            
            # The effective score is the actual min area
            effective_t = min(new_t, actual_min_area)
            
            if effective_t > best_score:
                best_score = effective_t
                best_bary = new_bary.copy()
                best_t = effective_t
                x0 = np.append(new_bary.flatten(), effective_t)
                
            # Increase penalty if violations remain
            violations = np.maximum(0, new_t - calc_areas_bary(new_bary))
            if np.sum(violations ** 2) > 1e-6:
                current_penalty *= 2.0
            else:
                # If feasible, we can try to push t up directly
                # Simple ascent on t
                # For now, just increase penalty to tighten constraint
                current_penalty *= 1.5
                
        except Exception:
            break
            
    return best_bary, best_t

def generate_hexagonal_init():
    """
    Generate a hexagonal lattice-inspired initial configuration.
    This is often a good starting point for Heilbronn problems.
    """
    # Hexagonal packing in a triangle
    # Layer 1 (bottom): 3 points
    # Layer 2: 2 points
    # Layer 3: 1 point
    # Plus some interior/edge adjustments for N=11
    
    # Let's use a known good structure for N=11 if possible, or generate one.
    # A common approach is to take the optimal for N=10 and add a point, 
    # but we don't have that. 
    # We'll use a perturbed grid.
    
    # Grid in barycentric coords
    u = np.array([0.0, 0.25, 0.5, 0.75, 1.0, 0.0, 0.25, 0.5, 0.75, 0.0, 0.5])
    v = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.25, 0.5, 0.75, 1.0, 0.5, 0.25])
    
    # Filter valid points (u+v <= 1)
    valid = u + v <= 1.0 + 1e-9
    u = u[valid]
    v = v[valid]
    
    # We need exactly 11 points. If we have more, subsample. If less, add random.
    # The above grid likely has more than 11 valid points.
    # Let's pick the most "spread out" ones or just take the first 11 valid.
    # Better: Use a specific known configuration or random sampling from the grid.
    
    # Let's just take the first 11 valid ones from a denser grid
    grid_size = 5
    u_grid, v_grid = np.meshgrid(np.linspace(0, 1, grid_size), np.linspace(0, 1, grid_size))
    u_flat = u_grid.flatten()
    v_flat = v_grid.flatten()
    
    valid_mask = (u_flat + v_flat <= 1.0 + 1e-9)
    u_valid = u_flat[valid_mask]
    v_valid = v_flat[valid_mask]
    
    # Randomly select 11 from valid grid points
    rng = np.random.default_rng(42)
    if len(u_valid) >= 11:
        indices = rng.choice(len(u_valid), 11, replace=False)
        u_sel = u_valid[indices]
        v_sel = v_valid[indices]
    else:
        # Pad with random if needed (unlikely with grid 5)
        u_sel = np.random.rand(11)
        v_sel = np.random.rand(11) * (1 - u_sel)
        
    bary = np.column_stack([u_sel, v_sel])
    return clamp_bary(bary)

def evolve_population(pop, time_limit=30.0):
    """
    Simple evolutionary loop:
    1. Evaluate all
    2. Keep top half
    3. Generate offspring from top half (crossover + mutation)
    4. Refine offspring with epigraph solver
    """
    start_time = time.time()
    best_bary = pop[0]
    best_val = get_min_area(best_bary)
    
    # Initial evaluation
    vals = [get_min_area(p) for p in pop]
    
    max_generations = 50
    pop_size = len(pop)
    elite_size = max(2, pop_size // 2)
    
    for gen in range(max_generations):
        if time.time() - start_time > time_limit:
            break
            
        # Sort by fitness
        sorted_indices = np.argsort(vals)[::-1]
        elite_indices = sorted_indices[:elite_size]
        elites = [pop[i] for i in elite_indices]
        elite_vals = [vals[i] for i in elite_indices]
        
        # Update global best
        if elite_vals[0] > best_val:
            best_val = elite_vals[0]
            best_bary = elites[0].copy()
            
        # Generate new population
        new_pop = []
        new_vals = []
        
        # Keep some elites
        for i in range(min(elite_size, pop_size // 2)):
            new_pop.append(elites[i].copy())
            new_vals.append(elite_vals[i])
            
        # Generate offspring
        while len(new_pop) < pop_size:
            # Crossover
            parent1 = elites[np.random.randint(elite_size)]
            parent2 = elites[np.random.randint(elite_size)]
            
            # Uniform crossover
            mask = np.random.rand(N, 2) > 0.5
            child = np.where(mask, parent1, parent2)
            
            # Mutation
            child += np.random.normal(0, 0.05, size=child.shape)
            child = clamp_bary(child)
            
            # Evaluate
            val = get_min_area(child)
            
            # Refine with epigraph solver if promising
            # Only refine if time allows and val is decent
            if time.time() - start_time < time_limit * 0.8:
                try:
                    refined_bary, refined_val = solve_epigraph(child, val, time_limit=0.5)
                    if refined_val > val:
                        child = refined_bary
                        val = refined_val
                except Exception:
                    pass
            
            new_pop.append(child)
            new_vals.append(val)
            
        pop = new_pop
        vals = new_vals

    return best_bary, best_val

def construct():
    start_time = time.time()
    deadline = start_time + TIME_LIMIT

    best_bary = None
    best_val = -np.inf

    def update_best(bary):
        nonlocal best_bary, best_val
        bary = clamp_bary(bary)
        val = get_min_area(bary)
        if val > best_val:
            best_val = val
            best_bary = bary.copy()
            return True
        return False

    # 1. Generate initial population
    pop = []
    
    # Structured init
    hex_init = generate_hexagonal_init()
    pop.append(hex_init)
    
    # Random inits
    rng = np.random.default_rng(42)
    for _ in range(10):
        u = rng.random(N)
        v = rng.random(N) * (1 - u)
        pop.append(np.column_stack([u, v]))
        
    # Perturbed hex
    for _ in range(5):
        perturbed = hex_init + rng.normal(0, 0.05, size=hex_init.shape)
        pop.append(clamp_bary(perturbed))
        
    # 2. Run Evolutionary Search
    evo_bary, evo_val = evolve_population(pop, time_limit=60.0)
    update_best(evo_bary)
    
    # 3. If time remains, do a more intensive epigraph refinement on the best found
    if best_bary is not None and time.time() < deadline - 30:
        # Increase t_init to current best to try to push higher
        try:
            # Use a longer time budget for the final polish
            remaining = deadline - 30 - time.time()
            refined_bary, refined_val = solve_epigraph(best_bary, best_val, time_limit=remaining)
            update_best(refined_bary)
        except Exception:
            pass
            
    # 4. Final Differential Evolution if time permits (as a backup global search)
    if time.time() < deadline - 40:
        try:
            de_bounds = [(0, 1)] * (N * 2)
            
            def de_obj(x):
                b = clamp_bary(x.reshape(N, 2))
                return -get_min_area(b)
                
            de_result = differential_evolution(
                de_obj,
                bounds=de_bounds,
                seed=42,
                maxiter=10,
                popsize=8,
                tol=1e-5,
                polish=True
            )
            
            if de_result is not None and de_result.x is not None:
                de_bary = clamp_bary(de_result.x.reshape(N, 2))
                de_val = get_min_area(de_bary)
                
                # Refine DE result
                if de_val > best_val * 0.9:
                    remaining_de = deadline - 20 - time.time()
                    if remaining_de > 5:
                        refined_de, _ = solve_epigraph(de_bary, de_val, time_limit=remaining_de)
                        update_best(refined_de)
                    else:
                        update_best(de_bary)
        except Exception:
            pass

    # 5. Final local polish with Nelder-Mead on the best
    if best_bary is not None and time.time() < deadline - 5:
        try:
            def nm_obj(x):
                b = clamp_bary(x.reshape(N, 2))
                return -get_min_area(b)
                
            res = minimize(
                nm_obj,
                best_bary.flatten(),
                method='Nelder-Mead',
                options={'maxiter': 100, 'xatol': 1e-10, 'fatol': 1e-12, 'adaptive': True}
            )
            update_best(clamp_bary(res.x.reshape(N, 2)))
        except Exception:
            pass

    if best_bary is None:
        best_bary = generate_hexagonal_init()

    cart = bary_to_cart(best_bary)
    return cart
