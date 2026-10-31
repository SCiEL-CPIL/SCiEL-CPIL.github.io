import numpy as np
from itertools import combinations
import time

N = 11
TIME_LIMIT = 145.0

# Pre-compute triangle indices
tri_indices = list(combinations(range(N), 3))
tri_idx = np.array(tri_indices)  # (165, 3)

V0 = np.array([0.0, 0.0])
V1 = np.array([1.0, 0.0])
V2 = np.array([0.5, np.sqrt(3) / 2])
H = np.sqrt(3) / 2

def bary_to_cart(bary):
    u = bary[:, 0]
    v = bary[:, 1]
    w = 1.0 - u - v
    x = v + 0.5 * w
    y = w * H
    return np.column_stack([x, y])

def calc_areas_bary(bary):
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
    areas = calc_areas_bary(bary)
    return np.min(areas)

def generate_hexagonal_init():
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
    rng = np.random.default_rng(seed)
    u = rng.random(N)
    v = rng.random(N) * (1 - u)
    return np.column_stack([u, v])

def generate_perturbed_init(base_bary, sigma=0.05, seed=None):
    rng = np.random.default_rng(seed)
    noise = rng.normal(0, sigma, base_bary.shape)
    return clamp_bary(base_bary + noise)

def refine_local(bary_init, time_limit=5.0):
    start_time = time.time()
    bounds = [(0, 1)] * (N * 2)
    
    bary_current = bary_init.copy()
    current_val = min_area(bary_current)
    
    # Stage 1: Nelder-Mead
    try:
        from scipy.optimize import minimize
        result1 = minimize(
            lambda x: -min_area(x.reshape(N, 2)),
            bary_current.flatten(),
            method='Nelder-Mead',
            options={'maxiter': 1000, 'xatol': 1e-14, 'fatol': 1e-18, 'adaptive': True}
        )
        candidate = clamp_bary(result1.x.reshape(N, 2))
        val = min_area(candidate)
        if val > current_val:
            bary_current = candidate
            current_val = val
    except Exception:
        pass
    
    if time.time() - start_time > time_limit * 0.4:
        return bary_current, current_val
        
    # Stage 2: SLSQP
    try:
        from scipy.optimize import minimize
        result2 = minimize(
            lambda x: -min_area(x.reshape(N, 2)),
            bary_current.flatten(),
            method='SLSQP',
            bounds=bounds,
            options={'maxiter': 500, 'ftol': 1e-15}
        )
        candidate = clamp_bary(result2.x.reshape(N, 2))
        val = min_area(candidate)
        if val > current_val:
            bary_current = candidate
            current_val = val
    except Exception:
        pass
    
    if time.time() - start_time > time_limit * 0.7:
        return bary_current, current_val
    
    # Stage 3: Coordinate Descent (cyclic)
    try:
        for _ in range(200):
            if time.time() - start_time > time_limit:
                break
            improved = False
            for i in range(N):
                for j in range(2):
                    old = bary_current[i, j]
                    # Try small steps
                    for step in [0.001, -0.001, 0.0005, -0.0005, 0.002, -0.002]:
                        candidate = bary_current.copy()
                        candidate[i, j] = old + step
                        candidate = clamp_bary(candidate)
                        val = min_area(candidate)
                        if val > current_val + 1e-15:
                            bary_current = candidate
                            current_val = val
                            improved = True
                            break
            if not improved:
                break
    except Exception:
        pass
    
    return bary_current, current_val

def differential_evolution_search(initial_population, rng, time_limit=30.0, pop_size=30, F=0.8, CR=0.9):
    """
    Differential Evolution for global search in barycentric space.
    """
    start_time = time.time()
    
    # Initialize population
    pop = [initial_population[i].copy() for i in range(min(pop_size, len(initial_population)))]
    while len(pop) < pop_size:
        # Fill with random or perturbed
        if rng.random() < 0.5:
            pop.append(generate_random_init(seed=int(rng.integers(0, 100000))))
        else:
            base = pop[rng.integers(0, len(pop))]
            pop.append(generate_perturbed_init(base, sigma=0.1, seed=int(rng.integers(0, 100000))))
    
    # Evaluate initial population
    pop_vals = [min_area(p) for p in pop]
    best_idx = np.argmax(pop_vals)
    best_bary = pop[best_idx].copy()
    best_val = pop_vals[best_idx]
    
    gen = 0
    while time.time() - start_time < time_limit:
        gen += 1
        # Generate trial population
        for i in range(pop_size):
            # Select three distinct random indices different from i
            r1, r2, r3 = rng.choice(pop_size, 3, replace=False)
            while r1 == i or r2 == i or r3 == i:
                r1, r2, r3 = rng.choice(pop_size, 3, replace=False)
            
            # DE/rand/1
            mutant = pop[r1] + F * (pop[r2] - pop[r3])
            
            # Crossover
            trial = np.zeros_like(mutant)
            j_rand = rng.integers(0, 2 * N)
            for j in range(2 * N):
                if rng.random() < CR or j == j_rand:
                    trial[j] = mutant[j]
                else:
                    trial[j] = pop[i][j]
            
            trial_bary = clamp_bary(trial.reshape(N, 2))
            trial_val = min_area(trial_bary)
            
            # Selection
            if trial_val >= pop_vals[i]:
                pop[i] = trial_bary
                pop_vals[i] = trial_val
                if trial_val > best_val:
                    best_val = trial_val
                    best_bary = trial_bary.copy()
        
        # Adaptive parameter adjustment
        if gen % 10 == 0:
            F = max(0.3, F * 0.95)
            CR = min(0.95, CR * 1.02)
    
    return best_bary, best_val

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

    # 1. Generate diverse initial population
    rng = np.random.default_rng(42)
    
    hex_init = generate_hexagonal_init()
    update_best(hex_init)
    
    population_inits = [hex_init.copy()]
    
    # Add random starts
    for i in range(15):
        pop_member = generate_random_init(seed=1000 + i)
        population_inits.append(pop_member)
    
    # Add perturbed versions of hex_init
    for i in range(5):
        pop_member = generate_perturbed_init(hex_init, sigma=0.05, seed=2000 + i)
        population_inits.append(pop_member)
    
    # Add more structured inits
    for scale in [0.2, 0.3, 0.4]:
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
            for _ in range(3):
                idx = rng.choice(len(pts), N, replace=False)
                population_inits.append(np.array(pts)[idx])
    
    # 2. Differential Evolution global search
    if time.time() < deadline - 30:
        de_result, de_val = differential_evolution_search(
            population_inits, rng, time_limit=25.0, pop_size=30, F=0.8, CR=0.9
        )
        update_best(de_result)
    
    # 3. Local refinement of the best found
    if best_bary is not None and time.time() < deadline - 10:
        refined, refined_val = refine_local(best_bary, time_limit=10.0)
        update_best(refined)
    
    # 4. Final high-precision polish
    if best_bary is not None and time.time() < deadline - 5:
        try:
            from scipy.optimize import minimize
            bounds = [(0, 1)] * (N * 2)
            result_slsqp = minimize(
                lambda x: -min_area(x.reshape(N, 2)),
                best_bary.flatten(),
                method='SLSQP',
                bounds=bounds,
                options={'maxiter': 1000, 'ftol': 1e-16}
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
                    options={'maxiter': 3000, 'xatol': 1e-15, 'fatol': 1e-19, 'adaptive': True}
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
                    options={'maxiter': 2000, 'ftol': 1e-17}
                )
                update_best(clamp_bary(result_final.x.reshape(N, 2)))
            except Exception:
                pass

    # Convert to cartesian
    cart = bary_to_cart(best_bary)
    
    return cart

# EVOLVE-BLOCK-END
