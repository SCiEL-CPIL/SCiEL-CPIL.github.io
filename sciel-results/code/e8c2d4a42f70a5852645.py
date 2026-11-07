import numpy as np
import time
from scipy.optimize import minimize

N = 50
M = 2 * N - 1  # length of g = convolve(h, h)

def compute_value(h):
    """Compute the value metric for given heights."""
    h = np.asarray(h, dtype=float)
    if h.size != N:
        h = h[:N]
        if h.size < N:
            h = np.pad(h, (0, N - h.size), 'constant')
    if np.any(h < 0) or not np.all(np.isfinite(h)):
        return 0.0
    g = np.convolve(h, h)
    m = len(g)
    y = np.zeros(m + 2)
    y[1:-1] = g
    w = 1.0 / (m + 1)
    y_sq_terms = y[:m+1]**2 + y[:m+1]*y[1:m+2] + y[1:m+2]**2
    l2 = (w / 3.0) * np.sum(y_sq_terms)
    norm_1 = np.sum(np.abs(g)) / (m + 1)
    norm_inf = np.max(np.abs(g))
    if norm_1 == 0 or norm_inf == 0:
        return 0.0
    return l2 / (norm_1 * norm_inf)

def neg_value(h):
    """Negative of compute_value for minimization."""
    return -compute_value(h)

def refine_local(h):
    """Refine a candidate using L-BFGS-B."""
    try:
        result = minimize(
            neg_value,
            h,
            method='L-BFGS-B',
            bounds=[(0, None)] * N,
            options={'maxiter': 100, 'ftol': 1e-12, 'gtol': 1e-9}
        )
        h_opt = np.maximum(0, result.x)
        return h_opt
    except Exception:
        return h

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # safety margin under 150s
    
    # Initialize population with diverse structures
    # 1. Seed (all ones)
    # 2. Random uniform
    # 3. Gaussian bump
    # 4. Two bumps
    # 5. Random sparse
    
    rng = np.random.default_rng(42)
    
    def make_individual():
        choice = rng.integers(0, 5)
        if choice == 0:
            return np.ones(N)
        elif choice == 1:
            return rng.uniform(0, 1, N)
        elif choice == 2:
            x = np.linspace(-1, 1, N)
            return np.exp(-((x - 0)**2) / (2 * 0.3**2))
        elif choice == 3:
            x = np.linspace(-1, 1, N)
            return np.exp(-((x - 0.4)**2) / (2 * 0.2**2)) + np.exp(-((x + 0.4)**2) / (2 * 0.2**2))
        else:
            h = np.zeros(N)
            indices = rng.choice(N, size=rng.integers(5, 15), replace=False)
            h[indices] = rng.uniform(0.5, 2.0, size=len(indices))
            return h

    pop_size = 20
    population = [make_individual() for _ in range(pop_size)]
    
    # Evaluate initial population
    scores = [compute_value(h) for h in population]
    best_idx = np.argmax(scores)
    best_h = population[best_idx].copy()
    best_val = scores[best_idx]
    
    # Sort population
    sorted_indices = np.argsort(scores)[::-1]
    population = [population[i] for i in sorted_indices]
    scores = [scores[i] for i in sorted_indices]
    
    generation = 0
    while time.time() < deadline - 10:
        generation += 1
        
        # Selection: Keep top 50%
        keep = pop_size // 2
        parents = population[:keep]
        
        # Crossover and Mutation
        children = []
        while len(children) < pop_size - keep:
            # Pick two parents
            p1_idx = rng.integers(0, keep)
            p2_idx = rng.integers(0, keep)
            if p1_idx == p2_idx:
                continue
            p1 = parents[p1_idx]
            p2 = parents[p2_idx]
            
            # Uniform Crossover
            mask = rng.random(N) < 0.5
            child = np.where(mask, p1, p2)
            
            # Mutation: Add noise to some elements
            if rng.random() < 0.5:
                mutate_mask = rng.random(N) < 0.2
                noise = rng.normal(0, 0.5, N)
                child[mutate_mask] += noise[mutate_mask]
            
            # Ensure non-negative
            child = np.maximum(0, child)
            
            # Local Refinement (expensive, but effective)
            # Only refine if we have time budget
            if time.time() < deadline - 15:
                child = refine_local(child)
            
            children.append(child)
        
        # Combine population
        population = parents + children
        scores = [compute_value(h) for h in population]
        
        # Update best
        current_best_idx = np.argmax(scores)
        if scores[current_best_idx] > best_val:
            best_val = scores[current_best_idx]
            best_h = population[current_best_idx].copy()
            
        # Sort and keep top
        sorted_indices = np.argsort(scores)[::-1]
        population = [population[i] for i in sorted_indices[:pop_size]]
        scores = [scores[i] for i in sorted_indices[:pop_size]]
        
        # Elitism is handled by keeping parents, but let's ensure best is in population
        # It already is because we kept top half which includes best
        
    # Final refinement on best
    if time.time() < deadline - 2:
        best_h = refine_local(best_h)
        final_val = compute_value(best_h)
        if final_val > best_val:
            best_val = final_val
            best_h = best_h.copy()

    return best_h.tolist()
