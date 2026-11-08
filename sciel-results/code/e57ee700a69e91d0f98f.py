"""Direct per-point optimization with coordinate descent and genetic algorithm."""
import time
import numpy as np

N = 100

# EVOLVE-BLOCK-START
def _compute_max_overlap(h):
    """Compute max over all circular shifts k of sum_j h[j]*(1-h[(j+k)%N]) * (2/N)."""
    n = len(h)
    dx = 2.0 / n
    H = np.fft.rfft(h, n=2*n)
    autocorr = np.fft.irfft(H * np.conj(H), n=2*n)[:n]
    sum_h = np.sum(h)
    C = np.zeros(n)
    C[0] = autocorr[0]
    if n > 1:
        C[1:] = autocorr[n-1:0:-1]
    overlaps = (sum_h - C) * dx
    return np.max(overlaps)

def _project(h):
    """Project h onto feasible set: [0,1]^N with sum = N/2."""
    h = np.clip(h, 0.0, 1.0)
    diff = N / 2.0 - np.sum(h)
    if abs(diff) > 1e-12:
        if diff > 0:
            mask = h < 1.0 - 1e-10
            cnt = np.sum(mask)
            if cnt > 0:
                h[mask] += diff / cnt
        else:
            mask = h > 1e-10
            cnt = np.sum(mask)
            if cnt > 0:
                h[mask] += diff / cnt
        h = np.clip(h, 0.0, 1.0)
        # Final correction if still off
        diff2 = N / 2.0 - np.sum(h)
        if abs(diff2) > 1e-10:
            # Distribute evenly among non-saturated points
            if diff2 > 0:
                mask = h < 1.0 - 1e-10
            else:
                mask = h > 1e-10
            cnt = np.sum(mask)
            if cnt > 0:
                h[mask] += diff2 / cnt
            h = np.clip(h, 0.0, 1.0)
    return h

def _coordinate_descent(h, max_iter=500, tol=1e-8):
    """Coordinate descent: optimize each coordinate while maintaining feasibility."""
    h = _project(h.copy())
    best = _compute_max_overlap(h)
    
    for it in range(max_iter):
        improved = False
        for j in range(N):
            old_j = h[j]
            # Try moving coordinate j
            # When we change h[j], the sum changes by delta, so we need to compensate
            # Keep sum fixed: if h[j] increases by d, another point must decrease by d
            # Simple approach: try small perturbations and re-project
            for d in [-0.1, -0.05, -0.01, 0.01, 0.05, 0.1, -0.2, 0.2]:
                h_try = h.copy()
                h_try[j] = old_j + d
                h_try = _project(h_try)
                score = _compute_max_overlap(h_try)
                if score < best - tol:
                    h = h_try
                    best = score
                    improved = True
                    break
        if not improved:
            break
    
    return h, best

def _genetic_algorithm(pop_size=40, n_gen=200, mutation_rate=0.3):
    """Simple genetic algorithm for global exploration."""
    np.random.seed(42)
    # Initialize population
    pop = []
    scores = []
    
    # Add some structured individuals
    x = np.linspace(0, 2, N, endpoint=False)
    
    # Uniform
    h0 = np.full(N, 0.5)
    pop.append(h0)
    scores.append(_compute_max_overlap(h0))
    
    # Sinusoidal patterns
    for freq in [1, 2, 3, 4, 5]:
        for phase in [0, np.pi/2, np.pi, 3*np.pi/2]:
            h = 0.5 + 0.3 * np.sin(np.pi * freq * x + phase)
            h = _project(h)
            pop.append(h)
            scores.append(_compute_max_overlap(h))
    
    # Square waves
    for width in [0.2, 0.3, 0.4, 0.5]:
        for offset in [0, 0.5, 1.0]:
            h = ((x + offset) % 2.0 < width).astype(float)
            h = _project(h)
            pop.append(h)
            scores.append(_compute_max_overlap(h))
    
    # Random individuals
    while len(pop) < pop_size:
        h = np.random.uniform(0, 1, N)
        h = _project(h)
        pop.append(h)
        scores.append(_compute_max_overlap(h))
    
    scores = np.array(scores)
    
    best_h = pop[np.argmin(scores)]
    best_score = scores.min()
    
    for gen in range(n_gen):
        # Selection: keep top half
        n_keep = pop_size // 2
        indices = np.argsort(scores)[:n_keep]
        new_pop = [pop[i].copy() for i in indices]
        
        # Crossover and mutation to fill population
        while len(new_pop) < pop_size:
            i1 = np.random.randint(0, n_keep)
            i2 = np.random.randint(0, n_keep)
            # Uniform crossover
            mask = np.random.uniform(0, 1, N) > 0.5
            child = np.where(mask, pop[indices[i1]], pop[indices[i2]])
            # Mutation
            if np.random.random() < mutation_rate:
                n_mut = np.random.randint(1, 10)
                mut_idx = np.random.choice(N, n_mut, replace=False)
                child[mut_idx] += np.random.uniform(-0.3, 0.3, n_mut)
            child = _project(child)
            new_pop.append(child)
        
        # Evaluate new population
        for i in range(len(new_pop)):
            new_pop[i] = _project(new_pop[i])
            s = _compute_max_overlap(new_pop[i])
            if s < best_score:
                best_score = s
                best_h = new_pop[i].copy()
        
        pop = new_pop
        scores = np.array([_compute_max_overlap(h) for h in pop])
    
    return best_h, best_score

def construct():
    start_time = time.time()
    deadline = start_time + 145.0
    
    # Phase 1: Genetic algorithm for global exploration
    ga_h, ga_score = _genetic_algorithm(pop_size=40, n_gen=150, mutation_rate=0.3)
    
    if time.time() > deadline:
        return _project(ga_h)
    
    # Phase 2: Coordinate descent refinement from GA best
    cd_h, cd_score = _coordinate_descent(ga_h, max_iter=300, tol=1e-8)
    
    if time.time() > deadline:
        return _project(cd_h)
    
    # Phase 3: Try multiple restarts of coordinate descent from different starts
    best_h = cd_h
    best_score = cd_score
    
    x = np.linspace(0, 2, N, endpoint=False)
    np.random.seed(123)
    
    for restart in range(10):
        if time.time() > deadline:
            break
        
        # Generate diverse starting points
        if restart == 0:
            h0 = np.full(N, 0.5)
        elif restart == 1:
            h0 = 0.5 + 0.4 * np.sin(np.pi * x)
        elif restart == 2:
            h0 = 0.5 + 0.4 * np.cos(np.pi * x)
        elif restart == 3:
            h0 = (x < 1.0).astype(float)
        elif restart == 4:
            h0 = 0.5 + 0.3 * np.sin(2 * np.pi * x)
        elif restart == 5:
            h0 = 0.5 + 0.3 * np.cos(2 * np.pi * x)
        else:
            h0 = np.random.uniform(0, 1, N)
        
        h0 = _project(h0)
        cd_h, cd_score = _coordinate_descent(h0, max_iter=200, tol=1e-8)
        
        if cd_score < best_score:
            best_score = cd_score
            best_h = cd_h
    
    # Final validation
    best_h = _project(best_h)
    if abs(np.sum(best_h) - N/2) > 1e-6:
        best_h = np.full(N, 0.5)
    
    return best_h
# EVOLVE-BLOCK-END
