import numpy as np

N = 600

def _score(h):
    """Compute the normalized maximum autoconvolution value."""
    h = np.asarray(h, dtype=float)
    total = h.sum()
    if total <= 0:
        return float('inf')
    conv = np.convolve(h, h, mode='full')
    max_conv = conv.max()
    return max_conv / (total * total)

def _coordinate_descent(h, max_iter=50):
    """
    Coordinate descent: for each coordinate, try setting it to 0 or scaling
    to reduce the score. This is a simple but effective local search.
    """
    h = np.array(h, dtype=float)
    best_score = _score(h)
    
    for iteration in range(max_iter):
        improved = False
        for i in range(N):
            # Try setting h[i] to 0
            if h[i] > 1e-10:
                h[i] = 0
                s0 = _score(h)
                if s0 < best_score:
                    best_score = s0
                    improved = True
                else:
                    h[i] = h[i]  # restore
            else:
                # Try setting h[i] to small positive values
                for val in [0.01, 0.1, 0.5, 1.0]:
                    h[i] = val
                    s1 = _score(h)
                    if s1 < best_score:
                        best_score = s1
                        improved = True
                        break
                    h[i] = 0
        if not improved:
            break
    
    return h, best_score

def _evolve_population(pop, scores, n_offspring=100, mutation_rate=0.3):
    """
    Evolutionary step: select parents, crossover, mutate.
    """
    n = len(pop)
    offspring = []
    
    # Tournament selection
    def tournament_select():
        idxs = np.random.choice(n, size=3, replace=False)
        return pop[idxs[np.argmin(scores[idxs])]].copy()
    
    for _ in range(n_offspring):
        # Crossover: uniform
        p1 = tournament_select()
        p2 = tournament_select()
        mask = np.random.rand(N) < 0.5
        child = np.where(mask, p1, p2)
        
        # Mutation
        if np.random.rand() < mutation_rate:
            # Add small Gaussian noise
            noise = np.random.randn(N) * 0.1
            child = np.maximum(child + noise, 0)
        
        # Sometimes do a block mutation: zero out a random block
        if np.random.rand() < 0.2:
            start = np.random.randint(0, N - 10)
            length = np.random.randint(5, 30)
            child[start:start+length] = 0
        
        offspring.append(child)
    
    return offspring

def construct():
    np.random.seed(12345)
    
    # Initialize diverse population
    pop = []
    
    # 1. Constant
    pop.append(np.ones(N))
    
    # 2. Blocky structures
    for n_blocks in [2, 3, 5, 10]:
        block_size = N // n_blocks
        h = np.zeros(N)
        for b in range(n_blocks):
            if b % 2 == 0:
                h[b*block_size:(b+1)*block_size] = 1.0
        pop.append(h)
    
    # 3. Sinusoidal
    for freq in [1, 2, 3, 5]:
        x = np.linspace(0, 2*np.pi*freq, N)
        h = 0.5 + 0.5*np.sin(x)
        pop.append(np.maximum(h, 0))
    
    # 4. Peaked (triangular)
    for peak_width in [50, 100, 200]:
        h = np.zeros(N)
        center = N // 2
        h[center-peak_width:center+peak_width] = np.linspace(1, 0, peak_width)
        h[center:center+peak_width] = np.linspace(1, 0, peak_width)[::-1]
        pop.append(h)
    
    # 5. Random
    for _ in range(10):
        pop.append(np.random.rand(N))
    
    # 6. Sparse random (many zeros)
    for _ in range(5):
        h = np.zeros(N)
        nonzero_idx = np.random.choice(N, size=N//5, replace=False)
        h[nonzero_idx] = np.random.rand(len(nonzero_idx))
        pop.append(h)
    
    # Evaluate initial population
    scores = [_score(h) for h in pop]
    
    best_h = pop[np.argmin(scores)].copy()
    best_score = min(scores)
    
    # Evolutionary loop
    max_generations = 20
    for gen in range(max_generations):
        # Create offspring
        offspring = _evolve_population(pop, scores, n_offspring=50, mutation_rate=0.4)
        
        # Evaluate offspring
        offspring_scores = [_score(h) for h in offspring]
        
        # Combine and select
        all_h = pop + offspring
        all_scores = scores + offspring_scores
        
        # Keep top 50
        n_keep = 50
        idxs = np.argsort(all_scores)[:n_keep]
        pop = [all_h[i] for i in idxs]
        scores = [all_scores[i] for i in idxs]
        
        # Track best
        if scores[0] < best_score:
            best_score = scores[0]
            best_h = pop[0].copy()
        
        # Periodically apply coordinate descent to best few
        if gen % 5 == 4:
            for i in range(min(5, len(pop))):
                refined, refined_score = _coordinate_descent(pop[i], max_iter=10)
                if refined_score < best_score:
                    best_score = refined_score
                    best_h = refined.copy()
                # Replace in population
                pop[i] = refined
                scores[i] = refined_score
            # Re-sort
            idxs = np.argsort(scores)[:n_keep]
            pop = [pop[i] for i in idxs]
            scores = [scores[i] for i in idxs]
    
    # Final coordinate descent on best
    best_h, final_score = _coordinate_descent(best_h, max_iter=20)
    
    return best_h.tolist()
