"""Erdos minimum overlap: minimize max overlap of h with shifted 1-h.
Approach: Genetic Algorithm (Evolutionary Strategy) with tournament selection,
uniform crossover, and Gaussian mutation with constraint projection.
"""
import time
import numpy as np

N = 100
_DEADLINE = time.time() + 115  # Safety margin before 120s kill


def _linear_overlap_values(h):
    """
    Compute the linear overlap values C_k for k=0..N-1.
    C_k = (2/N) * (N/2 - sum_{i=k}^{N-1} h_i h_{i-k})
    Vectorized using FFT for speed.
    """
    n = len(h)
    n_fft = 2 * n
    h_pad = np.zeros(n_fft)
    h_pad[:n] = h
    H = np.fft.rfft(h_pad, n=n_fft)
    corr = np.fft.irfft(H * np.conj(H), n=n_fft)
    r = corr[:n]
    cs = (2.0 / n) * (n / 2.0 - r)
    return cs


def _score(h):
    """
    Compute the maximum linear overlap.
    """
    h = np.asarray(h, dtype=np.float64)
    cs = _linear_overlap_values(h)
    return float(np.max(cs))


def _project(h):
    """
    Project h onto the set {x : 0 <= x_i <= 1, sum(x) = N/2}.
    Uses iterative clipping and scaling.
    """
    h = np.array(h, dtype=np.float64, copy=True)
    target = N / 2.0
    
    # Iterate to satisfy both constraints
    for _ in range(50):
        # Clip to [0, 1]
        h = np.clip(h, 0.0, 1.0)
        
        # Adjust sum
        current_sum = np.sum(h)
        if abs(current_sum - target) < 1e-12:
            break
            
        # Find indices that can be adjusted (not at bounds 0 or 1)
        # If all are at bounds, we have a problem, but with sum=N/2 and N even, 
        # it's unlikely to be stuck if we started reasonably.
        mask = (h > 1e-6) & (h < 1.0 - 1e-6)
        
        if np.sum(mask) == 0:
            # Fallback: adjust the one with most room
            if current_sum < target:
                idx = np.argmin(h)
                h[idx] = min(1.0, h[idx] + (target - current_sum))
            else:
                idx = np.argmax(h)
                h[idx] = max(0.0, h[idx] + (target - current_sum))
            h = np.clip(h, 0.0, 1.0)
            continue
            
        diff = target - current_sum
        h[mask] += diff / np.sum(mask)
        
    # Final precise correction
    diff = target - np.sum(h)
    if abs(diff) > 1e-14:
        # Adjust the element furthest from bounds
        if diff > 0:
            idx = np.argmin(h)
        else:
            idx = np.argmax(h)
        h[idx] += diff
        h = np.clip(h, 0.0, 1.0)
        
    return h


def construct():
    """Construct step heights minimizing maximum overlap with shifted complement."""
    t0 = time.time()
    rng = np.random.default_rng(42)
    
    population_size = 60
    elite_count = 5
    tournament_size = 5
    mutation_rate = 0.8
    mutation_scale_start = 0.15
    mutation_scale_end = 0.01
    
    # Initialize population
    population = []
    
    # Add some structured starts
    i = np.arange(N)
    
    # Uniform
    population.append(np.full(N, 0.5))
    
    # Cosine waves
    for freq in [1, 2, 3]:
        h_cos = 0.5 + 0.5 * np.cos(2.0 * np.pi * freq * i / N)
        population.append(_project(h_cos))
        
    # Random block structures
    for _ in range(10):
        # Create random blocks of high/low values
        h = np.zeros(N)
        num_blocks = rng.integers(2, 8)
        block_size = N // num_blocks
        for b in range(num_blocks):
            start_idx = b * block_size
            end_idx = (b + 1) * block_size if b < num_blocks - 1 else N
            val = rng.random()
            h[start_idx:end_idx] = val
        population.append(_project(h))
        
    # Fill rest with random
    while len(population) < population_size:
        h = rng.uniform(0, 1, N)
        population.append(_project(h))
        
    # Evaluate initial population
    scores = np.array([_score(h) for h in population])
    
    best_h = population[np.argmin(scores)]
    best_score = np.min(scores)
    
    generation = 0
    max_generations = 500 # Upper bound, will stop by time
    
    while time.time() < _DEADLINE - 2:
        generation += 1
        
        # Anneal mutation scale
        progress = min(1.0, generation / max_generations)
        mutation_scale = mutation_scale_start * (1.0 - progress) + mutation_scale_end * progress
        
        new_population = []
        
        # Elitism: copy best individuals
        elite_indices = np.argsort(scores)[:elite_count]
        for idx in elite_indices:
            new_population.append(population[idx].copy())
            
        # Generate offspring
        while len(new_population) < population_size:
            # Tournament selection
            p1_idx = _tournament_select(population, scores, rng, tournament_size)
            p2_idx = _tournament_select(population, scores, rng, tournament_size)
            
            p1 = population[p1_idx]
            p2 = population[p2_idx]
            
            # Uniform crossover
            mask = rng.random(N) < 0.5
            child = np.where(mask, p1, p2)
            
            # Mutation
            if rng.random() < mutation_rate:
                noise = rng.normal(0, mutation_scale, N)
                child += noise
            
            # Project to feasible set
            child = _project(child)
            new_population.append(child)
            
        # Update population and scores
        population = new_population
        scores = np.array([_score(h) for h in population])
        
        # Update best
        min_idx = np.argmin(scores)
        if scores[min_idx] < best_score:
            best_score = scores[min_idx]
            best_h = population[min_idx].copy()
            
    return _project(best_h)


def _tournament_select(population, scores, rng, size):
    """
    Select an individual index using tournament selection.
    """
    indices = rng.choice(len(population), size=size, replace=False)
    best_idx_in_tournament = indices[np.argmin(scores[indices])]
    return best_idx_in_tournament
