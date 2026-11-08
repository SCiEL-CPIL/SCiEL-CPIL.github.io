import numpy as np
import time

N = 400

# EVOLVE-BLOCK-START
def _score(h):
    """
    Compute the upper bound on C3.
    Metric: max(|autocorr_sidelobe|) / N
    """
    h = np.asarray(h, dtype=np.float64)
    n = len(h)
    size = 2 * n
    H = np.fft.rfft(h, n=size)
    C = np.fft.irfft(H * H, n=size)[:n]
    if n > 1:
        max_side = np.max(np.abs(C[1:]))
    else:
        max_side = 0.0
    return max_side / n

def _get_cost(h):
    size = 2 * N
    H = np.fft.rfft(h, n=size)
    C = np.fft.irfft(H * H, n=size)[:N]
    return np.max(np.abs(C[1:]))

def _local_search(h, max_iters=500):
    """
    Greedy local search to polish a sequence.
    """
    current_cost = _get_cost(h)
    for _ in range(max_iters):
        improved = False
        # Try flipping a few random bits
        for _ in range(20):
            idx = np.random.randint(N)
            old_val = h[idx]
            h[idx] = -old_val
            new_cost = _get_cost(h)
            if new_cost < current_cost:
                current_cost = new_cost
                improved = True
            else:
                h[idx] = old_val
        if not improved:
            break
    return h, current_cost

def construct():
    deadline = time.time() + 110.0
    np.random.seed(42)
    
    pop_size = 30
    # Initialize population
    population = np.random.choice([-1.0, 1.0], size=(pop_size, N))
    costs = np.array([_get_cost(ind) for ind in population])
    
    best_h = population[np.argmin(costs)].copy()
    best_cost = np.min(costs)
    
    mutation_rate = 0.05
    tournament_size = 3
    
    while time.time() < deadline - 5.0:
        new_population = []
        
        # Keep the best individual (elitism)
        best_idx = np.argmin(costs)
        # Polish the best individual
        polished_best, polished_cost = _local_search(population[best_idx].copy())
        if polished_cost < best_cost:
            best_cost = polished_cost
            best_h = polished_best
        new_population.append(polished_best)
        
        for _ in range(pop_size - 1):
            # Tournament Selection
            parents = []
            for _ in range(2):
                candidates = np.random.choice(pop_size, tournament_size, replace=False)
                parent_idx = candidates[np.argmin(costs[candidates])]
                parents.append(population[parent_idx])
            
            # Crossover (Uniform)
            mask = np.random.rand(N) < 0.5
            child = np.where(mask, parents[0], parents[1])
            
            # Mutation
            mut_mask = np.random.rand(N) < mutation_rate
            child[mut_mask] *= -1.0
            
            # Light local search on child? Maybe too expensive.
            # Just add to population.
            new_population.append(child)
        
        population = np.array(new_population)
        costs = np.array([_get_cost(ind) for ind in population])
        
        current_best_idx = np.argmin(costs)
        if costs[current_best_idx] < best_cost:
            best_cost = costs[current_best_idx]
            best_h = population[current_best_idx].copy()
            
    # Final polish on the best found
    final_h, final_cost = _local_search(best_h, max_iters=1000)
    if final_cost < best_cost:
        best_h = final_h
        best_cost = final_cost
        
    return best_h
# EVOLVE-BLOCK-END
