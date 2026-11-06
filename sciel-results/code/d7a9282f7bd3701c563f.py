"""
Genetic Algorithm with vectorized local search to minimize monochromatic 4-term APs in [1, 256].
"""
import numpy as np
import time

N = 256

# Precompute all 4-term arithmetic progressions (a, a+d, a+2d, a+3d) with 1 <= a, a+3d <= 256, d >= 1
# Represent each AP as 0-based indices
def _precompute_aps():
    aps = []
    for d in range(1, N):
        for a in range(1, N - 3 * d + 1):
            aps.append((a - 1, a + d - 1, a + 2 * d - 1, a + 3 * d - 1))
    return np.array(aps, dtype=np.int32)

AP_IDX = _precompute_aps()
NUM_APS = AP_IDX.shape[0]

# Precompute for each position i, which APs contain i, and the other 3 indices per AP.
# Store as arrays for fast access.
AP_CONTAINING_POS = [None] * N
OTHER_3_FOR_POS = [None] * N

for i in range(N):
    mask = (AP_IDX == i).any(axis=1)
    ap_indices = np.where(mask)[0]
    AP_CONTAINING_POS[i] = ap_indices
    # For each AP containing i, get the other 3 positions
    if len(ap_indices) > 0:
        sub = AP_IDX[ap_indices]
        # For each row, replace the column where value == i with -1, then remove -1s
        # Simpler: build list of tuples
        others = []
        for k in ap_indices:
            row = AP_IDX[k]
            o = [row[j] for j in range(4) if row[j] != i]
            others.append(o)
        OTHER_3_FOR_POS[i] = np.array(others, dtype=np.int32)
    else:
        OTHER_3_FOR_POS[i] = np.empty((0, 3), dtype=np.int32)

def _count_mono_aps(colours):
    """Count monochromatic 4-term APs."""
    cols = colours[AP_IDX]
    mono = (cols[:, 0] == cols[:, 1]) & (cols[:, 1] == cols[:, 2]) & (cols[:, 2] == cols[:, 3])
    return int(np.sum(mono))

def _local_search(colours, max_sweeps=100, rng=None):
    """
    Fast vectorized local search. Returns (improved_colours, score).
    """
    if rng is None:
        rng = np.random.RandomState(0)
    
    colours = colours.astype(np.int8).copy()
    cols = colours[AP_IDX]
    mono_status = (cols[:, 0] == cols[:, 1]) & (cols[:, 1] == cols[:, 2]) & (cols[:, 2] == cols[:, 3])
    current_count = int(np.sum(mono_status))
    
    for sweep in range(max_sweeps):
        improved = False
        order = np.arange(N)
        rng.shuffle(order)
        
        for pos in order:
            old_val = colours[pos]
            new_val = 1 - old_val
            
            affected_aps = AP_CONTAINING_POS[pos]
            others = OTHER_3_FOR_POS[pos]
            
            if len(affected_aps) == 0:
                continue
            
            # Get current values of the other 3 positions for each affected AP
            # others has shape (num_affected, 3)
            other_vals = colours[others]  # shape (num_affected, 3)
            
            # Check if all 3 others match new_val
            all_match_new = (other_vals == new_val).all(axis=1)
            # Check if all 3 others match old_val
            all_match_old = (other_vals == old_val).all(axis=1)
            
            mono_aff = mono_status[affected_aps]
            
            # Delta: +1 if flipping creates a new mono AP, -1 if it breaks one
            # If all others == new_val:
            #   If AP was NOT mono, flipping makes it mono => delta += 1
            #   If AP WAS mono, it stays mono (but now with new_val) => no change... wait
            # Actually need to think more carefully.
            # Before flip: AP has colours [old_val, v1, v2, v3] in some order
            # After flip: AP has colours [new_val, v1, v2, v3]
            # mono_before = (old_val == v1 == v2 == v3)
            # mono_after = (new_val == v1 == v2 == v3)
            # delta = mono_after - mono_before
            
            # mono_before for affected APs is mono_aff
            # mono_after: all_match_new
            
            delta = int(np.sum(all_match_new) - np.sum(mono_aff))
            
            if delta < 0:
                colours[pos] = new_val
                current_count += delta
                # Update mono_status for affected APs
                # Recompute for affected APs only
                new_cols = colours[AP_IDX[affected_aps]]
                mono_status[affected_aps] = (new_cols[:, 0] == new_cols[:, 1]) & \
                                             (new_cols[:, 1] == new_cols[:, 2]) & \
                                             (new_cols[:, 2] == new_cols[:, 3])
                improved = True
        
        if not improved:
            break
    
    return colours, current_count

def _generate_diverse_initial(pop_size, rng):
    """Generate a diverse initial population."""
    population = []
    
    # Random
    for _ in range(pop_size // 4):
        population.append(rng.randint(0, 2, size=N).astype(np.int8))
    
    # Thue-Morse
    tm = [0]
    while len(tm) < N:
        tm = tm + [1 - x for x in tm]
    population.append(np.array(tm[:N], dtype=np.int8))
    
    # Block patterns
    for block_size in [2, 3, 4, 5, 6, 7, 8, 10, 12, 16, 20, 32]:
        c = np.array([(i // block_size) % 2 for i in range(N)], dtype=np.int8)
        population.append(c)
    
    # Periodic patterns
    for period in [3, 4, 5, 6, 7, 8, 9, 10, 12, 15, 16, 20, 24, 32]:
        half = period // 2
        c = np.array([(i % period) < half for i in range(N)], dtype=np.int8)
        population.append(c)
    
    # XOR patterns
    for shift in range(1, 8):
        c = np.array([(i & 1) ^ ((i >> shift) & 1) for i in range(N)], dtype=np.int8)
        population.append(c)
    
    # More random with bias
    for bias in [0.3, 0.4, 0.6, 0.7]:
        c = (rng.rand(N) < bias).astype(np.int8)
        population.append(c)
    
    # Fill remaining with random
    while len(population) < pop_size:
        population.append(rng.randint(0, 2, size=N).astype(np.int8))
    
    return population[:pop_size]

def construct():
    """
    Genetic algorithm to minimize monochromatic 4-term APs in [1, 256].
    Returns 256 colours in {0, 1}.
    """
    start_time = time.time()
    time_limit = 115
    
    rng = np.random.RandomState(42)
    
    pop_size = 50
    population = _generate_diverse_initial(pop_size, rng)
    
    # Initial local search on all
    scores = []
    for i, c in enumerate(population):
        c, s = _local_search(c, max_sweeps=30, rng=rng)
        population[i] = c
        scores.append(s)
    
    scores = np.array(scores)
    best_idx = np.argmin(scores)
    best_score = scores[best_idx]
    best_individual = population[best_idx].copy()
    
    generation = 0
    stagnant = 0
    
    while time.time() - start_time < time_limit:
        generation += 1
        
        # Sort by score
        order = np.argsort(scores)
        population = [population[i] for i in order]
        scores = scores[order]
        
        elite_size = 8
        new_population = [population[i].copy() for i in range(elite_size)]
        new_scores = list(scores[:elite_size])
        
        while len(new_population) < pop_size:
            # Tournament selection
            cands = rng.randint(0, len(population), size=4)
            best_c = cands[np.argmin(scores[cands])]
            p1 = population[best_c]
            
            cands2 = rng.randint(0, len(population), size=4)
            best_c2 = cands2[np.argmin(scores[cands2])]
            p2 = population[best_c2]
            
            # Uniform crossover
            mask = rng.randint(0, 2, size=N)
            child = np.where(mask == 1, p1, p2).copy()
            
            # Adaptive mutation rate
            mut_rate = 0.05 + 0.05 * (generation % 10) / 10.0
            num_mut = max(1, int(N * mut_rate * rng.uniform(0.5, 1.5)))
            mut_positions = rng.randint(0, N, size=num_mut)
            child[mut_positions] = 1 - child[mut_positions]
            
            child, child_score = _local_search(child, max_sweeps=40, rng=rng)
            
            new_population.append(child)
            new_scores.append(child_score)
        
        population = new_population
        scores = np.array(new_scores)
        
        new_best_idx = np.argmin(scores)
        if scores[new_best_idx] < best_score:
            best_score = scores[new_best_idx]
            best_individual = population[new_best_idx].copy()
            stagnant = 0
        else:
            stagnant += 1
        
        # Periodic random restarts to escape local minima
        if stagnant > 5 and time.time() - start_time < time_limit - 15:
            for _ in range(5):
                c = rng.randint(0, 2, size=N).astype(np.int8)
                c, s = _local_search(c, max_sweeps=25, rng=rng)
                worst_idx = np.argmax(scores)
                population[worst_idx] = c
                scores[worst_idx] = s
                if s < best_score:
                    best_score = s
                    best_individual = c.copy()
                    stagnant = 0
            
            order = np.argsort(scores)
            population = [population[i] for i in order]
            scores = scores[order]
    
    return [int(x) for x in best_individual]

if __name__ == "__main__":
    result = construct()
    assert len(result) == 256
    assert all(x in (0, 1) for x in result)
    count = _count_mono_aps(np.array(result, dtype=np.int8))
    print(f"Monochromatic 4-APs: {count}")
