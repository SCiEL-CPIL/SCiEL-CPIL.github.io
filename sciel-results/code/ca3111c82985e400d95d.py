"""Rewritten program for minimizing monochromatic 4-term APs in [1,256]."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import random

N = 256

def _count_mono4(c):
    """Count monochromatic 4-term APs in a 0/1 numpy array of length N."""
    count = 0
    for d in range(1, N // 3 + 1):
        max_a = N - 3 * d
        if max_a <= 0:
            break
        c0 = c[:max_a]
        c1 = c[d:d + max_a]
        c2 = c[2 * d:2 * d + max_a]
        c3 = c[3 * d:3 * d + max_a]
        mask = (c0 == c1) & (c0 == c2) & (c0 == c3)
        count += np.sum(mask)
    return int(count)

def _thue_morse(n):
    """Generate Thue-Morse sequence."""
    c = np.zeros(n, dtype=np.int8)
    for i in range(n):
        c[i] = bin(i).count('1') % 2
    return c

def _precompute_ap_membership():
    """Precompute for each position, the list of (APs containing it) as indices into a flat AP list.
    Each AP is (a, d) with a in 0..N-3d, d in 1..N//3.
    Returns:
        ap_list: list of tuples (a, d) for all 4-term APs
        pos_to_aps: list of lists, pos_to_aps[p] = list of AP indices that contain position p
    """
    ap_list = []
    for d in range(1, N // 3 + 1):
        for a in range(N - 3 * d):
            ap_list.append((a, d))
    
    num_aps = len(ap_list)
    pos_to_aps = [[] for _ in range(N)]
    
    for idx, (a, d) in enumerate(ap_list):
        for k in range(4):
            pos = a + k * d
            pos_to_aps[pos].append(idx)
    
    return ap_list, pos_to_aps

# Precompute once at module level
_AP_LIST, _POS_TO_APS = _precompute_ap_membership()

def _delta_flip(c, p):
    """Compute change in mono-4AP count when flipping position p.
    Uses precomputed AP membership for efficiency."""
    old_val = c[p]
    new_val = 1 - old_val
    delta = 0
    
    for ap_idx in _POS_TO_APS[p]:
        a, d = _AP_LIST[ap_idx]
        # Get the 4 positions of this AP
        positions = [a, a + d, a + 2 * d, a + 3 * d]
        
        # Check if AP was monochromatic before flip
        was_mono = all(c[pp] == old_val for pp in positions)
        # Check if AP will be monochromatic after flip
        now_mono = all(c[pp] == new_val for pp in positions)
        
        if was_mono and not now_mono:
            delta -= 1
        elif now_mono and not was_mono:
            delta += 1
    
    return delta

def _local_search(c, time_limit, random_seed=42):
    """Local search with incremental delta computation."""
    rng = random.Random(random_seed)
    start = time.time()
    best_c = c.copy()
    best_count = _count_mono4(best_c)
    
    improved = True
    while improved and (time.time() - start) < time_limit:
        improved = False
        order = list(range(N))
        rng.shuffle(order)
        for pos in order:
            if (time.time() - start) >= time_limit:
                break
            delta = _delta_flip(best_c, pos)
            if delta < 0:
                best_c[pos] = 1 - best_c[pos]
                best_count += delta
                improved = True
    
    return best_c, best_count

def _simulated_annealing(c, time_limit, random_seed=42):
    """Simulated annealing with incremental delta computation."""
    rng = random.Random(random_seed)
    start = time.time()
    cur_c = c.copy()
    cur_count = _count_mono4(cur_c)
    best_c = cur_c.copy()
    best_count = cur_count
    
    T_start = 3.0
    T_end = 0.05
    
    # Estimate steps per second
    # Each step is roughly O(num_aps_containing_p) which is small
    # Let's just run until time limit
    step = 0
    while (time.time() - start) < time_limit:
        step += 1
        progress = min(step / 100000.0, 1.0)
        T = T_start * (T_end / T_start) ** progress
        
        pos = rng.randint(0, N - 1)
        delta = _delta_flip(cur_c, pos)
        
        if delta <= 0:
            cur_c[pos] = 1 - cur_c[pos]
            cur_count += delta
            if cur_count < best_count:
                best_count = cur_count
                best_c = cur_c.copy()
        else:
            # Accept with probability exp(-delta/T)
            if T > 0 and rng.random() < np.exp(-delta / T):
                cur_c[pos] = 1 - cur_c[pos]
                cur_count += delta
                if cur_count < best_count:
                    best_count = cur_count
                    best_c = cur_c.copy()
    
    return best_c, best_count

def _genetic_algorithm(starts, time_limit, random_seed=42):
    """Genetic algorithm with local search refinement."""
    rng = random.Random(random_seed)
    start = time.time()
    
    # Initialize population
    pop_size = 20
    population = [s.copy() for s in starts[:pop_size]]
    while len(population) < pop_size:
        population.append(np.random.RandomState(rng.randint(0, 2**31)).randint(0, 2, N).astype(np.int8))
    
    # Evaluate initial population
    scores = [_count_mono4(ind) for ind in population]
    best_idx = np.argmin(scores)
    best_c = population[best_idx].copy()
    best_count = scores[best_idx]
    
    generation = 0
    while (time.time() - start) < time_limit:
        generation += 1
        
        # Select parents (tournament selection)
        new_pop = []
        
        # Keep best individual
        new_pop.append(population[best_idx].copy())
        
        while len(new_pop) < pop_size:
            # Tournament selection
            tournament = rng.sample(range(pop_size), 3)
            winner = min(tournament, key=lambda i: scores[i])
            
            if rng.random() < 0.5:
                # Crossover: uniform
                other = rng.randint(0, pop_size - 1)
                if other == winner:
                    continue
                mask = rng.randint(0, 2, N).astype(np.int8)
                child = np.where(mask == 1, population[winner], population[other]).astype(np.int8)
            else:
                child = population[winner].copy()
            
            # Mutation
            if rng.random() < 0.3:
                num_mut = rng.randint(1, 5)
                for _ in range(num_mut):
                    child[rng.randint(0, N - 1)] = 1 - child[rng.randint(0, N - 1)]
            
            # Local search refinement (short)
            remaining = time_limit - (time.time() - start)
            if remaining > 1:
                child, child_count = _local_search(child, min(0.5, remaining * 0.1), rng.randint(0, 2**31))
            else:
                child_count = _count_mono4(child)
            
            new_pop.append(child)
            new_pop_scores_placeholder = child_count
        
        # Re-evaluate entire population
        population = new_pop
        scores = [_count_mono4(ind) for ind in population]
        best_idx = np.argmin(scores)
        if scores[best_idx] < best_count:
            best_count = scores[best_idx]
            best_c = population[best_idx].copy()
    
    return best_c, best_count

def construct():
    deadline = time.time() + 145  # Safety margin before 150s kill
    rng_global = random.Random(99999)
    
    # Generate diverse starting points
    starts = []
    
    # Thue-Morse and variants
    tm = _thue_morse(N)
    starts.append(tm.copy())
    for offset in range(1, 32):
        starts.append(np.roll(tm, offset).copy())
    
    # Random with bias toward 50/50
    for _ in range(20):
        r = np.random.RandomState(rng_global.randint(0, 2**31))
        starts.append(r.randint(0, 2, N).astype(np.int8))
    
    # Alternating patterns
    for period in [2, 3, 4, 5, 6, 7, 8, 10, 12, 16]:
        pat = np.zeros(N, dtype=np.int8)
        for i in range(N):
            pat[i] = (i // period) % 2
        starts.append(pat.copy())
    
    # Striped patterns with different phases
    for period in [4, 8, 16, 32]:
        for phase in range(period):
            pat = np.zeros(N, dtype=np.int8)
            for i in range(N):
                pat[i] = ((i + phase) // period) % 2
            starts.append(pat.copy())
    
    best_c = None
    best_count = float('inf')
    
    # Evaluate all starts
    for s in starts:
        cnt = _count_mono4(s)
        if cnt < best_count:
            best_count = cnt
            best_c = s.copy()
    
    # Phase 1: Local search on best start
    remaining = deadline - time.time()
    if remaining > 10:
        best_c, best_count = _local_search(best_c, min(remaining - 10, 30), 42)
    
    # Phase 2: Simulated annealing
    remaining = deadline - time.time()
    if remaining > 15:
        sa_c, sa_count = _simulated_annealing(best_c, min(remaining - 15, 40), 777)
        if sa_count < best_count:
            best_count = sa_count
            best_c = sa_c
    
    # Phase 3: Local search again after SA
    remaining = deadline - time.time()
    if remaining > 10:
        best_c, best_count = _local_search(best_c, min(remaining - 10, 25), 999)
    
    # Phase 4: Genetic algorithm with diverse starts
    remaining = deadline - time.time()
    if remaining > 30:
        ga_c, ga_count = _genetic_algorithm(starts, min(remaining - 30, 50), 555)
        if ga_count < best_count:
            best_count = ga_count
            best_c = ga_c
    
    # Phase 5: Final local search
    remaining = deadline - time.time()
    if remaining > 5:
        best_c, best_count = _local_search(best_c, min(remaining - 5, 20), 111)
    
    # Phase 6: One more SA pass
    remaining = deadline - time.time()
    if remaining > 10:
        sa_c, sa_count = _simulated_annealing(best_c, min(remaining - 10, 20), 222)
        if sa_count < best_count:
            best_count = sa_count
            best_c = sa_c
    
    # Final local search
    remaining = deadline - time.time()
    if remaining > 3:
        best_c, best_count = _local_search(best_c, min(remaining - 3, 10), 333)
    
    # Convert to list of ints
    result = [int(x) for x in best_c]
    assert len(result) == 256
    assert all(x in (0, 1) for x in result)
    return result
# EVOLVE-BLOCK-END
