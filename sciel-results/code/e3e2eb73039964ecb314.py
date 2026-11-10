import random
import time
import numpy as np

N = 256

def _build_aps():
    """Build list of all 4-term arithmetic progressions in 1..256 (0-indexed)."""
    aps = []
    for d in range(1, N // 3 + 1):
        for a in range(0, N - 3 * d):
            aps.append((a, a + d, a + 2 * d, a + 3 * d))
    return aps

def _count_mono(aps, colors):
    """Count monochromatic 4-APs."""
    count = 0
    for a, b, c, d in aps:
        if colors[a] == colors[b] == colors[c] == colors[d]:
            count += 1
    return count

def _local_search(aps, colors, max_iter=5000):
    """Greedy local search: flip bits that reduce mono-AP count."""
    colors = list(colors)
    n = len(colors)
    improved = True
    iters = 0
    while improved and iters < max_iter:
        improved = False
        iters += 1
        for i in range(n):
            # Try flipping bit i
            old = colors[i]
            colors[i] = 1 - old
            new_count = _count_mono(aps, colors)
            if new_count < _count_mono(aps, colors):
                pass
            # Restore
            colors[i] = old
    return colors

def construct():
    start_time = time.time()
    deadline = start_time + 140  # 10s safety margin
    
    aps = _build_aps()
    
    # Precompute for each position, which APs it participates in
    # This allows faster incremental scoring
    
    # Position to list of (ap_index, position_in_ap)
    pos_to_aps = [[] for _ in range(N)]
    for idx, (a, b, c, d) in enumerate(aps):
        pos_to_aps[a].append((idx, 0))
        pos_to_aps[b].append((idx, 1))
        pos_to_aps[c].append((idx, 2))
        pos_to_aps[d].append((idx, 3))
    
    def count_mono_fast(colors):
        """Fast count using numpy vectorization."""
        c = np.array(colors)
        count = 0
        for a, b, cc, d in aps:
            if c[a] == c[b] == c[cc] == c[d]:
                count += 1
        return count
    
    def count_mono_numpy(colors):
        """Vectorized count."""
        c = np.array(colors, dtype=np.int8)
        # For each AP, check if all 4 are equal
        # We can do this by checking: c[a]==c[b] and c[b]==c[c] and c[c]==c[d]
        # Or: c[a]==c[b] and c[a]==c[c] and c[a]==c[d]
        # Let's use: for each AP, the 4 values must all be 0 or all be 1
        # Count where all are 0: (c[a]==0) & (c[b]==0) & (c[c]==0) & (c[d]==0)
        # Plus where all are 1: (c[a]==1) & (c[b]==1) & (c[c]==1) & (c[d]==1)
        # This is hard to vectorize over all APs at once easily
        # Let's just loop but it's fast enough for 256
        total = 0
        for a, b, cc, d in aps:
            if c[a] == c[b] and c[b] == c[cc] and c[cc] == c[d]:
                total += 1
        return total
    
    # Use a simpler but efficient approach
    def score(colors):
        """Count monochromatic 4-APs efficiently."""
        c = colors  # list of 0/1
        total = 0
        for a, b, cc, d in aps:
            if c[a] == c[b] == c[cc] == c[d]:
                total += 1
        return total
    
    # Genetic Algorithm
    random.seed(12345)
    pop_size = 40
    mutation_rate = 0.05
    elitism = 2
    
    # Initialize population with diverse starting points
    population = []
    scores = []
    
    # Some structured initial candidates
    # 1. Random
    for _ in range(10):
        pop = [random.randint(0, 1) for _ in range(N)]
        population.append(pop)
    
    # 2. Block patterns
    for block_size in [2, 4, 8, 16]:
        pop = [(i // block_size) % 2 for i in range(N)]
        population.append(pop)
    
    # 3. Thue-Morse-like
    def thue_morse(n):
        # Thue-Morse sequence: t(n) = popcount(n) % 2
        return [bin(i).count('1') % 2 for i in range(n)]
    population.append(thue_morse(N))
    
    # 4. More random with bias
    for _ in range(10):
        pop = [random.randint(0, 1) for _ in range(N)]
        population.append(pop)
    
    # Pad to pop_size if needed
    while len(population) < pop_size:
        pop = [random.randint(0, 1) for _ in range(N)]
        population.append(pop)
    
    # Evaluate initial population
    for pop in population:
        scores.append(score(pop))
    
    # Sort by score
    order = sorted(range(len(population)), key=lambda i: scores[i])
    population = [population[i] for i in order]
    scores = [scores[i] for i in order]
    
    best_score = scores[0]
    best_pop = population[0][:]
    
    # Local search on the best few
    for i in range(min(5, len(population))):
        if time.time() > deadline - 20:
            break
        improved_pop = _local_search(aps, population[i], max_iter=2000)
        s = score(improved_pop)
        if s < best_score:
            best_score = s
            best_pop = improved_pop[:]
    
    # Main GA loop
    gen = 0
    while time.time() < deadline - 30:
        gen += 1
        
        # Selection: tournament selection
        def tournament_select():
            idx1 = random.randint(0, len(population) - 1)
            idx2 = random.randint(0, len(population) - 1)
            if scores[idx1] <= scores[idx2]:
                return population[idx1][:]
            else:
                return population[idx2][:]
        
        new_population = []
        
        # Elitism
        for i in range(elitism):
            new_population.append(population[i][:])
        
        # Generate offspring
        while len(new_population) < pop_size:
            if time.time() > deadline - 10:
                break
            p1 = tournament_select()
            p2 = tournament_select()
            
            # Uniform crossover
            child = [p1[i] if random.random() < 0.5 else p2[i] for i in range(N)]
            
            # Mutation
            for i in range(N):
                if random.random() < mutation_rate:
                    child[i] = 1 - child[i]
            
            new_population.append(child)
        
        # Evaluate new population
        new_scores = []
        for pop in new_population:
            s = score(pop)
            new_scores.append(s)
        
        # Combine and select
        all_pops = new_population
        all_scores = new_scores
        
        order = sorted(range(len(all_pops)), key=lambda i: all_scores[i])
        population = [all_pops[i] for i in order]
        scores = [all_scores[i] for i in order]
        
        if scores[0] < best_score:
            best_score = scores[0]
            best_pop = population[0][:]
        
        # Local search on best few each generation
        for i in range(min(3, len(population))):
            if time.time() > deadline - 15:
                break
            improved_pop = _local_search(aps, population[i], max_iter=1000)
            s = score(improved_pop)
            if s < best_score:
                best_score = s
                best_pop = improved_pop[:]
    
    # Final local search on best
    if time.time() < deadline - 5:
        final_pop = _local_search(aps, best_pop, max_iter=5000)
        s = score(final_pop)
        if s < best_score:
            best_score = s
            best_pop = final_pop[:]
    
    return best_pop
