import time
import random
import numpy as np

def construct():
    N = 256
    start_time = time.time()
    deadline = start_time + 145  # Safety margin before 150s kill
    
    # 1. Precompute all 4-term arithmetic progressions
    # A 4-AP: a, a+d, a+2d, a+3d with a >= 1, a+3d <= N, d >= 1
    # Indices are 0-based: a-1, a+d-1, a+2d-1, a+3d-1
    ap_list = []
    for d in range(1, (N - 3) // 3 + 1):
        for a in range(1, N - 3 * d + 1):
            ap_list.append((a - 1, a + d - 1, a + 2 * d - 1, a + 3 * d - 1))
    
    total_aps = len(ap_list)
    aps = np.array(ap_list, dtype=np.int32)
    
    # 2. Precompute for each position p, the "others" in every AP containing p.
    # others_for_p[p] is a numpy array of shape (K, 3)
    others_for_p = [None] * N
    for p in range(N):
        others_list = []
        # Iterate through all APs to find those containing p
        # Optimization: We can do this more efficiently by building an index, 
        # but for N=256 and ~26k APs, a simple loop is fast enough at startup.
        for i in range(total_aps):
            row = aps[i]
            if p in row:
                # Find index of p in row
                idx = np.where(row == p)[0][0]
                others = [row[j] for j in range(4) if j != idx]
                others_list.append(others)
        
        if others_list:
            others_for_p[p] = np.array(others_list, dtype=np.int32)
        else:
            others_for_p[p] = np.empty((0, 3), dtype=np.int32)

    def count_mono(colors):
        """Count monochromatic 4-APs using numpy vectorization."""
        c = colors[aps]  # shape (total_aps, 4)
        mono = (c.max(axis=1) == c.min(axis=1))
        return int(mono.sum())

    def fast_delta(colors, p):
        """Compute change in mono count if we flip position p."""
        old_color = colors[p]
        others = others_for_p[p]
        n_aps = others.shape[0]
        if n_aps == 0:
            return 0
        
        # Get colors of the other 3 positions in each AP
        other_colors = colors[others] # shape (n_aps, 3)
        
        # Check if the 3 other positions are all the same color
        all_same = (other_colors[:, 0] == other_colors[:, 1]) & (other_colors[:, 1] == other_colors[:, 2])
        
        # Before flip: AP is mono if all 3 others are same AND that color is old_color
        mono_before = all_same & (other_colors[:, 0] == old_color)
        
        # After flip: AP is mono if all 3 others are same AND that color is new_color
        new_color = 1 - old_color
        mono_after = all_same & (other_colors[:, 0] == new_color)
        
        return int(mono_after.sum()) - int(mono_before.sum())

    def local_search(colors, max_passes=10):
        """Perform hill-climbing local search. Returns (colors, count)."""
        count = count_mono(colors)
        for _ in range(max_passes):
            if time.time() > deadline - 2:
                break
            improved = False
            # Shuffle order to avoid bias
            order = np.random.permutation(N)
            for p in order:
                if time.time() > deadline - 2:
                    break
                delta = fast_delta(colors, p)
                if delta < 0:
                    colors[p] ^= 1
                    count += delta
                    improved = True
            if not improved:
                break
        return colors, count

    def simulated_annealing(colors, T_start=2.0, T_end=0.05, steps=2000):
        """Perform Simulated Annealing to escape local minima."""
        current_count = count_mono(colors)
        T = T_start
        cooling = (T_end / T_start) ** (1.0 / steps)
        
        for _ in range(steps):
            if time.time() > deadline - 5:
                break
            p = np.random.randint(0, N)
            delta = fast_delta(colors, p)
            
            if delta < 0 or random.random() < np.exp(-delta / T):
                colors[p] ^= 1
                current_count += delta
            
            T = max(T_end, T * cooling)
        
        return colors, current_count

    best_colors = None
    best_count = float('inf')
    
    def update_best(colors, count):
        nonlocal best_count, best_colors
        if count < best_count:
            best_count = count
            best_colors = colors.copy()

    # Initialize population with diverse starting points
    pop_size = 40
    population = []
    
    # Diverse initial individuals
    init_patterns = [
        np.array([(i // 2) % 2 for i in range(N)], dtype=np.int8),      # Alternating pairs
        np.array([(i // 3) % 2 for i in range(N)], dtype=np.int8),      # Blocks of 3
        np.array([(i // 4) % 2 for i in range(N)], dtype=np.int8),      # Blocks of 4
        np.array([(i // 5) % 2 for i in range(N)], dtype=np.int8),      # Blocks of 5
        np.array([(i // 7) % 2 for i in range(N)], dtype=np.int8),      # Blocks of 7
        np.random.randint(0, 2, size=N).astype(np.int8),                # Random 1
        np.random.randint(0, 2, size=N).astype(np.int8),                # Random 2
        np.random.randint(0, 2, size=N).astype(np.int8),                # Random 3
        np.random.randint(0, 2, size=N).astype(np.int8),                # Random 4
        np.random.randint(0, 2, size=N).astype(np.int8),                # Random 5
    ]
    
    for i in range(pop_size):
        if i < len(init_patterns):
            colors = init_patterns[i].copy()
        else:
            colors = np.random.randint(0, 2, size=N).astype(np.int8)
        
        # Apply local search to each initial individual
        colors, cnt = local_search(colors, max_passes=5)
        population.append((colors, cnt))
        update_best(colors, cnt)
    
    population.sort(key=lambda x: x[1])
    
    # Main Loop: Genetic Algorithm with SA refinements
    elite_size = 5
    mutation_rate = 0.08
    
    while time.time() < deadline - 10:
        next_population = []
        
        # Elitism
        for i in range(min(elite_size, len(population))):
            next_population.append(population[i])
        
        # Fill rest with offspring
        while len(next_population) < pop_size:
            if time.time() > deadline - 10:
                break
            
            # Tournament Selection
            def tournament_select(k=3):
                candidates = random.sample(range(len(population)), k)
                best_idx = min(candidates, key=lambda i: population[i][1])
                return population[best_idx][0]
            
            parent1 = tournament_select(3)
            parent2 = tournament_select(3)
            
            # Uniform Crossover
            mask = np.random.randint(0, 2, size=N).astype(np.int8)
            child = parent1 * mask + parent2 * (1 - mask)
            
            # Mutation
            flip_mask = (np.random.rand(N) < mutation_rate).astype(np.int8)
            child[flip_mask] ^= 1
            
            # Refine with Local Search
            child, child_count = local_search(child, max_passes=3)
            
            next_population.append((child, child_count))
            update_best(child, child_count)
        
        population = next_population
        population.sort(key=lambda x: x[1])
        
        # Periodic SA on the best individual to escape local minima
        if random.random() < 0.2: # 20% chance each generation
            if time.time() > deadline - 15:
                break
            best_ind = population[0][0].copy()
            best_cnt = population[0][1]
            
            # Run SA
            sa_colors, sa_count = simulated_annealing(best_ind, T_start=1.5, T_end=0.1, steps=1000)
            # Refine SA result
            sa_colors, sa_count = local_search(sa_colors, max_passes=3)
            
            update_best(sa_colors, sa_count)
            
            # Insert if it's better than the worst in population
            if sa_count < population[-1][1]:
                population[-1] = (sa_colors, sa_count)
                population.sort(key=lambda x: x[1])
        
        # Periodic diversity injection
        if len(population) > 10 and random.random() < 0.1:
            new_ind = np.random.randint(0, 2, size=N).astype(np.int8)
            new_ind, new_cnt = local_search(new_ind, max_passes=3)
            population[-1] = (new_ind, new_cnt)
            population.sort(key=lambda x: x[1])

    # Final Polish
    if best_colors is not None:
        final_colors, final_count = local_search(best_colors.copy(), max_passes=10)
        update_best(final_colors, final_count)
    
    if best_colors is not None:
        return [int(c) for c in best_colors]
    else:
        return [(i // 2) % 2 for i in range(N)]
