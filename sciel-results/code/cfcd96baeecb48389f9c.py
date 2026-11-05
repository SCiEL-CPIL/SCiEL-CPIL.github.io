import time
import random
import numpy as np

def construct():
    # Time management
    start_time = time.time()
    deadline = start_time + 110  # Safety margin for 120s limit

    N = 256
    
    # 1. Precompute all 4-term APs
    # A 4-AP is (a, a+d, a+2d, a+3d) with 1 <= a and a+3d <= N, d >= 1
    # 0-indexed: (i0, i1, i2, i3)
    aps = []
    for d in range(1, (N // 3) + 1):
        max_start = N - 3 * d
        for a in range(1, max_start + 1):
            i0 = a - 1
            i1 = i0 + d
            i2 = i0 + 2 * d
            i3 = i0 + 3 * d
            aps.append((i0, i1, i2, i3))
    
    aps = np.array(aps, dtype=np.int32)
    n_aps = len(aps)
    
    # 2. Precompute for each position, the APs it belongs to and the other indices
    # pos_aps[i] = list of (ap_idx, other_indices_tuple)
    pos_aps = [[] for _ in range(N)]
    for ap_idx in range(n_aps):
        i0, i1, i2, i3 = aps[ap_idx]
        pos_aps[i0].append((ap_idx, (i1, i2, i3)))
        pos_aps[i1].append((ap_idx, (i0, i2, i3)))
        pos_aps[i2].append((ap_idx, (i0, i1, i3)))
        pos_aps[i3].append((ap_idx, (i0, i1, i2)))
        
    # Convert to lists of tuples for faster iteration
    # pos_aps[i] is a list of (ap_idx, (j, k, l))
    
    def count_mono(colors):
        """
        Vectorized count of monochromatic 4-APs.
        colors: np.array of 0s and 1s
        """
        c0 = colors[aps[:, 0]]
        c1 = colors[aps[:, 1]]
        c2 = colors[aps[:, 2]]
        c3 = colors[aps[:, 3]]
        mono = (c0 == c1) & (c1 == c2) & (c2 == c3)
        return int(mono.sum())

    def calc_delta(colors, pos, new_color):
        """
        Calculate delta in mono count if we set colors[pos] to new_color.
        Assumes colors[pos] != new_color.
        """
        old_color = colors[pos]
        delta = 0
        for ap_idx, others in pos_aps[pos]:
            j, k, l = others
            cj = colors[j]
            ck = colors[k]
            cl = colors[l]
            
            # Current state: (old_color, cj, ck, cl)
            was_mono = (old_color == cj) and (cj == ck) and (ck == cl)
            
            # New state: (new_color, cj, ck, cl)
            is_mono = (new_color == cj) and (cj == ck) and (ck == cl)
            
            if was_mono and not is_mono:
                delta -= 1
            elif not was_mono and is_mono:
                delta += 1
        return delta

    def local_search(colors, time_limit=0.05):
        """
        Improve a coloring using hill climbing.
        Returns improved colors and count.
        """
        current_count = count_mono(colors)
        best_colors = colors.copy()
        best_count = current_count
        t_start = time.time()
        
        # Random order of positions to break symmetry
        order = list(range(N))
        
        improved = True
        while improved and (time.time() - t_start) < time_limit:
            improved = False
            random.shuffle(order)
            for pos in order:
                new_color = 1 - colors[pos]
                delta = calc_delta(colors, pos, new_color)
                if delta < 0:
                    colors[pos] = new_color
                    current_count += delta
                    improved = True
                    if current_count < best_count:
                        best_count = current_count
                        best_colors = colors.copy()
                elif delta == 0:
                    # Optional: accept neutral moves to escape plateaus?
                    # For now, stick to strict improvement for stability
                    pass
                
        return best_colors, best_count

    # 3. Genetic Algorithm Setup
    pop_size = 50
    mutation_rate = 0.05
    tournament_size = 5
    
    # Initialize population
    population = []
    for _ in range(pop_size):
        c = np.array([random.randint(0, 1) for _ in range(N)], dtype=np.int8)
        c, count = local_search(c, time_limit=0.02)
        population.append((c, count))
        
    # Sort initial population
    population.sort(key=lambda x: x[1])
    best_colors, best_count = population[0]
    
    # GA Loop
    generation = 0
    while time.time() < deadline:
        gen_start = time.time()
        new_population = []
        
        # Elitism: keep the top few
        elite_count = 5
        for i in range(elite_count):
            new_population.append(population[i])
            
        # Generate offspring
        while len(new_population) < pop_size:
            # Tournament Selection
            candidates = random.sample(population, tournament_size)
            parent1 = min(candidates, key=lambda x: x[1])[0]
            
            candidates = random.sample(population, tournament_size)
            parent2 = min(candidates, key=lambda x: x[1])[0]
            
            # Crossover (Uniform)
            child = np.empty(N, dtype=np.int8)
            for i in range(N):
                if random.random() < 0.5:
                    child[i] = parent1[i]
                else:
                    child[i] = parent2[i]
            
            # Mutation
            for i in range(N):
                if random.random() < mutation_rate:
                    child[i] = 1 - child[i]
            
            # Local Search on Child
            child, count = local_search(child, time_limit=0.01)
            new_population.append((child, count))
            
        # Update population
        population = new_population
        population.sort(key=lambda x: x[1])
        
        if population[0][1] < best_count:
            best_count = population[0][1]
            best_colors = population[0][0].copy()
            
        generation += 1
        
        # If time is running out, do a final polish on the best solution
        if time.time() > deadline - 10:
            break

    # Final polish on the best solution found
    best_colors, best_count = local_search(best_colors, time_limit=deadline - time.time() - 1)

    return [int(c) for c in best_colors]
