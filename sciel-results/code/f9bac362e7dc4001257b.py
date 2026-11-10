import time
import random
import numpy as np

# EVOLVE-BLOCK-START
def construct():
    N = 256
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin for 120s limit
    
    # 1. Precompute all 4-term arithmetic progressions
    # Store as indices 0..N-1
    aps = []
    for d in range(1, (N - 3) // 3 + 1):
        for a in range(1, N - 3 * d + 1):
            i1 = a - 1
            i2 = a + d - 1
            i3 = a + 2 * d - 1
            i4 = a + 3 * d - 1
            aps.append((i1, i2, i3, i4))
    
    # Convert to numpy array for vectorized counting
    aps_arr = np.array(aps, dtype=np.int32)
    
    # 2. Precompute neighbors for fast delta calculation
    # neighbors[i] contains indices of other elements in APs with i
    neighbors = [[] for _ in range(N)]
    for i1, i2, i3, i4 in aps:
        neighbors[i1].append((i2, i3, i4))
        neighbors[i2].append((i1, i3, i4))
        neighbors[i3].append((i1, i2, i4))
        neighbors[i4].append((i1, i2, i3))
        
    # Convert to numpy arrays
    neighbors_np = []
    for i in range(N):
        if neighbors[i]:
            neighbors_np.append(np.array(neighbors[i], dtype=np.int32))
        else:
            neighbors_np.append(np.empty((0, 3), dtype=np.int32))

    def count_mono(colors):
        """Count monochromatic 4-APs using vectorized operations."""
        c1 = colors[aps_arr[:, 0]]
        c2 = colors[aps_arr[:, 1]]
        c3 = colors[aps_arr[:, 2]]
        c4 = colors[aps_arr[:, 3]]
        return np.sum((c1 == c2) & (c2 == c3) & (c3 == c4))

    best_score = float('inf')
    best_coloring = None

    # 3. Structured Initialization Strategy
    # Diverse base patterns
    patterns = [
        np.array([(i // 2) % 2 for i in range(N)], dtype=np.int8),
        np.array([(i // 3) % 2 for i in range(N)], dtype=np.int8),
        np.array([(i % 4) < 2 for i in range(N)], dtype=np.int8),
        np.array([1 - ((i // 2) % 2) for i in range(N)], dtype=np.int8),
        np.array([(i // 5) % 2 for i in range(N)], dtype=np.int8),
        np.array([(i % 6) < 3 for i in range(N)], dtype=np.int8),
        np.array([bin(i).count('1') % 2 for i in range(N)], dtype=np.int8),
        np.array([(i // 7) % 2 for i in range(N)], dtype=np.int8),
        np.array([((i // 2) % 2) if (i // 8) % 2 == 0 else ((i // 3) % 2) for i in range(N)], dtype=np.int8),
        np.array([(i // 11) % 2 for i in range(N)], dtype=np.int8),
    ]

    # 4. Local Search (Polish)
    def polish(colors, score, time_limit=5.0):
        """Perform exact local search: only accept improving moves."""
        polish_start = time.time()
        improved = True
        # Randomize the order of checks to avoid bias
        order = list(range(N))
        while improved and time.time() < polish_start + time_limit:
            improved = False
            random.shuffle(order)
            for i in order:
                if time.time() > polish_start + time_limit:
                    break
                old_c = colors[i]
                new_c = 1 - old_c
                
                others = neighbors_np[i]
                if len(others) == 0:
                    continue
                cols = colors[others]
                c0, c1, c2 = cols[:, 0], cols[:, 1], cols[:, 2]
                all_eq = (c0 == c1) & (c1 == c2)
                val = c0
                was_mono = all_eq & (val == old_c)
                becomes_mono = all_eq & (val == new_c)
                delta = np.sum(becomes_mono) - np.sum(was_mono)
                
                if delta < 0:
                    colors[i] = new_c
                    score += delta
                    improved = True
        return score

    def fast_polish(colors, score):
        """Very fast polish, single pass, no time limit check inside loop for speed if called often? 
        No, keep time check but tight."""
        improved = True
        while improved:
            improved = False
            for i in range(N):
                old_c = colors[i]
                new_c = 1 - old_c
                others = neighbors_np[i]
                if len(others) == 0:
                    continue
                cols = colors[others]
                c0, c1, c2 = cols[:, 0], cols[:, 1], cols[:, 2]
                all_eq = (c0 == c1) & (c1 == c2)
                val = c0
                was_mono = all_eq & (val == old_c)
                becomes_mono = all_eq & (val == new_c)
                delta = np.sum(becomes_mono) - np.sum(was_mono)
                if delta < 0:
                    colors[i] = new_c
                    score += delta
                    improved = True
        return score

    # 5. Genetic Algorithm Components
    def crossover(p1, p2):
        """Uniform crossover"""
        mask = (random.random() < 0.5 for _ in range(N))
        # Vectorized crossover
        mask_arr = np.random.random(N) < 0.5
        return np.where(mask_arr, p1, p2)

    def mutate(ind, rate=0.05):
        """Random bit flip mutation"""
        ind = ind.copy()
        num_flips = int(N * rate)
        if num_flips > 0:
            indices = np.random.choice(N, num_flips, replace=False)
            ind[indices] = 1 - ind[indices]
        return ind

    # 6. Main GA Loop
    POP_SIZE = 30
    population = []
    
    # Initialize population with diverse patterns
    for p in range(POP_SIZE):
        base_idx = p % len(patterns)
        colors = patterns[base_idx].copy()
        # Add random noise
        if p > 0: # Keep first few pure
            num_perturb = random.randint(5, 15)
            perturb_indices = random.sample(range(N), num_perturb)
            for idx in perturb_indices:
                colors[idx] = 1 - colors[idx]
        
        score = count_mono(colors)
        # Polish to local optimum
        score = polish(colors, score, time_limit=1.0)
        
        population.append((score, colors))
        
        if score < best_score:
            best_score = score
            best_coloring = colors.copy()

    # Sort population by score
    population.sort(key=lambda x: x[0])

    generation = 0
    while time.time() < deadline - 15.0:
        # Select parents using tournament selection
        def tournament_select(pop, k=3):
            candidates = random.sample(pop, k)
            return min(candidates, key=lambda x: x[0])[1]
        
        parent1 = tournament_select(population)
        parent2 = tournament_select(population)
        
        # Crossover
        child = crossover(parent1, parent2)
        
        # Mutation: adaptive rate, higher if stagnant
        if generation % 50 == 0:
            mut_rate = 0.1
        else:
            mut_rate = 0.02
        child = mutate(child, rate=mut_rate)
        
        # Evaluate and Polish
        child_score = count_mono(child)
        child_score = polish(child, child_score, time_limit=0.5)
        
        # Replacement: Replace the worst individual in the population
        # Find worst
        worst_idx = len(population) - 1
        if child_score < population[worst_idx][0]:
            population[worst_idx] = (child_score, child)
            population.sort(key=lambda x: x[0])
            
            if child_score < best_score:
                best_score = child_score
                best_coloring = child.copy()
                
        generation += 1
        
        # Periodically, try to improve the best solution with a longer SA or deeper polish
        if generation % 20 == 0 and time.time() < deadline - 20.0:
            # Deep polish on best
            deep_colors = best_coloring.copy()
            deep_score = polish(deep_colors, best_score, time_limit=2.0)
            if deep_score < best_score:
                best_score = deep_score
                best_coloring = deep_colors
                # Inject improved best back into population
                population[0] = (best_score, best_coloring.copy())
                population.sort(key=lambda x: x[0])

    # 7. Final High-Precision Polish
    # Use remaining time for aggressive polishing of the best solution
    if best_coloring is not None:
        final_colors = best_coloring.copy()
        final_score = polish(final_colors, best_score, time_limit=10.0)
        if final_score < best_score:
            best_score = final_score
            best_coloring = final_colors

        # Try a few more perturbations + polish
        remaining_time = deadline - time.time()
        if remaining_time > 2.0:
            for _ in range(3):
                if time.time() > deadline - 1.0:
                    break
                perturbed = best_coloring.copy()
                num_pert = random.randint(2, 6)
                pert_idx = random.sample(range(N), num_pert)
                for idx in pert_idx:
                    perturbed[idx] = 1 - perturbed[idx]
                
                pert_score = count_mono(perturbed)
                # Only polish if it's not catastrophically worse
                if pert_score < best_score + 10:
                    polished_score = polish(perturbed, pert_score, time_limit=max(1.0, remaining_time / 3.0))
                    if polished_score < best_score:
                        best_score = polished_score
                        best_coloring = perturbed.copy()
                        remaining_time = deadline - time.time()

    return best_coloring.tolist()
# EVOLVE-BLOCK-END
