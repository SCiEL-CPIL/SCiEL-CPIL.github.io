import numpy as np
import itertools
import time
import random
from collections import Counter

def construct(v, k, t):
    start_time = time.time()
    time_limit = 115.0
    
    points = range(v)
    all_ts = list(itertools.combinations(points, t))
    n_ts = len(all_ts)
    idx_map = {comb: i for i, comb in enumerate(all_ts)}
    
    all_ks = list(itertools.combinations(points, k))
    n_ks = len(all_ks)
    
    def nCr(n, r):
        if r > n: return 0
        if r == 0: return 1
        res = 1
        for i in range(r):
            res *= (n - i)
            res //= (i + 1)
        return res
            
    t_count = nCr(k, t)
    
    # Precompute: for each block, which t-subsets it covers
    block_t_indices = np.zeros((n_ks, t_count), dtype=np.int32)
    for i, block in enumerate(all_ks):
        subs = itertools.combinations(block, t)
        for j, sub in enumerate(subs):
            block_t_indices[i, j] = idx_map[sub]

    # Binary matrix: block_t_mat[i, j] = 1 iff block i covers t-subset j
    block_t_mat = np.zeros((n_ks, n_ts), dtype=np.int8)
    for i in range(n_ks):
        block_t_mat[i, block_t_indices[i]] = 1

    # Inverse index: for each t-subset, which blocks cover it
    triple_to_blocks = [[] for _ in range(n_ts)]
    for i in range(n_ks):
        for j in range(t_count):
            triple_to_blocks[int(block_t_indices[i, j])].append(i)
    triple_to_blocks = [np.array(lst, dtype=np.int32) if lst else np.array([], dtype=np.int32) for lst in triple_to_blocks]
            
    best_solution = None
    best_score = float('inf')
    
    def aggressive_deletion(current_blocks, coverage_count):
        """Try harder to remove blocks, including blocks where most subsets are over-covered."""
        improved = True
        while improved:
            if time.time() - start_time > time_limit - 3.0:
                break
            improved = False
            new_blocks = []
            
            for b_idx in current_blocks:
                subs_in_block = block_t_indices[b_idx]
                # Count how many of this block's t-subsets are covered by other blocks
                other_covers = coverage_count[subs_in_block] - 1
                if np.all(other_covers > 0):
                    # Safe to remove
                    improved = True
                    coverage_count[subs_in_block] -= 1
                else:
                    new_blocks.append(b_idx)
                    
            current_blocks = new_blocks
            
            # Second pass: try removing blocks where >= 90% of subsets are over-covered
            # and verify the few uncovered ones are covered elsewhere
            improved = False
            new_blocks = []
            for b_idx in current_blocks:
                subs_in_block = block_t_indices[b_idx]
                other_covers = coverage_count[subs_in_block] - 1
                # If at least 95% are covered by others, try removing
                if np.mean(other_covers > 0) >= 0.95:
                    uncovered_mask = other_covers <= 0
                    if np.any(uncovered_mask):
                        uncovered_subs = subs_in_block[uncovered_mask]
                        # Fast check using precomputed inverse index
                        can_remove = True
                        for unc_sub in uncovered_subs:
                            unc_sub = int(unc_sub)
                            # coverage_count[unc_sub] == 1 means only b_idx covers it
                            if coverage_count[unc_sub] == 1:
                                can_remove = False
                                break
                        if can_remove:
                            improved = True
                            coverage_count[subs_in_block] -= 1
                            continue
                    else:
                        improved = True
                        coverage_count[subs_in_block] -= 1
                        continue
                new_blocks.append(b_idx)
            current_blocks = new_blocks
            
        return current_blocks
    
    def run_greedy(seed):
        random.seed(seed)
        np.random.seed(seed)
        
        covered = np.zeros(n_ts, dtype=bool)
        blocks_indices = []
        remaining = n_ts
        
        # Track how many times each t-subset has been covered (for weighted scoring)
        cover_freq = np.zeros(n_ts, dtype=np.int32)
        
        while remaining > 0:
            if time.time() - start_time > time_limit - 5.0:
                return None, None

            # Compute gain via single vectorized mat-vec: how many uncovered triples each block covers
            gains = block_t_mat @ (~covered).astype(np.int8)

            max_gain = int(gains.max())
            if max_gain == 0:
                break

            # Among blocks with max gain, use weighted score based on rarity
            max_gain_blocks = np.where(gains == max_gain)[0]

            if len(max_gain_blocks) > 40:
                candidates = max_gain_blocks[random.sample(range(len(max_gain_blocks)), min(40, len(max_gain_blocks)))]
            else:
                candidates = max_gain_blocks

            # Score candidates: gain + bonus for covering rare subsets
            best_score_candidate = -1.0
            best_candidate = int(candidates[0])
            for cand in candidates:
                subs = block_t_indices[cand]
                newly_covered = ~covered[subs]
                if np.any(newly_covered):
                    bonus = float(np.sum(1.0 / (1.0 + cover_freq[subs[newly_covered]])))
                else:
                    bonus = 0.0
                score_cand = max_gain + bonus * 0.1
                if score_cand > best_score_candidate:
                    best_score_candidate = score_cand
                    best_candidate = int(cand)

            blocks_indices.append(best_candidate)

            subs_to_cover = block_t_indices[best_candidate]
            newly_covered = ~covered[subs_to_cover]
            new_covers = int(np.sum(newly_covered))
            remaining -= new_covers
            cover_freq[subs_to_cover] += 1
            covered[subs_to_cover] = True

        current_blocks = list(blocks_indices)
        
        # Build coverage count
        coverage_count = np.zeros(n_ts, dtype=np.int32)
        for b_idx in current_blocks:
            coverage_count[block_t_indices[b_idx]] += 1
        
        # Aggressive deletion
        current_blocks = aggressive_deletion(current_blocks, coverage_count)
        
        # Rebuild coverage count after deletion
        coverage_count = np.zeros(n_ts, dtype=np.int32)
        for b_idx in current_blocks:
            coverage_count[block_t_indices[b_idx]] += 1
        
        # Final deletion pass
        coverage_count = np.zeros(n_ts, dtype=np.int32)
        for b_idx in current_blocks:
            coverage_count[block_t_indices[b_idx]] += 1
        current_blocks = aggressive_deletion(current_blocks, coverage_count)
        
        return current_blocks, len(current_blocks)
    
    def compute_fitness(block_set):
        """
        Compute fitness for a block set.
        Fitness = number of blocks + penalty for uncovered triples.
        Lower is better.
        """
        coverage_count = np.zeros(n_ts, dtype=np.int32)
        for b_idx in block_set:
            coverage_count[block_t_indices[b_idx]] += 1
        
        uncovered = np.sum(coverage_count == 0)
        # Penalty: each uncovered triple adds 1.0 to the score
        # This encourages complete coverage
        fitness = len(block_set) + uncovered * 1.0
        return fitness, coverage_count, uncovered
    
    def genetic_algorithm(seed, pop_size=20, generations=50):
        """
        Genetic Algorithm that evolves a population of block sets.
        """
        random.seed(seed)
        np.random.seed(seed)
        
        # Initialize population with greedy solutions
        population = []
        for i in range(pop_size):
            blocks, _ = run_greedy(seed + i * 1000)
            if blocks is not None:
                population.append(blocks)
        
        if len(population) == 0:
            return None, float('inf')
        
        # Evaluate initial population
        best_fitness = float('inf')
        best_individual = population[0]
        for ind in population:
            fit, _, _ = compute_fitness(ind)
            if fit < best_fitness:
                best_fitness = fit
                best_individual = ind
        
        for gen in range(generations):
            if time.time() - start_time > time_limit - 10.0:
                break
            
            # Evaluate all individuals
            fitnesses = []
            for ind in population:
                fit, _, _ = compute_fitness(ind)
                fitnesses.append(fit)
                if fit < best_fitness:
                    best_fitness = fit
                    best_individual = ind
            
            # Selection: tournament selection
            new_population = []
            for _ in range(pop_size):
                # Tournament selection
                tournament_size = 3
                candidates = random.sample(range(len(population)), tournament_size)
                winner = min(candidates, key=lambda i: fitnesses[i])
                
                # Crossover: uniform crossover between two parents
                if len(population) >= 2 and random.random() < 0.7:
                    parent2_idx = random.choice([i for i in range(len(population)) if i != winner])
                    parent1 = population[winner]
                    parent2 = population[parent2_idx]
                    
                    # Uniform crossover: for each position, pick from parent1 or parent2
                    child = []
                    for pos in range(min(len(parent1), len(parent2))):
                        if random.random() < 0.5:
                            child.append(parent1[pos])
                        else:
                            child.append(parent2[pos])
                    
                    # Ensure no duplicates
                    child = list(set(child))
                    
                    # Mutation: randomly swap some blocks
                    if random.random() < 0.3 and len(child) > 0:
                        n_mutations = random.randint(1, max(1, len(child) // 5))
                        for _ in range(n_mutations):
                            idx_to_swap = random.randint(0, len(child) - 1)
                            new_block = random.randint(0, n_ks - 1)
                            if new_block not in child:
                                child[idx_to_swap] = new_block
                    
                    new_population.append(child)
                else:
                    # Just copy the winner with some mutation
                    child = list(population[winner])
                    if random.random() < 0.2 and len(child) > 0:
                        idx_to_swap = random.randint(0, len(child) - 1)
                        new_block = random.randint(0, n_ks - 1)
                        if new_block not in child:
                            child[idx_to_swap] = new_block
                    new_population.append(child)
            
            # Local search on best individuals
            for i in range(min(3, len(new_population))):
                if time.time() - start_time > time_limit - 15.0:
                    break
                ind = new_population[i]
                coverage_count = np.zeros(n_ts, dtype=np.int32)
                for b_idx in ind:
                    coverage_count[block_t_indices[b_idx]] += 1
                new_population[i] = aggressive_deletion(ind, coverage_count)
            
            population = new_population
        
        # Final evaluation
        best_fitness = float('inf')
        best_individual = population[0]
        for ind in population:
            fit, _, _ = compute_fitness(ind)
            if fit < best_fitness:
                best_fitness = fit
                best_individual = ind
        
        return best_individual, best_fitness

    # Phase 1: Run multiple greedy restarts
    restart_count = 0
    while time.time() - start_time < time_limit - 30.0:
        seed = restart_count
        blocks, score = run_greedy(seed)
        
        if blocks is None:
            break
            
        if score < best_score:
            best_score = score
            best_solution = blocks
            
        restart_count += 1
    
    # Phase 2: Run genetic algorithm if time permits
    if time.time() - start_time < time_limit - 20.0:
        ga_seed = restart_count
        ga_blocks, ga_score = genetic_algorithm(ga_seed, pop_size=15, generations=30)
        
        if ga_blocks is not None and ga_score < best_score:
            best_score = ga_score
            best_solution = ga_blocks
    
    # Phase 3: Final greedy restarts with remaining time
    if time.time() - start_time < time_limit - 5.0:
        while time.time() - start_time < time_limit - 2.0:
            seed = restart_count + 1000
            blocks, score = run_greedy(seed)
            
            if blocks is None:
                break
                
            if score < best_score:
                best_score = score
                best_solution = blocks
                
            restart_count += 1
    
    if best_solution is None:
        blocks, score = run_greedy(0)
        if blocks is not None:
            best_solution = blocks
            best_score = score
        else:
            return []

    final_blocks = [list(all_ks[i]) for i in best_solution]
    
    return final_blocks
