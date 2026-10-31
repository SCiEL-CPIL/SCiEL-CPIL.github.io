import itertools
import random
import time
import numpy as np

def construct(v, k, t):
    start_time = time.time()
    time_limit = 110
    
    points = list(range(v))
    all_t_subsets = list(itertools.combinations(points, t))
    n_t_subsets = len(all_t_subsets)
    
    t_subset_index = {ts: i for i, ts in enumerate(all_t_subsets)}
    
    # Pre-compute all k-subsets and their covered t-subsets
    all_k_subsets = list(itertools.combinations(points, k))
    n_k_subsets = len(all_k_subsets)
    
    # Map block index to list of t-subset indices it covers
    # Also compute degree of each t-subset
    block_t_indices = []
    degree = np.zeros(n_t_subsets, dtype=np.int32)
    
    # For faster lookup in local search, we might want a list of t-subsets per block
    # and a list of blocks per t-subset
    ts_to_blocks = [[] for _ in range(n_t_subsets)]
    
    for b_idx, block in enumerate(all_k_subsets):
        covered_ts = []
        for ts in itertools.combinations(block, t):
            idx = t_subset_index[ts]
            covered_ts.append(idx)
            degree[idx] += 1
        block_t_indices.append(covered_ts)
        for ts_idx in covered_ts:
            ts_to_blocks[ts_idx].append(b_idx)
            
    # 1. LP Duality / Probabilistic Packing Lower Bound Approximation
    # Weights for t-subsets: 1/degree. Low degree = high weight = critical.
    # We will use these weights in the GA fitness and greedy construction.
    weights = np.zeros(n_t_subsets, dtype=np.float64)
    non_zero = degree > 0
    weights[non_zero] = 1.0 / degree[non_zero]
    
    # Identify critical subsets: top N by weight
    # Let's pick the top 10-20% as "hard" constraints to focus the GA
    sorted_indices = np.argsort(weights)[::-1]
    n_critical = max(10, n_t_subsets // 10)
    critical_ts_set = set(sorted_indices[:n_critical].tolist())
    
    # Pre-compute a boolean mask for critical subsets for fast fitness calc?
    # Or just use the set.
    
    def get_fitness(block_set):
        """
        Returns a tuple (uncovered_critical_count, total_blocks).
        Lower is better.
        """
        if not block_set:
            return (n_critical, 0)
        
        # For GA purposes, we only strictly enforce critical coverage in fitness.
        # The local search will ensure full coverage.
        # However, to guide the GA, we count uncovered criticals.
        
        # Optimization: Instead of iterating all blocks and all their t-subsets,
        # we can just check which criticals are covered.
        # But we need to know if a critical is covered by ANY block in the set.
        
        # Efficient way: 
        # 1. Create a set of covered t-subsets from the block_set.
        # 2. Intersect with critical_ts_set.
        
        covered = set()
        for b_idx in block_set:
            # Only add critical ones to the set to save memory/time?
            # No, we need to know if a critical is covered.
            for ts_idx in block_t_indices[b_idx]:
                if ts_idx in critical_ts_set:
                    covered.add(ts_idx)
        
        uncovered_critical = n_critical - len(covered)
        return (uncovered_critical, len(block_set))

    def local_search(block_set):
        """
        Optimized local search to minimize block count while maintaining full coverage.
        Uses NumPy for coverage counts.
        """
        current_blocks = list(block_set)
        coverage = np.zeros(n_t_subsets, dtype=np.int32)
        
        for b_idx in current_blocks:
            for ts_idx in block_t_indices[b_idx]:
                coverage[ts_idx] += 1
        
        # 1. Repair: Ensure full coverage
        uncovered_mask = coverage == 0
        if np.any(uncovered_mask):
            uncovered_indices = np.where(uncovered_mask)[0]
            while len(uncovered_indices) > 0 and time.time() - start_time < time_limit * 0.9:
                # Pick a random uncovered target
                target = random.choice(uncovered_indices)
                
                # Find best block covering this target
                candidates = ts_to_blocks[target]
                if not candidates:
                    # Should not happen in a valid covering problem
                    break
                    
                best_b = -1
                best_gain = -1
                # We want a block that covers the most currently uncovered subsets
                for b in candidates:
                    gain = 0
                    for ts in block_t_indices[b]:
                        if coverage[ts] == 0:
                            gain += 1
                    if gain > best_gain:
                        best_gain = gain
                        best_b = b
                        
                if best_b == -1:
                    break
                if best_b not in current_blocks:
                    current_blocks.append(best_b)
                    for ts in block_t_indices[best_b]:
                        coverage[ts] += 1
                    
                    # Update uncovered mask efficiently? 
                    # For simplicity, recompute if many changes, or just track count.
                    # Let's just recompute the list of uncovered indices if it's small enough
                    # or use a set.
                    # Given n_t_subsets is ~3876, recomputing is O(N).
                    # Let's just check if any remain.
                    still_uncovered = coverage == 0
                    if not np.any(still_uncovered):
                        uncovered_indices = []
                    else:
                        uncovered_indices = np.where(still_uncovered)[0]

        # 2. Minimize: Deletion and Swap
        improved = True
        while improved and time.time() - start_time < time_limit * 0.95:
            improved = False
            
            # Deletion Pass
            # Try to remove any block that doesn't cover any unique t-subset
            # A block is removable if for all ts in block, coverage[ts] > 1
            for i in range(len(current_blocks) - 1, -1, -1):
                b_idx = current_blocks[i]
                can_remove = True
                for ts in block_t_indices[b_idx]:
                    if coverage[ts] == 1:
                        can_remove = False
                        break
                if can_remove:
                    for ts in block_t_indices[b_idx]:
                        coverage[ts] -= 1
                    current_blocks.pop(i)
                    improved = True
                    # Restart deletion pass? Or continue?
                    # Continuing is fine, but the indices shifted.
                    # Since we popped i, we should break and restart the loop
                    # to ensure we don't skip blocks.
                    break
            
            if not improved and time.time() - start_time < time_limit * 0.95:
                # Swap Pass
                # Try to replace a block with another that covers its unique subsets
                # and potentially more.
                # To avoid O(N^2), we sample a few blocks.
                indices = list(range(len(current_blocks)))
                random.shuffle(indices)
                
                for i in indices:
                    if time.time() - start_time > time_limit * 0.95:
                        break
                    old_b = current_blocks[i]
                    # Find unique subsets covered by old_b
                    unique_ts = [ts for ts in block_t_indices[old_b] if coverage[ts] == 1]
                    if not unique_ts:
                        continue
                    
                    # Find candidates that cover ALL unique_ts
                    # Intersection of ts_to_blocks for each unique_ts
                    # Start with the smallest list for efficiency
                    unique_ts.sort(key=lambda x: len(ts_to_blocks[x]))
                    if not ts_to_blocks[unique_ts[0]]:
                        continue
                        
                    candidates = set(ts_to_blocks[unique_ts[0]])
                    for ts in unique_ts[1:]:
                        candidates.intersection_update(ts_to_blocks[ts])
                        if not candidates:
                            break
                    
                    candidates.discard(old_b)
                    if not candidates:
                        continue
                    
                    # Pick the best candidate: one that covers the most other subsets with coverage 1
                    # (i.e., it helps "fix" other potential swaps or is just "better")
                    # Actually, the goal is just to swap if it maintains coverage.
                    # Since candidates cover all unique_ts, swapping maintains coverage.
                    # We prefer a candidate that covers MORE subsets that are currently coverage 1?
                    # No, we want to minimize size. Swap is size-neutral.
                    # However, a "better" swap might set up future deletions.
                    # Let's pick a random candidate to explore, or the one with max overlap with low-coverage sets.
                    
                    best_new = -1
                    best_score = -1
                    cand_list = list(candidates)
                    if len(cand_list) > 20:
                        cand_list = random.sample(cand_list, 20)
                        
                    for nb in cand_list:
                        # Score: how many other coverage-1 subsets does nb cover?
                        # This might allow removing more blocks later.
                        score = 0
                        for ts in block_t_indices[nb]:
                            if coverage[ts] == 1:
                                score += 1
                        if score > best_score:
                            best_score = score
                            best_new = nb
                            
                    if best_new != -1:
                        # Perform swap
                        for ts in block_t_indices[old_b]:
                            coverage[ts] -= 1
                        for ts in block_t_indices[best_new]:
                            coverage[ts] += 1
                        current_blocks[i] = best_new
                        improved = True
                        # Restart swap pass? 
                        break
                        
        return current_blocks

    # 2. Genetic Algorithm Initialization
    # Create initial population using Greedy Set Cover with different random seeds/orderings
    population = []
    pop_size = 12
    
    for _ in range(pop_size):
        # Greedy construction
        coverage = np.zeros(n_t_subsets, dtype=np.int32)
        selected = []
        uncovered_count = n_t_subsets
        
        # Randomize the order of selection slightly to create diversity
        # We pick a random uncovered target, then best block
        while uncovered_count > 0 and time.time() - start_time < time_limit * 0.15:
            # Find an uncovered target
            # Instead of scanning all, pick random until we find one?
            # Or maintain a set of uncovered.
            # Let's just pick random index and check.
            # With 3876 subsets, this is fast.
            target = -1
            for _ in range(50):
                idx = random.randint(0, n_t_subsets - 1)
                if coverage[idx] == 0:
                    target = idx
                    break
            if target == -1:
                # Fallback: scan
                uncovered_indices = np.where(coverage == 0)[0]
                if len(uncovered_indices) == 0:
                    break
                target = random.choice(uncovered_indices.tolist())
            
            best_block = -1
            best_gain = -1.0
            for b_idx in ts_to_blocks[target]:
                gain = 0.0
                for ts_idx in block_t_indices[b_idx]:
                    if coverage[ts_idx] == 0:
                        gain += weights[ts_idx]
                if gain > best_gain:
                    best_gain = gain
                    best_block = b_idx
            
            if best_block == -1:
                break
            if best_block not in selected:
                selected.append(best_block)
                for ts_idx in block_t_indices[best_block]:
                    if coverage[ts_idx] == 0:
                        uncovered_count -= 1
                    coverage[ts_idx] += 1
        
        # Repair and Local Search
        optimized = local_search(selected)
        population.append(optimized)

    # 3. Genetic Algorithm Loop
    best_solution = min(population, key=lambda x: len(x))
    
    generations = 0
    max_generations = 300 # Limit by time
    
    while time.time() - start_time < time_limit * 0.85 and generations < max_generations:
        generations += 1
        
        # Selection: Tournament
        def tournament():
            p1 = random.choice(population)
            p2 = random.choice(population)
            # Fitness: primarily size, but ensure critical coverage?
            # Local search ensures validity. So size is the main driver.
            # However, if a solution is invalid (shouldn't happen), penalize.
            return p1 if len(p1) < len(p2) else p2
        
        new_population = []
        
        for _ in range(pop_size // 2):
            p1 = tournament()
            p2 = tournament()
            
            # Crossover: Merge and Reduce
            # Union of blocks
            child_blocks = list(set(p1) | set(p2))
            
            # Local Search to optimize
            optimized_child = local_search(child_blocks)
            new_population.append(optimized_child)
            
            # Elitism: Keep parents
            new_population.append(p1)
            new_population.append(p2)
            
        # Update population with new candidates, keeping the best pop_size
        population.extend(new_population)
        
        # Sort by size
        population.sort(key=lambda x: len(x))
        population = population[:pop_size]
        
        # Update best
        current_best = population[0]
        if len(current_best) < len(best_solution):
            best_solution = current_best

    # Final cleanup on best_solution
    final_blocks = local_search(best_solution)
    
    return [list(all_k_subsets[i]) for i in final_blocks]
