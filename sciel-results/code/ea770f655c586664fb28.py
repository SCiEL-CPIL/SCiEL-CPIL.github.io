import itertools
import random
import time
import numpy as np

def construct(v, k, t):
    """
    Construct a covering design C(v,k,t) using fast greedy multi-start
    followed by genetic algorithm refinement.
    """
    deadline = time.time() + 110.0  # Safety margin

    # Precompute t-subsets
    all_tsubs = list(itertools.combinations(range(v), t))
    n_tsubs = len(all_tsubs)
    tsub_to_idx = {ts: i for i, ts in enumerate(all_tsubs)}
    
    # Precompute for each t-subset, the list of k-subsets that contain it
    # For v=19, k=8, t=5: C(14, 3) = 364 k-subsets per t-subset
    # Precompute all k-subsets containing each t-subset for faster lookup
    tsub_to_ksubs = {}
    for ts in all_tsubs:
        ts_set = set(ts)
        remaining = [p for p in range(v) if p not in ts_set]
        ksubs = []
        for extra in itertools.combinations(remaining, k - t):
            block = tuple(sorted(ts + tuple(extra)))
            ksubs.append(block)
        tsub_to_ksubs[ts] = ksubs
    
    # Precompute for each k-subset, the list of t-subset indices it covers
    # We'll compute this on the fly but cache it
    ksub_to_tsub_indices = {}
    
    def get_ksub_indices(block):
        block = tuple(block)
        if block not in ksub_to_tsub_indices:
            idxs = [tsub_to_idx[comb] for comb in itertools.combinations(block, t)]
            ksub_to_tsub_indices[block] = idxs
        return ksub_to_tsub_indices[block]
    
    def greedy_once(seed):
        rng = random.Random(seed)
        uncovered = np.ones(n_tsubs, dtype=bool)
        blocks = []
        
        while np.any(uncovered):
            if time.time() > deadline:
                break
            
            # Pick a random uncovered t-subset
            uncovered_indices = np.where(uncovered)[0]
            if len(uncovered_indices) == 0:
                break
            start_idx = rng.choice(uncovered_indices)
            S = all_tsubs[start_idx]
            
            # Get precomputed k-subsets containing S
            candidates = tsub_to_ksubs[S]
            
            # Evaluate gain for each candidate
            best_block = None
            best_gain = -1
            
            # Shuffle for tie-breaking
            candidates = list(candidates)
            rng.shuffle(candidates)
            
            for B in candidates:
                idxs = get_ksub_indices(B)
                gain = np.sum(uncovered[idxs])
                
                if gain > best_gain:
                    best_gain = gain
                    best_block = B
            
            if best_block is None:
                break
                
            blocks.append(best_block)
            
            # Update uncovered mask
            idxs = get_ksub_indices(best_block)
            uncovered[idxs] = False
        
        return blocks

    def remove_redundant(blocks, max_iter=3):
        if not blocks:
            return blocks
        
        for _ in range(max_iter):
            if time.time() > deadline:
                break
                
            # Count coverage of each t-subset
            cover_count = np.zeros(n_tsubs, dtype=np.int32)
            block_indices = []
            for b in blocks:
                idxs = get_ksub_indices(tuple(b))
                block_indices.append(idxs)
                cover_count[idxs] += 1
            
            valid_blocks = []
            removed_any = False
            
            for i, b in enumerate(blocks):
                if time.time() > deadline:
                    # Return original blocks if we run out of time
                    return blocks
                
                idxs = block_indices[i]
                if np.all(cover_count[idxs] > 1):
                    # Redundant: remove it
                    cover_count[idxs] -= 1
                    removed_any = True
                else:
                    valid_blocks.append(b)
            
            if not removed_any:
                break
            blocks = valid_blocks
        
        return blocks

    def validate_covering(blocks):
        """Check if blocks form a valid covering."""
        if not blocks:
            return n_tsubs == 0
        covered = np.zeros(n_tsubs, dtype=bool)
        for b in blocks:
            idxs = get_ksub_indices(tuple(b))
            covered[idxs] = True
        return np.all(covered)

    def ga_mutation(individual, rng, deadline):
        """
        Mutate a covering by replacing one block with a new block
        that maximizes gain on under-covered t-subsets.
        """
        if not individual:
            return individual[:]
        
        # Compute coverage counts
        cover_count = np.zeros(n_tsubs, dtype=np.int32)
        for b in individual:
            idxs = get_ksub_indices(tuple(b))
            cover_count[idxs] += 1
        
        # Find under-covered t-subsets (covered exactly once)
        under_covered = np.where(cover_count == 1)[0]
        if len(under_covered) == 0:
            return individual[:]
        
        # Pick a random under-covered t-subset
        target_idx = rng.choice(under_covered)
        target_tsub = all_tsubs[target_idx]
        
        # Find blocks that cover this t-subset
        blocks_covering_target = []
        for i, b in enumerate(individual):
            idxs = get_ksub_indices(tuple(b))
            if target_idx in idxs:
                blocks_covering_target.append(i)
        
        if not blocks_covering_target:
            return individual[:]
        
        # Pick a random block to replace
        replace_idx = rng.choice(blocks_covering_target)
        old_block = individual[replace_idx]
        
        # Find a new block that covers the target t-subset and has high gain
        candidates = tsub_to_ksubs[target_tsub]
        candidates = list(candidates)
        rng.shuffle(candidates)
        
        best_new_block = None
        best_gain = -1
        
        for B in candidates:
            if B == tuple(old_block):
                continue
            idxs = get_ksub_indices(B)
            # Gain: number of under-covered t-subsets this block covers
            gain = np.sum(cover_count[idxs] == 1)
            # Penalize if it doesn't cover the target
            if target_idx not in idxs:
                gain = -1
            if gain > best_gain:
                best_gain = gain
                best_new_block = B
        
        if best_new_block is None:
            return individual[:]
        
        new_individual = individual[:]
        new_individual[replace_idx] = best_new_block
        
        # Validate
        if validate_covering(new_individual):
            return new_individual
        else:
            return individual[:]

    def ga_crossover(parent1, parent2, rng, deadline):
        """
        Crossover: merge two parents' blocks, then reduce.
        """
        # Combine all blocks from both parents
        combined = list(parent1) + list(parent2)
        
        # Remove duplicates
        seen = set()
        unique_combined = []
        for b in combined:
            key = tuple(b)
            if key not in seen:
                seen.add(key)
                unique_combined.append(b)
        
        # Reduce
        reduced = remove_redundant(unique_combined, max_iter=2)
        
        # Validate
        if validate_covering(reduced):
            return reduced
        else:
            # Fallback to smaller parent
            if len(parent1) <= len(parent2):
                return parent1[:]
            else:
                return parent2[:]

    def ga_evolve(population, num_generations, deadline):
        """
        Genetic algorithm to evolve a population of coverings.
        """
        best_individual = min(population, key=len)
        best_len = len(best_individual)
        
        for gen in range(num_generations):
            if time.time() > deadline:
                break
            
            new_population = []
            
            # Elitism: keep the best individual
            new_population.append(best_individual[:])
            
            # Generate offspring
            while len(new_population) < len(population):
                if time.time() > deadline:
                    break
                
                # Crossover
                p1_idx = rng.randint(0, len(population) - 1)
                p2_idx = rng.randint(0, len(population) - 1)
                p1 = population[p1_idx]
                p2 = population[p2_idx]
                
                child = ga_crossover(p1, p2, rng, deadline)
                
                # Mutation
                if rng.random() < 0.5:
                    child = ga_mutation(child, rng, deadline)
                
                new_population.append(child)
            
            population = new_population
            
            # Update best
            for ind in population:
                if len(ind) < best_len:
                    best_len = len(ind)
                    best_individual = ind[:]
        
        return best_individual

    best_blocks = None
    best_len = float('inf')
    
    # Phase 1: Multi-start greedy
    num_starts = 0
    start_time = time.time()
    
    # Run greedy starts until we have enough time for GA
    ga_time_budget = 30.0  # Reserve 30s for GA
    greedy_deadline = deadline - ga_time_budget
    
    while time.time() < greedy_deadline:
        seed = num_starts
        blocks = greedy_once(seed)
        
        if not blocks:
            num_starts += 1
            continue
            
        # Reduce
        reduced_blocks = remove_redundant(blocks)
        
        if len(reduced_blocks) < best_len:
            best_len = len(reduced_blocks)
            best_blocks = reduced_blocks[:]
            
        num_starts += 1
    
    if best_blocks is None:
        best_blocks = greedy_once(0)
        best_blocks = remove_redundant(best_blocks)
        best_len = len(best_blocks)
    
    # Phase 2: Genetic algorithm refinement
    if time.time() < deadline - 5.0:
        rng = random.Random(12345)
        
        # Build initial population from best greedy results
        # We only have the best, so create variations
        population = [best_blocks[:]]
        
        # Add some more greedy starts to the population if time permits
        for _ in range(5):
            if time.time() > deadline - 10.0:
                break
            seed = num_starts
            num_starts += 1
            blocks = greedy_once(seed)
            if blocks:
                reduced = remove_redundant(blocks)
                population.append(reduced)
        
        # Run GA
        ga_deadline = deadline - 5.0
        ga_best = ga_evolve(population, num_generations=100, deadline=ga_deadline)
        
        if len(ga_best) < best_len:
            best_len = len(ga_best)
            best_blocks = ga_best[:]
    
    return best_blocks
