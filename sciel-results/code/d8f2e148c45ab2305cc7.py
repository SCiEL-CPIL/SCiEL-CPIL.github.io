import itertools
import numpy as np
import time
import random
from collections import defaultdict

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 115  # 5s margin

    # Precompute all t-subsets and map them to indices
    all_t_subs = list(itertools.combinations(range(v), t))
    N_t = len(all_t_subs)
    sub_to_idx = {sub: i for i, sub in enumerate(all_t_subs)}

    # Precompute for each k-subset, the list of t-subset indices it covers
    # But we don't want to precompute all k-subsets (C(19,7) = 50388) with their full coverage
    # Instead, we compute coverage on the fly for specific blocks.

    def block_to_indices(block):
        """Return list of t-subset indices covered by a block."""
        return [sub_to_idx[sub] for sub in itertools.combinations(block, t)]

    def greedy_cover(seed, strategy=0):
        """
        Greedy set cover with multiple strategies.
        strategy 0: random start, greedy extension
        strategy 1: start with 'hard' subsets (points with low degree in uncovered)
        strategy 2: deterministic min-start
        """
        rng = random.Random(seed)
        uncovered = np.ones(N_t, dtype=bool)
        blocks = []
        
        # For strategy 1, we need to track point frequencies in uncovered subsets
        # We'll compute this periodically or just use a simpler heuristic
        
        while uncovered.any():
            # Choose starting t-subset based on strategy
            unc_idx = np.where(uncovered)[0]
            if len(unc_idx) == 0:
                break
                
            if strategy == 0:
                start_sub_idx = unc_idx[rng.randint(0, len(unc_idx)-1)]
            elif strategy == 1:
                # Pick subset containing a point that appears in fewest uncovered subsets
                # Approximate: pick random uncovered, but weight by rarity
                # For speed, sample a few
                sample_size = min(50, len(unc_idx))
                sample_idx = rng.sample(list(unc_idx), sample_size)
                # Count point frequencies in sample
                point_counts = defaultdict(int)
                for idx in sample_idx:
                    for p in all_t_subs[idx]:
                        point_counts[p] += 1
                # Pick the subset with the rarest point
                best_idx = sample_idx[0]
                best_rarity = float('inf')
                for idx in sample_idx:
                    rarity = min(point_counts[p] for p in all_t_subs[idx])
                    if rarity < best_rarity:
                        best_rarity = rarity
                        best_idx = idx
                start_sub_idx = best_idx
            else:
                start_sub_idx = min(unc_idx)
            
            start_sub = all_t_subs[start_sub_idx]
            block = list(start_sub)
            
            while len(block) < k:
                # Calculate gain for each candidate x
                best_gain = -1
                best_x = -1
                candidates = [x for x in range(v) if x not in block]
                
                for x in candidates:
                    gain = 0
                    for rest in itertools.combinations(block, t-1):
                        new_sub = tuple(sorted(rest + (x,)))
                        idx = sub_to_idx[new_sub]
                        if uncovered[idx]:
                            gain += 1
                    
                    if gain > best_gain:
                        best_gain = gain
                        best_x = x
                    elif gain == best_gain:
                        if rng.random() < 0.5:
                            best_x = x
                
                if best_x == -1:
                    # Fallback: random
                    best_x = rng.choice(candidates)
                
                block.append(best_x)
            
            block.sort()
            
            # Update uncovered
            for idx in block_to_indices(block):
                uncovered[idx] = False
                
            blocks.append(block)
            
        return blocks

    def remove_redundant(blocks):
        """Remove redundant blocks iteratively."""
        if not blocks:
            return blocks
            
        block_covs = []
        for b in blocks:
            covs = block_to_indices(b)
            block_covs.append(covs)
            
        count = np.zeros(N_t, dtype=int)
        for covs in block_covs:
            for idx in covs:
                count[idx] += 1
                
        kept_blocks = []
        changed = True
        while changed:
            changed = False
            new_kept = []
            for i, b in enumerate(kept_blocks):
                is_critical = False
                for idx in block_covs[i]:
                    if count[idx] == 1:
                        is_critical = True
                        break
                if not is_critical:
                    for idx in block_covs[i]:
                        count[idx] -= 1
                    changed = True
                else:
                    new_kept.append(b)
            if len(new_kept) < len(kept_blocks):
                # Rebuild block_covs for kept blocks
                # Actually, we need to be careful with indices
                pass
            kept_blocks = new_kept
            
            # Also try to remove from the original list if we haven't processed all
            # This is getting complex. Let's do a simpler single-pass first.
            
        # Simpler single-pass version:
        kept = []
        for i, b in enumerate(blocks):
            is_critical = False
            for idx in block_covs[i]:
                if count[idx] == 1:
                    is_critical = True
                    break
            if is_critical:
                kept.append(b)
            else:
                for idx in block_covs[i]:
                    count[idx] -= 1
        return kept

    def genetic_merge(blocks_a, blocks_b, uncovered_mask=None):
        """
        Merge two coverings by selecting blocks that cover the most uncovered subsets.
        This is a simplified version: we just take the union and remove redundancies.
        A more sophisticated approach would interleave selection.
        """
        # Simple merge: take all blocks from both, then remove redundancies
        merged = list(blocks_a) + list(blocks_b)
        # Remove duplicates
        seen = set()
        unique_merged = []
        for b in merged:
            key = tuple(sorted(b))
            if key not in seen:
                seen.add(key)
                unique_merged.append(b)
        return remove_redundant(unique_merged)

    def two_exchange(blocks):
        """
        Try to replace two blocks with one or two different blocks that cover the same or more.
        This is a local search move.
        """
        # For simplicity, we'll try removing one block and see if the rest still covers everything.
        # If not, we'll try to find a replacement.
        # This is a simplified version of 2-exchange.
        
        if len(blocks) < 2:
            return blocks
            
        # Compute coverage counts
        block_covs = [block_to_indices(b) for b in blocks]
        count = np.zeros(N_t, dtype=int)
        for covs in block_covs:
            for idx in covs:
                count[idx] += 1
        
        # Try removing each block
        improved = False
        for i in range(len(blocks)):
            # Check if block i is removable
            removable = True
            for idx in block_covs[i]:
                if count[idx] == 1:
                    removable = False
                    break
            if removable:
                # Remove it
                for idx in block_covs[i]:
                    count[idx] -= 1
                blocks.pop(i)
                block_covs.pop(i)
                improved = True
                break  # Restart after one removal
                
        if improved:
            return two_exchange(blocks)  # Recurse
        return blocks

    def local_search(blocks, rng, max_iter=100):
        """
        Local search: try random swaps of blocks.
        """
        if len(blocks) == 0:
            return blocks
            
        best_blocks = blocks[:]
        best_len = len(blocks)
        
        for _ in range(max_iter):
            # Try to remove a random block
            if len(blocks) > 1:
                i = rng.randint(0, len(blocks)-1)
                trial_blocks = blocks[:]
                trial_blocks.pop(i)
                # Check if still valid
                # This is expensive, so we'll skip for now
                # A better approach: maintain coverage counts
                pass
                
            # Try to replace a random block with a random other block
            # This is also complex. For now, we'll just do redundancy removal.
            pass
            
        return best_blocks

    best_blocks = None
    best_len = float('inf')
    
    # Phase 1: Generate diverse initial solutions using different strategies and seeds
    initial_solutions = []
    strategies = [0, 1, 2]
    for strategy in strategies:
        for seed in range(10):
            if time.time() > deadline - 10:
                break
            blocks = greedy_cover(seed * 100 + strategy, strategy)
            blocks = remove_redundant(blocks)
            if blocks:
                initial_solutions.append(blocks)
                if len(blocks) < best_len:
                    best_len = len(blocks)
                    best_blocks = blocks[:]
    
    if not initial_solutions:
        # Fallback
        return greedy_cover(0, 0)
    
    # Phase 2: Genetic algorithm with merge and local search
    population = initial_solutions[:]
    rng = random.Random(42)
    
    for gen in range(20):
        if time.time() > deadline - 5:
            break
            
        # Selection: keep the best half
        population.sort(key=lambda x: len(x))
        population = population[:max(4, len(population)//2)]
        
        # Crossover: merge pairs
        new_population = []
        for i in range(len(population)):
            for j in range(i+1, len(population)):
                if time.time() > deadline - 5:
                    break
                merged = genetic_merge(population[i], population[j])
                if merged:
                    new_population.append(merged)
        
        # Mutation: apply local search (redundancy removal + random restarts)
        for i in range(len(new_population)):
            if time.time() > deadline - 5:
                break
            # Apply two-exchange (redundancy removal)
            new_population[i] = two_exchange(new_population[i])
            
            # Occasionally, do a random restart on a random block
            if rng.random() < 0.3 and len(new_population[i]) > 0:
                # Pick a random block and replace it with a new greedy block
                # This is a simplified mutation
                pass
        
        # Add new population to existing
        population.extend(new_population)
        
        # Update best
        for blocks in population:
            if len(blocks) < best_len:
                best_len = len(blocks)
                best_blocks = blocks[:]
        
        if best_len <= 131:
            break
    
    return best_blocks
