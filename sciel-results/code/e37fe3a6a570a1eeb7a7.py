import time
import random
import itertools
import numpy as np

def construct(v, k, t):
    t0 = time.time()
    deadline = t0 + 115.0

    # Precomputation
    points = range(v)
    all_t_subsets = list(itertools.combinations(points, t))
    N_t = len(all_t_subsets)
    t_to_idx = {s: i for i, s in enumerate(all_t_subsets)}

    all_blocks = list(itertools.combinations(points, k))
    N_b = len(all_blocks)

    # For each block, which t-subsets does it cover?
    block_cover = []
    for b in all_blocks:
        covered = [t_to_idx[s] for s in itertools.combinations(b, t)]
        block_cover.append(covered)

    # Inverse index
    t_to_blocks = [[] for _ in range(N_t)]
    for bi, covered in enumerate(block_cover):
        for ti in covered:
            t_to_blocks[ti].append(bi)

    # Dual-guided covering heuristic
    def dual_guided_cover(seed_rng):
        """
        Primal-dual heuristic: maintain dual prices for uncovered t-subsets,
        iteratively select the block with maximum marginal gain per unit cost,
        then update prices for newly uncovered subsets.
        """
        # Initialize: all t-subsets uncovered
        covered_mask = np.zeros(N_t, dtype=bool)
        selected_blocks = []
        # Dual prices: start at 1 for all uncovered
        dual_prices = np.ones(N_t, dtype=np.float64)
        # Track coverage counts for pruning
        coverage_counts = np.zeros(N_t, dtype=np.int32)
        
        num_uncovered = N_t
        remaining_blocks = list(range(N_b))
        
        while num_uncovered > 0:
            if time.time() > deadline - 3.0:
                break
                
            # Find uncovered indices
            uncovered_idx = np.where(~covered_mask)[0]
            if len(uncovered_idx) == 0:
                break
            
            # For each candidate block, compute its "dual weight":
            # sum of dual_prices[ti] for ti in block_cover[bi] where ti is uncovered
            # We want the block with highest dual weight per block (unit cost)
            
            # Optimization: only consider blocks that cover at least one uncovered subset
            # Sample a few uncovered subsets to find candidate blocks
            sample_size = min(5, len(uncovered_idx))
            sampled = seed_rng.sample(uncovered_idx.tolist(), sample_size)
            
            candidate_blocks = set()
            for ti in sampled:
                candidate_blocks.update(t_to_blocks[ti])
            
            # Remove blocks already selected
            selected_set = set(selected_blocks)
            candidate_blocks -= selected_set
            
            if not candidate_blocks:
                # Fallback: pick any block covering a random uncovered t
                ti = seed_rng.choice(uncovered_idx.tolist())
                best_block = seed_rng.choice(t_to_blocks[ti])
            else:
                # Score candidates by dual weight
                best_block = -1
                best_weight = -1
                for bi in candidate_blocks:
                    w = 0.0
                    for ti in block_cover[bi]:
                        if not covered_mask[ti]:
                            w += dual_prices[ti]
                    if w > best_weight:
                        best_weight = w
                        best_block = bi
                
                if best_block == -1:
                    ti = seed_rng.choice(uncovered_idx.tolist())
                    best_block = seed_rng.choice(t_to_blocks[ti])
            
            # Select the block
            selected_blocks.append(best_block)
            
            # Update covered mask and coverage counts
            for ti in block_cover[best_block]:
                if not covered_mask[ti]:
                    covered_mask[ti] = True
                    num_uncovered -= 1
                coverage_counts[ti] += 1
            
            # Update dual prices: decrease prices for newly covered subsets,
            # increase prices for subsets that are still uncovered but "rare"
            # Simple heuristic: decay prices of covered, boost uncovered slightly
            for ti in block_cover[best_block]:
                if covered_mask[ti]:
                    dual_prices[ti] *= 0.8  # Decay covered
            # For uncovered subsets, boost those with fewer covering blocks
            # But this is expensive, so just do a global slight boost
            # Actually, let's not modify uncovered prices here for speed
            
        return selected_blocks

    def prune_redundant(blocks):
        """Remove blocks that don't uniquely cover any t-subset."""
        current_blocks = list(blocks)
        if not current_blocks:
            return current_blocks
            
        counts = np.zeros(N_t, dtype=np.int32)
        for bi in current_blocks:
            for ti in block_cover[bi]:
                counts[ti] += 1
                
        changed = True
        while changed:
            changed = False
            unique_counts = {}
            for bi in current_blocks:
                uc = 0
                for ti in block_cover[bi]:
                    if counts[ti] == 1:
                        uc += 1
                unique_counts[bi] = uc
                
            removable = [bi for bi in current_blocks if unique_counts[bi] == 0]
            if removable:
                for bi in removable:
                    current_blocks.remove(bi)
                    for ti in block_cover[bi]:
                        counts[ti] -= 1
                changed = True
                
        return current_blocks

    def local_search_optimize(blocks, rng, max_steps=3000):
        """
        Swap-based local search: try to replace blocks to enable further pruning.
        """
        current_blocks = list(blocks)
        if not current_blocks:
            return current_blocks
            
        counts = np.zeros(N_t, dtype=np.int32)
        for bi in current_blocks:
            for ti in block_cover[bi]:
                counts[ti] += 1
                
        best_len = len(current_blocks)
        best_solution = list(current_blocks)
        
        for step in range(max_steps):
            if time.time() > deadline - 3.0:
                break
                
            if not current_blocks:
                break
                
            bi_idx = rng.randrange(len(current_blocks))
            bi = current_blocks[bi_idx]
            
            # Find unique t-subsets covered by bi
            unique_ts = []
            for ti in block_cover[bi]:
                if counts[ti] == 1:
                    unique_ts.append(ti)
                    
            if not unique_ts:
                # Redundant block! Remove it.
                current_blocks.pop(bi_idx)
                for ti in block_cover[bi]:
                    counts[ti] -= 1
                if len(current_blocks) < best_len:
                    best_len = len(current_blocks)
                    best_solution = list(current_blocks)
                continue
                
            # Find a replacement block that covers all unique_ts
            if len(unique_ts) == 1:
                candidates = set(t_to_blocks[unique_ts[0]])
            else:
                min_len = float('inf')
                min_ti = unique_ts[0]
                for ti in unique_ts:
                    if len(t_to_blocks[ti]) < min_len:
                        min_len = len(t_to_blocks[ti])
                        min_ti = ti
                candidates = set(t_to_blocks[min_ti])
                for ti in unique_ts[1:]:
                    candidates &= set(t_to_blocks[ti])
                    
            candidates.discard(bi)
            if not candidates:
                continue
                
            # Pick a candidate that maximizes coverage of low-count subsets
            new_bi = rng.choice(list(candidates))
            
            # Perform swap
            for ti in block_cover[bi]:
                counts[ti] -= 1
            for ti in block_cover[new_bi]:
                counts[ti] += 1
            current_blocks[bi_idx] = new_bi
            
            # Try to prune affected blocks
            affected = set()
            for ti in block_cover[new_bi]:
                affected.update(t_to_blocks[ti])
            affected = [b for b in affected if b in current_blocks]
            
            for b in affected:
                is_red = True
                for ti in block_cover[b]:
                    if counts[ti] == 1:
                        is_red = False
                        break
                if is_red:
                    current_blocks.remove(b)
                    for ti in block_cover[b]:
                        counts[ti] -= 1
                    if len(current_blocks) < best_len:
                        best_len = len(current_blocks)
                        best_solution = list(current_blocks)
                        
        return best_solution

    # Main search: multi-start with dual-guided construction + local search
    best_solution = None
    best_len = float('inf')
    rng = random.Random(42)
    
    # Phase 1: Dual-guided construction with multiple restarts
    num_restarts = 15
    for i in range(num_restarts):
        if time.time() > deadline - 5.0:
            break
        local_rng = random.Random(rng.randint(0, 10**6))
        blocks = dual_guided_cover(local_rng)
        blocks = prune_redundant(blocks)
        
        if len(blocks) < best_len:
            best_len = len(blocks)
            best_solution = blocks
    
    # Phase 2: Local search on the best solution
    if best_solution is not None:
        local_rng = random.Random(rng.randint(0, 10**6))
        optimized = local_search_optimize(best_solution, local_rng, max_steps=4000)
        if len(optimized) < best_len:
            best_len = len(optimized)
            best_solution = optimized
    
    # Fallback
    if best_solution is None:
        local_rng = random.Random(123)
        best_solution = dual_guided_cover(local_rng)
        best_solution = prune_redundant(best_solution)
    
    return [list(all_blocks[bi]) for bi in best_solution]
