import numpy as np
import itertools
import time
import random

def construct(v, k, t):
    start_time = time.time()
    time_limit = 145.0  # Safety margin for 150s limit
    
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
    # Using a list of lists for flexibility, then converting to array if needed for speed
    # But for swap logic, we need inverted index: t_subset -> list of blocks
    
    # 1. Forward map: block_idx -> array of t_subset indices
    block_t_indices = np.zeros((n_ks, t_count), dtype=np.int32)
    for i, block in enumerate(all_ks):
        subs = itertools.combinations(block, t)
        for j, sub in enumerate(subs):
            block_t_indices[i, j] = idx_map[sub]
            
    # 2. Inverted map: t_subset_idx -> list of block indices
    # This is crucial for efficient swap search
    t_block_indices = [[] for _ in range(n_ts)]
    for b_idx in range(n_ks):
        subs = block_t_indices[b_idx]
        for t_idx in subs:
            t_block_indices[t_idx].append(b_idx)
    
    # Convert to numpy arrays for faster access if possible, 
    # but lists are fine for small intersections.
    # Let's keep them as lists for now, or convert to np.array for vectorization?
    # np.array is better for intersection checks if we use sets.
    
    best_solution = None
    best_score = float('inf')
    
    def run_greedy(seed):
        random.seed(seed)
        np.random.seed(seed)
        
        covered = np.zeros(n_ts, dtype=bool)
        blocks_indices = []
        remaining = n_ts
        cover_freq = np.zeros(n_ts, dtype=np.int32)
        
        while remaining > 0:
            if time.time() - start_time > time_limit - 10.0:
                return None, None
            
            # Compute gain
            covered_counts = covered[block_t_indices].sum(axis=1)
            gains = t_count - covered_counts
            
            max_gain = np.max(gains)
            if max_gain == 0:
                break
            
            max_gain_blocks = np.where(gains == max_gain)[0]
            
            # Randomize choice among max gain blocks to allow diversity
            # Pick a random subset of top candidates
            if len(max_gain_blocks) > 10:
                candidates = max_gain_blocks[random.sample(range(len(max_gain_blocks)), 10)]
            else:
                candidates = max_gain_blocks
            
            # Simple random choice from candidates, or weighted by rarity
            # Weighted: prefer blocks covering rarer triples
            best_cand = None
            best_score = -1
            for cand in candidates:
                subs = block_t_indices[cand]
                newly = ~covered[subs]
                if np.any(newly):
                    # Score: max_gain + small bonus for rarity
                    bonus = np.sum(1.0 / (1.0 + cover_freq[subs[newly]])) * 0.01
                    s = max_gain + bonus
                    if s > best_score:
                        best_score = s
                        best_cand = cand
            
            if best_cand is None:
                best_cand = candidates[0]
                
            blocks_indices.append(best_cand)
            
            subs_to_cover = block_t_indices[best_cand]
            newly_covered = ~covered[subs_to_cover]
            new_covers = np.sum(newly_covered)
            remaining -= new_covers
            cover_freq[subs_to_cover] += 1
            covered[subs_to_cover] = True
            
        current_blocks = list(blocks_indices)
        
        # Build coverage count
        coverage_count = np.zeros(n_ts, dtype=np.int32)
        for b_idx in current_blocks:
            coverage_count[block_t_indices[b_idx]] += 1
        
        # Local Search: Deletion
        current_blocks = local_search_deletion(current_blocks, coverage_count)
        
        return current_blocks, len(current_blocks)
    
    def local_search_deletion(blocks, coverage_count):
        """Remove redundant blocks."""
        improved = True
        while improved:
            if time.time() - start_time > time_limit - 5.0:
                break
            improved = False
            new_blocks = []
            for b_idx in blocks:
                subs = block_t_indices[b_idx]
                if np.all(coverage_count[subs] > 1):
                    # Redundant
                    coverage_count[subs] -= 1
                    improved = True
                else:
                    new_blocks.append(b_idx)
            blocks = new_blocks
        return blocks
    
    def local_search_swap(blocks, coverage_count, max_iters=1000):
        """
        Try to swap blocks to enable further deletions or just find a different local optimum.
        Strategy: 
        1. Try to delete.
        2. If stuck, pick a random block B_out.
        3. Remove B_out, find uncovered triples U.
        4. If |U| is small, find B_in that covers U.
        5. If B_in exists and is not in blocks, swap.
        6. Repeat.
        """
        blocks_set = set(blocks)
        current_blocks = list(blocks)
        
        for _ in range(max_iters):
            if time.time() - start_time > time_limit - 5.0:
                break
            
            # 1. Try deletion
            improved = False
            new_blocks = []
            for b_idx in current_blocks:
                subs = block_t_indices[b_idx]
                if np.all(coverage_count[subs] > 1):
                    coverage_count[subs] -= 1
                    improved = True
                else:
                    new_blocks.append(b_idx)
            
            if improved:
                current_blocks = new_blocks
                blocks_set = set(current_blocks)
                continue
            
            # 2. Try swap
            if not current_blocks:
                break
                
            # Pick a random block to remove
            b_out = random.choice(current_blocks)
            
            # Temporarily remove
            coverage_count[block_t_indices[b_out]] -= 1
            blocks_set.remove(b_out)
            
            # Find uncovered triples
            uncovered_mask = coverage_count == 0
            if not np.any(uncovered_mask):
                # Deletion was valid, we missed it in the previous loop? 
                # Actually, the previous loop checks ALL blocks. If it didn't delete, 
                # it means for every block, at least one triple has count 1.
                # But if we remove b_out, maybe it becomes valid?
                # Wait, the deletion loop above already tried to delete ANY block.
                # If it didn't delete, it means NO block was fully redundant.
                # So removing b_out MUST leave some triples uncovered.
                pass
            
            uncovered_indices = np.where(uncovered_mask)[0]
            n_unc = len(uncovered_indices)
            
            if n_unc == 0:
                # Should not happen if deletion loop was exhaustive
                current_blocks = list(blocks_set)
                continue
            
            # If too many uncovered, skip swap (too expensive to find B_in)
            if n_unc > 8:
                coverage_count[block_t_indices[b_out]] += 1
                blocks_set.add(b_out)
                continue
            
            # Find candidate B_in
            # B_in must cover ALL uncovered_indices
            # Intersect t_block_indices for each uncovered index
            
            # Get list of blocks covering the first uncovered triple
            candidates = set(t_block_indices[uncovered_indices[0]])
            
            # Intersect with blocks covering other uncovered triples
            for u_idx in uncovered_indices[1:]:
                if not candidates:
                    break
                candidates &= set(t_block_indices[u_idx])
            
            # Filter candidates: must not be in current solution
            valid_candidates = [c for c in candidates if c not in blocks_set]
            
            if valid_candidates:
                # Pick one at random
                b_in = random.choice(valid_candidates)
                
                # Perform swap
                coverage_count[block_t_indices[b_in]] += 1
                blocks_set.add(b_in)
                
                # Update current_blocks list
                # Remove b_out, add b_in
                # Since we removed b_out from set, we just add b_in
                # But we need to update the list structure for next iteration
                current_blocks = list(blocks_set)
                
            else:
                # Revert
                coverage_count[block_t_indices[b_out]] += 1
                blocks_set.add(b_out)
                # If no swap possible for this random b_out, try another one?
                # To avoid getting stuck, we can just continue to next iteration of the loop
                # which will pick a new random b_out.
                
        return current_blocks, coverage_count

    def iterated_local_search(initial_blocks, initial_coverage, n_iters=50):
        """
        Run ILS starting from initial solution.
        """
        current_blocks = list(initial_blocks)
        current_coverage = initial_coverage.copy()
        best_blocks = list(current_blocks)
        best_size = len(current_blocks)
        
        for i in range(n_iters):
            if time.time() - start_time > time_limit - 10.0:
                break
            
            # Local Search (Deletion + Swaps)
            current_blocks, current_coverage = local_search_swap(current_blocks, current_coverage, max_iters=200)
            
            if len(current_blocks) < best_size:
                best_size = len(current_blocks)
                best_blocks = list(current_blocks)
                # Rebuild coverage for best to ensure consistency? 
                # local_search_swap updates coverage_count correctly.
            
            # Perturbation
            if len(current_blocks) > 0:
                # Randomly swap 2-5 blocks
                n_perturb = random.randint(2, min(5, len(current_blocks)))
                for _ in range(n_perturb):
                    if time.time() - start_time > time_limit - 10.0:
                        break
                    b_out = random.choice(current_blocks)
                    # Find any block not in current set
                    # Randomly pick from all_ks
                    b_in = random.randint(0, n_ks - 1)
                    while b_in in current_blocks:
                        b_in = random.randint(0, n_ks - 1)
                    
                    # Swap
                    current_blocks.remove(b_out)
                    current_blocks.append(b_in)
                    
                    # Update coverage
                    current_coverage[block_t_indices[b_out]] -= 1
                    current_coverage[block_t_indices[b_in]] += 1
                    
                    # Check if valid? 
                    # If invalid, the next local_search_swap will try to fix it 
                    # (by swapping back or finding a fix). 
                    # However, if we introduce invalidity, "deletion" in local_search_swap 
                    # might fail, but "swap" logic assumes we start from a valid state 
                    # to find uncovered triples. 
                    # If state is invalid (uncovered triples exist), 
                    # the "uncovered_mask" logic in swap will try to cover them.
                    # So it acts as a repair step.
            
        return best_blocks, best_size

    # Phase 1: Greedy Restart + ILS
    restart_count = 0
    while time.time() - start_time < time_limit - 20.0:
        seed = restart_count
        blocks, score = run_greedy(seed)
        
        if blocks is None:
            break
        
        # Build coverage for ILS
        coverage = np.zeros(n_ts, dtype=np.int32)
        for b_idx in blocks:
            coverage[block_t_indices[b_idx]] += 1
            
        # Run ILS
        best_blocks, best_size = iterated_local_search(blocks, coverage, n_iters=20)
        
        if best_size < best_score:
            best_score = best_size
            best_solution = best_blocks
            
        restart_count += 1
    
    if best_solution is None:
        # Fallback
        blocks, score = run_greedy(0)
        if blocks is not None:
            best_solution = blocks
            best_score = score
        else:
            return []

    final_blocks = [list(all_ks[i]) for i in best_solution]
    
    return final_blocks
