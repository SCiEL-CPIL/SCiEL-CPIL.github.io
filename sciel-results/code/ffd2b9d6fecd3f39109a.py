"""Rewritten program for covering design C(v,k,t) with greedy + local search + redundancy removal."""
# EVOLVE-BLOCK-START
import time
import random
import itertools
import numpy as np

def construct(v, k, t):
    """
    Construct a covering design C(v, k, t): a set of k-subsets (blocks) of {0..v-1}
    such that every t-subset is contained in at least one block.
    Minimize the number of blocks.
    """
    deadline = time.time() + 110.0  # Safety margin before 120s limit
    
    # 1. Precompute all t-subsets and assign an index to each
    all_t_subsets = list(itertools.combinations(range(v), t))
    n_t = len(all_t_subsets)
    t_subset_index = {ts: i for i, ts in enumerate(all_t_subsets)}
    
    # 2. Precompute the bitmask for every possible k-subset (block)
    all_k_subsets = list(itertools.combinations(range(v), k))
    n_blocks = len(all_k_subsets)
    
    # Build block_masks as a list of Python ints (bitmasks)
    block_masks = []
    for block in all_k_subsets:
        mask = 0
        for ts in itertools.combinations(block, t):
            idx = t_subset_index[ts]
            mask |= (1 << idx)
        block_masks.append(mask)
    
    def popcount(x):
        if hasattr(x, 'bit_count'):
            return x.bit_count()
        return bin(x).count('1')

    def run_greedy(seed):
        rng = random.Random(seed)
        unc = (1 << n_t) - 1
        sel = []
        sel_idx_set = set()
        
        while unc:
            if time.time() > deadline:
                return None
            
            best_gain = -1
            best_indices = []
            
            for i, mask in enumerate(block_masks):
                if i in sel_idx_set:
                    continue
                x = unc & mask
                if x == 0:
                    continue
                gain = popcount(x)
                
                if gain > best_gain:
                    best_gain = gain
                    best_indices = [i]
                elif gain == best_gain and gain > 0:
                    best_indices.append(i)
            
            if not best_indices:
                break
            
            chosen_idx = rng.choice(best_indices)
            
            sel.append(all_k_subsets[chosen_idx])
            sel_idx_set.add(chosen_idx)
            unc &= ~block_masks[chosen_idx]
            
            if time.time() > deadline - 5.0:
                return None
        
        return sel

    def remove_redundant(blocks):
        """Remove blocks that don't cover any unique t-subset."""
        if not blocks:
            return blocks
        
        # Count how many blocks cover each t-subset
        t_counts = [0] * n_t
        for block in blocks:
            block_set = set(block)
            for ts in itertools.combinations(block, t):
                idx = t_subset_index[ts]
                t_counts[idx] += 1
        
        # Keep only blocks that cover at least one t-subset uniquely
        kept = []
        for block in blocks:
            block_set = set(block)
            has_unique = False
            for ts in itertools.combinations(block, t):
                idx = t_subset_index[ts]
                if t_counts[idx] == 1:
                    has_unique = True
                    break
            if has_unique:
                kept.append(block)
        
        return kept

    def local_search(blocks, max_time=30.0):
        """Try to improve by removing and replacing blocks."""
        if not blocks:
            return blocks
        
        start_time = time.time()
        current = list(blocks)
        
        # Try removing each block and see if coverage is still complete
        for i in range(len(current)):
            if time.time() - start_time > max_time:
                break
            
            # Try removing block i
            candidate = current[:i] + current[i+1:]
            
            # Check if coverage is still complete
            covered = set()
            for block in candidate:
                block_set = set(block)
                for ts in itertools.combinations(block, t):
                    covered.add(ts)
            
            if len(covered) == n_t:
                current = candidate
                # Restart the loop since indices changed
                i = 0
        
        return current

    best_blocks = None
    best_count = float('inf')
    
    # First run: deterministic (seed=0)
    res = run_greedy(0)
    if res:
        res = remove_redundant(res)
        best_blocks = res
        best_count = len(res)
    
    # Try more seeds if time allows
    for seed in range(1, 50):
        if time.time() > deadline - 15.0:
            break
        res = run_greedy(seed)
        if res:
            res = remove_redundant(res)
            if len(res) < best_count:
                best_blocks = res
                best_count = len(res)
    
    # Local search on the best solution
    if best_blocks is not None and time.time() < deadline - 10.0:
        improved = local_search(best_blocks, max_time=min(30.0, deadline - time.time() - 5.0))
        if len(improved) < best_count:
            best_blocks = improved
            best_count = len(best_blocks)
    
    if best_blocks is None:
        # Fallback: should not happen if first run succeeds
        return []

    return best_blocks
# EVOLVE-BLOCK-END
