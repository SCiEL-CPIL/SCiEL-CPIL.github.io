"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import itertools
import time
import random
import numpy as np

def construct(v, k, t):
    """
    Constructs a covering design C(v, k, t) using multi-start greedy with efficient reduction.
    """
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin
    
    points = list(range(v))
    all_t_subsets = list(itertools.combinations(points, t))
    N_t = len(all_t_subsets)
    subset_to_id = {s: i for i, s in enumerate(all_t_subsets)}
    
    def greedy_cover(uncovered_mask, seed=None):
        """
        Run one greedy pass.
        uncovered_mask: np.array of bools, length N_t. True if uncovered.
        Returns list of blocks (lists of ints).
        """
        if seed is not None:
            random.seed(seed)
        
        if not np.any(uncovered_mask):
            return []
            
        blocks = []
        unc = uncovered_mask.copy()
        
        while np.any(unc):
            # Pick a random uncovered t-subset to start a new block
            unc_indices = np.where(unc)[0]
            if len(unc_indices) == 0:
                break
            start_idx = unc_indices[random.randint(0, len(unc_indices) - 1)]
            start_subset = all_t_subsets[start_idx]
            
            block = list(start_subset)
            
            while len(block) < k:
                # Find point that maximizes coverage of uncovered t-subsets
                best_gain = -1
                best_point = None
                
                candidates = [x for x in range(v) if x not in block]
                
                # Precompute (t-1) subsets of current block to reuse for all candidates
                block_t_minus_1 = list(itertools.combinations(block, t - 1))
                
                for x in candidates:
                    gain = 0
                    for rest in block_t_minus_1:
                        cand_tuple = tuple(sorted(rest + (x,)))
                        cid = subset_to_id[cand_tuple]
                        if unc[cid]:
                            gain += 1
                    if gain > best_gain:
                        best_gain = gain
                        best_point = x
                        
                if best_point is None:
                    best_point = candidates[0]
                    
                block.append(best_point)
            
            block.sort()
            blocks.append(block)
            
            # Update uncovered mask
            for comb in itertools.combinations(block, t):
                cid = subset_to_id[comb]
                unc[cid] = False
                
        return blocks

    def reduce_blocks(blocks, seed=None):
        """
        Reduce blocks by checking redundancy using coverage counts.
        """
        if not blocks:
            return blocks
            
        if seed is not None:
            random.seed(seed)
            
        reduced = list(blocks)
        
        # Build initial coverage counts
        cover_count = np.zeros(N_t, dtype=np.int32)
        
        # Precompute t-subsets indices for each block
        block_t_subsets = []
        for b in reduced:
            subs = [subset_to_id[comb] for comb in itertools.combinations(b, t)]
            block_t_subsets.append(subs)
            for s in subs:
                cover_count[s] += 1
        
        changed = True
        while changed and time.time() < deadline - 5.0:
            changed = False
            indices = list(range(len(reduced)))
            random.shuffle(indices)
            
            for i in indices:
                if i >= len(reduced):
                    continue
                    
                # Check if reduced[i] is redundant
                subs = block_t_subsets[i]
                
                # Decrement counts
                for s in subs:
                    cover_count[s] -= 1
                
                # Check if any count in subs became 0
                redundant = True
                for s in subs:
                    if cover_count[s] == 0:
                        redundant = False
                        break
                
                if redundant:
                    # Remove block i
                    reduced.pop(i)
                    block_t_subsets.pop(i)
                    changed = True
                    break 
                else:
                    # Restore counts
                    for s in subs:
                        cover_count[s] += 1
            
        return reduced

    best_blocks = None
    best_count = float('inf')
    
    initial_mask = np.ones(N_t, dtype=bool)
    
    restart_count = 0
    while time.time() < deadline - 2.0:
        restart_count += 1
        seed = restart_count * 12345 + 7
        blocks = greedy_cover(initial_mask, seed=seed)
        
        if len(blocks) < best_count:
            best_count = len(blocks)
            best_blocks = blocks
        
        # Try reduction
        if time.time() < deadline - 10.0:
            reduced = reduce_blocks(best_blocks, seed=seed + 1)
            if len(reduced) < best_count:
                best_count = len(reduced)
                best_blocks = reduced

    if best_blocks is None:
        best_blocks = greedy_cover(initial_mask, seed=1)
        
    return best_blocks
# EVOLVE-BLOCK-END
