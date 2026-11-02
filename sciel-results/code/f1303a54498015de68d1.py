import itertools
import time
import random

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 110  # 10s buffer for safety
    
    # Pre-compute all t-subsets
    all_t_subsets = list(itertools.combinations(range(v), t))
    all_t_set = set(all_t_subsets)
    
    # Map each t-subset to an index for efficient bitmask operations
    subset_to_idx = {sub: i for i, sub in enumerate(all_t_subsets)}
    n_subsets = len(all_t_subsets)
    
    # Pre-compute for each possible k-block, the bitmask of t-subsets it covers
    # But there are C(24,7) = 346104 blocks, which is a lot to precompute.
    # Instead, we'll compute on the fly but use bitmasks for the uncovered set.
    
    # Use Python integers as bitmasks (arbitrary precision)
    # Bit i is set if t-subset i is uncovered
    
    def block_mask(block):
        """Compute bitmask of t-subsets covered by a block."""
        mask = 0
        for sub in itertools.combinations(block, t):
            mask |= (1 << subset_to_idx[sub])
        return mask
    
    best_blocks = None
    best_score = float('inf')
    
    # We will run the greedy algorithm multiple times with different randomizations
    for run in range(100): # Cap runs, break on time
        if time.time() > deadline:
            break
            
        # uncovered_mask: bit i set means t-subset i is uncovered
        uncovered_mask = (1 << n_subsets) - 1
        blocks = []
        
        # Greedy construction with bitmask
        while uncovered_mask:
            if time.time() > deadline:
                break
            
            # Pick a random uncovered subset to base the block on
            # Find an index where bit is set
            # Use bit_length to find highest set bit, then randomize
            # For speed, just pick a random position and check if set
            target_idx = -1
            for _ in range(100):  # Try up to 100 times to find a set bit
                pos = random.randint(0, n_subsets - 1)
                if (uncovered_mask >> pos) & 1:
                    target_idx = pos
                    break
            if target_idx == -1:
                # Fallback: find the lowest set bit
                temp = uncovered_mask
                target_idx = (temp & -temp).bit_length() - 1
            
            target = all_t_subsets[target_idx]
            
            # Find all k-blocks containing 'target'
            remaining_points = [p for p in range(v) if p not in target]
            
            best_block = None
            best_gain = -1
            
            # Iterate through combinations of remaining points to form blocks
            for combo in itertools.combinations(remaining_points, k - t):
                candidate_block = tuple(sorted(target + combo))
                
                # Calculate gain using bitmask
                cand_mask = block_mask(candidate_block)
                gain = bin(uncovered_mask & cand_mask).count('1')
                
                if gain > best_gain:
                    best_gain = gain
                    best_block = candidate_block
            
            if best_block is None:
                break # Should not happen
            
            blocks.append(list(best_block))
            
            # Update uncovered mask
            best_mask = block_mask(best_block)
            uncovered_mask &= ~best_mask
        
        if not uncovered_mask: # Valid covering found
            # Post-processing: Remove redundant blocks
            coverage_count = {}
            for b in blocks:
                for sub in itertools.combinations(b, t):
                    coverage_count[sub] = coverage_count.get(sub, 0) + 1
            
            changed = True
            while changed:
                changed = False
                new_blocks = []
                for b in blocks:
                    # Check if b covers any subset uniquely
                    is_redundant = True
                    for sub in itertools.combinations(b, t):
                        if coverage_count[sub] == 1:
                            is_redundant = False
                            break
                    
                    if is_redundant:
                        # Remove b
                        changed = True
                        for sub in itertools.combinations(b, t):
                            coverage_count[sub] -= 1
                    else:
                        new_blocks.append(b)
                
                blocks = new_blocks
                
                if time.time() > deadline:
                    break
            
            if len(blocks) < best_score:
                best_score = len(blocks)
                best_blocks = blocks.copy()
            
    if best_blocks is None:
        # Fallback: deterministic greedy
        uncovered_mask = (1 << n_subsets) - 1
        blocks = []
        while uncovered_mask:
            # Find the lowest set bit
            temp = uncovered_mask
            target_idx = (temp & -temp).bit_length() - 1
            target = all_t_subsets[target_idx]
            
            remaining_points = [p for p in range(v) if p not in target]
            best_block = None
            best_gain = -1
            for combo in itertools.combinations(remaining_points, k - t):
                candidate_block = tuple(sorted(target + combo))
                cand_mask = block_mask(candidate_block)
                gain = bin(uncovered_mask & cand_mask).count('1')
                if gain > best_gain:
                    best_gain = gain
                    best_block = candidate_block
            blocks.append(list(best_block))
            best_mask = block_mask(best_block)
            uncovered_mask &= ~best_mask
        best_blocks = blocks

    return best_blocks
