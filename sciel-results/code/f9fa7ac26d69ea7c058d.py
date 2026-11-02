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
    
    def local_search(blocks, time_limit):
        """Try to reduce block count by swapping blocks."""
        if not blocks:
            return blocks
        
        # Compute coverage count for each t-subset
        coverage_count = {}
        for b in blocks:
            for sub in itertools.combinations(b, t):
                coverage_count[sub] = coverage_count.get(sub, 0) + 1
        
        improved = True
        while improved:
            improved = False
            if time.time() > time_limit:
                break
            
            # Try removing each block and see if we can fix coverage
            for i in range(len(blocks)):
                if time.time() > time_limit:
                    break
                
                b = blocks[i]
                # Check if removing b leaves any uncovered subset
                newly_uncovered = []
                for sub in itertools.combinations(b, t):
                    if coverage_count[sub] == 1:
                        newly_uncovered.append(sub)
                
                if not newly_uncovered:
                    # Block is redundant, remove it
                    for sub in itertools.combinations(b, t):
                        coverage_count[sub] -= 1
                    blocks.pop(i)
                    improved = True
                    break
                else:
                    # Try to find a replacement block that covers all newly_uncovered
                    # and doesn't create new issues
                    # We need a block that contains at least one newly_uncovered subset
                    # and covers all of them
                    
                    # Get all points in newly_uncovered subsets
                    candidate_points = set()
                    for sub in newly_uncovered:
                        candidate_points.update(sub)
                    
                    # We need to find a k-block that covers all newly_uncovered subsets
                    # This means for each newly_uncovered subset, it must be contained in the block
                    # So the block must contain the union of all newly_uncovered subsets
                    # But that union might have more than k elements
                    
                    # Alternative: try all blocks that contain at least one element from each newly_uncovered
                    # This is expensive. Instead, try a heuristic:
                    # Pick a newly_uncovered subset and try blocks containing it
                    
                    if len(newly_uncovered) <= 3:  # Only try if few uncovered
                        # Try all blocks that contain the first newly_uncovered subset
                        target = newly_uncovered[0]
                        remaining_points = [p for p in range(v) if p not in target]
                        
                        best_replacement = None
                        best_gain = -1
                        
                        for combo in itertools.combinations(remaining_points, k - t):
                            candidate_block = tuple(sorted(target + combo))
                            
                            # Check if this block covers all newly_uncovered
                            cand_set = set()
                            for sub in itertools.combinations(candidate_block, t):
                                cand_set.add(sub)
                            
                            if all(sub in cand_set for sub in newly_uncovered):
                                # Calculate gain: how many currently covered subsets would be lost
                                # vs how many newly covered
                                gain = len(newly_uncovered)
                                # Subtract subsets that would become uncovered if we remove b
                                # (but we're not removing b, we're adding a replacement)
                                # Actually, we want to add a block that covers newly_uncovered
                                # The gain is the number of newly_uncovered it covers
                                
                                if gain > best_gain:
                                    best_gain = gain
                                    best_replacement = candidate_block
                        
                        if best_replacement is not None:
                            # Add the replacement block
                            new_block = list(best_replacement)
                            for sub in itertools.combinations(new_block, t):
                                coverage_count[sub] = coverage_count.get(sub, 0) + 1
                            blocks.append(new_block)
                            improved = True
                            break
        
        return blocks
    
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
            
            # Try local search to reduce block count further
            if time.time() < deadline - 10:  # Leave some time buffer
                blocks = local_search(blocks, deadline - 5)
            
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
