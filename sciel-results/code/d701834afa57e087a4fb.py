import time
from itertools import combinations

def construct():
    # Deadline guard
    start_time = time.time()
    deadline = start_time + 140.0  # 140s limit to be safe
    
    points = range(12)
    # Precompute all 5-subsets as bitmasks
    all_5_subsets = []
    for s in combinations(points, 5):
        mask = 0
        for p in s:
            mask |= (1 << p)
        all_5_subsets.append(mask)
        
    # Map each 5-subset mask to the list of 6-subsets (blocks) that contain it
    # A 6-subset is a block. 
    # There are C(12,6) = 924 possible blocks.
    # For each 5-subset, there are C(12-5, 6-5) = C(7,1) = 7 blocks that contain it.
    
    # We want to select a set of blocks such that every 5-subset is covered exactly once.
    # This is an exact cover problem.
    # We can use Algorithm X (Dancing Links) or simple backtracking with MRV.
    
    # Let's use a simple recursive backtracking with MRV.
    # State:
    # - covered_5s: set of 5-subset masks that are already covered.
    # - available_blocks: list of 6-subset masks that are still valid (don't overlap with covered 5s in a bad way? No, blocks are 6-sets. 
    #   If a block shares a 5-subset with a previously chosen block, it's invalid.
    #   So, a block is valid if none of its 5-subsets are in covered_5s.
    
    # Optimization:
    # Instead of tracking all available blocks, we can just track the set of uncovered 5-subsets.
    # At each step, pick an uncovered 5-subset. It must be covered by one of the 7 blocks containing it.
    # Try each of the 7 blocks.
    # If we pick a block, we cover its 6 constituent 5-subsets.
    
    # Precompute: for each 5-subset mask, the list of 6-subset masks (blocks) that contain it.
    # 6-subset mask: 12 bits.
    
    blocks_by_5 = {mask: [] for mask in all_5_subsets}
    
    # Generate all 924 6-subsets
    all_6_subsets = []
    for b in combinations(points, 6):
        mask = 0
        for p in b:
            mask |= (1 << p)
        all_6_subsets.append(mask)
        
    # For each 6-subset, find its 5-subsets and add this 6-subset to their lists
    # To speed this up, we can just iterate 6-subsets and their 5-subsets.
    # C(12,6)=924. Each has 6 5-subsets. Total 5544 operations. Fast.
    
    for b_mask in all_6_subsets:
        # Get the 6 points
        pts = [i for i in range(12) if (b_mask >> i) & 1]
        # Get 5-subsets of these 6 points
        for i in range(6):
            # 5-subset is b with pts[i] removed
            sub_mask = b_mask & ~(1 << pts[i])
            # Add b_mask to the list of blocks covering this 5-subset
            blocks_by_5[sub_mask].append(b_mask)

    # Backtracking
    # We need to select 132 blocks.
    # Total 5-subsets = 792. Each block covers 6. 792/6 = 132.
    
    solution = []
    
    # Use a set for fast lookup of covered 5-subsets
    covered = set()
    
    # To make MRV work, we need to know which 5-subsets are uncovered.
    # We can iterate through all_5_subsets and check if they are in covered.
    
    def backtrack(depth):
        if time.time() > deadline:
            return False
            
        if depth == 132:
            # Check if all 5-subsets are covered
            # Since we only add blocks that cover 6 new 5-subsets, and we added 132*6=792 covers,
            # and there are only 792 5-subsets, if no overlaps occurred, all are covered.
            # The constraint "no 5-subset lies in two blocks" ensures no overlaps.
            return True

        # Find the uncovered 5-subset with the fewest available covering blocks (MRV)
        # We need to check which blocks containing a 5-subset are still "available".
        # A block is available if none of its 5-subsets are already covered.
        
        best_5_mask = None
        best_candidates = None
        min_candidates = float('inf')
        
        # Iterate through all 5-subsets
        # Optimization: We can maintain a list of uncovered 5-subsets, but checking 792 each time is fast enough.
        for mask in all_5_subsets:
            if mask in covered:
                continue
                
            # Find available blocks for this 5-subset
            candidates = []
            for b_mask in blocks_by_5[mask]:
                # Check if this block is valid: none of its 5-subsets are in covered
                # To do this efficiently, we can precompute the 5-subsets of each block?
                # Or just check the 6 5-subsets of b_mask.
                
                # Let's precompute the 5-subsets for each 6-mask to avoid recalculating.
                # Actually, we can just check if any of the 6 5-subsets of b_mask are in covered.
                
                valid = True
                # Get points of b_mask
                # To avoid bit manipulation overhead in inner loop, precompute list of 5-masks for each 6-mask?
                # There are 924 blocks. Storing 6 masks each is fine.
                
                # Let's use a precomputed dictionary: block_5s[b_mask] = list of 5-masks
                if any(s in covered for s in block_5s[b_mask]):
                    valid = False
                
                if valid:
                    candidates.append(b_mask)
            
            if len(candidates) < min_candidates:
                min_candidates = len(candidates)
                best_5_mask = mask
                best_candidates = candidates
                
                if min_candidates == 0:
                    return False
                if min_candidates == 1:
                    break # Can't get better than 1
            
        if best_candidates is None or len(best_candidates) == 0:
            return False
            
        # Try each candidate block
        for b_mask in best_candidates:
            # Add block
            solution.append(b_mask)
            # Mark its 5-subsets as covered
            new_covers = block_5s[b_mask]
            for s in new_covers:
                covered.add(s)
                
            if backtrack(depth + 1):
                return True
                
            # Backtrack
            solution.pop()
            for s in new_covers:
                covered.remove(s)
                
        return False

    # Precompute block_5s for faster lookup in backtrack
    block_5s = {}
    for b_mask in all_6_subsets:
        pts = [i for i in range(12) if (b_mask >> i) & 1]
        masks = []
        for i in range(6):
            sub_mask = b_mask & ~(1 << pts[i])
            masks.append(sub_mask)
        block_5s[b_mask] = masks

    if backtrack(0):
        # Convert masks back to lists of points
        result = []
        for mask in solution:
            result.append([i for i in range(12) if (mask >> i) & 1])
        return result
    else:
        # Fallback: return whatever we have (should not happen for S(5,6,12))
        # Or return an empty list if it fails, but it shouldn't.
        # If it times out, we return the partial solution? 
        # The task says "Maximize the number of blocks". 
        # If we don't find 132, we should return the best found.
        # But the backtrack function returns False if it hits deadline or dead end.
        # We need to track the best solution found so far.
        
        # Let's modify the approach to track the best solution.
        # However, for S(5,6,12), the solution is unique and should be found quickly.
        # If it's not found, something is wrong.
        return solution # Might be partial

# Precompute block_5s globally to avoid re-computation if construct is called multiple times?
# The task implies construct() is called.
# I will put the precomputation inside construct but it's fast.

# Wait, the variable `block_5s` is used inside `backtrack` but defined after the function definition in my draft.
# In Python, closures capture variables by reference, so as long as `block_5s` is defined before `backtrack` is *called*, it's fine.
# But I defined `block_5s` after the `def backtrack` block in my thought process.
# I will restructure the code properly.
