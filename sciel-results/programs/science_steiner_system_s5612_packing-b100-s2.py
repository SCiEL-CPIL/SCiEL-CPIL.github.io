import time
from itertools import combinations

def construct():
    # Deadline guard
    start_time = time.time()
    deadline = start_time + 110.0
    
    # We know the optimal number of blocks is 132.
    # We will use a backtracking search with symmetry breaking.
    
    points = range(12)
    all_5_subsets = list(combinations(points, 5))
    
    # Map 5-subsets to indices for fast lookup? Or just use a set of tuples.
    covered = set()
    blocks = []
    
    # Precompute all 6-subsets
    all_6_subsets = list(combinations(points, 6))
    
    # For each 5-subset, which 6-subsets contain it?
    # This is useful for finding candidates.
    # But we can just iterate.
    
    # Symmetry breaking: 
    # The first block must contain {0,1,2,3,4}. 
    # By symmetry, we can assume the 6th element is 5.
    # So Block 0 is {0,1,2,3,4,5}.
    
    first_block = (0, 1, 2, 3, 4, 5)
    first_block_5s = list(combinations(first_block, 5))
    
    # Check if we can start with this. Yes.
    blocks.append(first_block)
    covered.update(first_block_5s)
    
    # Now we need 131 more blocks.
    # We will search for the next block.
    
    def backtrack():
        if len(blocks) == 132:
            return True
        
        if time.time() > deadline:
            return False
            
        # Find the first uncovered 5-subset in lexicographic order
        # This ensures we fill gaps systematically.
        # However, iterating all 792 5-subsets each step is $132 \times 792 \approx 100k$ ops, which is fast.
        
        target_5 = None
        for s in all_5_subsets:
            if s not in covered:
                target_5 = s
                break
                
        if target_5 is None:
            # All covered, but we don't have 132 blocks?
            # This shouldn't happen if we stop at 132.
            return False
            
        # Find all 6-subsets that contain target_5
        candidates = []
        for b in all_6_subsets:
            if set(target_5).issubset(set(b)):
                # Check if this block is valid (no overlap in 5-subsets)
                block_5s = list(combinations(b, 5))
                if not any(s in covered for s in block_5s):
                    candidates.append(b)
                    
        # Try each candidate
        for b in candidates:
            # Add block
            blocks.append(b)
            block_5s = list(combinations(b, 5))
            covered.update(block_5s)
            
            if backtrack():
                return True
                
            # Undo
            blocks.pop()
            covered.difference_update(block_5s)
            
        return False
        
    if backtrack():
        return [list(b) for b in blocks]
    else:
        # Fallback: Return what we have or a greedy result?
        # If backtrack fails, it might be due to time or logic error.
        # But S(5,6,12) exists, so it should find it.
        # If it times out, we return the partial list?
        # The task says "Maximize the number of blocks".
        # Returning a partial list is better than nothing.
        return [list(b) for b in blocks]
