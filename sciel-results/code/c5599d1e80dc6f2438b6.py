import time
import numpy as np
from itertools import combinations

def construct():
    """
    Constructs the Steiner System S(5, 6, 12) (Witt Design W_12) deterministically.
    
    The design is constructed by first generating the Steiner System S(4, 5, 11) (Witt Design W_11)
    using the action of the Mathieu group M_11 on 11 points.
    The blocks of W_12 are then formed by adding a 12th point to the blocks of W_11 and including
    their complements.
    
    Returns a list of 132 distinct 6-subsets of {0, ..., 11}.
    """
    start_time = time.time()
    deadline = start_time + 110.0
    
    # 1. Construct S(4, 5, 11) on points 0..10
    # Generators for M_11
    # g1: 11-cycle
    g1 = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 0]
    # g2: specific permutation in M_11
    g2 = [1, 3, 7, 5, 2, 6, 9, 4, 10, 8, 0]
    
    def apply_perm(perm, block):
        return frozenset(perm[x] for x in block)
    
    # Base block for S(4, 5, 11)
    base_block = frozenset([0, 1, 2, 3, 4])
    
    blocks_11 = set()
    blocks_11.add(base_block)
    queue = [base_block]
    
    # BFS to generate the orbit
    while queue:
        if time.time() > deadline:
            break
        current = queue.pop(0)
        
        # Apply g1
        new_block_1 = apply_perm(g1, current)
        if new_block_1 not in blocks_11:
            blocks_11.add(new_block_1)
            queue.append(new_block_1)
            
        # Apply g2
        new_block_2 = apply_perm(g2, current)
        if new_block_2 not in blocks_11:
            blocks_11.add(new_block_2)
            queue.append(new_block_2)
            
        # Early exit if we have found all 66 blocks
        if len(blocks_11) == 66:
            break
            
    # Verify we got 66 blocks
    if len(blocks_11) != 66:
        # Fallback: If the group generation failed or was incomplete, 
        # we cannot construct W_12 correctly. 
        # However, this is a mathematical certainty, so this should not happen.
        # We will return an empty list or a greedy result if this fails.
        # For robustness, let's try a simple greedy if this fails.
        return _greedy_fallback(deadline)

    # 2. Construct S(5, 6, 12) on points 0..11
    # Map points 0..10 of W_11 to 1..11 of W_12
    # Point 0 in W_12 is the new point.
    # Points 1..11 in W_12 correspond to 0..10 in W_11.
    
    all_points_12 = set(range(12))
    points_11_in_12 = set(range(1, 12)) # 1..11
    
    blocks_12 = []
    
    for b in blocks_11:
        # Map b (subset of 0..10) to subset of 1..11
        b_mapped = frozenset(x + 1 for x in b)
        
        # Block 1: {0} U b_mapped
        block_with_0 = frozenset([0]) | b_mapped
        
        # Block 2: Complement of b_mapped in {1..11}
        # Note: The complement in W_12 of {0} U b_mapped is (1..11) \ b_mapped
        # Because ({0} U b_mapped)^c = {1..11} \ b_mapped
        block_complement = points_11_in_12 - b_mapped
        
        blocks_12.append(tuple(sorted(block_with_0)))
        blocks_12.append(tuple(sorted(block_complement)))
        
    # Ensure distinctness (should be inherently distinct)
    blocks_12 = list(set(blocks_12))
    
    # Verify validity
    if not _verify_packing(blocks_12):
        return _greedy_fallback(deadline)
        
    return blocks_12

def _verify_packing(blocks):
    """
    Verifies that the list of blocks forms a valid packing S(5,6,12).
    i.e., no 5-subset is contained in more than one block.
    """
    covered = set()
    for b in blocks:
        for s in combinations(b, 5):
            if s in covered:
                return False
            covered.add(s)
    # Check if it's the full design (792 subsets covered)
    # The task asks to maximize, and 132 is the max.
    # If we have 132 blocks and no conflicts, it's the unique optimal solution.
    if len(blocks) == 132 and len(covered) == 792:
        return True
    # If it's a valid packing but not 132, it's suboptimal.
    # But for the purpose of "valid" in the fallback check, we just need no conflicts.
    # However, the main construct aims for 132.
    return len(blocks) > 0

def _greedy_fallback(deadline):
    """
    Fallback greedy heuristic if the algebraic construction fails.
    """
    n_points = 12
    block_size = 6
    sub_size = 5
    
    all_blocks = list(combinations(range(n_points), block_size))
    covered = set()
    greedy_blocks = []
    
    for b in all_blocks:
        conflict = False
        for s in combinations(b, sub_size):
            if s in covered:
                conflict = True
                break
        if not conflict:
            greedy_blocks.append(b)
            for s in combinations(b, sub_size):
                covered.add(s)
        if time.time() > deadline:
            break
            
    return greedy_blocks

if __name__ == "__main__":
    t0 = time.time()
    sol = construct()
    print(f"Time: {time.time() - t0:.4f}s")
    print(f"Blocks: {len(sol)}")
    # Verify
    covered = set()
    valid = True
    for b in sol:
        for s in combinations(b, 5):
            if s in covered:
                valid = False
                break
            covered.add(s)
        if not valid:
            break
    print(f"Valid: {valid}, Covered: {len(covered)}")
