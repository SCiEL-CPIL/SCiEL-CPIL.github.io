import time
import sys

def construct():
    start_time = time.time()
    timeout = 110  # Safety margin
    
    # Points are 0..11
    # Represent 6-subsets as bitmasks (12 bits)
    # Represent 5-subsets as bitmasks (12 bits)
    
    # Precompute all 5-subsets and their bitmasks
    # There are C(12,5) = 792 5-subsets
    # There are C(12,6) = 924 6-subsets
    
    from itertools import combinations
    
    # Map 5-subset to index
    # We need to track which 5-subsets are covered.
    # 792 bits can fit in 13 integers of 64 bits, or a Python set of ints.
    # Using a set of 5-subset tuples is slower.
    # Using a bitset is faster.
    
    # Let's use a list of bitmasks for the 5-subsets.
    # Index i corresponds to the i-th 5-subset in lexicographic order.
    five_subsets = []
    five_subset_to_idx = {}
    idx = 0
    for s in combinations(range(12), 5):
        mask = 0
        for p in s:
            mask |= (1 << p)
        five_subsets.append(mask)
        five_subset_to_idx[s] = idx
        idx += 1
        
    num_five = 792
    
    # Precompute for each 6-subset, the list of 5-subset indices it covers
    six_subsets = []
    six_to_fives = []
    for b in combinations(range(12), 6):
        fives = []
        for s in combinations(b, 5):
            fives.append(five_subset_to_idx[s])
        six_subsets.append(b)
        six_to_fives.append(fives)
        
    num_six = 924
    
    # We want to select a subset of 6-subsets (blocks) such that
    # no 5-subset index appears in more than one selected block.
    # Maximize number of blocks.
    
    # This is a maximum set packing problem.
    # Since the optimal solution is known to be 132, we can try to find it.
    
    # Backtracking with symmetry breaking
    # Fix block 0: {0,1,2,3,4,5}
    # Index of {0,1,2,3,4,5} in six_subsets
    # It's the first one in lexicographic order?
    # combinations(range(12), 6) starts with (0,1,2,3,4,5). Yes.
    
    block_0_idx = 0
    covered_fives = set(five_subset_to_idx[s] for s in combinations(six_subsets[block_0_idx], 5))
    
    blocks = [list(six_subsets[block_0_idx])]
    
    # To speed up, we can use a greedy heuristic with backtracking.
    # However, a simple greedy might fail.
    # Given the time limit, a randomized greedy with many restarts might be safer than a full backtracking that could explode.
    # But S(5,6,12) is unique, so backtracking with symmetry breaking should be fast.
    
    # Let's try a simple recursive backtracking.
    # State: current list of blocks, set of covered 5-subsets.
    # To avoid deep recursion, we can use a stack.
    
    # Optimization: 
    # At each step, consider only 6-subsets that don't conflict with covered_fives.
    # Order them by a heuristic (e.g., lexicographic).
    
    # Since 132 is the max, and we start with 1, we need 131 more.
    # Depth is 131. Branching factor is high.
    # Without strong pruning, this is slow.
    
    # Alternative: Use the known structure.
    # The S(5,6,12) is the unique design.
    # I will use a hardcoded list of the 132 blocks if I can generate it reliably in the code.
    # But I can't.
    
    # Let's try a different strategy: 
    # The problem is small enough that a well-optimized greedy with random restarts might hit 132.
    # Or, I can use the fact that the design is related to the icosahedron and generate it via geometry?
    # No, let's stick to combinatorics.
    
    # I will implement a backtracking search that uses the following pruning:
    # 1. If the number of remaining uncovered 5-subsets / 6 < number of blocks needed, prune.
    # 2. Symmetry breaking: 
    #    - Block 1 is fixed.
    #    - Block 2 must contain 0 and 1. 
    #    - Among blocks containing 0 and 1, pick the lexicographically smallest one that doesn't conflict.
    #    - This drastically reduces the search space.
    
    # Let's implement this.
    
    # Precompute list of block indices containing 0 and 1
    blocks_with_01 = [i for i, b in enumerate(six_subsets) if 0 in b and 1 in b]
    
    def backtrack(current_blocks, covered_fives, next_block_idx_constraint=None):
        if time.time() - start_time > timeout:
            return None
        
        if len(current_blocks) == 132:
            return current_blocks[:]
        
        # Pruning: 
        # Max possible blocks = len(current_blocks) + (792 - len(covered_fives)) // 6
        if len(current_blocks) + (792 - len(covered_fives)) // 6 < 132:
            return None
            
        # Determine candidate blocks
        # If we just added a block, we can restrict the next block to break symmetry.
        # For simplicity, we'll just iterate through all valid blocks in lex order.
        # To break symmetry, we can enforce that the next block is "greater" in some sense?
        # No, that's hard.
        
        # Let's just use the heuristic: pick the first valid block in lex order.
        # If it leads to a dead end, backtrack.
        
        for i in range(num_six):
            if i in current_blocks: # We track blocks by index
                continue
                
            fives = six_to_fives[i]
            # Check if any of these fives are already covered
            if any(f in covered_fives for f in fives):
                continue
                
            # Try this block
            current_blocks.append(i)
            covered_fives.update(fives)
            
            result = backtrack(current_blocks, covered_fives)
            if result is not None:
                return result
                
            current_blocks.pop()
            covered_fives.difference_update(fives)
            
        return None

    # Initial call
    # We already have block 0.
    # We need to find 131 more.
    # To make it faster, we can force the second block to contain 0 and 1.
    # In S(5,6,12), every pair is in 30 blocks.
    # Block 0 contains 0,1.
    # We can restrict the search for the second block to those containing 0 and 1.
    
    # However, a simple backtracking over 924 blocks at each step is too slow.
    # We need better pruning.
    
    # Given the complexity, and the fact that the seed program might already be close,
    # I will use a **randomized greedy** approach with many restarts.
    # If it finds 132, it returns it.
    
    import random
    
    best_blocks = []
    
    for attempt in range(1000):
        if time.time() - start_time > timeout:
            break
            
        random.seed(attempt)
        blocks = []
        covered = set()
        
        # Shuffle the order of 6-subsets
        order = list(range(num_six))
        random.shuffle(order)
        
        for i in order:
            fives = six_to_fives[i]
            if not any(f in covered for f in fives):
                blocks.append(six_subsets[i])
                covered.update(fives)
                
        if len(blocks) > len(best_blocks):
            best_blocks = blocks[:]
            if len(best_blocks) == 132:
                return best_blocks
                
    return best_blocks
