"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import itertools
import time

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 115.0  # Leave 5s margin for 120s limit
    
    # Map t-subsets to unique integers for fast membership testing
    # We can use a dictionary mapping tuple(sorted(subset)) to id, or just use the tuple as key in a set.
    # Using tuples in a set is fast enough for 11628 elements.
    
    all_subsets = [tuple(sorted(s)) for s in itertools.combinations(range(v), t)]
    uncovered = set(all_subsets)
    
    blocks = []
    
    while uncovered:
        if time.time() > deadline:
            break
            
        # Pick an uncovered subset to start a new block
        # Heuristic: pick one that is "most constrained" or just arbitrary?
        # Arbitrary is fine for greedy.
        target = next(iter(uncovered))
        
        block = list(target)
        
        while len(block) < k:
            if time.time() > deadline:
                break
                
            best_point = -1
            best_gain = -1
            block_set = set(block)
            
            # Pre-compute (t-1) subsets of current block for gain calculation
            # We need to check if P U {x} is in uncovered.
            # P is a (t-1) subset of block.
            t_minus_1_subs = list(itertools.combinations(block, t - 1))
            
            for x in range(v):
                if x in block_set:
                    continue
                
                gain = 0
                for p in t_minus_1_subs:
                    # Check if p U {x} is an uncovered t-subset
                    # p is a tuple of t-1 elements. x is an int.
                    # We need to check if the set {x} U p is in uncovered.
                    # Since uncovered contains sorted tuples, we need to sort p U {x}.
                    # But p is already sorted. We can insert x and check.
                    # Optimization: if x < p[0], new_sub = (x,) + p
                    # if x > p[-1], new_sub = p + (x,)
                    # else, need to find insertion point.
                    # Given small size (t=5), just creating a set and sorting is okay, 
                    # or manual insertion.
                    
                    # Manual insertion for speed
                    new_sub_list = list(p)
                    # Insert x into sorted p
                    inserted = False
                    for i in range(len(new_sub_list)):
                        if x < new_sub_list[i]:
                            new_sub_list.insert(i, x)
                            inserted = True
                            break
                    if not inserted:
                        new_sub_list.append(x)
                    
                    new_sub_tuple = tuple(new_sub_list)
                    
                    if new_sub_tuple in uncovered:
                        gain += 1
                
                if gain > best_gain:
                    best_gain = gain
                    best_point = x
            
            if best_point != -1:
                block.append(best_point)
            else:
                # Should not happen if uncovered is not empty and k > t
                # But if no point adds gain, we might be stuck?
                # Actually, if uncovered is not empty, there is some subset S in uncovered.
                # If S is not subset of block, there is some x in S \ block.
                # Adding x will cover S (if S \ block == {x})?
                # Wait, the logic "gain of x is count of P in C(block, t-1) such that P U {x} in uncovered"
                # is correct for subsets that become covered *by the addition of x* and *were not covered before*.
                # If S is uncovered, and S is not subset of block, let x be an element in S \ block.
                # Then S \ {x} is a subset of block?
                # If |S \ block| > 1, then adding just x doesn't cover S.
                # So gain counts only subsets S where S \ block = {x}.
                # If there are no such subsets for any x, it means all uncovered subsets have at least 2 elements outside the current block.
                # In that case, adding any x will have gain 0?
                # If gain is 0 for all x, we should still pick one to grow the block.
                # We should pick one that is in some uncovered subset.
                if best_gain == 0:
                    # Pick a point that appears in some uncovered subset
                    # To be efficient, just pick the first point from the first uncovered subset not in block
                    for s in uncovered:
                        for x in s:
                            if x not in block_set:
                                best_point = x
                                break
                        if best_point != -1:
                            break
                    if best_point == -1:
                         # Fallback, shouldn't happen
                         best_point = 0 if 0 not in block_set else 1
                
                block.append(best_point)
        
        block.sort()
        blocks.append(block)
        
        # Update uncovered set
        for s in itertools.combinations(block, t):
            uncovered.discard(tuple(sorted(s)))
            
    return blocks
# EVOLVE-BLOCK-END
