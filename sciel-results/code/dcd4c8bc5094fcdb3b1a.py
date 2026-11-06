import time
from itertools import combinations

def construct():
    # Deadline guard
    start_time = time.time()
    deadline = start_time + 110.0  # 10 second margin before 120s limit

    points = range(12)
    all_blocks = list(combinations(points, 6))
    
    # Precompute the 5-subsets for each 6-subset
    # Represent 5-subsets as frozensets for easy set operations
    block_to_5subs = {}
    for b in all_blocks:
        block_to_5subs[b] = [frozenset(s) for s in combinations(b, 5)]

    # Symmetry breaking: The first block is fixed to (0, 1, 2, 3, 4, 5)
    # This reduces the search space significantly.
    first_block = (0, 1, 2, 3, 4, 5)
    initial_covered = set(block_to_5subs[first_block])
    
    # We need to find 132 blocks total.
    # We have 1. Need 131 more.
    
    # Filter candidate blocks:
    # A block is a valid candidate if it doesn't share any 5-subset with existing blocks.
    # Also, for symmetry breaking, we can enforce an ordering on the blocks.
    # E.g., the second block must be lexicographically greater than the first?
    # Or simpler: just iterate through all_blocks and pick valid ones.
    # To avoid finding the same solution multiple times (permutations of blocks),
    # we enforce that blocks are added in lexicographic order.
    
    blocks_found = [first_block]
    covered = initial_covered
    
    # List of all other blocks
    candidates = [b for b in all_blocks if b != first_block]
    
    def backtrack(idx, current_blocks, current_covered):
        if time.time() > deadline:
            return None # Timeout
        
        if len(current_blocks) == 132:
            return current_blocks
        
        # Pruning: If not enough candidates left
        # (Not strictly necessary if we just search, but good)
        
        # Try adding candidates in order
        for i in range(idx, len(candidates)):
            b = candidates[i]
            # Check if b conflicts with current_covered
            b_5subs = block_to_5subs[b]
            if any(s in current_covered for s in b_5subs):
                continue
            
            # If valid, add it
            new_covered = current_covered.union(b_5subs)
            result = backtrack(i + 1, current_blocks + [b], new_covered)
            if result:
                return result
        
        return None

    # Start search
    solution = backtrack(0, blocks_found, covered)
    
    if solution:
        return [list(b) for b in solution]
    else:
        # Fallback: Return what we have (should not happen for S(5,6,12) with good symmetry breaking)
        # But to be safe, return the greedy result if backtracking fails/times out?
        # The prompt says "Maximize". If we fail to find 132, we return less.
        # However, with fixed first block, it should find 132 quickly.
        
        # Fallback to greedy if backtracking fails (unlikely)
        covered_greedy, blocks_greedy = set(), []
        for b in all_blocks:
            subsets = block_to_5subs[b]
            if not any(s in covered_greedy for s in subsets):
                blocks_greedy.append(list(b))
                covered_greedy.update(subsets)
        return blocks_greedy
