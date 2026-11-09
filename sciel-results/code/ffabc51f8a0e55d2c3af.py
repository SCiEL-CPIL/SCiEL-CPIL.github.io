import time
from itertools import combinations
from functools import lru_cache

# Precomputation
POINTS = 12
BLOCK_SIZE = 6
SUBSET_SIZE = 5

# Generate all 6-subsets
all_blocks = [frozenset(c) for c in combinations(range(POINTS), BLOCK_SIZE)]

# Map each 5-subset to the list of 6-subsets containing it
blocks_containing_5set = {}
# Map each 6-subset to its 5-subsets
subsets_of_block = {}

for b in all_blocks:
    subs = list(combinations(b, SUBSET_SIZE))
    froz_subs = [frozenset(s) for s in subs]
    subsets_of_block[b] = froz_subs
    for s in froz_subs:
        if s not in blocks_containing_5set:
            blocks_containing_5set[s] = []
        blocks_containing_5set[s].append(b)

# List of all 5-subsets for iteration
all_5sets = list(blocks_containing_5set.keys())

def construct():
    start_time = time.time()
    deadline = 110.0 # 10 seconds safety margin
    
    covered = set()
    selected_blocks = []
    
    # We will use a recursive backtracking with MRV heuristic
    # To avoid deep recursion issues or overhead, we can use a stack or just recursion (depth 132 is fine)
    
    def is_valid_block(b, covered):
        for s in subsets_of_block[b]:
            if s in covered:
                return False
        return True

    def find_mrv_5set(covered):
        min_count = float('inf')
        best_5set = None
        best_candidates = []
        
        for s in all_5sets:
            if s in covered:
                continue
            
            # Count valid blocks containing s
            valids = []
            for b in blocks_containing_5set[s]:
                if is_valid_block(b, covered):
                    valids.append(b)
            
            if len(valids) == 0:
                return None, [] # Fail state
            
            if len(valids) < min_count:
                min_count = len(valids)
                best_5set = s
                best_candidates = valids
                
                if min_count == 1:
                    break # Can't get better than 1
            
        return best_5set, best_candidates

    def backtrack(covered, selected_blocks, depth):
        if time.time() > deadline:
            return False # Timeout, handled by outer function returning best effort? 
                         # Actually, if we time out, we should return the current best.
                         # But the function must return a list. 
                         # We'll use a non-local variable to store the best solution found.
        
        if len(selected_blocks) == 132:
            return True
        
        # If we have reached a point where no more blocks can be added, but we haven't reached 132,
        # this is a dead end.
        
        s, candidates = find_mrv_5set(covered)
        
        if s is None:
            return False
        
        for b in candidates:
            # Add b
            new_covered = set(covered)
            new_covered.update(subsets_of_block[b])
            new_selected = selected_blocks + [b]
            
            if backtrack(new_covered, new_selected, depth + 1):
                # Update global best if we found a solution (should be 132)
                # For now, just return True to unwind
                return True
            
        return False

    # Initial call
    # To break symmetry, we can fix the first block.
    # The all-zero block {0,1,2,3,4,5} is a good candidate.
    initial_block = frozenset({0, 1, 2, 3, 4, 5})
    initial_covered = set(subsets_of_block[initial_block])
    initial_selected = [initial_block]
    
    # We need to handle the case where the search might be slow.
    # If backtrack returns False, we might have a partial solution.
    # However, for S(5,6,12), it should find it.
    
    # Let's store the best solution found in case of timeout or failure to complete 132.
    # But the problem says "Maximize the number of blocks".
    # If we can't find 132, we should return the largest packing we found.
    # The backtrack above only returns True if 132 is found.
    # We need to modify it to track the best partial solution.
    
    best_solution = [initial_block]
    
    def backtrack_best(covered, selected_blocks, depth):
        nonlocal best_solution
        
        if time.time() > deadline:
            return False
        
        if len(selected_blocks) > len(best_solution):
            best_solution = list(selected_blocks)
            
        if len(selected_blocks) == 132:
            return True
        
        s, candidates = find_mrv_5set(covered)
        
        if s is None:
            return False
        
        for b in candidates:
            new_covered = set(covered)
            new_covered.update(subsets_of_block[b])
            new_selected = selected_blocks + [b]
            
            if backtrack_best(new_covered, new_selected, depth + 1):
                return True
            
        return False
    
    backtrack_best(initial_covered, initial_selected, 1)
    
    # Convert frozensets to lists for the required output format
    # The task asks for "list of distinct 6-subsets". Lists of integers are standard.
    return [list(b) for b in best_solution]
