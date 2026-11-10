import time
import random
from itertools import combinations

def is_valid_set(S):
    """
    Check if S is a valid set.
    No element a in S divides the sum of any 3 distinct other elements.
    """
    S_list = list(S)
    n = len(S_list)
    if n < 4:
        return True
    
    # Precompute sums of 3 distinct elements? 
    # No, the condition is: for each a in S, a does NOT divide sum(b,c,d) for any b,c,d in S\{a}.
    
    for a in S_list:
        others = [x for x in S_list if x != a]
        # Check all combinations of 3 from others
        for b, c, d in combinations(others, 3):
            s = b + c + d
            if s % a == 0:
                return False
    return True

def greedy_construct(order):
    """
    Construct a set by greedily adding elements in the given order.
    """
    selected = []
    for x in order:
        candidate = selected + [x]
        if is_valid_set(candidate):
            selected.append(x)
    return selected

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin for 90s limit
    
    best_set = []
    
    # 1. Generate initial candidates
    # Strategy 1: Descending
    order_desc = list(range(192, 0, -1))
    s1 = greedy_construct(order_desc)
    
    # Strategy 2: Ascending
    order_asc = list(range(1, 193))
    s2 = greedy_construct(order_asc)
    
    # Strategy 3: Random permutations (a few)
    random.seed(42)
    for _ in range(10):
        order_rand = list(range(1, 193))
        random.shuffle(order_rand)
        s_rand = greedy_construct(order_rand)
        if len(s_rand) > len(best_set):
            best_set = s_rand
            
    if len(s1) > len(best_set):
        best_set = s1
    if len(s2) > len(best_set):
        best_set = s2
        
    # 2. Local Search: Try to improve best_set
    # We want to increase the size.
    # Move: Remove R (subset of S), Add A (subset of complement) such that |A| > |R| and new set is valid.
    # Start with R of size 1, A of size 2.
    
    current_set = set(best_set)
    full_range = set(range(1, 193))
    
    improved = True
    while improved and time.time() < deadline:
        improved = False
        current_set = set(best_set)
        complement = full_range - current_set
        
        # Try Remove 1, Add 2
        # For each x in current_set, remove it, then try to add two elements from complement
        # This is O(N^3) roughly. N=48. 48 * (144^2) ~ 1M checks.
        # Each check is is_valid_set which is ~700k ops. Too slow for 1M checks.
        # We need a faster validity check or a smarter search.
        
        # Optimized is_valid_set for incremental addition?
        # Or just check the specific constraints affected by the change.
        
        # Let's try a simpler local search first:
        # Try to add any single element from complement.
        added = False
        for y in complement:
            if time.time() > deadline: break
            if is_valid_set(current_set | {y}):
                current_set.add(y)
                best_set = list(current_set)
                improved = True
                added = True
                # Restart loop to check for more additions
                break
        
        if added:
            continue
            
        # If no single addition, try Remove 1, Add 2.
        # To speed up, we can pre-check.
        # If we remove x, the set becomes S' = S \ {x}.
        # We want to find y1, y2 in complement such that S' U {y1, y2} is valid.
        
        # This is still expensive. Let's limit the scope of the search.
        # Only try a few random swaps per iteration?
        # Or iterate through x in current_set, and for each, try a few pairs (y1, y2).
        
        # Given the time constraint, let's stick to single additions if possible.
        # If stuck, do a limited number of R1A2 moves.
        
        moves_made = 0
        for x in list(current_set):
            if time.time() > deadline: break
            if moves_made > 20: break # Limit swaps per pass
            
            S_minus_x = current_set - {x}
            
            # Try to find y1, y2 in complement
            # We can iterate y1, y2
            # To save time, shuffle complement and take first few
            comp_list = list(complement)
            random.shuffle(comp_list)
            
            found_pair = False
            for i in range(min(50, len(comp_list))): # Check first 50 candidates for y1
                y1 = comp_list[i]
                S_temp = S_minus_x | {y1}
                # Check if S_temp is valid? No, we need to add y2 too.
                # But S_temp must be valid *before* adding y2? No, the final set must be valid.
                # However, if S_temp is already invalid, adding y2 won't fix it (constraints are monotonic? 
                # No, adding an element can only ADD constraints. It cannot remove existing violations.
                # So if S_temp is invalid, S_temp U {y2} will also be invalid.
                # Thus, we can prune: only consider y1 such that S_minus_x U {y1} is valid.
                
                if not is_valid_set(S_temp):
                    continue
                
                # Now find y2
                for j in range(min(50, len(comp_list))):
                    if i == j: continue
                    y2 = comp_list[j]
                    candidate_set = S_temp | {y2}
                    if is_valid_set(candidate_set):
                        # Success
                        current_set = candidate_set
                        best_set = list(current_set)
                        improved = True
                        moves_made += 1
                        found_pair = True
                        break
                if found_pair:
                    break
        
        if improved:
            # Restart the while loop to check for single additions again
            continue

    return best_set
