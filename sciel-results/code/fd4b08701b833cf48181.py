import time
import random

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 80.0  # Safety margin
    
    # Helper to check if a set has a subset summing to 200 using bitmask
    def is_valid(S):
        mask = 1
        for x in S:
            mask |= (mask << x)
            if (mask >> 200) & 1:
                return False
        return True

    def get_conflict(S):
        """Returns a list of indices in S that sum to 200, or None if valid."""
        # Use DP to track parent or just find one subset
        pass

    # Greedy construction given an order of elements using bitmask
    def greedy_build(order):
        S = []
        mask = 1
        
        for x in order:
            new_mask = mask | (mask << x)
            if not ((new_mask >> 200) & 1):
                mask = new_mask
                S.append(x)
        return S

    # Local search: try to add elements from the complement using bitmask
    def local_search_improve(S):
        S_set = set(S)
        all_elements = set(range(1, 101))
        complement = list(all_elements - S_set)
        improved = True
        current_S = list(S)
        
        # Compute initial mask
        mask = 1
        for x in current_S:
            mask |= (mask << x)
            
        while improved:
            improved = False
            # Shuffle complement to randomize which elements we try first
            random.shuffle(complement)
            for x in complement:
                # Check if adding x to current_S is valid
                new_mask = mask | (mask << x)
                if not ((new_mask >> 200) & 1):
                    current_S.append(x)
                    S_set.add(x)
                    complement.remove(x)
                    mask = new_mask
                    improved = True
        return current_S

    best_set = []
    best_size = -1
    
    # Strategy 1: Seed's greedy (Descending)
    order = list(range(100, 0, -1))
    S = greedy_build(order)
    S = local_search_improve(S)
    if len(S) > best_size:
        best_size = len(S)
        best_set = S.copy()
        
    if time.time() > deadline:
        return best_set if best_set else [100]

    # Strategy 2: Randomized Greedy
    for _ in range(500):
        if time.time() > deadline:
            break
        order = list(range(1, 101))
        random.shuffle(order)
        S = greedy_build(order)
        # Local search improvement
        S = local_search_improve(S)
        if len(S) > best_size:
            best_size = len(S)
            best_set = S.copy()
            
    if time.time() > deadline:
        return best_set if best_set else [100]

    # Strategy 3: Randomized Removal from Full Set
    # This finds a hitting set. We remove elements until no sum 200 exists.
    for _ in range(200):
        if time.time() > deadline:
            break
        S = list(range(1, 101))
        # Loop until valid
        # To find a conflict, we can use a simple DP that also stores the subset
        # But N=100, T=200 is small.
        valid = False
        while not valid:
            # Check validity and find a conflicting subset if invalid
            # DP for subset sum with reconstruction is O(N*T)
            # But we just need one conflict.
            # Simple approach: Randomly sample subsets? No, too slow.
            # Use DP to find if 200 is reachable.
            # If reachable, find which elements were used.
            
            # For speed, we'll just use is_valid first.
            # If invalid, we need to remove an element.
            # Which element?
            # We can find the conflict using a backtracking search limited by the DP path.
            # Or simpler: Just pick a random element to remove? 
            # That's too slow if we don't guide it.
            
            # Let's use a guided removal:
            # 1. Find any subset summing to 200.
            # 2. Remove a random element from that subset.
            
            # Find conflict:
            reachable = [None] * 201  # Store parent (prev_sum, element)
            reachable[0] = (0, None)
            conflict_found = False
            for x in S:
                # Iterate backwards for 0/1
                # We need to copy the state or iterate carefully
                # Since we want to find ONE path, we can do this:
                new_reachable = reachable[:]
                for s in range(200 - x, -1, -1):
                    if reachable[s] is not None:
                        if new_reachable[s + x] is None: # Only set if not already found
                            new_reachable[s + x] = (s, x)
                reachable = new_reachable
                if reachable[200] is not None:
                    conflict_found = True
                    break
                    
            if not conflict_found:
                valid = True
            else:
                # Reconstruct the subset
                subset = []
                current_sum = 200
                while reachable[current_sum] is not None and reachable[current_sum][1] is not None:
                    prev_sum, elem = reachable[current_sum]
                    subset.append(elem)
                    current_sum = prev_sum
                # Remove a random element from the conflict subset
                if subset:
                    elem_to_remove = random.choice(subset)
                    S.remove(elem_to_remove)
        
        # After loop, S is valid.
        # Apply local search to add back elements?
        # The removal process might have removed too many.
        S = local_search_improve(S)
        if len(S) > best_size:
            best_size = len(S)
            best_set = S.copy()

    # Final check
    if time.time() > deadline:
        return best_set if best_set else [100]
        
    # If no set found (should not happen), return a trivial valid set
    if not best_set:
        best_set = [100]
        
    return best_set
# EVOLVE-BLOCK-END
