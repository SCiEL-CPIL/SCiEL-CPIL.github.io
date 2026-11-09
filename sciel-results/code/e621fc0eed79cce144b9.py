import time
import random

def construct():
    deadline = time.time() + 110  # 10s margin
    initial_elements = [1, 2, 5, 11]
    
    # 1. Greedy Initialization
    best_set = list(initial_elements)
    best_sums = set()
    for i, x in enumerate(best_set):
        for y in best_set[i:]:
            best_sums.add(x + y)
    
    # Greedy expansion
    for x in range(12, 285):
        new_sums = {x + y for y in best_set} | {2 * x}
        if len(new_sums) == len(best_set) + 1 and not (new_sums & best_sums):
            best_set.append(x)
            best_sums |= new_sums
            
    if len(best_set) == 20:
        return best_set

    # 2. Randomized DFS to improve size
    # We will try to find a set of size len(best_set) + 1, then +2, etc.
    
    def dfs_search(target_size, current_set, current_sums, deadline):
        """
        Searches for a Sidon set of exactly target_size.
        Returns the set if found, None otherwise.
        Uses randomized backtracking.
        """
        if time.time() > deadline:
            return None
        
        if len(current_set) == target_size:
            return list(current_set)
        
        # Generate valid candidates
        # Optimization: Only consider x > max(current_set) ?
        # No, Sidon sets don't have to be contiguous or sorted in insertion order,
        # but the set itself is unordered. However, for uniqueness of search paths,
        # we can enforce an order?
        # If we enforce x > max(current_set), we might miss solutions where a smaller number is added later.
        # But since we start with {1, 2, 5, 11}, and we are adding more,
        # usually we add larger numbers.
        # However, it is possible that a valid set requires a number between existing ones.
        # Example: {1, 2, 5, 11, 13, ...}. 13 > 11.
        # What if we need 8?
        # If 8 was valid, greedy would have added it (or not, if it conflicted).
        # If greedy didn't add 8, it's because it conflicted with {1, 2, 5, 11}.
        # Adding more elements will only create MORE conflicts, so 8 will never become valid later.
        # Therefore, we only need to consider x > max(current_set) ??
        # Wait.
        # If x conflicts with current_set, it will conflict with any superset of current_set.
        # So if x was invalid at step k, it is invalid at step k+1.
        # Thus, we only need to consider candidates that were valid at some point?
        # No, we just need to consider x > max(current_set) is NOT sufficient.
        # We need to consider x that is NOT in current_set.
        # But if x < max(current_set) and was not added, it was invalid.
        # So we can restrict to x > max(current_set) ?
        # Let's verify.
        # Start {1, 2, 5, 11}.
        # Candidates > 11.
        # If we pick 13. Set {1, 2, 5, 11, 13}.
        # Next candidates > 13.
        # This assumes we always add the largest element.
        # Is it possible that a valid set of size 20 requires adding a number smaller than the current max later?
        # No, because if it was smaller than the current max, it would have been considered earlier.
        # If it was invalid then, it is invalid now.
        # So, YES, we can restrict to x > max(current_set).
        # This reduces the search space significantly.
        
        max_val = max(current_set)
        candidates = []
        for x in range(max_val + 1, 285):
            # Check conflicts
            conflict = False
            for y in current_set:
                if (x + y) in current_sums:
                    conflict = True
                    break
            if not conflict:
                if (2 * x) in current_sums:
                    conflict = True
            if not conflict:
                candidates.append(x)
        
        if not candidates:
            return None
        
        random.shuffle(candidates)
        
        for x in candidates:
            new_set = current_set | {x}
            new_sums = current_sums | {x + y for y in current_set} | {2 * x}
            res = dfs_search(target_size, new_set, new_sums, deadline)
            if res:
                return res
        
        return None

    current_best_size = len(best_set)
    
    while current_best_size < 20:
        if time.time() > deadline - 10:
            break
        
        target_size = current_best_size + 1
        found = None
        
        # Try a few restarts for this target size
        # Limit time per target size to avoid getting stuck
        start_time = time.time()
        while time.time() - start_time < 5:  # 5 seconds per size increment
            if time.time() > deadline - 5:
                break
            # Use the current best set as the start?
            # No, DFS from scratch is safer to avoid bias, 
            # but starting from best_set might be faster if we just extend it?
            # If we extend best_set, we assume best_set is a subset of the optimal solution.
            # This is a strong assumption.
            # Better to restart from initial_elements to explore the whole space.
            
            found = dfs_search(target_size, set(initial_elements), set(best_sums), deadline - 5)
            # Wait, best_sums corresponds to best_set, not initial_elements.
            # We need initial_sums.
            init_sums = set()
            for i, x in enumerate(initial_elements):
                for y in initial_elements[i:]:
                    init_sums.add(x + y)
            
            found = dfs_search(target_size, set(initial_elements), init_sums, deadline - 5)
            
            if found:
                break
        
        if found:
            best_set = found
            # Recalculate best_sums? Not strictly needed if we just return the set,
            # but if we continue to next size, we need it.
            best_sums = set()
            for i, x in enumerate(best_set):
                for y in best_set[i:]:
                    best_sums.add(x + y)
            current_best_size = len(best_set)
        else:
            # If we couldn't find size target_size, we might still find a larger one?
            # Unlikely. If we can't find 17, we probably can't find 18.
            # But due to randomness, we might have missed it.
            # However, to be safe, we break if we fail to find the next size.
            # Actually, it's possible that a specific path to 18 exists that doesn't go through the specific 17 we failed to find.
            # But DFS explores all paths (with randomization).
            # If we fail, it's likely the size is not attainable or we ran out of time.
            # Let's break to return the best found.
            break

    return best_set
