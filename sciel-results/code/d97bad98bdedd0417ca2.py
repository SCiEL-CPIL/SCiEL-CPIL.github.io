# EVOLVE-BLOCK-START
import time
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 82  # safety margin before 90s kill
    
    n = 4096
    max_sum = 3 * n  # maximum possible triple sum
    
    def check_and_add(x, selected, sum_flags):
        """
        Check if x can be added. Uses bytearray for O(1) sum lookups.
        Returns True if valid and updates sum_flags in-place, False otherwise.
        On failure, sum_flags is restored to its original state.
        """
        k = len(selected)
        new_sums = []
        
        # (x, x, x)
        s = 3 * x
        if sum_flags[s]:
            return False
        new_sums.append(s)
        
        # (x, x, y) and (x, y, y)
        for y in selected:
            s = 2 * x + y
            if sum_flags[s]:
                for ss in new_sums:
                    sum_flags[ss] = 0
                return False
            new_sums.append(s)
            
            s = x + 2 * y
            if sum_flags[s]:
                for ss in new_sums:
                    sum_flags[ss] = 0
                return False
            new_sums.append(s)
        
        # (x, y, z) for y <= z in selected
        for i in range(k):
            y = selected[i]
            for j in range(i, k):
                z = selected[j]
                s = x + y + z
                if sum_flags[s]:
                    for ss in new_sums:
                        sum_flags[ss] = 0
                    return False
                new_sums.append(s)
        
        # All valid - commit
        for s in new_sums:
            sum_flags[s] = 1
        selected.append(x)
        return True

    def rebuild_flags(selected):
        """Build bytearray sum flags from scratch."""
        flags = bytearray(max_sum + 1)
        k = len(selected)
        for i in range(k):
            a = selected[i]
            for j in range(i, k):
                b = selected[j]
                for l in range(j, k):
                    c = selected[l]
                    flags[a + b + c] = 1
        return flags

    def greedy_build(order, selected, flags):
        """Greedy build from given order, starting with existing selected/flags."""
        selected_set = set(selected)
        for x in order:
            if x in selected_set:
                continue
            if check_and_add(x, selected, flags):
                selected_set.add(x)
        return selected

    best_set = []
    best_size = 0
    
    # Phase 1: Multiple greedy restarts with different orderings
    np.random.seed(42)
    
    # Standard ascending greedy
    selected = []
    flags = bytearray(max_sum + 1)
    for x in range(1, n + 1):
        check_and_add(x, selected, flags)
    if len(selected) > best_size:
        best_size = len(selected)
        best_set = list(selected)
    
    # Randomized greedy restarts
    max_restarts = 500
    for i in range(max_restarts):
        if time.time() > deadline - 10:
            break
        order = np.random.permutation(n) + 1
        selected = []
        flags = bytearray(max_sum + 1)
        for x in order:
            check_and_add(x, selected, flags)
        if len(selected) > best_size:
            best_size = len(selected)
            best_set = list(selected)
    
    # Phase 2: Local search - swap moves to improve
    current_set = list(best_set)
    current_size = best_size
    current_flags = rebuild_flags(current_set)
    current_set_membership = set(current_set)
    
    improvement_found = True
    while improvement_found and current_size < 80:
        improvement_found = False
        
        if time.time() > deadline - 3:
            break
        
        # Try swap moves: remove one element, add another
        for attempt in range(300):
            if time.time() > deadline - 3:
                break
            
            # Pick a random element to potentially remove
            idx = np.random.randint(len(current_set))
            y_to_remove = current_set[idx]
            
            # Build reduced set
            reduced_set = [x for x in current_set if x != y_to_remove]
            reduced_flags = rebuild_flags(reduced_set)
            reduced_membership = set(reduced_set)
            
            # Try to add a new element
            candidates = set()
            for _ in range(40):
                c = int(np.random.randint(1, n + 1))
                candidates.add(c)
            
            for x in reduced_set:
                for delta in [-3, -2, -1, 1, 2, 3, 10, 20, 50, 100]:
                    c = x + delta
                    if 1 <= c <= n and c not in reduced_membership:
                        candidates.add(c)
                    c = x - delta
                    if 1 <= c <= n and c not in reduced_membership:
                        candidates.add(c)
            
            candidates = list(candidates)
            np.random.shuffle(candidates)
            
            found = False
            for x_new in candidates[:60]:
                if x_new in reduced_membership:
                    continue
                
                temp_set = list(reduced_set)
                temp_flags = bytearray(reduced_flags)
                if check_and_add(x_new, temp_set, temp_flags):
                    # Valid swap - now try to extend
                    extended_set = list(temp_set)
                    extended_flags = temp_flags
                    extended_membership = set(extended_set)
                    
                    # Try to add more elements greedily
                    for _ in range(40):
                        if time.time() > deadline - 3:
                            break
                        c = int(np.random.randint(1, n + 1))
                        if c in extended_membership:
                            continue
                        if check_and_add(c, extended_set, extended_flags):
                            extended_membership.add(c)
                    
                    if len(extended_set) > current_size:
                        current_set = extended_set
                        current_flags = extended_flags
                        current_size = len(extended_set)
                        current_set_membership = extended_membership
                        improvement_found = True
                        found = True
                        break
            
            if found:
                break
        
        if current_size > best_size:
            best_size = current_size
            best_set = list(current_set)
    
    # Phase 3: Final greedy fill on best set
    if best_size < 80:
        final_set = list(best_set)
        final_flags = rebuild_flags(final_set)
        final_set_set = set(final_set)
        
        # Try adding elements in ascending order
        for x in range(1, n + 1):
            if time.time() > deadline - 1:
                break
            if x in final_set_set:
                continue
            if check_and_add(x, final_set, final_flags):
                final_set_set.add(x)
        
        if len(final_set) > best_size:
            best_size = len(final_set)
            best_set = final_set
    
    # Ensure we return a valid sorted list
    return sorted(best_set)

# EVOLVE-BLOCK-END
