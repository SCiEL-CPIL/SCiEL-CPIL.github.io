import time
import random

def construct():
    N = 256
    start_time = time.time()
    time_limit = 50.0  # Safety margin for 60s limit

    # Precompute potential third elements for APs? 
    # Not strictly necessary with dynamic check, but can speed up.
    
    def can_add(S_set, x):
        """Check if x can be added to S_set without forming a 3-AP."""
        for y in S_set:
            # Case 1: x, y, z -> z = 2y - x
            z1 = 2 * y - x
            if z1 in S_set and z1 != x:
                return False
            # Case 2: y, x, z -> z = 2x - y
            z2 = 2 * x - y
            if z2 in S_set and z2 != y:
                return False
            # Case 3: x, z, y -> z = (x+y)/2
            if (x + y) % 2 == 0:
                z3 = (x + y) // 2
                if z3 in S_set and z3 != x and z3 != y:
                    return False
        return True

    def greedy_fill(S_set):
        """Add smallest possible valid elements."""
        changed = True
        while changed:
            changed = False
            for x in range(1, N + 1):
                if x not in S_set and can_add(S_set, x):
                    S_set.add(x)
                    changed = True
                    break
        return S_set

    def calculate_score(S_set):
        return sum(1.0 / x for x in S_set)

    # 1. Initial Greedy Solution
    S_best = set()
    S_best = greedy_fill(S_best)
    best_score = calculate_score(S_best)
    
    # 2. Local Search: Best Improvement Swap
    # We look for a swap (remove x, add y) that improves the score.
    # Since 1/y > 1/x iff y < x, we only care about swapping larger x for smaller y.
    
    improved = True
    while improved and (time.time() - start_time) < time_limit:
        improved = False
        S_current = S_best.copy()
        current_score = best_score
        
        # Generate list of elements to remove (sorted by value descending to prioritize removing large ones)
        # and elements to add (sorted by value ascending to prioritize adding small ones)
        to_remove = sorted([x for x in S_current], reverse=True)
        to_add = sorted([x for x in range(1, N + 1) if x not in S_current])
        
        best_swap = None
        best_new_score = current_score
        best_new_set = None
        
        # Try all pairs (x in S, y not in S) where y < x
        # Optimization: If y >= x, score doesn't improve.
        for x in to_remove:
            if x == 1: break # Can't remove 1 and add smaller
            for y in to_add:
                if y >= x:
                    break # No need to check larger y
                # Check if swapping x with y is valid
                # New set: (S \ {x}) U {y}
                # We need to check if y can be added to (S \ {x})
                # And if removing x doesn't break anything (it doesn't, removing always keeps validity)
                
                # Efficient check: 
                # 1. Remove x from a temporary set? 
                #    It's faster to check can_add(S_current - {x}, y)
                #    But can_add iterates over the set.
                #    Let's just create the set and check.
                
                S_temp = S_current - {x}
                if can_add(S_temp, y):
                    new_score = current_score - (1.0 / x) + (1.0 / y)
                    if new_score > best_new_score:
                        best_new_score = new_score
                        best_swap = (x, y)
                        best_new_set = S_temp | {y}
        
        if best_swap:
            S_best = best_new_set
            best_score = best_new_score
            improved = True
            # Optional: After a swap, try to greedily add more elements
            # Since we removed x, maybe we can add something else?
            S_best = greedy_fill(S_best)
            new_score_after_fill = calculate_score(S_best)
            if new_score_after_fill > best_score:
                best_score = new_score_after_fill

    # 3. Random Restarts with Perturbation
    # If stuck, perturb the best set and search again.
    while (time.time() - start_time) < time_limit:
        S_current = S_best.copy()
        
        # Remove 1-3 random elements
        if len(S_current) > 1:
            num_remove = random.randint(1, 3)
            if len(S_current) < num_remove:
                num_remove = len(S_current)
            to_remove = random.sample(list(S_current), num_remove)
            for x in to_remove:
                S_current.remove(x)
        
        # Greedy fill
        S_current = greedy_fill(S_current)
        
        # Local search on this new start
        local_improved = True
        while local_improved and (time.time() - start_time) < time_limit:
            local_improved = False
            to_remove = sorted([x for x in S_current], reverse=True)
            to_add = sorted([x for x in range(1, N + 1) if x not in S_current])
            
            best_swap = None
            best_new_score = calculate_score(S_current)
            best_new_set = None
            
            for x in to_remove:
                if x == 1: break
                for y in to_add:
                    if y >= x:
                        break
                    S_temp = S_current - {x}
                    if can_add(S_temp, y):
                        new_score = best_new_score - (1.0 / x) + (1.0 / y)
                        if new_score > best_new_score:
                            best_new_score = new_score
                            best_swap = (x, y)
                            best_new_set = S_temp | {y}
            
            if best_swap:
                S_current = best_new_set
                local_improved = True
                S_current = greedy_fill(S_current)
                if calculate_score(S_current) > best_new_score:
                    best_new_score = calculate_score(S_current)
        
        final_score = calculate_score(S_current)
        if final_score > best_score:
            best_score = final_score
            S_best = S_current

    # Convert to list
    result = sorted(list(S_best))
    if not result:
        result = [1]
    return result
