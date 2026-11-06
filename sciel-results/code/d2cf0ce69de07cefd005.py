"""Improved program: maximize reciprocal weight of 3-AP-free subset of [1,256]."""
# EVOLVE-BLOCK-START
import time
from fractions import Fraction

N = 256
TIME_LIMIT = 85.0  # safety margin before 90s kill

def can_add(x, S, in_set):
    """
    Check if adding x to set S (represented by set S and boolean array in_set)
    creates a non-constant 3-term arithmetic progression.
    Returns True if x can be added safely.
    """
    # Case 1: x is the middle term. a, x, b => a + b = 2x.
    # a, b in S, a != b, a != x, b != x.
    two_x = 2 * x
    for a in S:
        b = two_x - a
        if b != a and b != x and in_set[b]:
            return False
    
    # Case 2: x is an endpoint.
    # Subcase 2a: x, a, b => x + b = 2a => b = 2a - x.
    # a, b in S, a != b, a != x, b != x.
    # Subcase 2b: a, b, x => a + x = 2b => a = 2b - x.
    # This is symmetric to 2a. Just checking if there exists a,b in S 
    # such that 2a - x is in S (and distinct) covers both.
    # If 2a - x = b in S, then {x, a, b} or {b, a, x} is an AP.
    # We must ensure a != b and a != x and b != x.
    # b = 2a - x.
    # b != x => 2a - x != x => 2a != 2x => a != x. (True since x not in S)
    # b != a => 2a - x != a => a != x. (True)
    # So we just check if 2a - x is in S.
    for a in S:
        b = 2 * a - x
        if b != a and b != x and in_set[b]:
            return False
            
    return True

def reciprocal_sum(selected):
    """Compute exact reciprocal sum as Fraction."""
    total = Fraction(0, 1)
    for x in selected:
        total += Fraction(1, x)
    return total

def greedy_construct():
    """Greedy: include x if it doesn't create a 3-AP with previously included numbers."""
    selected = set()
    in_set = [False] * (N + 1)
    for x in range(1, N + 1):
        if can_add(x, selected, in_set):
            selected.add(x)
            in_set[x] = True
    return selected, in_set

def local_search(initial_set, time_limit):
    """Local search to improve the 3-AP-free set."""
    start_time = time.time()
    current = set(initial_set)
    in_set = [False] * (N + 1)
    for x in current:
        in_set[x] = True
        
    current_score = reciprocal_sum(current)
    best = set(current)
    best_score = current_score
    best_in_set = in_set.copy()
    
    # Precompute reciprocals for speed
    recip = [0.0] * (N + 1)
    for i in range(1, N + 1):
        recip[i] = 1.0 / i

    while True:
        if time.time() - start_time > time_limit:
            break
        
        improved = False
        
        # Phase 1: Try adding single elements
        for x in range(1, N + 1):
            if time.time() - start_time > time_limit:
                break
            if in_set[x]:
                continue
            if can_add(x, current, in_set):
                # Add x
                current.add(x)
                in_set[x] = True
                current_score += Fraction(1, x)
                improved = True
                if current_score > best_score:
                    best = set(current)
                    best_score = current_score
                    best_in_set = in_set.copy()
        
        # Phase 2: Try 2-swaps (remove x, add y, z)
        # To make this efficient, we iterate x in current, then y not in current,
        # then z not in current.
        # We can optimize by checking if the potential gain is worth it.
        # Gain = 1/y + 1/z - 1/x. If <= 0, skip.
        # But we need exact Fraction for comparison later, but float is good for pruning.
        
        for x in list(current):
            if time.time() - start_time > time_limit:
                break
            
            # Temporarily remove x
            current.remove(x)
            in_set[x] = False
            base_score_float = sum(recip[i] for i in current)
            
            # Try adding y
            for y in range(1, N + 1):
                if time.time() - start_time > time_limit:
                    break
                if in_set[y] or y == x:
                    continue
                
                # Prune: if 1/y + 1/(y+1) <= 1/x, no z can help (since z>y implies 1/z < 1/y)
                # Actually z can be smaller than y? No, we iterate y, then z.
                # We should iterate y and z such that y < z to avoid duplicates?
                # Or just any pair.
                # Let's just check can_add first, it's fast.
                
                if can_add(y, current, in_set):
                    # Add y temporarily
                    current.add(y)
                    in_set[y] = True
                    
                    # Try adding z
                    for z in range(1, N + 1):
                        if time.time() - start_time > time_limit:
                            break
                        if in_set[z] or z == x or z == y:
                            continue
                        
                        # Prune: if 1/y + 1/z <= 1/x, skip
                        if recip[y] + recip[z] <= recip[x]:
                            continue
                            
                        if can_add(z, current, in_set):
                            # Check if this swap improves the score
                            # New score = Old Score - 1/x + 1/y + 1/z
                            # We can use float for quick check, then Fraction for update
                            new_score_float = base_score_float + recip[y] + recip[z]
                            old_score_float = base_score_float + recip[x]
                            if new_score_float > old_score_float + 1e-12:
                                # Perform the swap
                                current.add(z)
                                in_set[z] = True
                                
                                # Update current_score exactly
                                current_score = current_score - Fraction(1, x) + Fraction(1, y) + Fraction(1, z)
                                improved = True
                                
                                if current_score > best_score:
                                    best = set(current)
                                    best_score = current_score
                                    best_in_set = in_set.copy()
                                
                                # Restart inner loops? Or break?
                                # Breaking z loop and restarting y loop with new state is safer
                                # But we are in a nested loop. Let's just break and let the outer while loop restart.
                                # However, we modified 'current' inside the loop.
                                # We need to be careful.
                                # Let's break out of all loops to restart the pass.
                                # Set a flag
                                break
                        # End if can_add z
                    # End for z
                    if improved and (z == N or in_set[z]): # Check if we broke due to improvement
                         # This logic is tricky. Let's use a flag.
                         pass
                    
                    # Remove y temporarily (if we didn't commit the swap)
                    # If we committed, current has y and z.
                    # If we didn't, current has y.
                    # We need to restore state to (current without x) for next y?
                    # Actually, if we found an improvement, we should restart the whole local search pass.
                    # So we can break out.
                    
                    current.remove(y)
                    in_set[y] = False
            # End for y
            
            # Restore x
            current.add(x)
            in_set[x] = True
            
            if improved:
                # If we made an improvement, we should restart the search from scratch
                # to take advantage of the new state.
                # Break out of x loop and while loop iteration
                break
        
        if not improved:
            # No improvement found in this pass.
            # Try a random restart/perturbation to escape local optima
            if time.time() - start_time > time_limit:
                break
            
            # Randomly remove 1-3 elements and re-optimize
            # This is a simple heuristic.
            if len(current) > 3:
                items_to_remove = list(current)
                import random
                random.seed(int(time.time() * 1000) % 10000) # Simple seed
                num_remove = random.randint(1, 3)
                to_remove = random.sample(items_to_remove, num_remove)
                for item in to_remove:
                    current.remove(item)
                    in_set[item] = False
                # Continue to next while iteration to re-optimize
            else:
                break

    return best, best_in_set

def construct():
    start_time = time.time()
    
    # Phase 1: Greedy construction
    greedy_set, greedy_in_set = greedy_construct()
    greedy_score = reciprocal_sum(greedy_set)
    
    # Phase 2: Local search
    remaining_time = TIME_LIMIT - (time.time() - start_time)
    if remaining_time > 2.0:
        improved_set, _ = local_search(greedy_set, remaining_time - 1.0)
        improved_score = reciprocal_sum(improved_set)
        if improved_score > greedy_score:
            greedy_set = improved_set
            greedy_score = improved_score
    
    # Convert to sorted list
    result = sorted(greedy_set)
    return result
# EVOLVE-BLOCK-END
