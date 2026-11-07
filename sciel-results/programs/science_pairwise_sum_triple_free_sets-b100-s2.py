import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 85  # 5s margin
    
    def can_add(S, x):
        """Check if S U {x} is valid."""
        # Any invalid triple in S U {x} must involve x.
        # Let the triple be (x, b, c) with b, c in S, b != c.
        # Sums: x+b, x+c, b+c.
        # Conditions for invalidity:
        # (x+b in S U {x}) AND (x+c in S U {x}) AND (b+c in S U {x})
        # Since x+b > x (assuming positive ints), x+b in S U {x} implies x+b in S.
        # Since x+c > x, x+c in S U {x} implies x+c in S.
        # b+c in S U {x} implies b+c in S OR b+c == x.
        
        S_list = list(S)
        n = len(S_list)
        
        for i in range(n):
            b = S_list[i]
            if (x + b) not in S:
                continue
            for j in range(n):
                if i == j:
                    continue
                c = S_list[j]
                if (x + c) not in S:
                    continue
                if (b + c) in S or (b + c) == x:
                    return False
        return True

    def is_valid(S):
        """Full validity check for debugging or final verification (slow)."""
        S_list = sorted(S)
        n = len(S_list)
        for i in range(n):
            a = S_list[i]
            for j in range(i + 1, n):
                b = S_list[j]
                if (a + b) not in S:
                    continue
                for k in range(n):
                    if k == i or k == j:
                        continue
                    c = S_list[k]
                    if (a + c) in S and (b + c) in S:
                        return False
        return True

    # --- Strategy: Randomized Greedy with Local Search ---
    # We will run multiple random restarts to find a good starting set,
    # then try to improve it via swaps and additions.
    
    best_set = set()
    best_size = 0
    
    # Pre-compute some helper structures? Maybe not needed for small N.
    
    # We'll try a few different random seeds and orders.
    # Since the problem is symmetric-ish but bounded by 128,
    # random ordering can lead to different local optima.
    
    num_restarts = 20
    random.seed(123)
    
    for restart in range(num_restarts):
        if time.time() > deadline:
            break
            
        # Randomize the order of candidates
        candidates = list(range(1, 129))
        random.shuffle(candidates)
        
        S = set()
        
        # Greedy Phase 1: Add elements in random order
        for x in candidates:
            if time.time() > deadline:
                break
            if can_add(S, x):
                S.add(x)
        
        # Greedy Phase 2: Try to add remaining elements in random order
        # (This handles cases where order mattered in Phase 1)
        # Actually, Phase 1 iterates through ALL candidates. If it couldn't add then,
        # it might be able to now? No, S only grows. If it couldn't add x to S_old,
        # it definitely can't add x to S_new (superset).
        # Wait, can_add checks if S U {x} is valid.
        # If S U {x} was invalid for S_old, then S_new U {x} is also invalid (superset).
        # So Phase 1 is sufficient for pure addition.
        
        current_size = len(S)
        
        # Local Search Phase: Try to improve via swaps
        # We maintain a list of elements not in S.
        not_in_S = [x for x in range(1, 129) if x not in S]
        
        improved = True
        swap_limit = 500 # Limit number of swap attempts to avoid infinite loops
        
        while improved and time.time() < deadline:
            improved = False
            random.shuffle(not_in_S)
            
            for x in not_in_S:
                if time.time() > deadline:
                    break
                
                # First, try to add x directly
                if can_add(S, x):
                    S.add(x)
                    not_in_S.remove(x)
                    current_size += 1
                    improved = True
                    if current_size > best_size:
                        best_size = current_size
                        best_set = set(S)
                    continue
                
                # If x can't be added, try swapping with some y in S
                # We want to remove y and add x, keeping size constant.
                # This might allow adding more elements later.
                
                # Identify potential blockers or just try random y's
                # To be efficient, we can try a limited number of y's.
                
                S_list = list(S)
                random.shuffle(S_list)
                
                found_swap = False
                # Try up to 10 random swaps for this x
                for _ in range(10):
                    if time.time() > deadline:
                        break
                    y = S_list[_ % len(S_list)]
                    
                    # Check if S - {y} + {x} is valid
                    # This is equivalent to checking if x can be added to S - {y}
                    S_temp = S - {y}
                    if can_add(S_temp, x):
                        # Perform swap
                        S = S_temp
                        S.add(x)
                        # Update not_in_S? x is now in S. y is now not in S.
                        # We are iterating over not_in_S, so x is handled.
                        # y needs to be added back to not_in_S.
                        not_in_S.append(y)
                        # Remove x from not_in_S? It's already there?
                        # No, x was in not_in_S. We are iterating over it.
                        # We should mark it as processed or remove it.
                        # Since we are in a for loop over a list, removing is tricky.
                        # Better to just let it be, but we already added it.
                        # If we encounter x again in this same shuffle iteration, 
                        # can_add(S, x) will be False (x in S).
                        # So it's safe.
                        
                        found_swap = True
                        # Size is same, so best_size doesn't change unless we find a larger one later.
                        # But we set improved = True to continue searching.
                        break
                
                if found_swap:
                    improved = True
            
            # If we didn't improve (add) in this pass, and we've done enough swaps, stop.
            # Actually, 'improved' is set on swap too.
            # We should probably track if we found a NEW best size.
            # Let's refine: only continue if we added an element OR we are doing swaps to escape local optima.
            # To prevent infinite swap loops, we can count swaps.
            
            if not any(x not in S for x in not_in_S):
                # All candidates tried?
                break

        # End Local Search for this restart
        
        if len(S) > best_size:
            best_size = len(S)
            best_set = set(S)

    # Final check: Try to add any remaining elements to best_set?
    # We might have found a good set, but maybe we can add more?
    # The local search above tries to add.
    # Let's do a final greedy pass on best_set to be sure.
    
    # Actually, the local search in the loop above does this.
    # But maybe we can do a specific "fill" phase.
    
    # Let's try to add any element from 1..128 to best_set that fits.
    # This is O(128 * N^2). Fast.
    for x in range(1, 129):
        if x in best_set:
            continue
        if can_add(best_set, x):
            best_set.add(x)
    
    # One more pass?
    changed = True
    while changed:
        changed = False
        for x in range(1, 129):
            if x in best_set:
                continue
            if can_add(best_set, x):
                best_set.add(x)
                changed = True

    # Verification (optional, slow, but good for safety)
    # if not is_valid(best_set):
    #     # Fallback to a known good set?
    #     # Or just return empty? No, must be valid.
    #     # Let's trust can_add.
    #     pass

    result = list(best_set)
    return [int(x) for x in result]
