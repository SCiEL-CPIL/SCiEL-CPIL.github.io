import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 85  # 5s margin
    
    def can_add(S, x):
        """Check if S U {x} is valid.
        Returns True if S U {x} does not contain a forbidden triple.
        A forbidden triple is distinct a,b,c such that a+b, a+c, b+c are all in the set.
        """
        # If S is empty, always valid
        if not S:
            return True
            
        # Optimization: Pre-compute sets of elements that 'block' x
        # For x to be invalid with b,c in S, we need:
        # 1. x+b in S
        # 2. x+c in S
        # 3. b+c in S or b+c == x
        
        # Let A = {b in S | x+b in S}
        # Let B = {c in S | x+c in S}
        # Note A == B because the condition is symmetric in b,c regarding x.
        # So let C = {y in S | x+y in S}
        
        C = set()
        for y in S:
            if x + y in S:
                C.add(y)
        
        # Now we need to check if there exist distinct b, c in C such that:
        # b != c AND (b+c in S OR b+c == x)
        
        # If len(C) < 2, no pair exists.
        if len(C) < 2:
            return True
            
        C_list = list(C)
        n = len(C_list)
        
        for i in range(n):
            b = C_list[i]
            for j in range(i + 1, n):
                c = C_list[j]
                # b and c are distinct by construction (i != j)
                if (b + c) in S or (b + c) == x:
                    return False
        return True

    def is_valid(S):
        """Full validity check. Used for final verification."""
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

    best_set = set()
    best_size = 0
    
    # We will try several different initial strategies (restarts)
    # Strategy 1: Greedy from high to low
    # Strategy 2: Greedy from low to high
    # Strategy 3: Randomized Greedy
    # Strategy 4: Start with a known large block (e.g., 63..128) and expand
    
    num_restarts = 50
    random.seed(42)
    
    for restart in range(num_restarts):
        if time.time() > deadline:
            break
        
        S = set()
        
        # Choose a strategy based on restart index
        if restart == 0:
            # High to Low
            candidates = list(range(128, 0, -1))
        elif restart == 1:
            # Low to High
            candidates = list(range(1, 129))
        elif restart == 2:
            # Start with 63..128
            S = set(range(63, 129))
            candidates = [i for i in range(1, 63)]
            # Note: We assume 63..128 is valid. Let's verify quickly or just trust it?
            # The problem statement implies large sets exist.
            # Let's verify 63..128 is valid.
            # If a,b,c in 63..128, min sum is 63+64=127, max is 127+128=255 (out of range).
            # Sums must be in S (63..128).
            # So a+b, a+c, b+c must be in 63..128.
            # This restricts a,b,c significantly.
            # Actually, let's just run the greedy on the remaining candidates.
            # But we must ensure the initial set is valid.
            if not is_valid(S):
                S = set() # Fallback if my assumption is wrong
                candidates = list(range(128, 0, -1))
        else:
            # Randomized
            candidates = list(range(1, 129))
            random.shuffle(candidates)
        
        # Greedy Phase
        for x in candidates:
            if time.time() > deadline:
                break
            if can_add(S, x):
                S.add(x)
        
        # Local Search Phase
        not_in_S = [x for x in range(1, 129) if x not in S]
        improved = True
        swap_attempts = 0
        max_swap_attempts = 2000 # Hard cap to prevent infinite loops
        
        while improved and time.time() < deadline and swap_attempts < max_swap_attempts:
            improved = False
            random.shuffle(not_in_S)
            
            for x in not_in_S:
                if time.time() > deadline:
                    break
                if swap_attempts >= max_swap_attempts:
                    break
                
                # Try to add x
                if can_add(S, x):
                    S.add(x)
                    not_in_S.remove(x)
                    improved = True
                    if len(S) > best_size:
                        best_size = len(S)
                        best_set = set(S)
                    continue
                
                # Try to swap
                # Identify blockers for x
                # x is blocked if there exist b,c in S such that x+b in S, x+c in S, and (b+c in S or b+c==x)
                # Removing b, c, or b+c (if in S) might unblock x.
                
                blockers = set()
                # Efficiently find blockers
                C = [y for y in S if x + y in S]
                if len(C) >= 2:
                    for i in range(len(C)):
                        b = C[i]
                        for j in range(i + 1, len(C)):
                            c = C[j]
                            if (b + c) in S or (b + c) == x:
                                blockers.add(b)
                                blockers.add(c)
                                if (b + c) in S:
                                    blockers.add(b + c)
                
                if not blockers:
                    continue
                    
                # Try removing a blocker
                blocker_list = list(blockers)
                random.shuffle(blocker_list)
                
                found_swap = False
                for y in blocker_list:
                    if time.time() > deadline:
                        break
                    if swap_attempts >= max_swap_attempts:
                        break
                        
                    S_temp = S - {y}
                    if can_add(S_temp, x):
                        # Perform swap
                        S = S_temp
                        S.add(x)
                        not_in_S.append(y)
                        # x is still in not_in_S list? We are iterating over it.
                        # We should remove x from not_in_S if it's there, but we are in the loop.
                        # Actually, we check `if x in not_in_S` at the start of next iterations?
                        # No, we iterate over a snapshot? No, `for x in not_in_S` iterates over the list object.
                        # Modifying the list during iteration is bad.
                        # Let's restructure to avoid modifying not_in_S during iteration.
                        
                        # Better approach:
                        # Create a new list of candidates to try in the next outer loop?
                        # Or just break and restart the shuffle?
                        
                        # Let's just break the inner loop and set improved=True.
                        # The next iteration of the `while improved` loop will shuffle not_in_S again.
                        # We need to update not_in_S correctly.
                        # x was in not_in_S. It is now in S.
                        # y was in S. It is now in not_in_S.
                        # We added y to not_in_S.
                        # We need to remove x from not_in_S.
                        if x in not_in_S:
                            not_in_S.remove(x)
                        
                        found_swap = True
                        improved = True
                        swap_attempts += 1
                        # Update best if size increased? No, size is same.
                        # But this swap might enable future additions.
                        break
                
                if found_swap:
                    # Continue to next x in not_in_S?
                    # Since we modified not_in_S, the iterator might be confused?
                    # In Python, `for x in list` iterates over a copy of the list?
                    # No, `for x in not_in_S` iterates over the list object.
                    # If we remove x and add y, the index shifts?
                    # It's safer to break out of the for loop and restart the while loop.
                    break
        
        # End Local Search for this restart
        
        if len(S) > best_size:
            best_size = len(S)
            best_set = set(S)

    # Final polish: Try to add any remaining elements to best_set
    # This is a simple greedy pass on the best set found so far.
    for x in range(1, 129):
        if x in best_set:
            continue
        if can_add(best_set, x):
            best_set.add(x)
            
    # One more local search pass on the final best_set?
    # Might be expensive, but we have time.
    # Let's try a quick swap phase on best_set.
    not_in_best = [x for x in range(1, 129) if x not in best_set]
    improved = True
    while improved and time.time() < deadline:
        improved = False
        random.shuffle(not_in_best)
        for x in not_in_best:
            if time.time() > deadline:
                break
            if can_add(best_set, x):
                best_set.add(x)
                not_in_best.remove(x)
                improved = True
                continue
            
            # Try swap
            C = [y for y in best_set if x + y in best_set]
            if len(C) >= 2:
                blockers = set()
                for i in range(len(C)):
                    b = C[i]
                    for j in range(i + 1, len(C)):
                        c = C[j]
                        if (b + c) in best_set or (b + c) == x:
                            blockers.add(b)
                            blockers.add(c)
                            if (b + c) in best_set:
                                blockers.add(b + c)
                
                if blockers:
                    blocker_list = list(blockers)
                    random.shuffle(blocker_list)
                    for y in blocker_list:
                        S_temp = best_set - {y}
                        if can_add(S_temp, x):
                            best_set = S_temp
                            best_set.add(x)
                            not_in_best.append(y)
                            if x in not_in_best:
                                not_in_best.remove(x)
                            improved = True
                            break
                    if improved:
                        break

    # Final verification
    if not is_valid(best_set):
        # Fallback to a known valid set if something went wrong
        # e.g., 1..10? Or just return empty?
        # Let's try a simple greedy from scratch as a last resort
        S_fallback = set()
        for i in range(128, 0, -1):
            if can_add(S_fallback, i):
                S_fallback.add(i)
        return [int(x) for x in S_fallback]

    result = list(best_set)
    return [int(x) for x in result]
