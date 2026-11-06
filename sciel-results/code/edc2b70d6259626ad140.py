import sys
import time
import random

def construct():
    start_time = time.time()
    timeout = 115  # Leave 5s buffer
    
    # Prescribed elements
    prescribed = [1, 2, 5, 11]
    
    def get_sums(S):
        sums = set()
        for i in range(len(S)):
            for j in range(i, len(S)):
                sums.add(S[i] + S[j])
        return sums

    best_set = prescribed
    best_size = len(prescribed)
    
    # Candidate pool: 1..284 excluding prescribed
    all_candidates = [x for x in range(1, 285) if x not in prescribed]
    
    def try_greedy(order_func):
        nonlocal best_set, best_size, start_time, timeout
        
        if time.time() - start_time > timeout:
            return False
            
        S = list(prescribed)
        sums = get_sums(S)
        
        # Create a list of candidates to try
        # order_func determines the order
        candidates = list(all_candidates)
        if order_func is not None:
            candidates.sort(key=order_func)
        
        for x in candidates:
            if time.time() - start_time > timeout:
                return False
            
            # Check if x is valid
            valid = True
            new_sums = set()
            for y in S:
                s = x + y
                if s in sums or s in new_sums:
                    valid = False
                    break
                new_sums.add(s)
            
            if valid:
                s2 = 2 * x
                if s2 in sums or s2 in new_sums:
                    valid = False
                else:
                    new_sums.add(s2)
            
            if valid:
                S.append(x)
                sums.update(new_sums)
                
        if len(S) > best_size:
            best_set = S
            best_size = len(S)
            
        return best_size < 20

    # Strategy 1: Randomized Greedy
    # Run many iterations with different random orders
    iterations = 0
    while time.time() - start_time < timeout:
        iterations += 1
        # Random shuffle
        random.shuffle(all_candidates)
        # Use the shuffled list as the order
        # We need to pass the shuffled list or a function that returns it?
        # Simpler: Just shuffle a copy inside the loop or pass a flag.
        
        # Let's refactor try_greedy to take the order directly or a seed
        
        # To avoid overhead, we'll just implement the loop here for random case
        S = list(prescribed)
        sums = get_sums(S)
        
        # Get a random permutation of candidates
        # Using random.sample is faster than shuffle if we just need the list
        # But we need to iterate through ALL candidates, just in random order.
        # So shuffle is correct.
        
        # Optimization: Pre-calculate some stats? No.
        
        candidates_iter = list(all_candidates)
        random.shuffle(candidates_iter)
        
        for x in candidates_iter:
            if time.time() - start_time > timeout:
                return best_set
            
            valid = True
            new_sums = set()
            for y in S:
                s = x + y
                if s in sums or s in new_sums:
                    valid = False
                    break
                new_sums.add(s)
            
            if valid:
                s2 = 2 * x
                if s2 in sums or s2 in new_sums:
                    valid = False
                else:
                    new_sums.add(s2)
            
            if valid:
                S.append(x)
                sums.update(new_sums)
                
        if len(S) > best_size:
            best_set = S
            best_size = len(S)
            
        if best_size == 20:
            break

    # If we didn't find 20, try a few structured orders
    if best_size < 20 and time.time() - start_time < timeout:
        # Try descending order
        candidates_desc = sorted(all_candidates, reverse=True)
        S = list(prescribed)
        sums = get_sums(S)
        for x in candidates_desc:
            if time.time() - start_time > timeout:
                break
            valid = True
            new_sums = set()
            for y in S:
                s = x + y
                if s in sums or s in new_sums:
                    valid = False
                    break
                new_sums.add(s)
            if valid:
                s2 = 2 * x
                if s2 in sums or s2 in new_sums:
                    valid = False
                else:
                    new_sums.add(s2)
            if valid:
                S.append(x)
                sums.update(new_sums)
        if len(S) > best_size:
            best_set = S
            best_size = len(S)
            
        # Try ascending order
        if best_size < 20 and time.time() - start_time < timeout:
            candidates_asc = sorted(all_candidates)
            S = list(prescribed)
            sums = get_sums(S)
            for x in candidates_asc:
                if time.time() - start_time > timeout:
                    break
                valid = True
                new_sums = set()
                for y in S:
                    s = x + y
                    if s in sums or s in new_sums:
                        valid = False
                        break
                    new_sums.add(s)
                if valid:
                    s2 = 2 * x
                    if s2 in sums or s2 in new_sums:
                        valid = False
                    else:
                        new_sums.add(s2)
                if valid:
                    S.append(x)
                    sums.update(new_sums)
            if len(S) > best_size:
                best_set = S
                best_size = len(S)

    return best_set
