import time
import random

N = 256

def _validate(S):
    """Return True iff S satisfies: for all a<b<=c in S, a does not divide b+c."""
    L = sorted(S)
    m = len(L)
    for i in range(m):
        a = L[i]
        for j in range(i + 1, m):
            b = L[j]
            for k in range(j, m):
                c = L[k]
                if (b + c) % a == 0:
                    return False
    return True

def _can_add(x, S):
    """
    Check if x can be added to S while maintaining validity.
    S is a set of integers.
    x is an integer not in S.
    """
    # We need to check all triples (a, b, c) in S U {x} that involve x.
    # The condition is: for a < b <= c, a !| (b+c).
    
    # Case 1: x is 'a' (smallest)
    # a=x, b in S, c in S, x < b <= c
    # Check if x | (b+c)
    # Optimization: 
    # If we precomputed nothing, we just iterate.
    # To make it faster, we can iterate b in S where b > x.
    # Then c in S where c >= b.
    # If (b+c) % x == 0, return False.
    
    # Case 2: x is 'b' (middle)
    # a in S, b=x, c in S, a < x <= c
    # Check if a | (x+c)
    
    # Case 3: x is 'c' (largest)
    # a in S, b in S, c=x, a < b <= x
    # Check if a | (b+x)
    
    # Iterate through S to find relevant elements.
    # For efficiency, we can split S into S_less_x, S_eq_x (none, x not in S), S_greater_x.
    # But simple loops are fine for N=256.
    
    S_list = list(S)
    
    # Case 1: x is a
    for b in S_list:
        if b <= x:
            continue
        for c in S_list:
            if c < b:
                continue
            if (b + c) % x == 0:
                return False
                
    # Case 2: x is b
    for a in S_list:
        if a >= x:
            continue
        for c in S_list:
            if c < x:
                continue
            if (x + c) % a == 0:
                return False
                
    # Case 3: x is c
    for b in S_list:
        if b > x:
            continue
        for a in S_list:
            if a >= b:
                continue
            if (b + x) % a == 0:
                return False
                
    return True

def _greedy_descending():
    n = N
    S = set()
    for x in range(n, 0, -1):
        if _can_add(x, S):
            S.add(x)
    return S

def _greedy_ascending():
    n = N
    S = set()
    for x in range(1, n + 1):
        if _can_add(x, S):
            S.add(x)
    return S

def _greedy_random(seed):
    random.seed(seed)
    n = N
    S = set()
    candidates = list(range(1, n + 1))
    random.shuffle(candidates)
    for x in candidates:
        if _can_add(x, S):
            S.add(x)
    return S

def _interval_seed(start_frac):
    n = N
    start = int(n * start_frac) + 1
    if start < 1: start = 1
    if start > n: return set()
    S = set(range(start, n + 1))
    # Filter out invalid ones? No, just return the set. 
    # It might be invalid. We should validate or repair.
    # For initialization, we just want a large set to start from.
    # We will use local search to repair/optimize.
    # But if it's very invalid, local search might get stuck.
    # Better to return a valid subset of the interval.
    valid_S = set()
    for x in range(n, start - 1, -1):
        if _can_add(x, valid_S):
            valid_S.add(x)
    return valid_S

def construct():
    start_time = time.time()
    deadline = start_time + 80.0  # Safety margin
    
    # Generate initial candidates
    candidates = []
    candidates.append(_greedy_descending())
    candidates.append(_greedy_ascending())
    
    # Random restarts
    for i in range(8):
        candidates.append(_greedy_random(i))
        
    # Interval seeds
    candidates.append(_interval_seed(0.5))
    candidates.append(_interval_seed(0.6))
    candidates.append(_interval_seed(0.7))
    candidates.append(_interval_seed(0.8))
    
    # Pick the best valid candidate
    best_set = set()
    best_size = 0
    for S in candidates:
        if _validate(S):
            if len(S) > best_size:
                best_size = len(S)
                best_set = S
        else:
            # Repair: remove elements that cause conflicts? 
            # Or just skip.
            pass
            
    if not best_set:
        # Fallback
        best_set = {N}
        best_size = 1
        
    # Local Search
    improved = True
    while improved and time.time() < deadline:
        improved = False
        # Try to add elements
        added_any = False
        for x in range(1, N + 1):
            if x in best_set:
                continue
            if _can_add(x, best_set):
                best_set.add(x)
                best_size += 1
                improved = True
                added_any = True
                
        if not added_any:
            # Try swaps: remove 1, add 2
            found_swap = False
            for y in list(best_set):
                if time.time() > deadline:
                    break
                S_temp = best_set - {y}
                # Try to add 2 elements from remaining
                added_count = 0
                new_elements = []
                for x in range(1, N + 1):
                    if x == y or x in best_set:
                        continue
                    if _can_add(x, S_temp):
                        S_temp.add(x)
                        new_elements.append(x)
                        added_count += 1
                        if added_count == 2:
                            break
                if added_count == 2:
                    best_set = S_temp
                    best_size += 1 # Removed 1, Added 2 -> Net +1
                    improved = True
                    found_swap = True
                    break # Restart outer loop
                    
            if found_swap:
                continue # Restart the while loop to try more additions
            
            # Try remove 2, add 3 swaps
            found_swap2 = False
            elements_list = list(best_set)
            for i in range(len(elements_list)):
                if time.time() > deadline:
                    break
                for j in range(i + 1, len(elements_list)):
                    y1 = elements_list[i]
                    y2 = elements_list[j]
                    S_temp = best_set - {y1, y2}
                    # Try to add 3 elements from remaining
                    added_count = 0
                    new_elements = []
                    for x in range(1, N + 1):
                        if x == y1 or x == y2 or x in best_set:
                            continue
                        if _can_add(x, S_temp):
                            S_temp.add(x)
                            new_elements.append(x)
                            added_count += 1
                            if added_count == 3:
                                break
                    if added_count == 3:
                        best_set = S_temp
                        best_size += 1 # Removed 2, Added 3 -> Net +1
                        improved = True
                        found_swap2 = True
                        break
                if found_swap2:
                    break
                    
            if found_swap2:
                continue # Restart the while loop to try more additions
            
            # If no swap found, try a random kick: remove 1 random element, 
            # then greedily add as many as possible
            kicked = False
            if time.time() < deadline - 5:  # Need time for the kick
                random.seed(int(time.time() * 1000) % 10000)
                y = random.choice(list(best_set))
                S_temp = best_set - {y}
                # Greedily add elements
                for x in range(1, N + 1):
                    if x == y or x in best_set:
                        continue
                    if _can_add(x, S_temp):
                        S_temp.add(x)
                if len(S_temp) > best_size:
                    best_set = S_temp
                    best_size = len(S_temp)
                    improved = True
                    kicked = True
                    
            if not kicked:
                # If no improvement, break
                break
            
    return sorted(best_set)

if __name__ == "__main__":
    _s = construct()
    assert _validate(set(_s)), "constructed set is invalid"
    print("size", len(_s), "first", _s[:5], "last", _s[-5:])
