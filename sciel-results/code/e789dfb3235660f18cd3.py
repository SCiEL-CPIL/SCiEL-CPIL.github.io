import time
import random
import sys

N = 128
MAX_TIME = 55.0  # Safety margin for 60s limit

def _fast_validity_check(s):
    """Faster check using precomputed pair sums."""
    s_set = set(s)
    s_list = sorted(s)
    n = len(s_list)
    
    # For each (a,b) pair, if a+b is in s, check if there's c > b with a+c and b+c in s
    for i in range(n):
        a = s_list[i]
        # Optimization: if a + s_list[i+1] > 128, no valid b exists for this a
        if i + 1 >= n:
            break
        if a + s_list[i+1] > 128:
            break
        for j in range(i+1, n):
            b = s_list[j]
            ab = a + b
            if ab > 128:
                break
            if ab not in s_set:
                continue
            # c must be > b and b+c <= 128
            # Also a+c must be in s_set
            max_c = min(128 - b, 128 - a)
            for c in range(b+1, max_c+1):
                if c not in s_set:
                    continue
                if a + c in s_set and b + c in s_set:
                    return False
    return True

def _count_conflicts(s_set):
    """
    Returns a dictionary mapping each element in s_set to the number of 
    forbidden triples (a,b,c) in s_set that contain that element.
    A forbidden triple is a,b,c distinct in s_set such that a+b, a+c, b+c are in s_set.
    """
    s_list = sorted(s_set)
    n = len(s_list)
    counts = {x: 0 for x in s_list}
    
    for i in range(n):
        a = s_list[i]
        if i + 1 >= n:
            break
        if a + s_list[i+1] > 128:
            break
            
        for j in range(i + 1, n):
            b = s_list[j]
            ab = a + b
            if ab > 128:
                break
            if ab not in s_set:
                continue
            # Now check c > b
            # We need a+c in s_set and b+c in s_set
            # c must be such that a+c <= 128 and b+c <= 128
            # So c <= 128 - b
            # Also c > b
            for k in range(j + 1, n):
                c = s_list[k]
                if b + c > 128:
                    break
                # Check if a+c and b+c are in s_set
                # We already know ab is in s_set
                if a + c in s_set and b + c in s_set:
                    counts[a] += 1
                    counts[b] += 1
                    counts[c] += 1
    return counts

def _can_add(s_set, cand):
    """Check if adding cand to s_set keeps it valid. O(n^2) but with early termination."""
    s_list = sorted(s_set)
    n = len(s_list)
    
    # New violations must involve cand as one of a, b, or c
    # Case 1: cand is c (largest). Need a, b in s_set with a < b < cand,
    #         and a+b in s_set, a+cand in s_set, b+cand in s_set
    for i in range(n):
        a = s_list[i]
        ac = a + cand
        if ac not in s_set:
            continue
        for j in range(i+1, n):
            b = s_list[j]
            if b >= cand:
                break
            bc = b + cand
            if bc not in s_set:
                continue
            ab = a + b
            if ab in s_set:
                return False
    
    # Case 2: cand is b (middle). Need a < cand < c, a in s_set, c in s_set,
    #         a+cand in s_set, cand+c in s_set, a+c in s_set
    for a in s_list:
        if a >= cand:
            break
        ac_cand = a + cand
        if ac_cand not in s_set:
            continue
        # Need c > cand, c in s_set, cand+c in s_set, a+c in s_set
        max_c = min(128 - cand, 128 - a)
        for c in range(cand+1, max_c+1):
            if c not in s_set:
                continue
            if cand + c in s_set and a + c in s_set:
                return False
    
    # Case 3: cand is a (smallest). Need cand < b < c, b, c in s_set,
    #         cand+b in s_set, cand+c in s_set, b+c in s_set
    for j in range(n):
        b = s_list[j]
        if b <= cand:
            continue
        bc_cand = cand + b
        if bc_cand not in s_set:
            continue
        for k in range(j+1, n):
            c = s_list[k]
            cc_cand = cand + c
            if cc_cand not in s_set:
                continue
            if b + c in s_set:
                return False
    
    return True

def _greedy_add(s_set, start_time):
    """Try to greedily add elements to improve the set."""
    s_set = set(s_set)
    
    improved = True
    while improved and time.time() - start_time < MAX_TIME - 3.0:
        improved = False
        candidates = [x for x in range(1, N+1) if x not in s_set]
        candidates.sort()
        for cand in candidates:
            if time.time() - start_time > MAX_TIME - 3.0:
                break
            if _can_add(s_set, cand):
                s_set.add(cand)
                improved = True
    
    return s_set

def _remove_and_add(s_set, start_time):
    """Try removing one element and adding one or more others."""
    s_set = set(s_set)
    original_len = len(s_set)
    
    for remove_el in sorted(s_set):
        if time.time() - start_time > MAX_TIME - 3.0:
            break
        remaining_set = s_set - {remove_el}
        candidates = [x for x in range(1, N+1) if x not in remaining_set]
        candidates.sort()
        
        best_new_set = None
        for cand in candidates:
            if time.time() - start_time > MAX_TIME - 3.0:
                break
            if cand == remove_el:
                continue
            if _can_add(remaining_set, cand):
                new_s = remaining_set | {cand}
                # We can try to add more greedily after this successful swap
                # But for speed, we just check if this single swap improves size (it won't if 1 out 1 in)
                # So we should greedily add to new_s
                new_s_greedy = _greedy_add(new_s, start_time)
                if len(new_s_greedy) > original_len:
                    if best_new_set is None or len(new_s_greedy) > len(best_new_set):
                        best_new_set = new_s_greedy
        
        if best_new_set is not None:
            return best_new_set
    
    return s_set

def _greedy_remove_until_valid(s_set, start_time, randomize_ties=False):
    """
    Start with s_set and remove elements with highest conflict count until valid.
    """
    s_set = set(s_set)
    while True:
        if time.time() - start_time > MAX_TIME - 5.0:
            break
        
        # Check if valid first to avoid unnecessary counting if already valid
        if _fast_validity_check(s_set):
            break
            
        counts = _count_conflicts(s_set)
        if not counts:
            break
            
        max_count = max(counts.values())
        if max_count == 0:
            break # Should be valid if no conflicts counted, but check above just in case
            
        candidates = [x for x, v in counts.items() if v == max_count]
        
        if randomize_ties:
            remove_el = random.choice(candidates)
        else:
            # Tie-break: remove smallest to keep the set "dense" at the top? 
            # Or remove the largest? 
            # Removing smallest preserves higher numbers which might have fewer conflicts 
            # due to sum constraints (a+b > 128).
            remove_el = min(candidates)
            
        s_set.remove(remove_el)
        
    return s_set

def construct():
    start_time = time.time()
    
    best_s = set(range(63, 129)) # Fallback valid set (size 66)
    best_len = len(best_s)
    
    # Strategy 1: Hitting set removal from full set
    # Run multiple variations of the removal process
    num_removal_runs = 5
    for i in range(num_removal_runs):
        if time.time() - start_time > MAX_TIME - 10.0:
            break
            
        # Perturb the initial set slightly for diversity in later runs
        initial_set = set(range(1, N+1))
        if i > 0:
            # Randomly remove a few elements to break symmetry/local optima
            num_remove = random.randint(1, 5)
            to_remove = random.sample(list(initial_set), num_remove)
            initial_set -= set(to_remove)
            
        s_set = _greedy_remove_until_valid(initial_set, start_time, randomize_ties=(i > 2))
        
        # Refine with greedy add
        if time.time() - start_time < MAX_TIME - 5.0:
            s_set = _greedy_add(s_set, start_time)
            
        # Refine with remove-and-add
        if time.time() - start_time < MAX_TIME - 3.0:
            s_set = _remove_and_add(s_set, start_time)
            
        if len(s_set) > best_len:
            best_s = s_set
            best_len = len(s_set)
            
    # Strategy 2: Random Perturbation on the current best
    if time.time() - start_time < MAX_TIME - 5.0:
        for _ in range(10):
            if time.time() - start_time > MAX_TIME - 3.0:
                break
            s_perturbed = set(best_s)
            # Remove 1-3 random elements
            num_remove = random.randint(1, 3)
            removable = sorted(s_perturbed)
            if len(removable) > num_remove:
                removed = random.sample(removable, num_remove)
                s_perturbed -= set(removed)
                
            # Greedy add back
            s_new = _greedy_add(s_perturbed, start_time)
            if len(s_new) > best_len:
                best_s = s_new
                best_len = len(best_s)
                
            # Remove-add on new best
            if time.time() - start_time < MAX_TIME - 2.0:
                s_new2 = _remove_and_add(s_new, start_time)
                if len(s_new2) > best_len:
                    best_s = s_new2
                    best_len = len(best_s)

    # Final validation
    if not _fast_validity_check(best_s):
        # Fallback to known valid set
        best_s = set(range(63, 129))
    
    # Ensure all are Python ints in a list
    return [int(x) for x in sorted(best_s)]

if __name__ == "__main__":
    # Simple self-test
    res = construct()
    print(f"Size: {len(res)}")
    print(f"Valid: {_fast_validity_check(res)}")
