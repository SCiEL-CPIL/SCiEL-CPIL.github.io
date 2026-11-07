import time
import random
import itertools
from bisect import bisect_left, bisect_right

N = 128
MAX_TIME = 55.0  # safety margin for 60s limit

def _validity_check(s):
    """Check if set s is valid: no three distinct a,b,c in s have a+b, a+c, b+c all in s."""
    s_set = set(s)
    s_list = sorted(s)
    # For each pair (a, b) with a < b, check if there exists c > b such that
    # a+b, a+c, b+c are all in s_set
    # More efficiently: for each triple (a, b, c), check if a+b, a+c, b+c in s_set
    # This is O(n^3 * 3) which for n~80 is ~1.5M * 3 = 4.5M checks, too slow for many restarts.
    # Use pair-based: for each pair (a,b), compute ab=a+b. If ab in s_set,
    # then for any c in s_set with c>b, check if a+c in s_set AND b+c in s_set.
    # But that's still O(n^3). Let me think of a faster way.
    
    # Alternative: for each (a,b) pair, if a+b is in s, then we need to ensure
    # that there's no c such that both a+c and b+c are in s.
    # Precompute for each x in s, the set of y in s such that x+y in s.
    # Actually, let's just do the direct triple check but with early termination.
    
    n = len(s_list)
    for i in range(n):
        a = s_list[i]
        for j in range(i+1, n):
            b = s_list[j]
            ab = a + b
            if ab not in s_set:
                continue
            for k in range(j+1, n):
                c = s_list[k]
                if a + c in s_set and b + c in s_set:
                    return False
    return True

def _fast_validity_check(s):
    """Faster check using precomputed pair sums."""
    s_set = set(s)
    s_list = sorted(s)
    n = len(s_list)
    
    # For each element c, compute which pairs (a,b) with a<b<c have a+c in s_set AND b+c in s_set
    # AND a+b in s_set
    
    # Precompute: for each c in s, for each a in s with a<c, check if a+c in s
    # Then for pairs (a,b) both < c with a<b, if (a+c in s) AND (b+c in s) AND (a+b in s) -> invalid
    
    # Even faster: iterate over pairs (a,b) and check
    for i in range(n):
        a = s_list[i]
        for j in range(i+1, n):
            b = s_list[j]
            ab = a + b
            if ab not in s_set:
                continue
            # Now need to check if exists c > b with a+c in s_set and b+c in s_set
            # a+c in s_set means c in s_set - a
            # b+c in s_set means c in s_set - b
            # So c must be in (s_set - a) & (s_set - b) & {x : x > b, x in s_set}
            # i.e., c in s_set, c > b, a+c in s_set, b+c in s_set
            # Let's iterate over possible c values
            # c must be in s_set and c > b, and a+c in s_set, b+c in s_set
            # a+c <= 128 => c <= 128-a
            # b+c <= 128 => c <= 128-b
            # c > b
            max_c = min(128 - a, 128 - b)
            for c in range(b+1, max_c+1):
                if c not in s_set:
                    continue
                if a + c in s_set and b + c in s_set:
                    return False
    return True

def _try_greedy_add(s_set, s_list, start_time):
    """Try to greedily add elements to improve the set."""
    s_set = set(s_set)
    elapsed = time.time() - start_time
    remaining = MAX_TIME - elapsed
    if remaining < 5.0:
        return sorted(s_set)
    
    # Candidates to add: elements in 1..128 not in s_set
    candidates = [x for x in range(1, N+1) if x not in s_set]
    candidates.sort()
    
    improved = True
    while improved and time.time() - start_time < MAX_TIME - 3.0:
        improved = False
        for cand in candidates[:]:
            if time.time() - start_time > MAX_TIME - 3.0:
                break
            # Check if adding cand makes set invalid
            new_s = s_set | {cand}
            if _fast_validity_check(new_s):
                s_set = new_s
                candidates = [x for x in candidates if x != cand]
                improved = True
    
    return sorted(s_set)

def _try_greedy_remove_add(s_set, start_time):
    """Try removing one element and adding one or more others."""
    s_set = set(s_set)
    
    for remove_el in sorted(s_set):
        if time.time() - start_time > MAX_TIME - 3.0:
            break
        remaining_set = s_set - {remove_el}
        # Try adding candidates
        candidates = [x for x in range(1, N+1) if x not in remaining_set]
        candidates.sort()
        
        best_new_set = None
        for cand in candidates[:]:
            if time.time() - start_time > MAX_TIME - 3.0:
                break
            new_s = remaining_set | {cand}
            if len(new_s) > len(s_set) and _fast_validity_check(new_s):
                if best_new_set is None or len(new_s) > len(best_new_set):
                    best_new_set = new_s
        
        if best_new_set is not None:
            return sorted(best_new_set)
    
    return sorted(s_set)

def _random_perturbation(s, start_time, num_restarts=100):
    """Apply random perturbations to try to find better sets."""
    best_s = sorted(s)
    best_len = len(best_s)
    
    for _ in range(num_restarts):
        if time.time() - start_time > MAX_TIME - 3.0:
            break
        
        s_set = set(best_s)
        
        # Randomly remove 1-3 elements
        num_remove = random.randint(1, 3)
        removable = sorted(s_set)
        if len(removable) <= num_remove:
            continue
        removed = random.sample(removable, num_remove)
        s_set -= set(removed)
        
        # Try to greedily add back
        s_list = sorted(s_set)
        new_s = _try_greedy_add(s_set, s_list, start_time)
        
        if len(new_s) > best_len:
            best_s = new_s
            best_len = len(new_s)
        
        # Try remove-and-add on current best
        if time.time() - start_time > MAX_TIME - 3.0:
            break
        new_s2 = _try_greedy_remove_add(set(best_s), start_time)
        if len(new_s2) > best_len:
            best_s = new_s2
            best_len = len(new_s2)
    
    return best_s

def _construct_from_threshold(threshold):
    """Construct set {threshold, ..., 128} and try to improve."""
    base = set(range(threshold, N+1))
    # Verify base is valid
    if not _fast_validity_check(base):
        # Try to fix by removing problematic elements
        base = set()
        for x in range(threshold, N+1):
            new_base = base | {x}
            if _fast_validity_check(new_base):
                base = new_base
    
    return sorted(base)

def construct():
    start_time = time.time()
    
    # Best known: {63..128} = 66 elements
    best_s = list(range(63, 129))
    best_len = len(best_s)
    
    # Verify it's valid (should be)
    if not _fast_validity_check(set(best_s)):
        # This shouldn't happen, but just in case
        best_s = [x for x in range(1, N+1) if True]
        best_s = []
        for x in range(63, N+1):
            if _fast_validity_check(set(best_s) | {x}):
                best_s.append(x)
        best_len = len(best_s)
    
    # Try different thresholds
    for threshold in [62, 61, 60, 59, 58]:
        if time.time() - start_time > MAX_TIME - 5.0:
            break
        s = _construct_from_threshold(threshold)
        if len(s) > best_len:
            best_s = s
            best_len = len(s)
    
    # Try greedy add from current best
    if time.time() - start_time < MAX_TIME - 5.0:
        s2 = _try_greedy_add(set(best_s), best_s, start_time)
        if len(s2) > best_len:
            best_s = s2
            best_len = len(s2)
    
    # Try remove-add
    if time.time() - start_time < MAX_TIME - 5.0:
        s3 = _try_greedy_remove_add(set(best_s), start_time)
        if len(s3) > best_len:
            best_s = s3
            best_len = len(s3)
    
    # Random perturbation
    if time.time() - start_time < MAX_TIME - 3.0:
        s4 = _random_perturbation(best_s, start_time, num_restarts=50)
        if len(s4) > best_len:
            best_s = s4
            best_len = len(s4)
    
    # Final validation
    if not _fast_validity_check(set(best_s)):
        # Fallback to known valid set
        best_s = list(range(63, 129))
    
    # Ensure all are Python ints in a list
    return [int(x) for x in best_s]
