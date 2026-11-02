import time
import random
from itertools import combinations

DEADLINE = 140.0  # safety margin before 150s kill

def score_set(U):
    """Compute the score for a set U."""
    U = sorted(set(U))
    if not U or 0 not in U:
        return 0.0
    max_u = max(U)
    if max_u == 0:
        return 0.0
    # Compute sumset and difference set
    sumset = set()
    diffset = set()
    n = len(U)
    for i in range(n):
        for j in range(n):
            sumset.add(U[i] + U[j])
            diffset.add(U[i] - U[j])
    ratio = len(diffset) / len(sumset)
    if ratio <= 1.0:
        return 1.0
    import math
    log_denom = math.log(2 * max_u + 1)
    if log_denom <= 0:
        return 1.0
    return 1.0 + math.log(ratio) / log_denom

def construct():
    start_time = time.time()
    
    # Start with the seed set
    best_set = [0, 1, 3]
    best_score = score_set(best_set)
    
    # Local search: try to improve by adding, removing, or swapping elements
    # We'll use a simulated annealing-like approach
    
    def elapsed():
        return time.time() - start_time
    
    def time_left():
        return DEADLINE - elapsed()
    
    # Phase 1: Try small exhaustive search for sets of size 3-5
    # Try all sets of size 3 with max element up to some limit
    for max_val in range(4, 30):
        if time_left() < 5:
            break
        # Try sets of size 3: {0, a, b} where 1 <= a < b <= max_val
        for a in range(1, max_val):
            for b in range(a + 1, max_val + 1):
                if time_left() < 2:
                    break
                s = [0, a, b]
                sc = score_set(s)
                if sc > best_score:
                    best_score = sc
                    best_set = s[:]
    
    # Phase 2: Local search - add/remove/swap elements
    # We'll try to grow the set and then prune
    current_set = best_set[:]
    current_score = best_score
    
    # Try adding elements
    max_search_val = 50
    attempts = 0
    max_attempts = 5000
    while time_left() > 10 and attempts < max_attempts:
        attempts += 1
        # Try adding a random element
        if len(current_set) < 10:
            new_elem = random.randint(1, max_search_val)
            if new_elem not in current_set:
                new_set = current_set + [new_elem]
                new_score = score_set(new_set)
                if new_score > current_score:
                    current_set = new_set
                    current_score = new_score
                    if current_score > best_score:
                        best_score = current_score
                        best_set = current_set[:]
        
        # Try removing an element (if size > 2)
        if len(current_set) > 2:
            idx = random.randint(1, len(current_set) - 1)  # don't remove 0
            new_set = current_set[:idx] + current_set[idx+1:]
            new_score = score_set(new_set)
            if new_score > current_score:
                current_set = new_set
                current_score = new_score
                if current_score > best_score:
                    best_score = current_score
                    best_set = current_set[:]
        
        # Try swapping an element
        if len(current_set) >= 2:
            idx = random.randint(1, len(current_set) - 1)
            old_elem = current_set[idx]
            new_elem = random.randint(1, max_search_val)
            if new_elem not in current_set and new_elem != old_elem:
                new_set = current_set[:]
                new_set[idx] = new_elem
                new_score = score_set(new_set)
                if new_score > current_score:
                    current_set = new_set
                    current_score = new_score
                    if current_score > best_score:
                        best_score = current_score
                        best_set = current_set[:]
    
    # Phase 3: Try to find a good set by exploring structure
    # Sets with geometric-like spacing might work well
    # Try {0, 1, 3, 7, 15, ...} (2^k - 1)
    for k in range(3, 8):
        if time_left() < 5:
            break
        s = [0]
        val = 1
        for i in range(k):
            s.append(val)
            val = val * 2 + 1
        sc = score_set(s)
        if sc > best_score:
            best_score = sc
            best_set = s[:]
    
    # Try {0, 1, 2, 4, 8, ...} (powers of 2)
    for k in range(3, 10):
        if time_left() < 5:
            break
        s = [0] + [2**i for i in range(k)]
        sc = score_set(s)
        if sc > best_score:
            best_score = sc
            best_set = s[:]
    
    # Try {0, 1, 3, 6, 10, ...} (triangular numbers)
    for k in range(3, 8):
        if time_left() < 5:
            break
        s = [0]
        val = 0
        for i in range(1, k+1):
            val += i
            s.append(val)
        sc = score_set(s)
        if sc > best_score:
            best_score = sc
            best_set = s[:]
    
    # Final: return the best set found
    return best_set
