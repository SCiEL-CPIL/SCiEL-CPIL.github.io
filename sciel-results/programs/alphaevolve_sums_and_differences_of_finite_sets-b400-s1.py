"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from itertools import combinations

def _score_set(U):
    """Compute the score for a finite set U containing 0."""
    U_set = set(U)
    n = len(U_set)
    if n == 0:
        return 0.0
    max_u = max(U_set)
    if max_u == 0:
        return 1.0
    # Compute sumset
    sumset = set()
    for a in U_set:
        for b in U_set:
            sumset.add(a + b)
    # Compute difference set
    diffset = set()
    for a in U_set:
        for b in U_set:
            diffset.add(a - b)
    import math
    ratio = len(diffset) / len(sumset)
    if ratio <= 0:
        return 0.0
    log_ratio = math.log(ratio)
    log_denom = math.log(2 * max_u + 1)
    if log_denom == 0:
        return 1.0
    return 1.0 + log_ratio / log_denom

def _golomb_ruler(n):
    """Generate a Golomb ruler of order n using greedy construction."""
    if n <= 1:
        return [0]
    # Use known optimal Golomb rulers for small n
    known = {
        1: [0],
        2: [0, 1],
        3: [0, 1, 3],
        4: [0, 1, 4, 6],
        5: [0, 1, 4, 10, 11],
        6: [0, 1, 4, 10, 12, 17],
        7: [0, 1, 4, 10, 12, 17, 22],
        8: [0, 1, 4, 10, 12, 17, 22, 29],
        9: [0, 1, 4, 10, 12, 17, 22, 29, 36],
        10: [0, 1, 4, 10, 12, 17, 22, 29, 36, 44],
    }
    if n in known:
        return known[n]
    # Greedy construction
    ruler = [0]
    for i in range(1, n):
        # Find smallest position that doesn't create duplicate differences
        pos = 1
        while True:
            candidate = ruler + [pos]
            diffs = set()
            valid = True
            for j in range(len(candidate)):
                for k in range(j + 1, len(candidate)):
                    d = candidate[k] - candidate[j]
                    if d in diffs:
                        valid = False
                        break
                    diffs.add(d)
                if not valid:
                    break
            if valid:
                ruler.append(pos)
                break
            pos += 1
    return ruler

def _local_search(U, max_iters=5000, deadline=None):
    """Improve set U via local search (swap elements)."""
    U = sorted(set(U))
    best_score = _score_set(U)
    best_U = U[:]
    max_u = max(U)
    upper_bound = max_u + 50  # search space for swaps
    
    for it in range(max_iters):
        if deadline and time.time() > deadline:
            break
        # Try swapping one element
        improved = False
        random.shuffle(U)  # randomize order for diversity
        for i in range(len(U)):
            old_val = U[i]
            # Try replacing with new value
            for new_val in range(0, upper_bound + 1):
                if new_val == old_val:
                    continue
                new_U = U[:]
                new_U[i] = new_val
                new_U_set = set(new_U)
                if len(new_U_set) != len(new_U):
                    continue
                s = _score_set(new_U)
                if s > best_score + 1e-9:
                    best_score = s
                    best_U = new_U[:]
                    improved = True
                    break
            if improved:
                break
        if not improved:
            # Try adding an element
            if len(U) < 30:
                for new_val in range(0, upper_bound + 1):
                    if new_val in U:
                        continue
                    new_U = U + [new_val]
                    s = _score_set(new_U)
                    if s > best_score + 1e-9:
                        best_score = s
                        best_U = new_U[:]
                        improved = True
                        break
            if not improved:
                # Try removing an element (if size > 2)
                if len(U) > 2:
                    for i in range(len(U)):
                        if U[i] == 0:
                            continue  # must keep 0
                        new_U = U[:i] + U[i+1:]
                        s = _score_set(new_U)
                        if s > best_score + 1e-9:
                            best_score = s
                            best_U = new_U[:]
                            improved = True
                            break
        if not improved:
            break
    
    return sorted(set(best_U)), best_score

def construct():
    start_time = time.time()
    deadline = start_time + 110  # safety margin
    
    best_U = [0, 1, 3]
    best_score = _score_set(best_U)
    
    # Try different initial constructions
    candidates = []
    
    # Golomb rulers of various sizes
    for n in range(3, 25):
        if time.time() > deadline:
            break
        gr = _golomb_ruler(n)
        candidates.append(gr)
    
    # Random sets
    random.seed(42)
    for trial in range(20):
        if time.time() > deadline:
            break
        n = random.randint(3, 20)
        max_val = random.randint(10, 200)
        U = [0] + random.sample(range(1, max_val + 1), n - 1)
        U = sorted(U)
        candidates.append(U)
    
    # Evaluate candidates
    for U in candidates:
        if time.time() > deadline:
            break
        s = _score_set(U)
        if s > best_score:
            best_score = s
            best_U = U[:]
    
    # Local search on best candidate
    if time.time() < deadline:
        best_U, best_score = _local_search(best_U, max_iters=2000, deadline=deadline)
    
    # Try local search on a few other candidates
    for U in candidates[:5]:
        if time.time() > deadline:
            break
        if U == best_U:
            continue
        s = _score_set(U)
        if s > best_score - 0.01:  # only if close
            improved_U, improved_score = _local_search(U, max_iters=1000, deadline=deadline)
            if improved_score > best_score:
                best_score = improved_score
                best_U = improved_U
    
    return sorted(set(best_U))
# EVOLVE-BLOCK-END
