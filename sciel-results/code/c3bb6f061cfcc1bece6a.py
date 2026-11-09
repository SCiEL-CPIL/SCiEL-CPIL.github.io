"""Find largest dissociated reciprocal subset of [1,22]."""
import time
import random
from itertools import combinations

# LCM of 1..22
_LCM = 232792560

def _weights():
    """Return list of integer weights w_i = LCM/i for i in 1..22."""
    return [_LCM // i for i in range(1, 23)]

def _is_dissociated(weights, indices):
    """Check if subset of weights (by indices) has all subset sums distinct.
    Uses integer arithmetic for exactness."""
    ws = [weights[i] for i in indices]
    n = len(ws)
    if n == 0:
        return True
    # Generate all 2^n subset sums
    sums = [0]
    for w in ws:
        sums = sums + [s + w for s in sums]
    # Check all distinct
    if len(set(sums)) != len(sums):
        return False
    return True

def _min_gap(weights, indices):
    """Return the minimum absolute difference between any two distinct subset sums.
    Returns 0 if any collision exists."""
    ws = [weights[i] for i in indices]
    n = len(w) if False else len(ws)
    if n == 0:
        return float('inf')
    sums = [0]
    for w in ws:
        sums = sums + [s + w for s in sums]
    sums_sorted = sorted(sums)
    min_d = float('inf')
    for i in range(len(sums_sorted) - 1):
        d = sums_sorted[i+1] - sums_sorted[i]
        if d == 0:
            return 0
        if d < min_d:
            min_d = d
    return min_d

def _greedy_build(weights, max_k=None):
    """Greedy: start empty, add elements one by one that maintain dissociation.
    Try to maximize cardinality."""
    n = len(weights)
    selected = []
    # Try all starting points and orderings
    best = []
    for start in range(n):
        # Greedy from this start
        sel = [start]
        for j in range(start + 1, n):
            if _is_dissociated(weights, sel + [j]):
                sel.append(j)
        if len(sel) > len(best):
            best = sel
        if max_k and len(best) >= max_k:
            break
    return best

def _local_search(weights, start_set, time_limit=8.0):
    """Simulated annealing / local search to improve dissociated set.
    Objective: maximize cardinality, then maximize min_gap."""
    n = len(weights)
    current = list(start_set)
    best = list(current)
    best_score = (len(current), _min_gap(weights, current))
    
    # Score function: (cardinality, min_gap)
    def score(s):
        if not s:
            return (0, 0)
        if not _is_dissociated(weights, s):
            return (-1, 0)
        return (len(s), _min_gap(weights, s))
    
    t0 = time.time()
    temp = 1.0
    cooling = 0.999
    
    while time.time() - t0 < time_limit:
        # Try a random move: swap in/out or add/remove
        move_type = random.random()
        if move_type < 0.4 and len(current) < n:
            # Add a random element not in current
            candidates = [i for i in range(n) if i not in current]
            if not candidates:
                continue
            j = random.choice(candidates)
            new_set = current + [j]
        elif move_type < 0.7 and current:
            # Remove a random element
            j = random.choice(current)
            new_set = [x for x in current if x != j]
        elif current and len(current) < n:
            # Swap: remove one, add another
            j = random.choice(current)
            candidates = [i for i in range(n) if i not in current]
            if not candidates:
                continue
            k = random.choice(candidates)
            new_set = [x for x in current if x != j] + [k]
        else:
            continue
        
        new_score = score(new_set)
        cur_score = score(current)
        
        # Accept if better, or with probability based on temperature
        if new_score > cur_score:
            current = new_set
            if new_score > best_score:
                best_score = new_score
                best = list(current)
        else:
            # Accept worse with some probability
            delta = cur_score[0] - new_score[0]  # cardinality difference
            if delta <= 0:
                # Same cardinality, compare gap
                if new_score[1] >= cur_score[1]:
                    current = new_set
                elif random.random() < temp * 0.1:
                    current = new_set
            else:
                # Lower cardinality - only accept rarely
                if random.random() < temp * 0.01:
                    current = new_set
        
        temp *= cooling
        if temp < 0.01:
            temp = 0.01
    
    return best

def construct():
    t_start = time.time()
    deadline = t_start + 75.0  # Safety margin before 90s limit
    
    weights = _weights()
    n = len(weights)
    
    # Start with greedy
    best_set = _greedy_build(weights)
    
    # Try multiple random restarts with local search
    best_score = (len(best_set), _min_gap(weights, best_set))
    
    random.seed(42)
    
    # Try different initial seeds
    attempts = 0
    while time.time() < deadline - 5.0:
        attempts += 1
        # Random initial set of size around best
        target_size = min(len(best_set) + 1, n)
        if random.random() < 0.3:
            # Start from best and perturb
            init = list(best_set)
            # Randomly swap one element
            if init and random.random() < 0.5:
                j = random.choice(init)
                candidates = [i for i in range(n) if i not in init]
                if candidates:
                    k = random.choice(candidates)
                    init = [x for x in init if x != j] + [k]
        else:
            # Random initial set
            size = random.randint(max(1, len(best_set) - 2), min(n, len(best_set) + 2))
            init = random.sample(range(n), size)
        
        # Check if initial is dissociated, if not shrink it
        while init and not _is_dissociated(weights, init):
            init.pop(random.randrange(len(init)))
        
        if not init:
            continue
        
        result = _local_search(weights, init, time_limit=min(5.0, max(1.0, deadline - time.time() - 2.0)))
        
        if result:
            s = (len(result), _min_gap(weights, result))
            if s > best_score:
                best_score = s
                best_set = result
    
    # Convert indices back to values [1..22]
    return [i + 1 for i in sorted(best_set)]
