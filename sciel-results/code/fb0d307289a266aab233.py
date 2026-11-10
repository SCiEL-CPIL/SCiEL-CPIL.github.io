import time
import random
from itertools import combinations

_DEADLINE = 140.0  # safety margin before 150s kill

def _independence_number(triples, n=16):
    """Find the independence number: largest subset with no triple."""
    # Use a greedy + branch approach, or exact for small n
    # For n=16, we can use a max-independent-set approach
    # Represent as: a set S is independent if no triple is fully contained in S
    # We want max |S| such that no triple ⊆ S
    
    # Convert to a hitting set / set cover dual
    # Use branch and bound
    triple_sets = [set(t) for t in triples]
    
    best = [0]
    
    def _search(remaining_points, current_size, excluded):
        if not remaining_points:
            if current_size > best[0]:
                best[0] = current_size
            return
        # Upper bound: current_size + len(remaining_points)
        if current_size + len(remaining_points) <= best[0]:
            return
        # Try adding the first point
        p = remaining_points[0]
        rest = remaining_points[1:]
        
        # Option 1: don't include p
        _search(rest, current_size, excluded)
        
        # Option 2: include p - check if any triple becomes fully contained
        # A triple is fully contained if all its points are in current set
        # We need to track which points are "chosen"
        # Actually, let me reformulate: we're building set S, and S is independent
        # if no triple has all 3 points in S.
        # When we add p to S, we must ensure no triple containing p has its other
        # 2 points already in S.
        pass
    
    # Simpler: brute force with pruning for n=16
    # 2^16 = 65536 subsets, very fast
    best_size = 0
    for mask in range(1 << n):
        # Check if no triple is fully in this subset
        ok = True
        for t in triple_sets:
            a, b, c = t
            if (mask >> a) & 1 and (mask >> b) & 1 and (mask >> c) & 1:
                ok = False
                break
        if ok:
            sz = bin(mask).count('1')
            if sz > best_size:
                best_size = sz
    return best_size


def _is_linear(triples):
    """Check any two triples share at most one point."""
    for i in range(len(triples)):
        for j in range(i+1, len(triples)):
            common = set(triples[i]) & set(triples[j])
            if len(common) > 1:
                return False
    return True


def _can_add(triples, candidate):
    """Check if candidate triple can be added to maintain linearity."""
    cs = set(candidate)
    for t in triples:
        if len(cs & set(t)) > 1:
            return False
    return True


def _greedy_construct(seed=0, max_triples=30):
    """Greedy construction of a linear triple system."""
    rng = random.Random(seed)
    triples = []
    all_points = list(range(16))
    
    # Generate all possible triples
    all_triples = list(combinations(range(16), 3))
    rng.shuffle(all_triples)
    
    for t in all_triples:
        if len(triples) >= max_triples:
            break
        if _can_add(triples, t):
            triples.append(list(t))
    
    return triples


def _local_search(triples, time_limit):
    """Improve independence number via local search."""
    start = time.time()
    best_indep = _independence_number(triples)
    best_triples = [list(t) for t in triples]
    
    all_triples = list(combinations(range(16), 3))
    
    while time.time() - start < time_limit:
        # Try adding a random triple
        t = list(random.choice(all_triples))
        if _can_add(best_triples, t):
            new_triples = best_triples + [t]
            new_indep = _independence_number(new_triples)
            if new_indep <= best_indep:
                best_triples = new_triples
                best_indep = new_indep
        
        # Try swapping: remove one, add another
        if best_triples and random.random() < 0.5:
            idx = random.randint(0, len(best_triples) - 1)
            temp = best_triples[:idx] + best_triples[idx+1:]
            t = list(random.choice(all_triples))
            if _can_add(temp, t):
                new_triples = temp + [t]
                new_indep = _independence_number(new_triples)
                if new_indep < best_indep:
                    best_triples = new_triples
                    best_indep = new_indep
                elif new_indep == best_indep and random.random() < 0.3:
                    best_triples = new_triples
    
    return best_triples, best_indep


def construct():
    start_time = time.time()
    deadline = start_time + _DEADLINE
    
    best_triples = None
    best_score = 16  # worst case
    
    # Try multiple random seeds for greedy construction
    for seed in range(100):
        if time.time() > deadline - 20:
            break
        
        triples = _greedy_construct(seed=seed, max_triples=25)
        if not triples:
            continue
        
        score = _independence_number(triples)
        if score < best_score:
            best_score = score
            best_triples = [list(t) for t in triples]
        
        # Local search on this solution
        remaining = deadline - time.time()
        if remaining > 5:
            ls_triples, ls_score = _local_search(triples, min(remaining - 3, 10))
            if ls_score < best_score:
                best_score = ls_score
                best_triples = ls_triples
    
    # Final local search with remaining time
    if best_triples is None:
        best_triples = [[0,1,2],[3,4,5],[6,7,8],[9,10,11],[12,13,14]]
    
    remaining = deadline - time.time()
    if remaining > 3:
        best_triples, best_score = _local_search(best_triples, max(0, remaining - 2))
    
    return best_triples
