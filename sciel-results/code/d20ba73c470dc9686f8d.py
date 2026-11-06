import time

_START = time.time()
_DEADLINE = 90.0
_MARGIN = 1.0  # safety margin in seconds
_CAP = 200
_MASK = (1 << (_CAP + 1)) - 1

def _bitset_reach(elements):
    """Compute reachable sums as a bitset (Python int) using each element at most once."""
    bits = 1  # sum 0 reachable
    for x in elements:
        bits |= (bits << x) & _MASK
    return bits

def _valid(s, cap=_CAP):
    bits = _bitset_reach(s)
    return not (bits >> cap) & 1

def _greedy_construct():
    """Greedy: add elements in descending order if 200 is not newly reachable."""
    selected = []
    bits = 1  # only sum 0 reachable
    for x in range(100, 0, -1):
        if (time.time() - _START) > (_DEADLINE - _MARGIN):
            break
        new_bits = bits | ((bits << x) & _MASK)
        if (new_bits >> _CAP) & 1:
            continue
        selected.append(x)
        bits = new_bits
    return selected

def _try_add_all(bits, selected, excluded):
    """Try to add any excluded element that doesn't create sum 200. Returns (bits, selected, changed)."""
    changed = False
    for x in sorted(excluded, reverse=True):
        new_bits = bits | ((bits << x) & _MASK)
        if (new_bits >> _CAP) & 1:
            continue
        selected.append(x)
        bits = new_bits
        changed = True
    return bits, selected, changed

def _local_search(best):
    """Try swaps (remove one or two, add one or more) that enable adding more elements."""
    best_set = set(best)
    excluded = [x for x in range(100, 0, -1) if x not in best_set]
    bits = _bitset_reach(best)
    
    # First, try to add all possible
    bits, best, _ = _try_add_all(bits, best, excluded)
    best_set = set(best)
    excluded = [x for x in range(100, 0, -1) if x not in best_set]
    
    improved = True
    while improved and (time.time() - _START) < (_DEADLINE - _MARGIN):
        improved = False
        best_set = set(best)
        excluded = [x for x in range(100, 0, -1) if x not in best_set]
        
        # Try 1-for-1 swaps: remove r, add a, then try to add more
        for r in list(best):
            if (time.time() - _START) > (_DEADLINE - _MARGIN):
                break
            # Remove r from best
            trial = [x for x in best if x != r]
            trial_set = best_set - {r}
            bits_r = _bitset_reach(trial)
            
            for a in excluded:
                if a == r:
                    continue
                # Try adding a
                new_bits = bits_r | ((bits_r << a) & _MASK)
                if (new_bits >> _CAP) & 1:
                    continue
                
                # Tentatively add a
                trial2 = trial + [a]
                trial_set2 = trial_set | {a}
                trial_bits = new_bits
                
                # Try to add more excluded elements
                trial_excluded = [x for x in range(100, 0, -1) if x not in trial_set2]
                trial_bits, trial2, _ = _try_add_all(trial_bits, trial2, trial_excluded)
                
                if len(trial2) > len(best):  # strict improvement
                    best = trial2
                    best_set = set(best)
                    excluded = [x for x in range(100, 0, -1) if x not in best_set]
                    improved = True
                    break
            
            if improved:
                break
        
        if improved:
            continue
        
        # Try 2-for-1 swaps: remove r1, r2, add a, then try to add more
        for i in range(len(best)):
            if (time.time() - _START) > (_DEADLINE - _MARGIN):
                break
            for j in range(i + 1, len(best)):
                if (time.time() - _START) > (_DEADLINE - _MARGIN):
                    break
                r1, r2 = best[i], best[j]
                trial = [x for x in best if x != r1 and x != r2]
                trial_set = best_set - {r1, r2}
                bits_r = _bitset_reach(trial)
                
                for a in excluded:
                    if a == r1 or a == r2:
                        continue
                    # Try adding a
                    new_bits = bits_r | ((bits_r << a) & _MASK)
                    if (new_bits >> _CAP) & 1:
                        continue
                    
                    # Tentatively add a
                    trial2 = trial + [a]
                    trial_set2 = trial_set | {a}
                    trial_bits = new_bits
                    
                    # Try to add more excluded elements
                    trial_excluded = [x for x in range(100, 0, -1) if x not in trial_set2]
                    trial_bits, trial2, _ = _try_add_all(trial_bits, trial2, trial_excluded)
                    
                    if len(trial2) > len(best):  # strict improvement
                        best = trial2
                        best_set = set(best)
                        excluded = [x for x in range(100, 0, -1) if x not in best_set]
                        improved = True
                        break
                
                if improved:
                    break
            
            if improved:
                break
    
    return best

def construct():
    # Greedy construction
    best = _greedy_construct()
    
    # Local search to try to improve
    if (time.time() - _START) < (_DEADLINE - _MARGIN):
        best = _local_search(best)
    
    # Final exact verification
    if not _valid(best):
        best = [1]
    
    if (time.time() - _START) > (_DEADLINE - _MARGIN):
        if not _valid(best):
            best = [1]
    
    return best
