import time
import itertools
import random

_LCM = 232792560
_MOD = 1000000007

def _get_sums(subset):
    """Calculate all subset sums of the scaled reciprocals."""
    scaled = [_LCM // x for x in subset]
    sums = {0}
    for v in scaled:
        sums |= {s + v for s in sums}
    return sums

def _is_dissociated(subset):
    """Check if the set is dissociated."""
    sums = _get_sums(subset)
    return len(sums) == (1 << len(subset))

def _mod_subset_sums(subset, mod=_MOD):
    """Compute all subset sums modulo mod. Returns the set of sums."""
    scaled = [(_LCM // x) % mod for x in subset]
    sums = {0}
    for v in scaled:
        sums = {(s + v) % mod for s in sums} | sums
    return sums

def _modular_dissociated(subset, mod=_MOD):
    """
    Necessary condition: all subset sums distinct modulo mod.
    Fast filter - if this fails, the set is definitely not dissociated.
    """
    n = len(subset)
    sums = _mod_subset_sums(subset, mod)
    return len(sums) == (1 << n)

def _fast_check(subset):
    """Two-stage: modular filter then exact check."""
    if not _modular_dissociated(subset):
        return False
    return _is_dissociated(subset)

def _greedy_construct(universe, order, deadline):
    """Greedy construction with a given element order."""
    S = []
    exact_sums = {0}
    for x in order:
        if time.time() > deadline:
            break
        v = _LCM // x
        is_valid = True
        for s in exact_sums:
            if (s + v) in exact_sums:
                is_valid = False
                break
        if is_valid:
            S.append(x)
            exact_sums |= {s + v for s in exact_sums}
    return S

def _local_search_1for2(best_S, universe, deadline):
    """Local search: try 1-for-2 swaps to increase size."""
    best_S = list(best_S)
    improved = True
    while improved and time.time() < deadline - 10:
        improved = False
        in_set = set(best_S)
        not_in_set = [x for x in universe if x not in in_set]
        
        moves = []
        for a in best_S:
            for x, y in itertools.combinations(not_in_set, 2):
                moves.append((a, x, y))
        
        random.shuffle(moves)
        
        for a, x, y in moves:
            if time.time() > deadline - 5:
                break
            
            candidate = [z for z in best_S if z != a] + [x, y]
            if _fast_check(candidate):
                best_S = candidate
                improved = True
                break
        
        if improved:
            continue
    
    return best_S

def _local_search_2for3(best_S, universe, deadline):
    """Local search: try 2-for-3 swaps to increase size."""
    best_S = list(best_S)
    improved = True
    while improved and time.time() < deadline - 10:
        improved = False
        in_set = set(best_S)
        not_in_set = [x for x in universe if x not in in_set]
        
        if len(not_in_set) < 3:
            break
        
        moves = []
        for a, b in itertools.combinations(best_S, 2):
            for x, y, z in itertools.combinations(not_in_set, 3):
                moves.append((a, b, x, y, z))
        
        random.shuffle(moves)
        
        max_tries = min(len(moves), 200)
        for i, (a, b, x, y, z) in enumerate(moves):
            if i >= max_tries or time.time() > deadline - 5:
                break
            
            candidate = [v for v in best_S if v not in (a, b)] + [x, y, z]
            if _fast_check(candidate):
                best_S = candidate
                improved = True
                break
    
    return best_S

def _backtracking_search(universe, deadline, target_size, current_set=None, current_sums=None, index=0):
    """
    Backtracking search to find a dissociated set of target_size.
    Uses modular filtering for early pruning.
    """
    if current_set is None:
        current_set = []
        current_sums = {0}
    
    if len(current_set) == target_size:
        return current_set
    
    if time.time() > deadline:
        return None
    
    remaining = target_size - len(current_set)
    if len(universe) - index < remaining:
        return None
    
    # Try to add elements from the remaining universe
    for i in range(index, len(universe)):
        if time.time() > deadline:
            return None
        
        x = universe[i]
        v = _LCM // x
        
        # Check if adding x keeps the set dissociated
        new_sums = {s + v for s in current_sums}
        if len(new_sums) != len(current_sums):
            continue  # Collision detected
        
        # Also check modular distinctness for faster pruning
        if len(current_set) + 1 <= 18:  # Only do modular check for smaller sets
            test_set = current_set + [x]
            if not _modular_dissociated(test_set):
                continue
        
        # Recurse
        result = _backtracking_search(universe, deadline, target_size, 
                                       current_set + [x], new_sums, i + 1)
        if result is not None:
            return result
    
    return None

def construct():
    start_time = time.time()
    deadline = start_time + 80  # Safety margin for 90s limit
    
    universe = list(range(1, 23))
    
    # 1. Multiple greedy starts with different orderings
    best_S = []
    best_size = 0
    
    # Descending greedy (deterministic)
    S_desc = _greedy_construct(universe, sorted(universe, reverse=True), deadline - 40)
    if len(S_desc) > best_size:
        best_S = list(S_desc)
        best_size = len(S_S)
    
    # Ascending greedy
    S_asc = _greedy_construct(universe, sorted(universe), deadline - 40)
    if len(S_asc) > best_size:
        best_S = list(S_asc)
        best_size = len(S_asc)
    
    # Randomized greedy restarts
    num_restarts = 15
    for i in range(num_restarts):
        if time.time() > deadline - 35:
            break
        candidates = list(universe)
        random.shuffle(candidates)
        S_rand = _greedy_construct(universe, candidates, deadline - 35)
        if len(S_rand) > best_size:
            best_S = list(S_rand)
            best_size = len(S_rand)
    
    # 2. Local search: 1-for-2 swaps
    if time.time() < deadline - 20:
        best_S = _local_search_1for2(best_S, universe, deadline - 15)
        best_size = len(best_S)
    
    # 3. Local search: 2-for-3 swaps
    if time.time() < deadline - 15:
        best_S = _local_search_2for3(best_S, universe, deadline - 10)
        best_size = len(best_S)
    
    # 4. Backtracking search for larger sets if time remains
    if time.time() < deadline - 15:
        # Try to find sets larger than current best
        for target in range(best_size + 1, min(best_size + 4, 14)):
            if time.time() > deadline - 10:
                break
            found = _backtracking_search(sorted(universe, reverse=True), deadline - 10, target)
            if found is not None and len(found) > best_size:
                best_S = found
                best_size = len(found)
    
    # Ensure we have a valid nonempty result
    if not best_S:
        best_S = _greedy_construct(universe, sorted(universe, reverse=True), deadline)
    
    return best_S
