"""Small inclusion-maximal Sidon sets in [1,256].

Strategy:
1. Build a large Sidon set via randomized greedy (multiple seeds/orders).
2. Prune to a minimal (inclusion-minimal) maximal Sidon set by repeatedly
   removing the element whose removal makes the fewest new elements addable.
   This is a heuristic for the minimum vertex cover on the maximality dependency
   graph: each missing element x is "covered" by the set S if adding x would
   create a sum collision. We want a small S such that every missing x is
   still blocked.

The key insight: after we have a maximal Sidon set, we can try to remove
elements one by one. Removing an element y from S is safe if S\{y} is still
maximal (i.e., every missing element x that was blocked only by y is now
still blocked by some other element in S\{y}). We greedily remove the element
whose removal "exposes" the fewest new addable elements, prioritizing removals
that keep maximality.
"""

import time
import random
from itertools import combinations

N = 256
_DEADLINE = 55.0  # seconds, safety margin under 60s limit
_START = None


def _elapsed():
    return time.time() - _START


def _is_valid_sidon(S):
    """Check that all unordered pair sums a+b with a<=b are distinct."""
    sums = set()
    for i, a in enumerate(S):
        for j in range(i, len(S)):
            b = S[j]
            s = a + b
            if s in sums:
                return False
            sums.add(s)
    return True


def _blocked_elements(S, sums):
    """Return set of missing elements x in [1,N] that CANNOT be added to S."""
    blocked = set()
    all_elements = set(range(1, N + 1))
    S_set = set(S)
    for x in all_elements - S_set:
        # Check if adding x would create a sum collision
        new_sums = [x + a for a in S] + [2 * x]
        if any(ns in sums for ns in new_sums) or len(set(new_sums)) < len(new_sums):
            blocked.add(x)
    return blocked


def _greedy_sidon(order, deadline):
    """Build a Sidon set by greedily adding elements in the given order."""
    selected = []
    sums = set()
    for x in order:
        if _elapsed() > deadline:
            break
        added = [x + a for a in selected] + [2 * x]
        if len(set(added)) == len(added) and not sums.intersection(added):
            selected.append(x)
            sums.update(added)
    return selected, sums


def _prune_to_minimal_maximal(S, sums, deadline):
    """
    Given a maximal Sidon set S with its sum set, iteratively remove elements
    whose removal keeps the set maximal. Remove the element whose removal
    exposes the fewest new addable elements (i.e., is most 'redundant' for
    blocking).
    """
    S = list(S)
    sums = set(sums)
    S_set = set(S)
    
    while _elapsed() < deadline and len(S) > 1:
        # For each element in S, compute what would happen if we removed it
        best_remove = None
        best_exposed = None
        best_new_sums = None
        
        for idx, y in enumerate(S):
            # Compute sums without y
            new_sums = set()
            for i in range(len(S)):
                if i == idx:
                    continue
                for j in range(i, len(S)):
                    if j == idx:
                        continue
                    a, b = S[i], S[j]
                    if a > b:
                        a, b = b, a
                    new_sums.add(a + b)
            # Also need sums involving y's partners... actually just recompute
            # Let's do it properly: sums of all pairs in S\{y}
            S_without = S[:idx] + S[idx+1:]
            new_sums = set()
            for i, a in enumerate(S_without):
                for j in range(i, len(S_without)):
                    new_sums.add(a + S_without[j])
            
            # Check if S\{y} is still maximal: every missing element must be blocked
            S_without_set = set(S_without)
            exposed = 0
            still_maximal = True
            for x in range(1, N + 1):
                if x in S_without_set:
                    continue
                # Check if x can be added to S_without
                new_sums_x = [x + a for a in S_without] + [2 * x]
                if any(ns in new_sums for ns in new_sums_x) or len(set(new_sums_x)) < len(new_sums_x):
                    # x is blocked
                    pass
                else:
                    # x can be added -> not maximal
                    still_maximal = False
                    exposed += 1
            
            if still_maximal:
                # This removal is safe and keeps maximality
                if best_remove is None or exposed < best_exposed:
                    best_remove = idx
                    best_exposed = 0
                    best_new_sums = new_sums
        
        if best_remove is not None:
            # Remove the best candidate
            y = S.pop(best_remove)
            sums = best_new_sums
        else:
            # No safe removal; try removing element that exposes fewest
            min_exposed = float('inf')
            min_idx = None
            min_sums = None
            for idx, y in enumerate(S):
                S_without = S[:idx] + S[idx+1:]
                new_sums = set()
                for i, a in enumerate(S_without):
                    for j in range(i, len(S_without)):
                        new_sums.add(a + S_without[j])
                
                S_without_set = set(S_without)
                exposed = 0
                for x in range(1, N + 1):
                    if x in S_without_set:
                        continue
                    new_sums_x = [x + a for a in S_without] + [2 * x]
                    if any(ns in new_sums for ns in new_sums_x) or len(set(new_sums_x)) < len(new_sums_x):
                        pass
                    else:
                        exposed += 1
                
                if exposed < min_exposed:
                    min_exposed = exposed
                    min_idx = idx
                    min_sums = new_sums
            
            if min_exposed == 0:
                y = S.pop(min_idx)
                sums = min_sums
            else:
                # Can't remove any without losing maximality
                break
    
    return S, sums


def construct():
    global _START
    _START = time.time()
    deadline = _START + _DEADLINE
    
    best_S = None
    best_len = float('inf')
    
    # Try multiple random orderings
    rng = random.Random(42)
    
    # First: ascending greedy as baseline
    order = list(range(1, N + 1))
    S, sums = _greedy_sidon(order, deadline)
    if len(S) < best_len:
        best_S = list(S)
        best_len = len(S)
    
    # Now try random permutations
    num_trials = 20
    for trial in range(num_trials):
        if _elapsed() > deadline - 5:
            break
        
        # Random permutation
        order = list(range(1, N + 1))
        rng.shuffle(order)
        
        S, sums = _greedy_sidon(order, deadline)
        if not S:
            continue
        
        # Verify maximality
        if not _is_valid_sidon(S):
            continue
        
        # Check maximality
        S_set = set(S)
        is_maximal = True
        for x in range(1, N + 1):
            if x in S_set:
                continue
            new_sums = [x + a for a in S] + [2 * x]
            if not any(ns in sums for ns in new_sums) and len(set(new_sums)) == len(new_sums):
                is_maximal = False
                break
        
        if not is_maximal:
            # Try to extend to maximal by adding elements that don't conflict
            changed = True
            while changed and _elapsed() < deadline - 2:
                changed = False
                for x in range(1, N + 1):
                    if x in S_set:
                        continue
                    new_sums = [x + a for a in S] + [2 * x]
                    if not any(ns in sums for ns in new_sums) and len(set(new_sums)) == len(new_sums):
                        S.append(x)
                        S_set.add(x)
                        sums.update(new_sums)
                        changed = True
                        break
        
        if len(S) < best_len:
            # Try to prune
            if _elapsed() < deadline - 3:
                S_pruned, sums_pruned = _prune_to_minimal_maximal(S, sums, deadline)
                if _is_valid_sidon(S_pruned):
                    # Verify maximality of pruned set
                    S_pruned_set = set(S_pruned)
                    is_max = True
                    for x in range(1, N + 1):
                        if x in S_pruned_set:
                            continue
                        new_sums = [x + a for a in S_pruned] + [2 * x]
                        if not any(ns in sums_pruned for ns in new_sums) and len(set(new_sums)) == len(new_sums):
                            is_max = False
                            break
                    if is_max and len(S_pruned) < best_len:
                        best_S = S_pruned
                        best_len = len(S_pruned)
                    elif is_max:
                        best_S = S_pruned
                        best_len = len(S_pruned)
                else:
                    best_S = list(S)
                    best_len = len(S)
            else:
                best_S = list(S)
                best_len = len(S)
    
    if best_S is None or not best_S:
        # Fallback: ascending greedy
        order = list(range(1, N + 1))
        S, sums = _greedy_sidon(order, deadline)
        best_S = S
    
    return best_S
