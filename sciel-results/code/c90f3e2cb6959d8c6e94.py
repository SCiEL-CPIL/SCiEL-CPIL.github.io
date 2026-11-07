"""Rewritten: dynamic conflict-degree greedy with local search for max set without forbidden 6-sets."""
import time
import random
from collections import defaultdict

N = 128
DEADLINE = 55.0

# Precompute forbidden 6-sets: {a, b, c, a+b, a+c, b+c} where a<b<c and b+c <= N
_TRIPLES = []
for a in range(1, N + 1):
    for b in range(a + 1, N + 1):
        for c in range(b + 1, N + 1):
            s1 = a + b
            s2 = a + c
            s3 = b + c
            if s3 <= N:
                _TRIPLES.append((a, b, c, s1, s2, s3))

# Index: for each value, which triples it belongs to
_val_to_triples = defaultdict(list)
for idx, t in enumerate(_TRIPLES):
    for v in t:
        if 1 <= v <= N:
            _val_to_triples[v].append(idx)

# For each triple, store the 6 members as a frozenset for quick lookup
_triple_sets = [frozenset(t) for t in _TRIPLES]

def _greedy_dynamic(deadline, rng=None):
    """Dynamic greedy: at each step, pick the unselected value with the fewest
    'near-complete' conflicts (triples where 5 of 6 are already selected)."""
    sel = set()
    result = []
    counts = [0] * len(_TRIPLES)
    
    # For efficiency, track for each unselected value how many triples it's in
    # that are at count 5 (i.e., adding it would complete a forbidden set)
    # We'll recompute the "danger" score periodically
    
    remaining = set(range(1, N + 1))
    
    while remaining and time.time() < deadline:
        best_v = None
        best_danger = None
        
        # Find the value with minimum danger (fewest near-complete triples)
        for v in remaining:
            danger = 0
            for ti in _val_to_triples[v]:
                if counts[ti] == 5:
                    danger += 1
            if best_danger is None or danger < best_danger:
                best_danger = danger
                best_v = v
            if danger == 0:
                break  # Can't do better than 0
        
        if best_v is None:
            break
        
        # Check if best_v is actually safe (danger should be 0, but verify)
        if best_danger > 0:
            # Try to find any value with danger 0
            found = False
            for v in remaining:
                safe = True
                for ti in _val_to_triples[v]:
                    if counts[ti] == 5:
                        safe = False
                        break
                if safe:
                    best_v = v
                    found = True
                    break
            if not found:
                break
        
        # Commit best_v
        sel.add(best_v)
        result.append(best_v)
        remaining.discard(best_v)
        for ti in _val_to_triples[best_v]:
            counts[ti] += 1
    
    return result

def _local_search(sel_list, deadline, rng=None):
    """Aggressive local search: repeatedly try to add any safe value, and if stuck,
    try removing one selected value to free up multiple additions."""
    if not sel_list:
        return []
    
    sel = set(sel_list)
    counts = [0] * len(_TRIPLES)
    for v in sel:
        for ti in _val_to_triples[v]:
            counts[ti] += 1
    
    best_set = set(sel)
    best_size = len(sel)
    
    changed = True
    while changed and time.time() < deadline:
        changed = False
        
        # Phase 1: Add any safe value
        for v in range(1, N + 1):
            if v in sel:
                continue
            safe = True
            for ti in _val_to_triples[v]:
                if counts[ti] == 5:
                    safe = False
                    break
            if safe:
                sel.add(v)
                for ti in _val_to_triples[v]:
                    counts[ti] += 1
                changed = True
        
        if not changed:
            # Phase 2: Try removing one element to enable more additions
            improved = False
            for v_remove in list(sel):
                # Tentatively remove v_remove
                sel.discard(v_remove)
                for ti in _val_to_triples[v_remove]:
                    counts[ti] -= 1
                
                # Try to add as many safe values as possible
                added = []
                for v_add in range(1, N + 1):
                    if v_add in sel:
                        continue
                    safe = True
                    for ti in _val_to_triples[v_add]:
                        if counts[ti] == 5:
                            safe = False
                            break
                    if safe:
                        sel.add(v_add)
                        added.append(v_add)
                        for ti in _val_to_triples[v_add]:
                            counts[ti] += 1
                
                new_size = len(sel)
                if new_size > best_size:
                    best_set = set(sel)
                    best_size = new_size
                    improved = True
                else:
                    # Revert: remove the added values and put back v_remove
                    for v_add in added:
                        sel.discard(v_add)
                        for ti in _val_to_triples[v_add]:
                            counts[ti] -= 1
                    sel.add(v_remove)
                    for ti in _val_to_triples[v_remove]:
                        counts[ti] += 1
                
                if time.time() > deadline:
                    # Revert and break
                    for v_add in added:
                        sel.discard(v_add)
                        for ti in _val_to_triples[v_add]:
                            counts[ti] -= 1
                    sel.add(v_remove)
                    for ti in _val_to_triples[v_remove]:
                        counts[ti] += 1
                    break
            
            if improved:
                changed = True
                # Reset sel to best_set and rebuild counts
                sel = set(best_set)
                counts = [0] * len(_TRIPLES)
                for v in sel:
                    for ti in _val_to_triples[v]:
                        counts[ti] += 1
    
    return list(best_set)

def construct():
    start = time.time()
    deadline = start + DEADLINE
    best = []
    best_size = 0
    
    # Strategy 1: Deterministic greedy with dynamic conflict-degree
    res = _greedy_dynamic(deadline)
    res = _local_search(res, deadline)
    if len(res) > best_size:
        best = res
        best_size = len(res)
    
    # Strategy 2: Randomized restarts with different seeds
    rng = random.Random(42)
    attempts = 0
    while time.time() < deadline - 2:
        attempts += 1
        # Vary the initial bias by shuffling the "danger" tie-breaking
        # Use a randomized version of dynamic greedy
        sel = set()
        result = []
        counts = [0] * len(_TRIPLES)
        remaining = set(range(1, N + 1))
        
        while remaining and time.time() < deadline - 1:
            # Find all values with minimum danger
            min_danger = None
            candidates = []
            for v in remaining:
                danger = 0
                for ti in _val_to_triples[v]:
                    if counts[ti] == 5:
                        danger += 1
                if min_danger is None or danger < min_danger:
                    min_danger = danger
                    candidates = [v]
                elif danger == min_danger:
                    candidates.append(v)
            
            if min_danger is None:
                break
            
            if min_danger > 0:
                # No safe value; try to find any with danger 0
                safe_found = False
                for v in remaining:
                    safe = True
                    for ti in _val_to_triples[v]:
                        if counts[ti] == 5:
                            safe = False
                            break
                    if safe:
                        candidates = [v]
                        min_danger = 0
                        safe_found = True
                        break
                if not safe_found:
                    break
            
            # Randomly pick from candidates
            best_v = rng.choice(candidates)
            sel.add(best_v)
            result.append(best_v)
            remaining.discard(best_v)
            for ti in _val_to_triples[best_v]:
                counts[ti] += 1
        
        res = _local_search(result, deadline)
        if len(res) > best_size:
            best = res
            best_size = len(res)
            if best_size >= 95:
                break
    
    # Final validity check
    if best:
        sel_set = set(best)
        for (a, b, c, s1, s2, s3) in _TRIPLES:
            if a in sel_set and b in sel_set and c in sel_set and s1 in sel_set and s2 in sel_set and s3 in sel_set:
                # Fallback: use the upper half which is always valid
                best = list(range(N // 2 + 1, N + 1))
                break
    
    return [int(x) for x in best]
