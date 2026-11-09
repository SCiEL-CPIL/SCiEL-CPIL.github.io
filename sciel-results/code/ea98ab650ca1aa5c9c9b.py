"""Max independent set in reciprocal-Schur hypergraph via structure-exploiting greedy + local search."""
import time
import random
import math
from collections import defaultdict

_DEADLINE = 85.0
N = 180


def _precompute_triples():
    """
    Find all triples (a, b, c) of distinct integers in [1, 180] such that
    1/a = 1/b + 1/c, with a < b < c (WLOG).
    
    Key identity: (b-a)(c-a) = a^2
    So for each a, factor pairs (d, e) of a^2 give b = a+d, c = a+e.
    """
    triples = []
    for a in range(1, N + 1):
        a2 = a * a
        for d in range(1, int(math.isqrt(a2)) + 1):
            if a2 % d != 0:
                continue
            e = a2 // d
            if d == e:
                continue
            b = a + d
            c = a + e
            if b > N or c > N:
                continue
            if a == b or a == c or b == c:
                continue
            # Ensure a < b < c
            if b > c:
                b, c = c, b
            triples.append((a, b, c))
    return triples


def _is_valid(set_bits, triples):
    for (a, b, c) in triples:
        if set_bits[a] and set_bits[b] and set_bits[c]:
            return False
    return True


def _greedy_max_set(triples):
    """
    Greedy approach: start with all elements, remove those causing conflicts.
    Use minimum-degree ordering for removal.
    """
    # Build conflict graph: edge (u,v) if they appear together in a triple
    conflict = defaultdict(set)
    for (a, b, c) in triples:
        conflict[a].add(b)
        conflict[a].add(c)
        conflict[b].add(a)
        conflict[b].add(c)
        conflict[c].add(a)
        conflict[c].add(b)
    
    # Count how many triples each element is in
    triple_count = defaultdict(int)
    for (a, b, c) in triples:
        triple_count[a] += 1
        triple_count[b] += 1
        triple_count[c] += 1
    
    # Start with all elements
    included = set(range(1, N + 1))
    
    # Iteratively remove elements that are in violating triples
    # Priority: remove element with highest "damage" (most conflicts)
    changed = True
    while changed:
        changed = False
        # Find all currently violated triples
        violated = []
        for (a, b, c) in triples:
            if a in included and b in included and c in included:
                violated.append((a, b, c))
        
        if not violated:
            break
        
        # For each element in violated triples, count how many violations it's in
        elem_violations = defaultdict(int)
        for (a, b, c) in violated:
            elem_violations[a] += 1
            elem_violations[b] += 1
            elem_violations[c] += 1
        
        # Remove the element with most violations
        if elem_violations:
            worst = max(elem_violations.keys(), key=lambda x: elem_violations[x])
            included.discard(worst)
            changed = True
    
    return included


def _local_search_improve(initial_set, triples, time_limit):
    """
    Local search: try to add back elements that were removed, if they don't create conflicts.
    """
    set_bits = [False] * (N + 1)
    current = set(initial_set)
    for x in current:
        set_bits[x] = True
    
    start_time = time.time()
    improved = True
    while improved and time.time() - start_time < time_limit:
        improved = False
        # Try to add each missing element
        for x in range(1, N + 1):
            if time.time() - start_time > time_limit:
                break
            if set_bits[x]:
                continue
            # Check if adding x creates a violation
            can_add = True
            for (a, b, c) in triples:
                if x in (a, b, c):
                    others = [v for v in (a, b, c) if v != x]
                    if set_bits[others[0]] and set_bits[others[1]]:
                        can_add = False
                        break
            if can_add:
                set_bits[x] = True
                current.add(x)
                improved = True
    
    return current


def _swap_improve(initial_set, triples, time_limit):
    """
    Try swapping: remove one element, see if we can add two or more others.
    """
    set_bits = [False] * (N + 1)
    current = set(initial_set)
    for x in current:
        set_bits[x] = True
    
    start_time = time.time()
    best = set(current)
    best_size = len(current)
    
    for x in list(current):
        if time.time() - start_time > time_limit:
            break
        # Temporarily remove x
        set_bits[x] = False
        current.discard(x)
        
        # Try to add as many elements as possible
        added = 0
        for y in range(1, N + 1):
            if time.time() - start_time > time_limit:
                break
            if set_bits[y]:
                continue
            can_add = True
            for (a, b, c) in triples:
                if y in (a, b, c):
                    others = [v for v in (a, b, c) if v != y]
                    if set_bits[others[0]] and set_bits[others[1]]:
                        can_add = False
                        break
            if can_add:
                set_bits[y] = True
                current.add(y)
                added += 1
        
        if len(current) > best_size:
            best_size = len(current)
            best = set(current)
        
        # Revert: remove added elements, restore x
        for y in list(current):
            if y != x and y not in initial_set:
                set_bits[y] = False
                current.discard(y)
        set_bits[x] = True
        current.add(x)
    
    return best


def construct():
    start_time = time.time()
    
    triples = _precompute_triples()
    
    # Strategy 1: Greedy removal
    greedy_set = _greedy_max_set(triples)
    
    # Strategy 2: Local search to add back elements
    ls_set = _local_search_improve(greedy_set, triples, min(10.0, _DEADLINE - (time.time() - start_time) - 5))
    
    # Strategy 3: Swap improvement
    swap_set = _swap_improve(ls_set, triples, min(10.0, _DEADLINE - (time.time() - start_time) - 5))
    
    best = max([greedy_set, ls_set, swap_set], key=len)
    
    # Validate and fix if needed
    set_bits = [False] * (N + 1)
    for x in best:
        set_bits[x] = True
    if not _is_valid(set_bits, triples):
        # Remove elements causing violations
        for x in sorted(best, reverse=True):
            set_bits[x] = False
            if _is_valid(set_bits, triples):
                break
        best = {x for x in range(1, N + 1) if set_bits[x]}
    
    return sorted(best)
