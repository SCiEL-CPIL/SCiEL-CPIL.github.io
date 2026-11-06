"""K(2,2,2)-free 3-graph on 16 vertices: maximize triples."""
import time
import random
from itertools import combinations

V = 16
ALL_TRIPLES = list(combinations(range(V), 3))

def _is_valid_add(trips_set, t):
    """Check if adding triple t to trips_set creates a K(2,2,2)."""
    a, b, c = t
    used = {a, b, c}
    
    # Iterate over possible a'
    for ap in range(V):
        if ap in used:
            continue
        # Check if {a', b, c} is in set
        t1 = (ap, b, c)
        if t1 not in trips_set:
            continue
            
        # Iterate over possible b'
        for bp in range(V):
            if bp in used or bp == ap:
                continue
            # Check if {a, b', c} is in set
            t2 = (a, bp, c)
            if t2 not in trips_set:
                continue
            
            # Iterate over possible c'
            for cp in range(V):
                if cp in used or cp == ap or cp == bp:
                    continue
                # Check if {a, b, c'} is in set
                t3 = (a, b, cp)
                if t3 not in trips_set:
                    continue
                
                # Check the remaining 4 transversal triples
                # {a', b', c}
                if (ap, bp, c) not in trips_set:
                    continue
                # {a', b, c'}
                if (ap, b, cp) not in trips_set:
                    continue
                # {a, b', c'}
                if (a, bp, cp) not in trips_set:
                    continue
                # {a', b', c'}
                if (ap, bp, cp) not in trips_set:
                    continue
                
                # All 7 other triples are present, so adding t completes K(2,2,2)
                return False
    return True

def _greedy(order, deadline):
    """Greedily add triples in given order, skipping those that create K(2,2,2)."""
    trips_set = set()
    result = []
    for t in order:
        if time.time() >= deadline:
            break
        if t in trips_set:
            continue
        if _is_valid_add(trips_set, t):
            trips_set.add(t)
            result.append(list(t))
    return result

def construct():
    start_time = time.time()
    deadline = start_time + 140  # Safety margin before 150s limit
    
    best = None
    best_len = -1
    
    # Start with the base: all triples through vertex 0
    base = [tuple([0, a, b]) for a in range(1, V) for b in range(a + 1, V)]
    base_set = set(base)
    
    # Other triples to consider
    other_triples = [t for t in ALL_TRIPLES if 0 not in t]
    
    # Try multiple random orders
    for attempt in range(200):
        if time.time() >= deadline:
            break
        
        # Create a random order: base triples first (in random order), then others
        shuffled_base = base[:]
        random.shuffle(shuffled_base)
        shuffled_other = other_triples[:]
        random.shuffle(shuffled_other)
        
        order = shuffled_base + shuffled_other
        
        result = _greedy(order, deadline)
        
        if len(result) > best_len:
            best_len = len(result)
            best = result
            # Early termination if we've used all remaining time budget
            if time.time() >= deadline - 5:
                break
    
    if best is None:
        # Fallback: just return the base
        return [list(t) for t in base]
    
    return best
