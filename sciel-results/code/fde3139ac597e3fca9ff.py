import time
import random
import itertools
import sys

DEADLINE = 110  # safety margin from 120s limit
_START = time.time()

def _time_left():
    return DEADLINE - (time.time() - _START)

def construct():
    _START_LOCAL = time.time()
    DEADLINE_LOCAL = 110  # seconds
    
    def time_left():
        return DEADLINE_LOCAL - (time.time() - _START_LOCAL)
    
    n = 16
    all_triples = list(itertools.combinations(range(n), 3))
    T = len(all_triples)  # 560
    triple_index = {t: i for i, t in enumerate(all_triples)}
    
    # Precompute: for each triple, which other triples form a K(2,2,2) with it
    # A K(2,2,2) is defined by 3 disjoint pairs. The 8 transversals are the 8 triples.
    # Two triples conflict if they share a K(2,2,2) configuration.
    
    # For triple (a,b,c), it participates in K(2,2,2) configurations where:
    # - (a,b,c) is one of the 8 transversals
    # - The 3 pairs are {a,a'}, {b,b'}, {c,c'} for some a'!=a, b'!=b, c'!=c
    # - All points distinct: a,b,c,a',b',c' all different
    
    # Precompute conflict: for each triple i, list of triples j that conflict with i
    # Two triples conflict if there exists a K(2,2,2) containing both.
    
    # More efficient: for triple (a,b,c), the conflicting triples are those that
    # complete a K(2,2,2). For each choice of a', b', c' (all distinct from a,b,c and each other),
    # the 8 transversals are determined. Any other triple in that set conflicts.
    
    # But we need: given triple (a,b,c), find all triples that share a K(2,2,2) with it.
    # A K(2,2,2) containing (a,b,c) has pairs {x,x'}, {y,y'}, {z,z'} where
    # (a,b,c) is one of the 8 transversals.
    
    # The 8 transversals for pairs {x,x'},{y,y'},{z,z'} are all combinations (p,q,r)
    # where p in {x,x'}, q in {y,y'}, r in {z,z'}.
    
    # For (a,b,c) to be a transversal: {a} subset {x,x'}, {b} subset {y,y'}, {c} subset {z,z'}
    # So x or x' = a, y or y' = b, z or z' = c.
    # The other points: x' (if x=a), y' (if y=b), z' (if z=c).
    # So the pairs are {a, a'}, {b, b'}, {c, c'} where a', b', c' are distinct from {a,b,c} and each other.
    
    # For each such (a', b', c'), the 8 transversals are:
    # (a,b,c), (a,b,c'), (a,b',c), (a,b',c'), (a',b,c), (a',b,c'), (a',b',c), (a',b',c')
    
    # So triple (a,b,c) conflicts with the other 7 triples in each such K(2,2,2).
    
    # Precompute conflicts
    conflicts = [set() for _ in range(T)]
    
    for (a, b, c) in all_triples:
        i = triple_index[(a, b, c)]
        # Choose a', b', c' all distinct from {a,b,c} and each other
        others = [p for p in range(n) if p not in (a, b, c)]
        for a_prime in others:
            for b_prime in others:
                if b_prime == a_prime:
                    continue
                for c_prime in others:
                    if c_prime in (a_prime, b_prime):
                        continue
                    # K(2,2,2) with pairs {a,a'}, {b,b'}, {c,c'}
                    # The 8 transversals:
                    trans = [
                        (a, b, c), (a, b, c_prime), (a, b_prime, c), (a, b_prime, c_prime),
                        (a_prime, b, c), (a_prime, b, c_prime), (a_prime, b_prime, c), (a_prime, b_prime, c_prime)
                    ]
                    for t in trans:
                        t_sorted = tuple(sorted(t))
                        if t_sorted != (a, b, c):
                            j = triple_index[t_sorted]
                            conflicts[i].add(j)
                            conflicts[j].add(i)
    
    # Now do greedy construction with randomization + local search
    best_set = set()
    best_count = 0
    
    # Greedy: repeatedly add a random valid triple
    def greedy_random(seed_offset=0):
        random.seed(42 + seed_offset)
        selected = set()
        # Order triples randomly
        order = list(range(T))
        random.shuffle(order)
        for i in order:
            # Check if adding i is valid: no conflict with existing selected triples
            # i.e., no j in selected such that j in conflicts[i]
            valid = True
            for j in selected:
                if j in conflicts[i]:
                    valid = False
                    break
            if valid:
                selected.add(i)
        return selected
    
    # Local search: try to add unselected triples, or swap
    def local_search(selected, max_steps=50000):
        selected = set(selected)
        unselected = set(range(T)) - selected
        improved = True
        steps = 0
        while improved and steps < max_steps and time_left() > 5:
            improved = False
            steps += 1
            # Try to add a random unselected triple
            if unselected:
                candidates = list(unselected)
                random.shuffle(candidates)
                for i in candidates[:50]:
                    # Check if i can be added
                    valid = True
                    for j in selected:
                        if j in conflicts[i]:
                            valid = False
                            break
                    if valid:
                        selected.add(i)
                        unselected.discard(i)
                        improved = True
                        break
            # Try swap: remove one, add another
            if not improved and selected and unselected:
                sel_list = list(selected)
                unsel_list = list(unselected)
                random.shuffle(sel_list)
                random.shuffle(unsel_list)
                for j in sel_list[:20]:
                    for i in unsel_list[:20]:
                        # Check if replacing j with i is valid
                        # i must not conflict with any in selected - {j}
                        valid = True
                        for k in selected:
                            if k != j and k in conflicts[i]:
                                valid = False
                                break
                        if valid:
                            selected.discard(j)
                            selected.add(i)
                            unselected.discard(i)
                            unselected.add(j)
                            improved = True
                            break
                    if improved:
                        break
        return selected
    
    # Run multiple random greedy + local search
    best = set()
    best_size = 0
    
    num_attempts = 50
    for attempt in range(num_attempts):
        if time_left() < 10:
            break
        sel = greedy_random(attempt)
        sel = local_search(sel, max_steps=20000)
        if len(sel) > best_size:
            best_size = len(sel)
            best = set(sel)
    
    # More intensive local search on best
    if time_left() > 20:
        best = local_search(best, max_steps=100000)
        best_size = len(best)
    
    # Convert to list of triples
    result = [list(all_triples[i]) for i in sorted(best)]
    return result

# Keep the original function name
def construct():
    return _construct_impl()

def _construct_impl():
    _START_LOCAL = time.time()
    DEADLINE_LOCAL = 110  # seconds
    
    def time_left():
        return DEADLINE_LOCAL - (time.time() - _START_LOCAL)
    
    n = 16
    all_triples = list(itertools.combinations(range(n), 3))
    T = len(all_triples)  # 560
    triple_index = {t: i for i, t in enumerate(all_triples)}
    
    # Precompute conflicts
    conflicts = [set() for _ in range(T)]
    
    for (a, b, c) in all_triples:
        i = triple_index[(a, b, c)]
        others = [p for p in range(n) if p not in (a, b, c)]
        for a_prime in others:
            for b_prime in others:
                if b_prime == a_prime:
                    continue
                for c_prime in others:
                    if c_prime in (a_prime, b_prime):
                        continue
                    trans = [
                        (a, b, c), (a, b, c_prime), (a, b_prime, c), (a, b_prime, c_prime),
                        (a_prime, b, c), (a_prime, b, c_prime), (a_prime, b_prime, c), (a_prime, b_prime, c_prime)
                    ]
                    for t in trans:
                        t_sorted = tuple(sorted(t))
                        if t_sorted != (a, b, c):
                            j = triple_index[t_sorted]
                            conflicts[i].add(j)
                            conflicts[j].add(i)
    
    # Convert conflicts to frozensets for faster lookup
    conflicts_frozen = [frozenset(s) for s in conflicts]
    
    def greedy_random(seed_offset=0):
        random.seed(42 + seed_offset)
        selected = set()
        order = list(range(T))
        random.shuffle(order)
        for i in order:
            valid = True
            for j in selected:
                if j in conflicts_frozen[i]:
                    valid = False
                    break
            if valid:
                selected.add(i)
        return selected
    
    def local_search(selected, max_steps=50000):
        selected = set(selected)
        unselected = set(range(T)) - selected
        improved = True
        steps = 0
        while improved and steps < max_steps and time_left() > 5:
            improved = False
            steps += 1
            if unselected:
                candidates = list(unselected)
                random.shuffle(candidates)
                for i in candidates[:100]:
                    valid = True
                    for j in selected:
                        if j in conflicts_frozen[i]:
                            valid = False
                            break
                    if valid:
                        selected.add(i)
                        unselected.discard(i)
                        improved = True
                        break
            if not improved and selected and unselected:
                sel_list = list(selected)
                unsel_list = list(unselected)
                random.shuffle(sel_list)
                random.shuffle(unsel_list)
                for j in sel_list[:30]:
                    for i in unsel_list[:30]:
                        valid = True
                        for k in selected:
                            if k != j and k in conflicts_frozen[i]:
                                valid = False
                                break
                        if valid:
                            selected.discard(j)
                            selected.add(i)
                            unselected.discard(i)
                            unselected.add(j)
                            improved = True
                            break
                    if improved:
                        break
        return selected
    
    best = set()
    best_size = 0
    
    num_attempts = 80
    for attempt in range(num_attempts):
        if time_left() < 10:
            break
        sel = greedy_random(attempt)
        sel = local_search(sel, max_steps=30000)
        if len(sel) > best_size:
            best_size = len(sel)
            best = set(sel)
    
    if time_left() > 20:
        best = local_search(best, max_steps=200000)
        best_size = len(best)
    
    result = [list(all_triples[i]) for i in sorted(best)]
    return result

# Re-define construct to call the implementation
def construct():
    return _construct_impl()
