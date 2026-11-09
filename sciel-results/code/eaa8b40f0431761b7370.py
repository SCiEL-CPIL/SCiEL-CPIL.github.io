"""Maximum independent set in a 3-uniform hypergraph via branch-and-bound + local search."""
import time
import random
import math
from collections import defaultdict

_DEADLINE = 85.0
N = 180

def _precompute_triples():
    """
    Find all triples (a, b, c) of distinct integers in [1, 180] such that
    1/a = 1/b + 1/c.
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
            if b < 1 or c < 1:
                continue
            if a == b or a == c or b == c:
                continue
            triples.append((a, b, c))
    return triples

def _build_conflict_graph(triples):
    """
    Build a conflict graph where edge (u,v) exists if there is a triple containing both u and v.
    Also store for each pair the list of triples containing them.
    """
    # conflict[u][v] = list of triples containing both u and v
    conflict = defaultdict(lambda: defaultdict(list))
    for t in triples:
        a, b, c = t
        conflict[a][b].append(t)
        conflict[a][c].append(t)
        conflict[b][c].append(t)
    return conflict

def _count_violations(set_bits, triples):
    count = 0
    for (a, b, c) in triples:
        if set_bits[a] and set_bits[b] and set_bits[c]:
            count += 1
    return count

def _is_valid(set_bits, triples):
    for (a, b, c) in triples:
        if set_bits[a] and set_bits[b] and set_bits[c]:
            return False
    return True

def _greedy_min_degree(triples, conflict, random_seed=None):
    """
    Minimum-degree greedy: iteratively select the vertex with minimum conflict degree
    among remaining candidates, add it to the set, and remove its conflicts.
    """
    if random_seed is not None:
        random.seed(random_seed)
    
    remaining = set(range(1, N + 1))
    selected = set()
    
    while remaining:
        # Compute conflict degree for each remaining vertex
        # Degree = number of other remaining vertices that conflict with it
        best_vertex = None
        best_degree = float('inf')
        
        for v in remaining:
            degree = 0
            for u in remaining:
                if u == v:
                    continue
                if u in conflict[v]:
                    degree += 1
            if degree < best_degree:
                best_degree = degree
                best_vertex = v
        
        if best_vertex is None:
            break
        
        selected.add(best_vertex)
        # Remove best_vertex and all its conflicts from remaining
        to_remove = {best_vertex}
        for u in conflict[best_vertex]:
            if u in remaining:
                to_remove.add(u)
        remaining -= to_remove
    
    return selected

def _local_search(initial_set, triples, time_limit):
    """
    Local search: try to add elements that don't create violations,
    and swap elements to improve size.
    """
    set_bits = [False] * (N + 1)
    current_set = set(initial_set)
    for x in current_set:
        set_bits[x] = True
    
    current_size = len(current_set)
    start_time = time.time()
    
    # First, try to add any element that doesn't create a violation
    changed = True
    while changed:
        if time.time() - start_time > time_limit:
            break
        changed = False
        for x in range(1, N + 1):
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
                current_set.add(x)
                current_size += 1
                changed = True
    
    # Now try swaps: remove one element, add another
    for _ in range(50):
        if time.time() - start_time > time_limit:
            break
        improved = False
        # Try removing each element and see if we can add more
        for x in list(current_set):
            if time.time() - start_time > time_limit:
                break
            set_bits[x] = False
            current_set.discard(x)
            
            # Try to add elements
            added = 0
            for y in range(1, N + 1):
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
                    current_set.add(y)
                    added += 1
            
            if added <= 1:
                # Revert
                set_bits[x] = True
                current_set.add(x)
            else:
                improved = True
                break
    
    return current_set

def _simulated_annealing(initial_set, triples, time_limit):
    """
    Simulated annealing with efficient violation counting.
    """
    set_bits = [False] * (N + 1)
    current_set = set(initial_set)
    for x in current_set:
        set_bits[x] = True
    
    current_size = len(current_set)
    current_violations = _count_violations(set_bits, triples)
    
    # Precompute for each element, which triples it's in
    elem_triples = defaultdict(list)
    for i, (a, b, c) in enumerate(triples):
        elem_triples[a].append(i)
        elem_triples[b].append(i)
        elem_triples[c].append(i)
    
    lambda_pen = 10000
    def score(size, viol):
        return size - lambda_pen * viol
    
    current_score = score(current_size, current_violations)
    best_set = set(current_set)
    best_size = current_size
    best_score = current_score
    
    T = 10.0
    cooling = 0.999
    start_time = time.time()
    
    while T > 0.01:
        if time.time() - start_time > time_limit:
            break
        
        x = random.randint(1, N)
        
        if set_bits[x]:
            # Remove x
            set_bits[x] = False
            new_size = current_size - 1
            # Count new violations: only triples containing x can change status
            new_violations = current_violations
            for ti in elem_triples[x]:
                a, b, c = triples[ti]
                if set_bits[a] and set_bits[b] and set_bits[c]:
                    new_violations += 1
            # Wait, that's wrong. Removing x can only reduce violations.
            # Violations that had x in them are now resolved.
            # So new_violations = current_violations - (number of violated triples containing x)
            # Let me recount properly
            set_bits[x] = True  # restore for counting
            violated_with_x = 0
            for ti in elem_triples[x]:
                a, b, c = triples[ti]
                if set_bits[a] and set_bits[b] and set_bits[c]:
                    violated_with_x += 1
            set_bits[x] = False
            new_violations = current_violations - violated_with_x
            new_score = score(new_size, new_violations)
            
            delta = new_score - current_score
            if delta >= 0 or random.random() < math.exp(delta / T):
                current_size = new_size
                current_violations = new_violations
                current_score = new_score
                current_set.discard(x)
                if new_violations == 0 and new_size > best_size:
                    best_size = new_size
                    best_score = new_score
                    best_set = set(current_set)
            else:
                set_bits[x] = True
        else:
            # Add x
            set_bits[x] = True
            new_size = current_size + 1
            # Count new violations: only triples containing x can become violated
            new_violations = current_violations
            for ti in elem_triples[x]:
                a, b, c = triples[ti]
                if set_bits[a] and set_bits[b] and set_bits[c]:
                    new_violations += 1
            new_score = score(new_size, new_violations)
            
            delta = new_score - current_score
            if delta >= 0 or random.random() < math.exp(delta / T):
                current_size = new_size
                current_violations = new_violations
                current_score = new_score
                current_set.add(x)
                if new_violations == 0 and new_size > best_size:
                    best_size = new_size
                    best_score = new_score
                    best_set = set(current_set)
            else:
                set_bits[x] = False
        
        T *= cooling
    
    return best_set

def _branch_and_bound(triples, time_limit):
    """
    Branch and bound for maximum independent set in hypergraph.
    """
    start_time = time.time()
    
    # Build adjacency: for each vertex, list of vertices it conflicts with
    # Two vertices conflict if they appear together in some triple
    conflict_with = [set() for _ in range(N + 1)]
    for (a, b, c) in triples:
        conflict_with[a].add(b)
        conflict_with[a].add(c)
        conflict_with[b].add(a)
        conflict_with[b].add(c)
        conflict_with[c].add(a)
        conflict_with[c].add(b)
    
    best_size = [0]
    best_set = [set()]
    
    def search(available, current_set, depth):
        if time.time() - start_time > time_limit:
            return
        
        if not available:
            if len(current_set) > best_size[0]:
                best_size[0] = len(current_set)
                best_set[0] = set(current_set)
            return
        
        # Upper bound: current size + number of available vertices
        upper_bound = len(current_set) + len(available)
        if upper_bound <= best_size[0]:
            return
        
        # Choose vertex with minimum degree in available
        min_deg = float('inf')
        chosen = None
        for v in available:
            deg = len(conflict_with[v] & available)
            if deg < min_deg:
                min_deg = deg
                chosen = v
        
        if chosen is None:
            return
        
        # Branch 1: include chosen
        new_available = available - {chosen} - conflict_with[chosen]
        current_set.add(chosen)
        search(new_available, current_set, depth + 1)
        current_set.discard(chosen)
        
        # Branch 2: exclude chosen
        search(available - {chosen}, current_set, depth + 1)
    
    available = set(range(1, N + 1))
    search(available, set(), 0)
    
    return best_set[0]

def construct():
    start_time = time.time()
    
    triples = _precompute_triples()
    conflict = _build_conflict_graph(triples)
    
    best_set = set()
    
    # Strategy 1: Multiple greedy min-degree with different random seeds
    for seed in range(50):
        if time.time() - start_time > _DEADLINE - 5:
            break
        g = _greedy_min_degree(triples, conflict, random_seed=seed)
        if len(g) > len(best_set):
            best_set = g
    
    # Strategy 2: Local search from best greedy
    if time.time() - start_time < _DEADLINE - 10:
        ls = _local_search(best_set, triples, _DEADLINE - (time.time() - start_time) - 10)
        if len(ls) > len(best_set):
            best_set = ls
    
    # Strategy 3: Simulated annealing from best set
    if time.time() - start_time < _DEADLINE - 15:
        sa = _simulated_annealing(best_set, triples, _DEADLINE - (time.time() - start_time) - 15)
        if len(sa) > len(best_set):
            best_set = sa
    
    # Strategy 4: More SA restarts
    for restart in range(10):
        if time.time() - start_time > _DEADLINE - 5:
            break
        random.seed(1000 + restart)
        alt = _greedy_min_degree(triples, conflict, random_seed=1000 + restart)
        sa = _simulated_annealing(alt, triples, max(0, _DEADLINE - (time.time() - start_time) - 5))
        if len(sa) > len(best_set):
            best_set = sa
    
    # Final validation
    set_bits = [False] * (N + 1)
    for x in best_set:
        set_bits[x] = True
    if not _is_valid(set_bits, triples):
        # Fallback: remove elements causing violations
        for x in sorted(best_set, reverse=True):
            set_bits[x] = False
            if _is_valid(set_bits, triples):
                break
        best_set = {x for x in range(1, N + 1) if set_bits[x]}
    
    return sorted(best_set)
