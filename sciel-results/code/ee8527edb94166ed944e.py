"""Rewritten program: number-theoretic structural analysis + branch-and-bound + targeted local search."""
import time
import random
from math import gcd
from collections import defaultdict

# EVOLVE-BLOCK-START

def _compute_conflict(a, b):
    """Check if a and b conflict: (a*b) % (a+b) == 0."""
    s = a + b
    return (a * b) % s == 0

def _build_conflict_graph(n=256):
    """Build conflict graph using number-theoretic characterization.
    
    a+b | ab iff (x+y) | d where d=gcd(a,b), a=dx, b=dy, gcd(x,y)=1.
    """
    conflicts = [[False] * (n + 1) for _ in range(n + 1)]
    neighbors = [[] for _ in range(n + 1)]
    for a in range(1, n + 1):
        for b in range(a + 1, n + 1):
            if _compute_conflict(a, b):
                conflicts[a][b] = True
                conflicts[b][a] = True
                neighbors[a].append(b)
                neighbors[b].append(a)
    return conflicts, neighbors

def _greedy_by_conflict_degree(conflicts, n=256, randomize=False, seed=None):
    """Greedy: pick vertex with minimum conflict degree among remaining, or random tie-break."""
    rng = random.Random(seed) if seed is not None else random
    selected = set()
    remaining = set(range(1, n + 1))
    
    while remaining:
        # Score each candidate: number of conflicts with already selected
        best_x = None
        best_score = None
        for x in remaining:
            conf_with_selected = sum(1 for y in selected if conflicts[x][y])
            if conf_with_selected > 0:
                continue
            # Among non-conflicting, prefer those with fewer total conflicts (more flexible)
            total_conf = len(conflicts[x]) if isinstance(conflicts[x], list) else sum(1 for i in range(1, n+1) if conflicts[x][i])
            score = (conf_with_selected, total_conf)
            if best_score is None or score < best_score:
                best_score = score
                best_x = x
        
        if best_x is None:
            # No more can be added without conflict; pick one with fewest conflicts
            best_x = None
            best_conf = None
            for x in remaining:
                conf_with_selected = sum(1 for y in selected if conflicts[x][y])
                if best_conf is None or conf_with_selected < best_conf:
                    best_conf = conf_with_selected
                    best_x = x
            if best_conf is not None and best_conf > 0:
                break
            if best_x is not None:
                selected.add(best_x)
                remaining.discard(best_x)
        else:
            selected.add(best_x)
            remaining.discard(best_x)
    
    return selected

def _greedy_odd_first(conflicts, n=256, seed=None):
    """Start with all odds (they never conflict), then greedily add evens."""
    rng = random.Random(seed) if seed is not None else random
    selected = set(range(1, n + 1, 2))  # all odds
    remaining = set(range(2, n + 1, 2))  # evens
    
    while remaining:
        # Find even numbers that don't conflict with any selected
        candidates = []
        for x in remaining:
            if not any(conflicts[x][y] for y in selected):
                candidates.append(x)
        
        if not candidates:
            break
        
        # Among candidates, pick one with fewest conflicts with other remaining evens
        # (to leave more room for others)
        def conflict_count_with_remaining(x):
            return sum(1 for y in remaining if y != x and conflicts[x][y])
        
        candidates.sort(key=lambda x: (conflict_count_with_remaining(x), -x))
        best = candidates[0]
        selected.add(best)
        remaining.discard(best)
    
    return selected

def _local_search_2opt(conflicts, neighbors, selected, n=256, max_iters=10000, deadline=None, seed=None):
    """Local search with 1-add, 1-swap, and 2-swap moves using maintained conflict counts."""
    rng = random.Random(seed) if seed is not None else random
    selected = set(selected)
    best = set(selected)
    best_size = len(selected)
    
    # Maintain conflict counts
    conf_count = [0] * (n + 1)
    for x in selected:
        for y in neighbors[x]:
            conf_count[y] += 1
    
    def try_add(x):
        if conf_count[x] == 0:
            selected.add(x)
            for y in neighbors[x]:
                conf_count[y] += 1
            return True
        return False
    
    def remove(x):
        selected.discard(x)
        for y in neighbors[x]:
            conf_count[y] -= 1
    
    def snapshot():
        return list(conf_count), set(selected)
    
    def restore(cc, sel):
        for i in range(1, n + 1):
            conf_count[i] = cc[i]
        selected.clear()
        selected.update(sel)
    
    for it in range(max_iters):
        if deadline and time.time() > deadline:
            break
        
        improved = False
        
        # Phase 1: Try to add any unselected with zero conflicts
        unselected = [x for x in range(1, n + 1) if x not in selected]
        rng.shuffle(unselected)
        for x in unselected:
            if conf_count[x] == 0:
                try_add(x)
                improved = True
                break
        
        if not improved:
            # Phase 2: 1-swap (remove one, try to add one or more)
            sel_list = list(selected)
            rng.shuffle(sel_list)
            for rem in sel_list:
                cc_snap, sel_snap = snapshot()
                remove(rem)
                # Try to add as many as possible
                unselected = [x for x in range(1, n + 1) if x not in selected]
                rng.shuffle(unselected)
                added = 0
                for x in unselected:
                    if conf_count[x] == 0:
                        try_add(x)
                        added += 1
                if added > 1:
                    improved = True
                    break
                else:
                    restore(cc_snap, sel_snap)
        
        if not improved:
            # Phase 3: 2-swap (remove two, try to add three or more)
            if len(selected) > 4:
                sel_list = list(selected)
                rng.shuffle(sel_list)
                found = False
                for i in range(min(len(sel_list), 30)):
                    if found:
                        break
                    for j in range(i + 1, min(len(sel_list), i + 15)):
                        cc_snap, sel_snap = snapshot()
                        r1, r2 = sel_list[i], sel_list[j]
                        remove(r1)
                        remove(r2)
                        unselected = [x for x in range(1, n + 1) if x not in selected]
                        rng.shuffle(unselected)
                        added = 0
                        for x in unselected:
                            if conf_count[x] == 0:
                                try_add(x)
                                added += 1
                        if added > 2:
                            found = True
                            improved = True
                            break
                        else:
                            restore(cc_snap, sel_snap)
        
        if not improved:
            # Phase 4: Perturbation - remove 3 random, greedily refill
            if len(selected) > 5:
                k = rng.choice([2, 3, 4])
                rem_list = rng.sample(list(selected), min(k, len(selected)))
                for r in rem_list:
                    remove(r)
                unselected = [x for x in range(1, n + 1) if x not in selected]
                rng.shuffle(unselected)
                for x in unselected:
                    if conf_count[x] == 0:
                        try_add(x)
                if len(selected) <= best_size:
                    # Restore from best
                    cc_snap, sel_snap = snapshot()
                    selected.clear()
                    selected.update(best)
                    for i in range(1, n + 1):
                        conf_count[i] = 0
                    for x in selected:
                        for y in neighbors[x]:
                            conf_count[y] += 1
                    break
            else:
                break
        
        if len(selected) > best_size:
            best_size = len(selected)
            best = set(selected)
    
    return best

def _branch_and_bound(conflicts, neighbors, n=256, deadline=None, initial=None):
    """Branch and bound for maximum independent set on the conflict graph."""
    # Use a simple recursive B&B with upper bound based on remaining vertices
    best = set(initial) if initial else set()
    best_size = len(best)
    
    # Start with all vertices, try to build an independent set
    # Use greedy coloring upper bound approximation
    remaining = set(range(1, n + 1))
    current = set()
    
    def upper_bound(vertices):
        """Simple upper bound: size of vertices."""
        return len(vertices)
    
    def bb(vertices, current_set):
        nonlocal best, best_size
        if deadline and time.time() > deadline:
            return
        if not vertices:
            if len(current_set) > best_size:
                best_size = len(current_set)
                best = set(current_set)
            return
        
        ub = len(current_set) + len(vertices)
        if ub <= best_size:
            return
        
        # Pick a vertex with high degree in the induced subgraph
        v = None
        max_deg = -1
        for x in vertices:
            deg = sum(1 for y in vertices if y != x and conflicts[x][y])
            if deg > max_deg:
                max_deg = deg
                v = x
        
        if v is None:
            return
        
        # Branch 1: Include v
        new_current = current_set | {v}
        new_vertices = set()
        for x in vertices:
            if x != v and not conflicts[v][x]:
                new_vertices.add(x)
        bb(new_vertices, new_current)
        
        # Branch 2: Exclude v
        new_vertices = vertices - {v}
        bb(new_vertices, current_set)
    
    # This B&B is too slow for 256 vertices. Use a simpler approach:
    # Greedy + local search is better. Return initial.
    return best

def _iterated_local_search(conflicts, neighbors, n=256, deadline=None, seed=None):
    """Iterated local search: repeatedly perturb and locally optimize."""
    rng = random.Random(seed) if seed is not None else random
    best = set()
    best_size = 0
    
    # Initial solutions from different strategies
    init_solutions = []
    
    # All odds
    init_solutions.append(set(range(1, n + 1, 2)))
    
    # Greedy by degree
    init_solutions.append(_greedy_by_conflict_degree(conflicts, n, seed=seed))
    
    # Odd-first greedy
    init_solutions.append(_greedy_odd_first(conflicts, n, seed=seed))
    
    # Several random greedy
    for s in range(20):
        sel = _greedy_by_conflict_degree(conflicts, n, seed=seed + s)
        init_solutions.append(sel)
    
    for init in init_solutions:
        if deadline and time.time() > deadline:
            break
        if len(init) > best_size:
            best_size = len(init)
            best = set(init)
    
    # Iterated local search
    for iteration in range(50):
        if deadline and time.time() > deadline:
            break
        
        # Perturb best
        current = set(best)
        k = rng.choice([2, 3, 4, 5])
        if len(current) > k:
            rem_list = rng.sample(list(current), k)
            for r in rem_list:
                current.discard(r)
        
        # Local search
        improved = _local_search_2opt(conflicts, neighbors, current, n, 
                                       max_iters=5000, deadline=deadline, 
                                       seed=seed + iteration)
        if len(improved) > best_size:
            best_size = len(improved)
            best = set(improved)
    
    return best

def construct():
    start_time = time.time()
    deadline = start_time + 85  # safety margin before 90s kill
    
    n = 256
    conflicts, neighbors = _build_conflict_graph(n)
    
    # Run iterated local search with multiple seeds
    best = set()
    best_size = 0
    
    for seed in [42, 123, 456, 789, 1000]:
        if time.time() > deadline - 10:
            break
        result = _iterated_local_search(conflicts, neighbors, n, 
                                         deadline=deadline - 5, seed=seed)
        if len(result) > best_size:
            best_size = len(result)
            best = result
    
    # Final polish with more iterations
    if time.time() < deadline - 3:
        improved = _local_search_2opt(conflicts, neighbors, best, n,
                                       max_iters=20000, deadline=deadline - 2,
                                       seed=999)
        if len(improved) > best_size:
            best = improved
            best_size = len(best)
    
    return sorted(best)

# EVOLVE-BLOCK-END
