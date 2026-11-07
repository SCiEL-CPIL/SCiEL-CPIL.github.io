"""Rewritten program using combinatorial search to maximize distinct induced degrees."""
from itertools import combinations
from collections import defaultdict

# EVOLVE-BLOCK-START
P = {'n': 17, 'edges': [[0, 1], [0, 2], [0, 4], [0, 8], [0, 9], [0, 13], [0, 15], [0, 16], [1, 2], [1, 3], [1, 5], [1, 9], [1, 10], [1, 14], [1, 16], [2, 3], [2, 4], [2, 6], [2, 10], [2, 11], [2, 15], [3, 4], [3, 5], [3, 7], [3, 11], [3, 12], [3, 16], [4, 5], [4, 6], [4, 8], [4, 12], [4, 13], [5, 6], [5, 7], [5, 9], [5, 13], [5, 14], [6, 7], [6, 8], [6, 10], [6, 14], [6, 15], [7, 8], [7, 9], [7, 11], [7, 15], [7, 16], [8, 9], [8, 10], [8, 12], [8, 16], [9, 10], [9, 11], [9, 13], [10, 11], [10, 12], [10, 14], [11, 12], [11, 13], [11, 15], [12, 13], [12, 14], [12, 16], [13, 14], [13, 15], [14, 15], [14, 16], [15, 16]], 'homogeneous_cap': 4, 'minimum_vertices': 8}

def _build_adj():
    n = P['n']
    adj = [set() for _ in range(n)]
    for u, v in P['edges']:
        adj[u].add(v)
        adj[v].add(u)
    return adj

_ADJ = _build_adj()
_N = P['n']
_MIN_V = P['minimum_vertices']

def _induced_degrees(vertices):
    """Compute the multiset of degrees in the induced subgraph on vertices."""
    vset = set(vertices)
    degs = []
    for v in vertices:
        d = sum(1 for u in vset if u in _ADJ[v])
        degs.append(d)
    return degs

def _score(vertices):
    degs = _induced_degrees(vertices)
    return len(set(degs))

def _greedy_construct():
    """Greedy: build up a set maximizing distinct degrees."""
    # Start with a random-ish seed and greedily add/remove
    import random
    random.seed(42)
    best = None
    best_score = 0
    for _ in range(200):
        # Random starting set of size _MIN_V
        start = set(random.sample(range(_N), _MIN_V))
        cur = list(start)
        cur_score = _score(cur)
        improved = True
        while improved:
            improved = False
            # Try adding a vertex
            for v in range(_N):
                if v in start:
                    continue
                trial = cur + [v]
                s = _score(trial)
                if s > cur_score:
                    cur = trial
                    start.add(v)
                    cur_score = s
                    improved = True
                    break
            # Try removing a vertex (to allow different composition)
            if len(cur) > _MIN_V:
                for v in list(start):
                    trial = [x for x in cur if x != v]
                    s = _score(trial)
                    if s > cur_score:
                        cur = trial
                        start.discard(v)
                        cur_score = s
                        improved = True
                        break
        if cur_score > best_score:
            best_score = cur_score
            best = list(start)
    return best

def _exhaustive_small():
    """Exhaustive over subsets of size 8-10 from a reduced candidate pool."""
    # First, find a good pool via greedy
    pool = _greedy_construct()
    # Expand pool slightly: include all vertices that appear in any good solution
    # For now, just do exhaustive over the greedy result + neighbors
    # Actually, let's do a more systematic search: try all subsets of size 8 from all 17
    # C(17,8) = 24310, which is fine
    best = None
    best_score = 0
    for comb in combinations(range(_N), _MIN_V):
        s = _score(list(comb))
        if s > best_score:
            best_score = s
            best = list(comb)
    return best, best_score

def _local_search(vertices):
    """Local search: try swaps to improve distinct degree count."""
    import random
    random.seed(123)
    cur = list(vertices)
    cur_score = _score(cur)
    for _ in range(500):
        improved = False
        # Try swapping one vertex
        for i, v in enumerate(cur):
            for w in range(_N):
                if w in cur:
                    continue
                trial = cur[:]
                trial[i] = w
                s = _score(trial)
                if s > cur_score:
                    cur = trial
                    cur_score = s
                    improved = True
                    break
            if improved:
                break
        if not improved:
            break
    return cur, cur_score

def construct():
    # Exhaustive search over all 8-vertex subsets (C(17,8)=24310 is fast)
    best = None
    best_score = 0
    for comb in combinations(range(_N), _MIN_V):
        s = _score(list(comb))
        if s > best_score:
            best_score = s
            best = list(comb)
    
    # Also try larger subsets (9-12 vertices) via local search from good starts
    for size in range(_MIN_V + 1, 13):
        # Start from best 8-set and add vertices greedily
        cur = list(best)
        while len(cur) < size:
            best_add = None
            best_add_score = _score(cur)
            for v in range(_N):
                if v in cur:
                    continue
                trial = cur + [v]
                s = _score(trial)
                if s > best_add_score:
                    best_add_score = s
                    best_add = v
            if best_add is not None:
                cur.append(best_add)
            else:
                break
        # Local search
        cur, sc = _local_search(cur)
        if sc > best_score:
            best_score = sc
            best = cur
    
    # Ensure we return at least _MIN_V vertices
    if best is None or len(best) < _MIN_V:
        best = list(range(_MIN_V))
    
    # Return as a list of Python ints
    return [int(v) for v in best]
# EVOLVE-BLOCK-END
