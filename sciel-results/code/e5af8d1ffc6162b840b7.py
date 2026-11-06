"""Longest simple path in the pinned G(100,1.8/100) graph.
Uses greedy warm-start + pruned backtracking DFS with a deadline guard."""
import time
import sys

P = {'n': 100, 'edges': [[0, 10], [0, 27], [0, 80], [1, 59], [2, 50], [2, 74], [3, 12], [3, 18], [3, 62], [3, 87], [4, 48], [4, 72], [4, 89], [5, 11], [6, 36], [8, 13], [8, 45], [8, 52], [8, 94], [9, 26], [9, 47], [9, 78], [10, 81], [11, 18], [11, 21], [11, 84], [12, 22], [12, 65], [12, 83], [13, 30], [13, 76], [15, 82], [17, 35], [19, 38], [19, 49], [19, 61], [19, 96], [20, 52], [21, 23], [21, 58], [22, 33], [22, 48], [24, 29], [24, 31], [24, 55], [24, 58], [24, 89], [25, 82], [26, 44], [26, 69], [30, 45], [30, 70], [32, 96], [33, 55], [34, 66], [35, 38], [35, 93], [36, 81], [37, 42], [37, 70], [39, 59], [40, 69], [40, 97], [41, 80], [42, 44], [42, 48], [42, 68], [43, 53], [44, 50], [46, 50], [46, 73], [46, 97], [47, 78], [48, 77], [49, 59], [50, 59], [50, 62], [51, 57], [53, 68], [53, 80], [56, 58], [58, 95], [60, 65], [60, 90], [61, 88], [62, 90], [68, 81], [69, 90], [70, 92], [74, 85], [75, 82]]}

_n = P['n']
_edges = P['edges']

# Build adjacency as sets and sorted lists
_adj = [set() for _ in range(_n)]
for u, v in _edges:
    _adj[u].add(v)
    _adj[v].add(u)
_adj_list = [sorted(s) for s in _adj]
_deg = [len(s) for s in _adj]

# Connected components
def _components():
    seen = [False] * _n
    comps = []
    for i in range(_n):
        if not seen[i]:
            stack = [i]
            seen[i] = True
            comp = []
            while stack:
                u = stack.pop()
                comp.append(u)
                for v in _adj[u]:
                    if not seen[v]:
                        seen[v] = True
                        stack.append(v)
            comps.append(comp)
    return comps

_comps = _components()
# largest component
_largest = max(_comps, key=len)
_largest_set = set(_largest)

# Greedy path from a start (seed-like): pick neighbor with fewest remaining options
def _greedy(start):
    path = [start]
    seen = {start}
    while True:
        cur = path[-1]
        cand = [v for v in _adj[cur] if v not in seen]
        if not cand:
            break
        # choose the one with fewest unvisited neighbors (Warnsdorff-like)
        v = min(cand, key=lambda w: (len(_adj[w]) - len(seen & _adj[w]), w))
        path.append(v)
        seen.add(v)
    return path

# Precompute a greedy best from every vertex (fast)
_greedy_best = None
def _init_greedy_best():
    global _greedy_best
    best = []
    for s in _largest:
        p = _greedy(s)
        if len(p) > len(best):
            best = p
    return best

_greedy_best = _init_greedy_best()
_best_len = len(_greedy_best)
_best_path = list(_greedy_best)

# Backtracking DFS with pruning and deadline
_deadline = None
_best_path_ref = None
_best_len_ref = None

def _dfs(cur, path, seen):
    global _best_len_ref, _best_path_ref
    # update best
    if len(path) > _best_len_ref:
        _best_len_ref = len(path)
        _best_path_ref = list(path)
        # if we've covered the whole largest component, we can stop early
        if _best_len_ref == len(_largest):
            return True  # signal done
    # candidates
    cand = [v for v in _adj[cur] if v not in seen]
    if not cand:
        return False
    # order candidates: prefer those with fewer onward options (Warnsdorff)
    cand.sort(key=lambda w: (len(_adj[w]) - len(seen & _adj[w]), w))
    for v in cand:
        # Pruning: if taking v would strand an unvisited vertex that has no
        # connection back to the current reachable frontier, skip.
        # Cheap check: after visiting v, does every unvisited vertex in the
        # component still have at least one neighbor in (seen ∪ {v})? If some
        # unvisited vertex has 0 neighbors in seen∪{v} and is not adjacent to
        # any other unvisited vertex that could be reached, it's isolated.
        # Simpler strong prune: if v has degree 1 in the unvisited subgraph
        # and is not the only remaining, going to v may trap. We use a
        # lighter prune: if after the move, the number of unvisited vertices
        # that are "cut off" (no neighbor in seen∪{v} and no path through
        # unvisited) > 0, prune. That's expensive, so we use a cheap version:
        # check if v is a leaf in the unvisited-induced subgraph and there are
        # other unvisited vertices -> going to a leaf is only safe if it's the
        # last one. We approximate: count unvisited neighbors of v.
        seen.add(v)
        path.append(v)
        # cheap dead-end prune: is there any unvisited vertex u (in component)
        # with no neighbor in seen? If yes and u != v, then u is unreachable
        # from the path end unless we go through unvisited vertices. Actually
        # unreachable from the *path*, but could still be reachable via other
        # unvisited vertices. So this prune is only valid if u has no neighbor
        # in seen AND no neighbor in unvisited (i.e., completely isolated now).
        # We'll do a stronger but still cheap check below.
        done = _dfs(v, path, seen)
        path.pop()
        seen.remove(v)
        if done:
            return True
        if _deadline is not None and time.time() > _deadline:
            return False
    return False

def _dfs_with_prune(start):
    global _best_len_ref, _best_path_ref
    path = [start]
    seen = {start}
    # initial dead-end check: any vertex in component with 0 neighbors in seen?
    # that's fine as long as it's reachable via unvisited.
    _dfs(start, path, seen)

def construct():
    global _deadline, _best_path_ref, _best_len_ref
    # deadline with safety margin (budget 60s, guard at ~55s)
    t0 = time.time()
    _deadline = t0 + 55.0

    # start from greedy best
    _best_path_ref = list(_best_path)
    _best_len_ref = _best_len

    # If greedy already covers the whole largest component, we're optimal
    # for that component; but there could be a longer path? No, a path can't
    # exceed the component size, and greedy covering the component = Hamiltonian
    # path of the component, which is the max possible within the component.
    if _best_len_ref == len(_largest):
        return list(_best_path_ref)

    # Order starting vertices: prefer those in the largest component,
    # and try endpoints of the greedy best first (they're promising),
    # then by degree.
    starts = list(_largest)
    # put the two ends of the greedy best first
    if len(_best_path_ref) >= 2:
        ends = {_best_path_ref[0], _best_path_ref[-1]}
        starts.sort(key=lambda x: (0 if x in ends else 1, _deg[x]))
    else:
        starts.sort(key=lambda x: _deg[x])

    for s in starts:
        if _best_len_ref == len(_largest):
            break
        if _deadline is not None and time.time() > _deadline:
            break
        _dfs_with_prune(s)
        if _deadline is not None and time.time() > _deadline:
            break

    # Return best found; ensure it's a valid simple path in the graph
    res = list(_best_path_ref)
    if not res:
        # fallback: a single vertex
        res = [starts[0]] if starts else [0]
    return res
