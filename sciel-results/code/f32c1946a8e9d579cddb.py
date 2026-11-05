"""Find a long simple path in the pinned G(100, 1.8/100) graph."""
import time
import sys
import random

sys.setrecursionlimit(1000000)

P = {'n': 100, 'edges': [[0, 10], [0, 27], [0, 80], [1, 59], [2, 50], [2, 74], [3, 12], [3, 18], [3, 62], [3, 87], [4, 48], [4, 72], [4, 89], [5, 11], [6, 36], [8, 13], [8, 45], [8, 52], [8, 94], [9, 26], [9, 47], [9, 78], [10, 81], [11, 18], [11, 21], [11, 84], [12, 22], [12, 65], [12, 83], [13, 30], [13, 76], [15, 82], [17, 35], [19, 38], [19, 49], [19, 61], [19, 96], [20, 52], [21, 23], [21, 58], [22, 33], [22, 48], [24, 29], [24, 31], [24, 55], [24, 58], [24, 89], [25, 82], [26, 44], [26, 69], [30, 45], [30, 70], [32, 96], [33, 55], [34, 66], [35, 38], [35, 93], [36, 81], [37, 42], [37, 70], [39, 59], [40, 69], [40, 97], [41, 80], [42, 44], [42, 48], [42, 68], [43, 53], [44, 50], [46, 50], [46, 73], [46, 97], [47, 78], [48, 77], [49, 59], [50, 59], [50, 62], [51, 57], [53, 68], [53, 80], [56, 58], [58, 95], [60, 65], [60, 90], [61, 88], [62, 90], [68, 81], [69, 90], [70, 92], [74, 85], [75, 82]], 'edge_probability': [9, 500]}

N = P['n']
ADJ = [set() for _ in range(N)]
for u, v in P['edges']:
    ADJ[u].add(v)
    ADJ[v].add(u)
# Precompute neighbor lists (sorted) for fast iteration
NEI = [sorted(a) for a in ADJ]

# Components
_comp = [-1] * N
_cid = 0
for s in range(N):
    if _comp[s] == -1:
        stack = [s]
        _comp[s] = _cid
        while stack:
            x = stack.pop()
            for y in NEI[x]:
                if _comp[y] == -1:
                    _comp[y] = _cid
                    stack.append(y)
        _cid += 1
COMP_SIZE = [0] * _cid
for s in range(N):
    COMP_SIZE[_comp[s]] += 1

DEADLINE = time.time() + 80.0  # safety margin before 90s kill
_best = []
_best_len = 0


def _greedy_from(start):
    """Greedy: at each step go to the neighbor with fewest unvisited options (Warnsdorff)."""
    path = [start]
    seen = {start}
    while True:
        cur = path[-1]
        cands = [w for w in NEI[cur] if w not in seen]
        if not cands:
            break
        v = min(cands, key=lambda w: (len([z for z in NEI[w] if z not in seen]), w))
        path.append(v)
        seen.add(v)
    return path


def _dfs_search(start, time_budget):
    """Time-limited DFS backtracking with upper-bound pruning to extend a path from start."""
    global _best, _best_len
    deadline = time.time() + time_budget
    seen = [False] * N
    seen[start] = True
    path = [start]
    # Count unvisited neighbors of current for pruning
    # Upper bound: current path length + number of unvisited vertices reachable.
    # We track total unvisited count = N - len(path).
    # Prune if len(path) + (N - len(path)) <= _best_len is never true, so use a
    # tighter bound: len(path) + (unvisited count in the same component).
    comp_of_start = _comp[start]
    comp_unvisited = COMP_SIZE[comp_of_start]  # will decrement as we visit

    def upper_bound():
        return len(path) + comp_unvisited

    def dfs():
        nonlocal comp_unvisited
        if time.time() > deadline:
            return False  # signal: stop
        if len(path) > _best_len:
            _best_len = len(path)
            _best = list(path)
        cur = path[-1]
        # candidates: unvisited neighbors
        cands = [w for w in NEI[cur] if not seen[w]]
        if not cands:
            return True
        # order: prefer neighbors that lead to more unvisited options (to find long paths fast)
        cands.sort(key=lambda w: -len([z for z in NEI[w] if not seen[z]]))
        for w in cands:
            # prune: even if we visit all remaining, can we beat best?
            if len(path) + 1 + (comp_unvisited - 1) <= _best_len:
                return True
            seen[w] = True
            path.append(w)
            comp_unvisited -= 1
            if not dfs():
                return False
            path.pop()
            seen[w] = False
            comp_unvisited += 1
            if time.time() > deadline:
                return False
        return True

    dfs()
    return _best


def _two_opt(path):
    """Improve a path's length via 2-opt-style local moves. Returns improved path."""
    global _best, _best_len
    path = list(path)
    seen = set(path)
    n = len(path)
    improved = True
    guard = 0
    while improved and guard < 2000:
        improved = False
        guard += 1
        # Try to extend from either end by inserting an unvisited vertex between
        # two consecutive vertices, or at the ends.
        # End extension:
        for end in (0, 1):
            if end == 0:
                cur = path[0]
            else:
                cur = path[-1]
            for w in NEI[cur]:
                if w not in seen:
                    if end == 0:
                        path.insert(0, w)
                    else:
                        path.append(w)
                    seen.add(w)
                    improved = True
                    break
        # Internal insertion: find i<j such that path[i] and path[j+1] are adjacent
        # and we can insert the subpath between. Simpler: try to insert a single
        # unvisited vertex v between consecutive a=path[i], b=path[i+1] if
        # a-v and v-b are edges.
        for i in range(len(path) - 1):
            a = path[i]
            b = path[i + 1]
            # common unvisited neighbor of a and b
            for v in NEI[a]:
                if v not in seen and v in ADJ[b]:
                    path.insert(i + 1, v)
                    seen.add(v)
                    improved = True
                    break
            if improved:
                break
    if len(path) > _best_len:
        _best_len = len(path)
        _best = list(path)
    return path


def construct():
    global _best, _best_len
    # Phase 1: greedy from every start (fast baseline)
    for s in range(N):
        p = _greedy_from(s)
        if len(p) > _best_len:
            _best_len = len(p)
            _best = p

    # Phase 2: 2-opt refine current best
    _two_opt(_best)

    # Phase 3: time-limited DFS backtracking from promising starts
    # Order starts by component size descending, and by degree.
    starts = sorted(range(N), key=lambda s: (-COMP_SIZE[_comp[s]], -len(NEI[s])))
    remaining = DEADLINE - time.time()
    if remaining > 1.0:
        # Give each start a slice of the remaining budget
        per_start = remaining / max(1, len(starts))
        for s in starts:
            if time.time() > DEADLINE - 1.0:
                break
            # skip if this component can't beat best
            if COMP_SIZE[_comp[s]] <= _best_len:
                continue
            budget = min(per_start, DEADLINE - time.time() - 0.5)
            if budget > 0.05:
                _dfs_search(s, budget)
            # refine after each DFS improvement
            if len(_best) > 0:
                _two_opt(_best)

    if not _best:
        # Fallback: return a single vertex (shouldn't happen)
        return [0]
    return list(_best)


if __name__ == "__main__":
    res = construct()
    print(len(res), res[:10], "...")
