"""Longest simple path in the pinned G(100,1.8/100) graph via multi-start greedy + 2-opt + extension."""
import time
import random

# EVOLVE-BLOCK-START
P = {'n': 100, 'edges': [[0, 10], [0, 27], [0, 80], [1, 59], [2, 50], [2, 74], [3, 12], [3, 18], [3, 62], [3, 87], [4, 48], [4, 72], [4, 89], [5, 11], [6, 36], [8, 13], [8, 45], [8, 52], [8, 94], [9, 26], [9, 47], [9, 78], [10, 81], [11, 18], [11, 21], [11, 84], [12, 22], [12, 65], [12, 83], [13, 30], [13, 76], [15, 82], [17, 35], [19, 38], [19, 49], [19, 61], [19, 96], [20, 52], [21, 23], [21, 58], [22, 33], [22, 48], [24, 29], [24, 31], [24, 55], [24, 58], [24, 89], [25, 82], [26, 44], [26, 69], [30, 45], [30, 70], [32, 96], [33, 55], [34, 66], [35, 38], [35, 93], [36, 81], [37, 42], [37, 70], [39, 59], [40, 69], [40, 97], [41, 80], [42, 44], [42, 48], [42, 68], [43, 53], [44, 50], [46, 50], [46, 73], [46, 97], [47, 78], [48, 77], [49, 59], [50, 59], [50, 62], [51, 57], [53, 68], [53, 80], [56, 58], [58, 95], [60, 65], [60, 90], [61, 88], [62, 90], [68, 81], [69, 90], [70, 92], [74, 85], [75, 82]], 'edge_probability': [9, 500]}

N = P['n']
EDGES = P['edges']
ADJ = [set() for _ in range(N)]
for u, v in EDGES:
    ADJ[u].add(v)
    ADJ[v].add(u)
ADJ_LIST = [sorted(ADJ[i]) for i in range(N)]
EDGE_SET = set()
for u, v in EDGES:
    EDGE_SET.add((min(u, v), max(u, v)))

def is_edge(u, v):
    if u == v:
        return False
    if u < v:
        return (u, v) in EDGE_SET
    return (v, u) in EDGE_SET

def is_valid_path(path):
    if not path:
        return False
    if len(set(path)) != len(path):
        return False
    for i in range(len(path) - 1):
        if not is_edge(path[i], path[i + 1]):
            return False
    return True

def extend_path(path, seen):
    """Greedy extend from both ends using Warnsdorff-like heuristic (prefer low remaining degree)."""
    changed = True
    while changed:
        changed = False
        # try extending front
        front = path[0]
        cands = [w for w in ADJ[front] if w not in seen]
        if cands:
            w = min(cands, key=lambda x: (len(ADJ[x]) - len(seen & ADJ[x]), x))
            path.insert(0, w)
            seen.add(w)
            changed = True
        # try extending back
        back = path[-1]
        cands = [w for w in ADJ[back] if w not in seen]
        if cands:
            w = min(cands, key=lambda x: (len(ADJ[x]) - len(seen & ADJ[x]), x))
            path.append(w)
            seen.add(w)
            changed = True
    return path

def greedy_path(start, rng):
    """Build a path starting at `start` using randomized Warnsdorff heuristic."""
    path = [start]
    seen = {start}
    while True:
        back = path[-1]
        cands = [w for w in ADJ[back] if w not in seen]
        if not cands:
            break
        # prefer vertices with fewest unvisited neighbors (Warnsdorff)
        cands.sort(key=lambda w: (len([x for x in ADJ[w] if x not in seen]), w))
        # randomize among the best few to escape local minima
        k = min(3, len(cands))
        chosen = rng.choice(cands[:k])
        path.append(chosen)
        seen.add(chosen)
    # extend from front too
    extend_path(path, seen)
    return path

def two_opt(path):
    """Apply 2-opt reversals: for each pair i<j, if reversing path[i..j] keeps validity and
    improves (exposes new extendable endpoints), do it. We accept a reversal if after reversal
    the path is still valid (it always is, since we only flip a contiguous segment and the
    internal edges are the same) and then we extend. We keep the best resulting length."""
    n = len(path)
    if n < 3:
        return path
    best = list(path)
    best_len = n
    for i in range(n - 1):
        for j in range(i + 2, n):
            # candidate: path[0:i] + reversed(path[i:j+1]) + path[j+1:]
            cand = path[:i] + path[i:j + 1][::-1] + path[j + 1:]
            # validity check: internal edges of the reversed segment are the same set, so only
            # the two boundary edges (i-1,i) and (j,j+1) might change.
            # Actually reversing keeps all internal adjacencies; the only new adjacencies are
            # between path[i-1] and path[j] (if i>0) and between path[i] and path[j+1] (if j<n-1).
            ok = True
            if i > 0:
                if not is_edge(cand[i - 1], cand[i]):
                    ok = False
            if ok and j + 1 < n:
                if not is_edge(cand[j], cand[j + 1]):
                    ok = False
            if not ok:
                continue
            # extend
            seen = set(cand)
            ext = extend_path(cand, seen)
            if len(ext) > best_len:
                best = ext
                best_len = len(ext)
                # restart loops with new length
                n = len(path)  # keep original n for outer loops; best may be longer
    return best

def local_search(path, rng, deadline):
    """Iterate 2-opt + extension until no improvement or deadline."""
    cur = list(path)
    seen = set(cur)
    cur = extend_path(cur, seen)
    improved = True
    while improved and time.time() < deadline:
        improved = False
        nxt = two_opt(cur)
        if len(nxt) > len(cur):
            cur = nxt
            improved = True
    return cur

def construct():
    deadline = time.time() + 75.0  # safety margin under 90s
    rng = random.Random(748)
    best = []

    # Phase 1: deterministic greedy from every start (Warnsdorff, no randomness)
    for start in range(N):
        if time.time() > deadline:
            break
        path = [start]
        seen = {start}
        while True:
            back = path[-1]
            cands = [w for w in ADJ[back] if w not in seen]
            if not cands:
                break
            cands.sort(key=lambda w: (len([x for x in ADJ[w] if x not in seen]), w))
            path.append(cands[0])
            seen.add(cands[0])
        path = extend_path(path, seen)
        if len(path) > len(best):
            best = list(path)

    # Phase 2: multi-start random greedy + 2-opt + extension
    attempts = 0
    max_attempts = 400
    while attempts < max_attempts and time.time() < deadline:
        attempts += 1
        start = rng.randrange(N)
        path = greedy_path(start, rng)
        path = local_search(path, rng, deadline)
        if len(path) > len(best):
            best = list(path)

    # Phase 3: 2-opt + extension on the best found
    if time.time() < deadline:
        best = local_search(best, rng, deadline)

    # Fallback: ensure nonempty valid
    if not best:
        best = [0]
    # sanity: if invalid, fall back to a single vertex
    if not is_valid_path(best):
        best = [0]
    return best
# EVOLVE-BLOCK-END
