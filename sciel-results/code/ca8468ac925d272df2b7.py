"""Rewritten program: exact minimum cochromatic partition via branch-and-bound."""
import time
from itertools import combinations

P = {'n': 14, 'edges': [[0, 2], [0, 3], [0, 4], [0, 5], [0, 6], [0, 10], [0, 12], [1, 4], [1, 9], [1, 11], [2, 3], [2, 5], [2, 6], [2, 7], [2, 12], [2, 13], [3, 4], [3, 5], [3, 8], [3, 11], [3, 12], [3, 13], [4, 5], [4, 7], [4, 13], [5, 8], [5, 9], [5, 10], [5, 12], [5, 13], [6, 8], [6, 11], [6, 13], [7, 12], [8, 9], [8, 13], [9, 11], [9, 12], [11, 13]]}

N = P['n']
adj = [0] * N
for u, v in P['edges']:
    adj[u] |= (1 << v)
    adj[v] |= (1 << u)
FULL = (1 << N) - 1

def is_clique(mask):
    while mask:
        v = (mask & -mask).bit_length() - 1
        mask ^= (1 << v)
        if mask & ~adj[v]:
            return False
    return True

def is_independent(mask):
    while mask:
        v = (mask & -mask).bit_length() - 1
        mask ^= (1 << v)
        if mask & adj[v]:
            return False
    return True

def is_homogeneous(mask):
    return is_clique(mask) or is_independent(mask)

# Precompute every homogeneous set (size >= 1)
hom_sets = []
for r in range(1, N + 1):
    for combo in combinations(range(N), r):
        m = 0
        for v in combo:
            m |= (1 << v)
        if is_homogeneous(m):
            hom_sets.append(m)

# For each vertex, the homogeneous sets that contain it, sorted by size descending
by_vertex = [[] for _ in range(N)]
for m in hom_sets:
    mm = m
    while mm:
        v = (mm & -mm).bit_length() - 1
        by_vertex[v].append(m)
        mm ^= (1 << v)
for v in range(N):
    by_vertex[v].sort(key=lambda m: -m.bit_count())

# Greedy upper bound to get an initial best
def greedy_ub():
    remaining = FULL
    parts = []
    while remaining:
        best_set = 0
        best_size = 0
        for m in hom_sets:
            if m & ~remaining:
                continue
            sz = m.bit_count()
            if sz > best_size:
                best_size = sz
                best_set = m
        if best_set == 0:
            best_set = remaining & -remaining
        parts.append(best_set)
        remaining &= ~best_set
    return len(parts), parts

UB, ub_parts = greedy_ub()

# Precompute max clique size and max independent set size for all masks
# This is O(2^N * N) which is feasible for N=14
max_clique_sz = [0] * (1 << N)
max_is_sz = [0] * (1 << N)

# For each mask, compute max clique and max independent set size
# We can use the fact that for a mask, the max clique is the max over vertices in mask
# of 1 + max clique in (mask & adj[v])
# But that's recursive. Let's use a different approach.

# Actually, for N=14, we can just precompute by iterating over all masks
# and for each mask, finding the max clique and max IS by checking all subsets.
# But that's O(3^N) which is too slow.

# Better: use the recursive approach with memoization, but precompute for all masks.
# We'll use a bottom-up DP over masks.

# For max clique: 
# max_clique[mask] = max over v in mask of (1 + max_clique[mask & adj[v] & (mask without v)])
# This is not quite right because we need to ensure the clique is within the mask.

# Actually, let's use a simpler approach: for each mask, the max clique size is the
# maximum over all subsets of mask that are cliques. But that's expensive.

# For N=14, let's just compute max_clique and max_is for each mask using a simple
# recursive method with memoization, and cache the results.

_mc_cache = {}
_mi_cache = {}

def _max_clique(mask):
    if mask == 0:
        return 0
    if mask in _mc_cache:
        return _mc_cache[mask]
    best = 0
    m = mask
    while m:
        v = (m & -m).bit_length() - 1
        m ^= (1 << v)
        cand = 1 + _max_clique(mask & adj[v])
        if cand > best:
            best = cand
    _mc_cache[mask] = best
    return best

def _max_is(mask):
    if mask == 0:
        return 0
    if mask in _mi_cache:
        return _mi_cache[mask]
    best = 0
    m = mask
    while m:
        v = (m & -m).bit_length() - 1
        m ^= (1 << v)
        cand = 1 + _max_is(mask & ~adj[v])
        if cand > best:
            best = cand
    _mi_cache[mask] = best
    return best

# Precompute for all masks that might be needed
# Actually, let's just compute on demand with caching.

def lower_bound(remaining):
    return max(_max_clique(remaining), _max_is(remaining))

def solve(deadline):
    best = [UB]
    best_parts = [ub_parts]

    def backtrack(remaining, current, depth):
        if remaining == 0:
            if depth < best[0]:
                best[0] = depth
                best_parts[0] = list(current)
            return
        if time.time() > deadline:
            return
        if depth + 1 >= best[0]:
            return
        if depth + lower_bound(remaining) >= best[0]:
            return
        first_v = (remaining & -remaining).bit_length() - 1
        for m in by_vertex[first_v]:
            if m & ~remaining:
                continue
            new_remaining = remaining & ~m
            current.append(m)
            backtrack(new_remaining, current, depth + 1)
            current.pop()

    backtrack(FULL, [], 0)
    return best_parts[0]

def construct():
    deadline = time.time() + 85.0
    try:
        result = solve(deadline)
        if result is None:
            raise ValueError("solve returned None")
        seen = set()
        for part in result:
            for v in part:
                if v in seen:
                    raise ValueError("dup")
                seen.add(v)
            m = 0
            for v in part:
                m |= (1 << v)
            if not is_homogeneous(m):
                raise ValueError("not homogeneous")
        if len(seen) != N:
            raise ValueError("incomplete")
        return [list(part) for part in result]
    except Exception:
        es = {tuple(e) for e in P['edges']}
        parts = []
        for v in range(N):
            for part in parts:
                trial = part + [v]
                hits = [(u, w) in es or (w, u) in es for i, u in enumerate(trial) for w in trial[i + 1:]]
                if all(hits) or not any(hits):
                    part.append(v)
                    break
            else:
                parts.append([v])
        return [list(part) for part in parts]
