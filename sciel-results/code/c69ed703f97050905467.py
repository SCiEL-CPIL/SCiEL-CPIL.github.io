import time
from itertools import combinations

# EVOLVE-BLOCK-START
P = {'n': 24, 'edges': [[0, 1], [1, 2], [2, 3], [3, 4], [4, 5], [5, 6], [6, 7], [7, 8], [8, 9], [9, 10], [10, 11], [11, 12], [12, 13], [13, 14], [14, 15], [15, 16], [16, 17], [17, 18], [18, 19], [19, 20], [20, 21], [21, 22], [22, 23]], 'diameter': 4}

_N = P['n']
_BASE_SET = frozenset((min(a, b), max(a, b)) for a, b in P['edges'])


def _diameter_leq4(adj):
    """Check if graph with adjacency list adj has diameter <= 4 and is connected."""
    n = _N
    for s in range(n):
        dist = [-1] * n
        dist[s] = 0
        q = [s]
        head = 0
        while head < len(q):
            u = q[head]
            head += 1
            du = dist[u]
            if du >= 4:
                continue
            nd = du + 1
            for v in adj[u]:
                if dist[v] == -1:
                    dist[v] = nd
                    q.append(v)
        if -1 in dist:
            return False
    return True


def _construct_direct():
    """Exhaustively search for minimum number of added edges."""
    n = _N
    base = _BASE_SET

    # Build candidate pool: even-odd pairs not in base, sorted by descending path distance
    pool = []
    for u in range(n):
        for v in range(u + 1, n):
            if (u, v) in base:
                continue
            if (u ^ v) & 1 == 1:
                pool.append((u, v))
    pool.sort(key=lambda e: -abs(e[0] - e[1]))

    # Precompute base neighbor lists
    base_adj = [set() for _ in range(n)]
    for u, v in base:
        base_adj[u].add(v)
        base_adj[v].add(u)

    def check_combo(combo):
        """Check if adding combo edges gives diameter <= 4."""
        # Build adjacency
        adj = [list(s) for s in base_adj]
        for u, v in combo:
            adj[u].append(v)
            adj[v].append(u)
        return _diameter_leq4(adj)

    # Try k = 1, 2, 3, 4, 5...
    for k in range(1, 6):
        for combo in combinations(pool, k):
            if check_combo(combo):
                return list(combo)
    return []


def construct():
    t0 = time.time()
    deadline = t0 + 80.0  # safety margin

    result = _construct_direct()

    if result and time.time() < deadline:
        # Validate
        base = _BASE_SET
        es = set()
        adj = [set() for _ in range(_N)]
        for u, v in P['edges']:
            adj[u].add(v)
            adj[v].add(u)
        for u, v in result:
            key = (min(u, v), max(u, v))
            if key in base or key in es:
                return []
            es.add(key)
            adj[u].add(v)
            adj[v].add(u)
        if not _diameter_leq4([list(s) for s in adj]):
            return []

        return [[int(u), int(v)] for u, v in sorted(result)]

    # Fallback: greedy
    pool = []
    for u in range(_N):
        for v in range(u + 1, _N):
            if (u, v) in _BASE_SET:
                continue
            if (u ^ v) & 1 == 1:
                pool.append((u, v))
    pool.sort(key=lambda e: -abs(e[0] - e[1]))

    es = set()
    adj = [set() for _ in range(_N)]
    for u, v in P['edges']:
        adj[u].add(v)
        adj[v].add(u)

    for u, v in pool:
        if _diameter_leq4([list(s) for s in adj]):
            break
        es.add((u, v))
        adj[u].add(v)
        adj[v].add(u)

    if not _diameter_leq4([list(s) for s in adj]):
        return []

    return [[int(u), int(v)] for u, v in sorted(es)]
# EVOLVE-BLOCK-END
