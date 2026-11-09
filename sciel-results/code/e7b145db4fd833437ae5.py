"""Rewrite: verify clique-subdivision candidates via max-flow (vertex-capacity)."""
# EVOLVE-BLOCK-START
import time
from collections import deque
import itertools

P = {'n': 24, 'edges': [[0, 1], [0, 3], [0, 5], [0, 9], [0, 10], [0, 12], [0, 20], [0, 22], [1, 4], [1, 5], [1, 6], [1, 14], [1, 15], [1, 17], [1, 19], [1, 20], [1, 22], [2, 3], [2, 5], [2, 7], [2, 8], [3, 7], [3, 9], [3, 10], [3, 11], [3, 13], [3, 14], [3, 16], [3, 19], [3, 20], [4, 6], [4, 9], [4, 11], [4, 12], [4, 13], [4, 14], [4, 15], [4, 18], [4, 19], [4, 20], [4, 21], [4, 23], [5, 7], [5, 8], [5, 9], [5, 12], [5, 14], [5, 15], [5, 17], [5, 18], [5, 19], [5, 20], [5, 21], [6, 10], [6, 13], [6, 16], [6, 20], [6, 21], [6, 22], [7, 15], [7, 16], [7, 19], [7, 22], [8, 9], [8, 15], [8, 18], [8, 20], [8, 22], [9, 11], [9, 12], [9, 15], [9, 17], [9, 21], [9, 23], [10, 12], [11, 13], [11, 14], [11, 17], [11, 18], [11, 20], [11, 22], [12, 13], [12, 15], [12, 16], [12, 22], [12, 23], [13, 16], [14, 16], [14, 17], [14, 18], [14, 19], [15, 16], [15, 19], [15, 20], [15, 22], [16, 18], [16, 21], [17, 22], [18, 19], [18, 20], [19, 21], [20, 22], [20, 23], [21, 22], [21, 23], [22, 23]]}

N = int(P['n'])
E = P['edges']
DEADLINE = time.time() + 75.0


def _build_adj():
    adj = [set() for _ in range(N)]
    for u, v in E:
        adj[u].add(v)
        adj[v].add(u)
    return adj


def _maxflow(cap, s, t, n):
    """Return (maxflow_value, parent) using Dinic on a residual cap matrix."""
    flow = 0
    INF = 10 ** 18
    while True:
        level = [-1] * n
        q = deque([s])
        level[s] = 0
        while q:
            u = q.popleft()
            lu = level[u]
            row = cap[u]
            for v in range(n):
                if level[v] < 0 and row[v] > 0:
                    level[v] = lu + 1
                    q.append(v)
        if level[t] < 0:
            break
        it = [0] * n
        while True:
            stack = [(s, INF)]
            while stack:
                u, f = stack[-1]
                if u == t:
                    break
                row = cap[u]
                i = it[u]
                while i < n and (level[i] != level[u] + 1 or row[i] <= 0):
                    i += 1
                it[u] = i
                if i < n:
                    stack.append((i, min(f, row[i])))
                else:
                    stack.pop()
            if stack[-1][0] != t:
                break
            f = stack[-1][1]
            for a, b in zip(stack, stack[1:]):
                cap[a[0]][b[0]] -= f
                cap[b[0]][a[0]] += f
            flow += f
    parent = {}
    for u in range(n):
        row = cap[u]
        for v in range(n):
            if v > u and row[v] > 0:
                parent[(u, v)] = row[v]
    return flow, parent


def _extract_paths(parent, branches, k, n):
    """Recover one simple path per branch pair from the max-flow parent map."""
    paths = {}
    used = set()
    for i in range(k):
        for j in range(i + 1, k):
            a, b = branches[i], branches[j]
            if (a, b) in parent:
                cap_ab = parent[(a, b)]
                if cap_ab == 1:
                    paths[(i, j)] = [a, b]
                    used.add((a, b))
                    continue
            path = [b]
            cur = b
            for _ in range(2 * n + 5):
                found = None
                for x in range(n):
                    if cur == x:
                        continue
                    if (x, cur) in parent:
                        found = x
                        break
                if found is None:
                    path = None
                    break
                path.append(found)
                cur = found
                if cur == a:
                    break
            if path is None or path[-1] != a or len(path) < 2:
                return None
            path.reverse()
            paths[(i, j)] = path
            for x in range(1, len(path) - 1):
                used.add(path[x])
    return paths, used


def _verify(branches, adj, k, n):
    """Return (paths, used) or None if no internally-disjoint path family exists."""
    B = set(branches)
    cap = [[0] * n for _ in range(n)]
    for u in range(n):
        if u in B:
            continue
        for v in adj[u]:
            if v < u:
                cap[u][v] = 1
                cap[v][u] = 1
    for i in range(k):
        for j in range(i + 1, k):
            a, b = branches[i], branches[j]
            if b in adj[a]:
                cap[a][b] += 1
                cap[b][a] += 1
    flow, parent = _maxflow(cap, branches[0], branches[1], n)
    need = k * (k - 1) // 2
    if flow < need:
        return None
    return _extract_paths(parent, branches, k, n)


def _greedy_branches(adj, k, deadline):
    """Grow a branch set by repeatedly adding the highest-degree non-branch vertex."""
    deg = [len(adj[i]) for i in range(N)]
    branches = [0]
    used = {0}
    while len(branches) < k:
        best = -1
        bestd = -1
        for i in range(N):
            if i in used:
                continue
            if deg[i] > bestd:
                bestd = deg[i]
                best = i
        if best < 0:
            return None
        branches.append(best)
        used.add(best)
        if time.time() > deadline:
            return None
    return branches


def _try_k(k, adj, deadline):
    if k < 2:
        return None
    need = k * (k - 1) // 2
    # 1) greedy degree-based branch set
    br = _greedy_branches(adj, k, deadline)
    if br is not None:
        res = _verify(br, adj, k, N)
        if res is not None:
            return br, res
    # 2) random restarts
    import random
    rng = random.Random(12345 + k)
    for _ in range(40):
        if time.time() > deadline:
            break
        cand = rng.sample(range(N), k)
        res = _verify(cand, adj, k, N)
        if res is not None:
            return cand, res
    return None


def _fallback(adj):
    """Fallback: find a maximal clique (branch set with direct edges) or a single edge."""
    best = [0, 1]
    for u in range(N):
        for v in range(u + 1, N):
            if v in adj[u]:
                if len(best) < 2:
                    best = [u, v]
    # greedy maximal clique
    cli = []
    for u in sorted(range(N), key=lambda x: -len(adj[x])):
        if all(u in adj[c] for c in cli):
            cli.append(u)
    if len(cli) >= 2:
        best = cli
    return best


def construct():
    start = time.time()
    adj = _build_adj()
    best = None
    # try large k down to 2, keep the first that verifies
    for k in range(N, 1, -1):
        if time.time() > DEADLINE:
            break
        res = _try_k(k, adj, DEADLINE)
        if res is not None:
            branches, (paths, used) = res
            if best is None or len(branches) > len(best[0]):
                best = (branches, paths)
            # found a valid one; stop (we go from large to small)
            break
    if best is None:
        branches = _fallback(adj)
        paths = {}
        for i in range(len(branches)):
            for j in range(i + 1, len(branches)):
                a, b = branches[i], branches[j]
                paths[(i, j)] = [a, b]
        best = (branches, paths)

    branches, paths = best
    out_paths = []
    for i in range(len(branches)):
        for j in range(i + 1, len(branches)):
            out_paths.append([int(x) for x in paths[(i, j)]])
    out_branches = [int(x) for x in branches]
    return {'branches': out_branches, 'paths': out_paths}
# EVOLVE-BLOCK-END
