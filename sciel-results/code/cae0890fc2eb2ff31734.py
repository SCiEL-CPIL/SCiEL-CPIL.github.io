import time
import random
from collections import deque

P = {'n': 24, 'edges': [[0, 1], [0, 3], [0, 5], [0, 9], [0, 10], [0, 12], [0, 20], [0, 22], [1, 4], [1, 5], [1, 6], [1, 14], [1, 15], [1, 17], [1, 19], [1, 20], [1, 22], [2, 3], [2, 5], [2, 7], [2, 8], [3, 7], [3, 9], [3, 10], [3, 11], [3, 13], [3, 14], [3, 16], [3, 19], [3, 20], [4, 6], [4, 9], [4, 11], [4, 12], [4, 13], [4, 14], [4, 15], [4, 18], [4, 19], [4, 20], [4, 21], [4, 23], [5, 7], [5, 8], [5, 9], [5, 12], [5, 14], [5, 15], [5, 17], [5, 18], [5, 19], [5, 20], [5, 21], [6, 10], [6, 13], [6, 16], [6, 20], [6, 21], [6, 22], [7, 15], [7, 16], [7, 19], [7, 22], [8, 9], [8, 15], [8, 18], [8, 20], [8, 22], [9, 11], [9, 12], [9, 15], [9, 17], [9, 21], [9, 23], [10, 12], [11, 13], [11, 14], [11, 17], [11, 18], [11, 20], [11, 22], [12, 13], [12, 15], [12, 16], [12, 22], [12, 23], [13, 16], [14, 16], [14, 17], [14, 18], [14, 19], [15, 16], [15, 19], [15, 20], [15, 22], [16, 18], [16, 21], [17, 22], [18, 19], [18, 20], [19, 21], [20, 22], [20, 23], [21, 22], [21, 23], [22, 23]]}


def construct():
    t_start = time.time()
    DEADLINE = 82.0
    n = P['n']
    edges = P['edges']

    adj = [set() for _ in range(n)]
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)
    deg = [len(adj[v]) for v in range(n)]

    def time_up():
        return (time.time() - t_start) > DEADLINE

    # ---------------- Dinic (iterative DFS to avoid recursion limits) --------
    class Dinic:
        def __init__(self, N):
            self.N = N
            self.head = [[] for _ in range(N)]
            self.to = []
            self.cap = []
            self.rev = []

        def add_edge(self, u, v, c):
            self.head[u].append(len(self.to))
            self.to.append(v); self.cap.append(c); self.rev.append(0)
            self.head[v].append(len(self.to))
            self.to.append(u); self.cap.append(0); self.rev.append(len(self.to) - 2)

        def max_flow(self, s, t):
            flow = 0
            INF = 10 ** 9
            N = self.N
            level = [-1] * N
            while True:
                for i in range(N):
                    level[i] = -1
                level[s] = 0
                q = deque([s])
                while q:
                    u = q.popleft()
                    for ei in self.head[u]:
                        if self.cap[ei] > 0:
                            v = self.to[ei]
                            if level[v] < 0:
                                level[v] = level[u] + 1
                                q.append(v)
                if level[t] < 0:
                    break
                it = [0] * N
                while True:
                    # iterative DFS pushing one augmenting path
                    stack = [(s, INF)]
                    path = []
                    found = False
                    while stack:
                        u, f = stack[-1]
                        if u == t:
                            found = True
                            break
                        advanced = False
                        while it[u] < len(self.head[u]):
                            ei = self.head[u][it[u]]
                            v = self.to[ei]
                            if self.cap[ei] > 0 and level[v] == level[u] + 1:
                                path.append((ei, v))
                                nf = f if f < self.cap[ei] else self.cap[ei]
                                stack.append((v, nf))
                                advanced = True
                                break
                            it[u] += 1
                        if not advanced:
                            stack.pop()
                            if path:
                                pei, pv = path.pop()
                                # edge pei goes into pv; its tail is u
                                it[u] += 1
                    if not found:
                        break
                    # compute bottleneck
                    bottleneck = stack[-1][1]
                    for (ei, v) in path:
                        if self.cap[ei] < bottleneck:
                            bottleneck = self.cap[ei]
                    for (ei, v) in path:
                        self.cap[ei] -= bottleneck
                        self.cap[self.rev[ei]] += bottleneck
                    flow += bottleneck
            return flow

    # ---------------- Flow packing: one unit per branch pair ----------------
    def pack(branches, order=None):
        m = len(branches)
        if m < 2:
            return None
        pairs = []
        for i in range(m):
            for j in range(i + 1, m):
                pairs.append((i, j))
        if order is not None:
            pairs = order
        K = len(pairs)
        base = 2 + n
        N = 2 + 2 * n + K
        S = 0
        T = 1
        def vin(v): return 2 + v
        def vout(v): return 2 + n + v
        def pin(k): return base + k
        def pout(k): return base + K + k
        d = Dinic(N)
        d.add_edge(S, T, K)  # super edge; flow here = number of pairs routed
        for k, (i, j) in enumerate(pairs):
            a, b = branches[i], branches[j]
            d.add_edge(S, pin(k), 1)
            d.add_edge(pout(k), T, 1)
            d.add_edge(pin(k), vin(a), 1)
            d.add_edge(pin(k), vin(b), 1)
            d.add_edge(vout(a), pout(k), 1)
            d.add_edge(vout(b), pout(k), 1)
        bset = set(branches)
        for v in range(n):
            if v in bset:
                d.add_edge(vin(v), vout(v), K)
            else:
                d.add_edge(vin(v), vout(v), 1)
        for u, v in edges:
            d.add_edge(vout(u), vin(v), 1)
            d.add_edge(vout(v), vin(u), 1)
        f = d.max_flow(S, T)
        if f < K:
            return None
        # extract paths
        paths = [None] * K
        for k in range(K):
            ei = None
            for e in d.head[pout(k)]:
                if d.to[e] == T and d.cap[self.rev[e]] > 0:
                    ei = e
                    break
            if ei is None:
                return None
            node = pout(k)
            while node != T:
                nxt = None
                for e in d.head[node]:
                    if d.to[e] == T and d.cap[self.rev[e]] > 0:
                        nxt = e
                        break
                if nxt is None:
                    return None
                node = d.to[nxt]
            # node is now T; re-walk from pout(k) collecting vertices
            seq = []
            node = pout(k)
            while node != T:
                nxt = None
                for e in d.head[node]:
                    if d.to[e] == T and d.cap[self.rev[e]] > 0:
                        nxt = e
                        break
                if nxt is None:
                    return None
                node = d.to[nxt]
                if node >= base:
                    continue
                if node < 2 + n:
                    seq.append(node - 2)
                else:
                    seq.append(node - (2 + n))
            # seq ends at the two branches (out nodes)
            # find the two branch vertices
            brs = [x for x in seq if x in bset]
            if len(brs) != 2:
                return None
            a, b = brs
            # interior = everything not a/b
            interior = [x for x in seq if x != a and x != b]
            # order the interior between a and b by graph traversal
            path = [a] + interior + [b]
            # verify adjacency
            ok = True
            for x, y in zip(path, path[1:]):
                if y not in adj[x]:
                    ok = False
                    break
            if not ok:
                return None
            paths[k] = path
        out = []
        for k, (i, j) in enumerate(pairs):
            out.append(paths[k])
        return out

    def pack_multi(branches, tries=6):
        p = pack(branches)
        if p is not None:
            return p
        m = len(branches)
        for _ in range(tries):
            pairs = []
            for i in range(m):
                for j in range(i + 1, m):
                    pairs.append((i, j))
            random.shuffle(pairs)
            p = pack(branches, pairs)
            if p is not None:
                return p
        return None

    # ---------------- Max clique (Bron–Kerbosch) ----------------
    def max_clique():
        best = []
        cand = list(range(n))
        cand.sort(key=lambda x: -deg[x])
        def expand(clique, cand):
            nonlocal best
            if len(clique) > len(best):
                best = list(clique)
            if not cand:
                return
            for i, v in enumerate(cand):
                nc = [u for u in cand[i + 1:] if u in adj[v]]
                expand(clique + [v], nc)
        expand([], cand)
        return best

    # ---------------- Search ----------------
    best_branches = None
    best_paths = None
    best_count = 0

    def update(b, p):
        nonlocal best_branches, best_paths, best_count
        if p is not None and len(b) > best_count:
            best_branches = list(b)
            best_paths = list(p)
            best_count = len(b)

    clique = max_clique()
    if len(clique) >= 2:
        update(clique, pack_multi(clique, 8))

    # Grow from clique
    current = list(clique)
    changed = True
    while changed and not time_up():
        changed = False
        cands = [v for v in range(n) if v not in current]
        cands.sort(key=lambda x: -deg[x])
        for v in cands:
            if time_up():
                break
            trial = current + [v]
            p = pack_multi(trial, 4)
            if p is not None:
                current = trial
                changed = True
                update(current, p)
                break

    # Local swap (1-for-1) to grow by replacing a "bad" branch
    if best_branches is not None and not time_up():
        improved = True
        while improved and not time_up():
            improved = False
            cur = list(best_branches)
            cset = set(cur)
            for idx in range(len(cur)):
                if time_up():
                    break
                for v in range(n):
                    if v in cset:
                        continue
                    nb = cur[:idx] + cur[idx + 1:] + [v]
                    nb.sort()
                    p = pack_multi(nb, 3)
                    if p is not None and len(nb) > best_count:
                        update(nb, p)
                        improved = True
                        break
                if improved:
                    break

    # Random restarts: seed near best, grow by flow-packing
    if not time_up():
        for r in range(400):
            if time_up():
                break
            seed_size = min(n, max(2, best_count + random.randint(-1, 2)))
            seed = random.sample(range(n), seed_size)
            cur = list(seed)
            if pack_multi(cur, 3) is not None:
                update(cur, pack_multi(cur, 3))
            changed = True
            while changed and not time_up():
                changed = False
                cands = [v for v in range(n) if v not in cur]
                cands.sort(key=lambda x: -deg[x])
                for v in cands:
                    if time_up():
                        break
                    trial = cur + [v]
                    p = pack_multi(trial, 3)
                    if p is not None:
                        cur = trial
                        changed = True
                        update(cur, p)
                        break

    # Direct backtracking: try to find a larger branch set by random target size
    if not time_up():
        target = best_count + 1
        while target <= n and not time_up():
            found = False
            for _ in range(150):
                if time_up():
                    break
                cand = random.sample(range(n), target)
                p = pack_multi(cand, 6)
                if p is not None:
                    update(cand, p)
                    found = True
                    break
            if found:
                target += 1
            else:
                break

    # Fallback
    if best_branches is None or len(best_branches) < 2:
        u, v = edges[0]
        best_branches = [u, v]
        best_paths = [[u, v]]

    branches_out = [int(b) for b in best_branches]
    paths_out = [[int(x) for x in p] for p in best_paths]
    return {'branches': branches_out, 'paths': paths_out}
