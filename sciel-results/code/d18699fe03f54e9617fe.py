import time
import random
import sys

# EVOLVE-BLOCK-START

def construct():
    deadline = time.time() + 110  # safety margin under 150s kill
    T0 = time.time()

    def time_ok():
        return time.time() < deadline - 3

    # ---------- max-flow (Dinic) ----------
    def maxflow(n, adj, s, t):
        if s == t:
            return 0
        N = n
        head = [-1] * N
        to = []
        nxt = []
        cap = []

        def add(u, v, c):
            to.append(v); cap.append(c); nxt.append(head[u]); head[u] = len(to) - 1
            to.append(u); cap.append(0); nxt.append(head[v]); head[v] = len(to) - 1

        for u in range(N):
            for v in adj[u]:
                add(u, v, 1)
        level = [0] * N
        it = [0] * N

        def bfs():
            for i in range(N):
                level[i] = -1
            q = [s]
            level[s] = 0
            qi = 0
            while qi < len(q):
                u = q[qi]; qi += 1
                e = head[u]
                while e != -1:
                    v = to[e]
                    if cap[e] > 0 and level[v] < 0:
                        level[v] = level[u] + 1
                        q.append(v)
                    e = nxt[e]
            return level[t] >= 0

        sys.setrecursionlimit(1000000)

        def dfs(u, f):
            if u == t:
                return f
            e = it[u]
            while e != -1:
                v = to[e]
                if cap[e] > 0 and level[v] == level[u] + 1:
                    ret = dfs(v, f if f < cap[e] else cap[e])
                    if ret > 0:
                        cap[e] -= ret
                        cap[e ^ 1] += ret
                        return ret
                e = nxt[e]
            it[u] = e
            return 0

        flow = 0
        while bfs():
            for i in range(N):
                it[i] = head[i]
            while True:
                f = dfs(s, 10**9)
                if f == 0:
                    break
                flow += f
        return flow

    # ---------- graph helpers ----------
    def to_edges(n, A):
        edges = []
        for i in range(n):
            for j in range(i + 1, n):
                if (A[j] >> i) & 1:
                    edges.append([i, j])
        return edges

    def has_k4_fast(n, A):
        # K4 exists iff there is a triangle with a common neighbour (max-flow test)
        for a in range(n):
            for b in range(a + 1, n):
                if not ((A[b] >> a) & 1):
                    continue
                for c in range(b + 1, n):
                    if not ((A[c] >> a) & 1 and (A[c] >> b) & 1):
                        continue
                    # triangle a,b,c ; need a vertex d in N(a)&N(b)&N(c)
                    inter = A[a] & A[b] & A[c]
                    if inter & ~((1 << a) | (1 << b) | (1 << c)):
                        return True
        return False

    def find_k4(n, A):
        for a in range(n):
            for b in range(a + 1, n):
                if not ((A[b] >> a) & 1):
                    continue
                for c in range(b + 1, n):
                    if not ((A[c] >> a) & 1 and (A[c] >> b) & 1):
                        continue
                    inter = A[a] & A[b] & A[c]
                    inter &= ~((1 << a) | (1 << b) | (1 << c))
                    if inter:
                        low = inter & -inter
                        d = low.bit_length() - 1
                        return (a, b, c, d)
        return None

    def find_independent_set(n, A, k):
        """Return an independent set of size k (list) or None. Greedy with backtracking."""
        comp = [0] * n
        all_mask = (1 << n) - 1
        for i in range(n):
            comp[i] = all_mask & ~(A[i] | (1 << i))
        found = [None]

        def bk(R, P, X):
            if found[0] is not None:
                return
            if len(R) == k:
                found[0] = list(R)
                return
            if not P:
                return
            if len(R) + bin(P).count("1") < k:
                return
            # pivot
            if X:
                u_bit = X & -X
                u = u_bit.bit_length() - 1
                cands = P & ~comp[u]
            else:
                cands = P
            c = cands
            while c and found[0] is None:
                low = c & -c
                v = low.bit_length() - 1
                R.append(v)
                bk(R, P & comp[v], X & comp[v])
                R.pop()
                P = P & ~low
                X = X | low
                c = c & ~low

        bk([], all_mask, 0)
        return found[0]

    def max_alpha(n, A):
        """Exact independence number via Bron-Kerbosch on complement."""
        comp = [0] * n
        all_mask = (1 << n) - 1
        for i in range(n):
            comp[i] = all_mask & ~(A[i] | (1 << i))
        best = [0]

        def bk(R, P, X):
            if not P:
                if len(R) > best[0]:
                    best[0] = len(R)
                return
            if len(R) + bin(P).count("1") <= best[0]:
                return
            if X:
                u_bit = X & -X
                u = u_bit.bit_length() - 1
                cands = P & ~comp[u]
            else:
                cands = P
            c = cands
            while c:
                low = c & -c
                v = low.bit_length() - 1
                bk(R + [v], P & comp[v], X & comp[v])
                P = P & ~low
                X = X | low
                c = c & ~low

        bk([], all_mask, 0)
        return best[0]

    def valid(n, A):
        if has_k4_fast(n, A):
            return False
        if find_independent_set(n, A, 6) is not None:
            return False
        return True

    # ---------- generators ----------
    def make_paley(q):
        if q < 5 or q % 4 != 1:
            return None
        for i in range(2, int(q ** 0.5) + 1):
            if q % i == 0:
                return None
        squares = {x * x % q for x in range(1, q)}
        n = q
        A = [0] * n
        for i in range(n):
            for j in range(n):
                if i != j and (j - i) % q in squares:
                    A[i] |= (1 << j)
        return A

    def random_graph(n, p, rng):
        A = [0] * n
        for i in range(n):
            for j in range(i + 1, n):
                if rng.random() < p:
                    A[i] |= (1 << j)
                    A[j] |= (1 << i)
        return A

    def rewire(A, rng, n_flips):
        n = len(A)
        A = A[:]
        for _ in range(n_flips):
            i = rng.randrange(n)
            j = rng.randrange(n)
            if i == j:
                continue
            if (A[j] >> i) & 1:
                A[i] &= ~(1 << j)
                A[j] &= ~(1 << i)
            else:
                A[i] |= (1 << j)
                A[j] |= (1 << i)
        return A

    # ---------- repair ----------
    def repair(n, A, rng, deadline):
        """Repair to K4-free and alpha<6. Returns (A, ok)."""
        A = A[:]
        for _ in range(300):
            if time.time() > deadline - 3:
                return A, False
            k4 = find_k4(n, A)
            if k4:
                a, b, c, d = k4
                u, v = rng.choice([(a, b), (a, c), (a, d), (b, c), (b, d), (c, d)])
                A[u] &= ~(1 << v)
                A[v] &= ~(1 << u)
                continue
            ind = find_independent_set(n, A, 6)
            if ind is not None:
                u, v = rng.sample(ind, 2)
                A[u] |= (1 << v)
                A[v] |= (1 << u)
                continue
            return A, True
        return A, False

    # ---------- grow via max-flow feasibility ----------
    def grow(n, A, rng, deadline):
        """Try to add one vertex. Returns (new_n, new_A) or None."""
        new_n = n + 1
        if new_n > 40:
            return None
        all_mask = (1 << n) - 1
        # precompute: for each vertex v, its neighbour mask
        for attempt in range(60):
            if not time_ok():
                return None
            verts = list(range(n))
            rng.shuffle(verts)
            neigh = 0
            for v in verts:
                # triangle-free test: N(v) & neigh must be independent in G
                nbrs = A[v] & neigh
                ok = True
                m = nbrs
                while m:
                    low = m & -m
                    u = low.bit_length() - 1
                    m ^= low
                    if A[u] & (nbrs & ~(1 << u)):
                        ok = False
                        break
                if not ok:
                    continue
                neigh |= (1 << v)
            # build new graph
            new_A = A + [0]
            for i in range(n):
                if (neigh >> i) & 1:
                    new_A[i] |= (1 << (new_n - 1))
                    new_A[new_n - 1] |= (1 << i)
            if valid(new_n, new_A):
                return new_n, new_A
        return None

    # ---------- main search ----------
    best_n = 0
    best_A = []

    # 1) Paley graphs (known valid small ones)
    for q in [5, 13, 17, 29, 37]:
        if not time_ok():
            break
        A = make_paley(q)
        if A is None:
            continue
        n = len(A)
        if valid(n, A):
            if n > best_n:
                best_n = n
                best_A = A[:]

    # 2) grow from the best valid graph
    if best_n > 0:
        cur_n = best_n
        cur_A = best_A[:]
        stall = 0
        while time_ok():
            res = grow(cur_n, cur_A, rng, deadline)
            if res:
                cur_n, cur_A = res
                if cur_n > best_n:
                    best_n = cur_n
                    best_A = cur_A[:]
                stall = 0
            else:
                stall += 1
                if stall > 20:
                    # perturb and repair, then keep trying to grow
                    pert = rewire(cur_A, rng, rng.randint(1, 4))
                    pert, ok = repair(cur_n, pert, rng, deadline)
                    if ok:
                        cur_A = pert
                        stall = 0
                    else:
                        break

    # 3) random restarts targeting n = best_n+1, +2
    for seed in [7, 11, 13, 17, 19, 23, 29, 31]:
        if not time_ok():
            break
        rng2 = random.Random(seed)
        for n_try in [best_n + 1, best_n + 2, best_n]:
            if n_try > 40 or n_try < 5:
                continue
            for _ in range(6):
                if not time_ok():
                    break
                A = random_graph(n_try, 0.3, rng2)
                A, ok = repair(n_try, A, rng2, deadline)
                if ok and valid(n_try, A):
                    if n_try > best_n:
                        best_n = n_try
                        best_A = A[:]

    if best_n == 0:
        # fallback: a small valid graph (triangle-free star)
        best_n = 5
        best_A = [0] * 5
        for i in range(5):
            for j in range(i + 1, 5):
                if i == 0:
                    best_A[i] |= (1 << j)
                    best_A[j] |= (1 << i)

    return {'n': best_n, 'edges': to_edges(best_n, best_A)}

# EVOLVE-BLOCK-END
