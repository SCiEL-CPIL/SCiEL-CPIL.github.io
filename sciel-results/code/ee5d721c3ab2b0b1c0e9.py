import time
import random
from collections import deque

def construct(k, g):
    t0 = time.time()
    deadline = 116.0  # safety margin before the 150 s wall-clock kill

    def elapsed():
        return time.time() - t0

    # ---------- Fallback: Lazebnik-Ustimenko-Woldar graph ----------
    def luw_construct(k, g):
        q = k
        d = max(3, g - 5 if (g - 5) % 2 else g - 4)

        if q == 4:
            table = [[0, 0, 0, 0], [0, 1, 2, 3], [0, 2, 3, 1], [0, 3, 1, 2]]
            add = lambda x, y: x ^ y
            sub = lambda x, y: x ^ y
            mul = lambda x, y: table[x][y]
        else:
            add = lambda x, y: (x + y) % q
            sub = lambda x, y: (x - y) % q
            mul = lambda x, y: x * y % q

        rule = {2: (1, 1), 3: (2, 1), 4: (1, 2)}
        for c in range(5, d + 1):
            r, c0 = (c - 5) % 4, c - (c - 5) % 4
            rule[c] = [(1, c0 - 2), (c0 - 1, 1), (c0, 1), (1, c0 + 1)][r]

        def neighbours(vertex):
            side, x = vertex
            for first in range(q):
                y = [None, first] + [0] * (d - 1)
                for c in range(2, d + 1):
                    a, b = rule[c]
                    if side == 0:
                        y[c] = add(x[c], mul(y[a], x[b]))
                    else:
                        y[c] = sub(x[c], mul(x[a], y[b]))
                yield 1 - side, tuple(y)

        start = (0, (None,) + (0,) * d)
        index, order, edges = {start: 0}, [start], []
        for vertex in order:
            for other in neighbours(vertex):
                if other not in index:
                    index[other] = len(order)
                    order.append(other)
                if vertex[0] == 0:
                    edges.append((index[vertex], index[other]))
        return len(order), edges

    # ---------- Girth certification via non-backtracking walks ----------
    def min_girth(n, edges, limit):
        adj = [[] for _ in range(n)]
        for i, j in edges:
            adj[i].append(j)
            adj[j].append(i)
        best = 10 ** 9
        for s in range(n):
            for t0 in adj[s]:
                seen = set()
                q = deque()
                q.append((t0, s))
                seen.add((t0, s))
                depth = 1
                while q:
                    if depth >= best or depth >= limit:
                        break
                    nxt = []
                    for cur, prev in q:
                        for nb in adj[cur]:
                            if nb == prev:
                                continue
                            if nb == s and depth + 1 >= 3:
                                best = min(best, depth + 1)
                                continue
                            st = (nb, cur)
                            if st in seen:
                                continue
                            seen.add(st)
                            nxt.append((nb, cur))
                    q = deque(nxt)
                    depth += 1
        return best

    # ---------- Local-search repair for k == 3 ----------
    def local_search_3(n, g, budget):
        # random 3-regular graph
        if n % 2 == 0:
            adj = [[] for _ in range(n)]
            for v in range(n):
                adj[v].append((v + 1) % n)
                adj[v].append((v - 1) % n)
            rem = list(range(n))
            random.shuffle(rem)
            for i in range(0, n, 2):
                a, b = rem[i], rem[i + 1]
                adj[a].append(b)
                adj[b].append(a)
        else:
            adj = [[] for _ in range(n)]
            perm = list(range(n))
            random.shuffle(perm)
            for i in range(n):
                adj[perm[i]].append(perm[(i + 1) % n])
                adj[perm[i]].append(perm[(i - 1) % n])
            rem = list(range(n))
            random.shuffle(rem)
            for i in range(0, n - 1, 2):
                a, b = rem[i], rem[i + 1]
                adj[a].append(b)
                adj[b].append(a)
            adj[rem[n - 1]].append(0)
            adj[0].append(rem[n - 1])

        def edges_of(adj):
            out = []
            for i in range(n):
                for j in adj[i]:
                    if i < j:
                        out.append((i, j))
            return out

        def girth_of(adj, limit):
            best = 10 ** 9
            for s in range(n):
                for t0 in adj[s]:
                    seen = set()
                    q = deque([(t0, s)])
                    seen.add((t0, s))
                    depth = 1
                    while q:
                        if depth >= best or depth >= limit:
                            break
                        nxt = []
                        for cur, prev in q:
                            for nb in adj[cur]:
                                if nb == prev:
                                    continue
                                if nb == s and depth + 1 >= 3:
                                    best = min(best, depth + 1)
                                    continue
                                st = (nb, cur)
                                if st in seen:
                                    continue
                                seen.add(st)
                                nxt.append((nb, cur))
                        q = deque(nxt)
                        depth += 1
            return best

        def short_cycle(adj):
            best = 10 ** 9
            for s in range(n):
                for t0 in adj[s]:
                    seen = set()
                    q = deque([(t0, s)])
                    seen.add((t0, s))
                    depth = 1
                    while q:
                        if depth >= best or depth >= g:
                            break
                        nxt = []
                        for cur, prev in q:
                            for nb in adj[cur]:
                                if nb == prev:
                                    continue
                                if nb == s and depth + 1 >= 3:
                                    if depth + 1 < best:
                                        best = depth + 1
                                    continue
                                st = (nb, cur)
                                if st in seen:
                                    continue
                                seen.add(st)
                                nxt.append((nb, cur))
                        q = deque(nxt)
                        depth += 1
            return best

        def try_swap(adj):
            e = random.randrange(n * 3)
            u = e // 3
            iu = e % 3
            v = random.randrange(n)
            if v == u:
                return False
            if v in adj[u]:
                return False
            w = adj[u][iu]
            if w in adj[v]:
                return False
            # candidate: remove (u,w), (v,?); add (u,v), (w,?)
            # We need a degree-preserving swap: remove (u,w) and (v,x), add (u,v) and (w,x)
            # pick x in adj[v], x != u, x != w
            x = random.choice(adj[v])
            if x == u or x == w:
                return False
            if x in adj[u]:
                return False
            if w in adj[v]:
                return False
            # apply
            adj[u].remove(w)
            adj[v].remove(x)
            adj[u].append(v)
            adj[v].append(u)
            adj[w].append(x)
            adj[x].append(w)
            # check
            newg = short_cycle(adj)
            oldg = short_cycle(adj)  # recompute after (this is the new one)
            # We want: newg >= g OR newg > oldg
            # But we already changed adj; oldg was lost. Track via return.
            return newg

        # Simpler: implement swap with explicit old-girth tracking
        def do_search():
            g0 = short_cycle(adj)
            if g0 >= g:
                return edges_of(adj), g0
            best_g = g0
            for _ in range(2000000):
                if elapsed() > budget:
                    break
                e = random.randrange(n * 3)
                u = e // 3
                iu = e % 3
                v = random.randrange(n)
                if v == u or v in adj[u]:
                    continue
                w = adj[u][iu]
                if w in adj[v]:
                    continue
                x = random.choice(adj[v])
                if x == u or x == w or x in adj[u] or w in adj[v]:
                    continue
                # record old girth (we need it before change)
                og = short_cycle(adj)
                if og >= g:
                    return edges_of(adj), og
                # apply swap
                adj[u].remove(w); adj[v].remove(x)
                adj[u].append(v); adj[v].append(u)
                adj[w].append(x); adj[x].append(w)
                ng = short_cycle(adj)
                if ng >= g:
                    return edges_of(adj), ng
                if ng <= og:
                    # revert (no improvement)
                    adj[u].remove(v); adj[v].remove(u)
                    adj[w].remove(x); adj[x].remove(w)
                    adj[u].append(w); adj[v].append(x)
                else:
                    # keep improvement
                    if ng > best_g:
                        best_g = ng
                    if best_g >= g:
                        return edges_of(adj), best_g
            return None, best_g

        return do_search()

    # ---------- Main ----------
    if k == 3:
        # Moore bound for (3, g): 2 * (2^{(g-2)/2} - 1) for even g
        if g % 2 == 0:
            n_moore = 2 * (2 ** ((g - 2) // 2) - 1)
        else:
            n_moore = 3 * (2 ** ((g - 1) // 2) - 1)
        # try a few sizes around the Moore bound with local search
        luw_n, luw_edges = None, None
        best = None
        # candidate sizes: Moore bound and a few slightly larger even sizes
        sizes = []
        start = n_moore
        if start % 2 != 0:
            start += 1
        for off in range(0, 8):
            sizes.append(start + 2 * off)
        for n in sizes:
            if elapsed() > deadline - 5:
                break
            budget = min(deadline - 2, elapsed() + 30)
            try:
                edges, gfound = local_search_3(n, g, budget)
            except Exception:
                edges = None
            if edges is not None and gfound >= g:
                # validate
                if min_girth(n, edges, g) >= g:
                    return n, edges
        # fallback to LUW
        return luw_construct(k, g)
    else:
        return luw_construct(k, g)
