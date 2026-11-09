import time
import random

N = 15
FULL_MASK = (1 << N) - 1
TIME_LIMIT = 75.0   # safety margin under the 90s wall-clock


def _is_valid(adj):
    # valid iff no two vertices at distance 2 share a degree
    for u in range(N):
        du = adj[u].bit_count()
        dist2 = 0
        for v in range(N):
            if v != u and (adj[u] >> v) & 1:
                dist2 |= adj[v]
        dist2 &= ~adj[u]
        dist2 &= ~(1 << u)
        if dist2 == 0:
            continue
        seen = set()
        m = dist2
        while m:
            lsb = m & (-m)
            w = lsb.bit_length() - 1
            dw = adj[w].bit_count()
            if dw == du:
                return False
            seen.add(dw)
            m -= lsb
    return True


def _score(adj):
    s = 0
    for u in range(N):
        s += adj[u].bit_count()
    return s // 2


def construct():
    start = time.time()
    best = [0] * N
    best_edges = 0

    def consider(adj):
        nonlocal best, best_edges
        if _is_valid(adj):
            e = _score(adj)
            if e > best_edges:
                best_edges = e
                best = [adj[i] for i in range(N)]

    # ---- Phase 1: greedy multi-restart ----
    for it in range(5000):
        if time.time() - start > TIME_LIMIT:
            break
        adj = [0] * N
        ok = True
        for u in range(N):
            for v in range(u + 1, N):
                if (adj[u] >> v) & 1:
                    continue
                nb_u = adj[u]
                nb_v = adj[v]
                adj[u] |= 1 << v
                adj[v] |= 1 << u
                if _is_valid(adj):
                    pass
                else:
                    adj[u] = nb_u
                    adj[v] = nb_v
        consider(adj)

    # ---- Phase 2: local search (add / swap / remove) from best ----
    base = [best[i] for i in range(N)]
    cur = base
    cur_edges = best_edges
    for it in range(200000):
        if time.time() - start > TIME_LIMIT:
            break
        if cur_edges == N * (N - 1) // 2:
            break
        moved = False
        # try additions
        for u in range(N):
            for v in range(u + 1, N):
                if (cur[u] >> v) & 1:
                    continue
                cur[u] |= 1 << v
                cur[v] |= 1 << u
                if _is_valid(cur):
                    cur_edges += 1
                    moved = True
                else:
                    cur[u] ^= 1 << v
                    cur[v] ^= 1 << u
                if moved:
                    break
            if moved:
                break
        if moved:
            if cur_edges > best_edges:
                best_edges = cur_edges
                best = [cur[i] for i in range(N)]
            continue
        # try swaps: remove (a,b), add (c,d)
        found = False
        for a in range(N):
            for b in range(a + 1, N):
                if not ((cur[a] >> b) & 1):
                    continue
                for c in range(N):
                    for d in range(c + 1, N):
                        if (c == a and d == b) or (c == b and d == a):
                            continue
                        if (cur[c] >> d) & 1:
                            continue
                        cur[a] ^= 1 << b
                        cur[b] ^= 1 << a
                        cur[c] |= 1 << d
                        cur[d] |= 1 << c
                        if _is_valid(cur):
                            found = True
                        else:
                            cur[c] ^= 1 << d
                            cur[d] ^= 1 << c
                            cur[a] |= 1 << b
                            cur[b] |= 1 << a
                        if found:
                            break
                    if found:
                        break
                if found:
                    break
            if found:
                break
        if found:
            if cur_edges > best_edges:
                best_edges = cur_edges
                best = [cur[i] for i in range(N)]
            continue
        # try removals (escape local minima)
        found = False
        for a in range(N):
            for b in range(a + 1, N):
                if not ((cur[a] >> b) & 1):
                    continue
                cur[a] ^= 1 << b
                cur[b] ^= 1 << a
                cur_edges -= 1
                found = True
                break
            if found:
                break
        if found:
            continue
        # stuck: restart from best
        cur = [best[i] for i in range(N)]
        cur_edges = best_edges

    edges = []
    for u in range(N):
        for v in range(u + 1, N):
            if (best[u] >> v) & 1:
                edges.append([int(u), int(v)])
    return edges
