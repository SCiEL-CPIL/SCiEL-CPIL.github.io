import time
import random

N = 15

def _degrees(adj):
    return [bin(adj[i]).count('1') for i in range(N)]

def _valid(adj, deg):
    # For each edge (u,v), forbid s in N(u)\{v}, t in N(v)\{u}, s!=t, deg[s]==deg[t]
    for u in range(N):
        au = adj[u]
        for v in range(u + 1, N):
            if (au >> v) & 1:
                su = au & ~(1 << v)          # N(u) without v
                tv = adj[v] & ~(1 << u)      # N(v) without u
                # group tv neighbors by degree
                tbd = {}
                tb = tv
                while tb:
                    lb = tb & (-tb)
                    t = lb.bit_length() - 1
                    d = deg[t]
                    tbd[d] = tbd.get(d, 0) | (1 << t)
                    tb ^= lb
                sb = su
                while sb:
                    lb = sb & (-sb)
                    s = lb.bit_length() - 1
                    m = tbd.get(deg[s], 0) & ~(1 << s)   # t with same degree, t != s
                    if m:
                        return False
                    sb ^= lb
    return True

def construct():
    t0 = time.time()
    deadline = t0 + 50.0  # safety margin under 60s

    best_edges = set()
    best_count = 0

    def consider(edges):
        nonlocal best_edges, best_count
        if len(edges) > best_count:
            best_count = len(edges)
            best_edges = set(edges)

    try:
        for seed in range(80):
            if time.time() > deadline - 2.0:
                break
            r_dead = min(deadline - 1.0, time.time() + 1.5)
            rng = random.Random(seed)
            adj = [0] * N
            edges = set()
            consider(edges)
            while time.time() < r_dead:
                added = False
                # random add attempts
                for _ in range(30):
                    u = rng.randrange(N); v = rng.randrange(N)
                    if u == v:
                        continue
                    if u > v:
                        u, v = v, u
                    if (u, v) in edges:
                        continue
                    adj[u] |= 1 << v; adj[v] |= 1 << u
                    deg = _degrees(adj)
                    if _valid(adj, deg):
                        edges.add((u, v))
                        consider(edges)
                        added = True
                        break
                    adj[u] ^= 1 << v; adj[v] ^= 1 << u
                if added:
                    continue
                # full scan for any valid add
                found = False
                for u in range(N):
                    if time.time() > r_dead:
                        break
                    for v in range(u + 1, N):
                        if (u, v) in edges:
                            continue
                        adj[u] |= 1 << v; adj[v] |= 1 << u
                        deg = _degrees(adj)
                        if _valid(adj, deg):
                            edges.add((u, v))
                            consider(edges)
                            found = True
                            break
                        adj[u] ^= 1 << v; adj[v] ^= 1 << u
                    if found:
                        break
                if found:
                    continue
                # stuck: remove a random edge to allow restructuring
                if edges:
                    e = rng.choice(sorted(edges))
                    u, v = e
                    adj[u] ^= 1 << v; adj[v] ^= 1 << u
                    edges.discard(e)
                else:
                    break

        # final polish: make best maximal
        if time.time() < deadline - 0.5:
            adj = [0] * N
            for (u, v) in best_edges:
                adj[u] |= 1 << v; adj[v] |= 1 << u
            edges = set(best_edges)
            changed = True
            while changed and time.time() < deadline - 0.5:
                changed = False
                for u in range(N):
                    for v in range(u + 1, N):
                        if (u, v) in edges:
                            continue
                        adj[u] |= 1 << v; adj[v] |= 1 << u
                        deg = _degrees(adj)
                        if _valid(adj, deg):
                            edges.add((u, v))
                            changed = True
                            consider(edges)
                            break
                        adj[u] ^= 1 << v; adj[v] ^= 1 << u
                    if changed:
                        break
    except Exception:
        pass

    out = []
    for (u, v) in sorted(best_edges):
        out.append([int(u), int(v)])
    return out
