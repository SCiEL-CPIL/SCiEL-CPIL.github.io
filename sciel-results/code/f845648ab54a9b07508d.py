import time
import random
import itertools

def construct():
    start = time.time()
    DEADLINE = 135.0

    def time_left():
        return DEADLINE - (time.time() - start)

    def make_random_graph(n, p=0.5):
        adj = [0] * n
        for u in range(n):
            for v in range(u + 1, n):
                if random.random() < p:
                    adj[u] |= 1 << v
                    adj[v] |= 1 << u
        return adj

    def sample_penalty(adj, n, sample):
        c = 0
        for mask, verts in sample:
            first = (adj[verts[0]] & mask).bit_count()
            ok = True
            for v in verts[1:]:
                if (adj[v] & mask).bit_count() != first:
                    ok = False
                    break
            if ok:
                c += 1
        return c

    def full_verify(adj, n):
        verts = list(range(n))
        for k in range(6, n + 1):
            for S in itertools.combinations(verts, k):
                mask = 0
                for v in S:
                    mask |= 1 << v
                first = (adj[S[0]] & mask).bit_count()
                ok = True
                for v in S[1:]:
                    if (adj[v] & mask).bit_count() != first:
                        ok = False
                        break
                if ok:
                    return False
        return True

    def try_local_search(n, p, time_slice):
        adj = make_random_graph(n, p)
        sample = []
        for _ in range(2500):
            s = tuple(random.sample(range(n), 6))
            mask = 0
            for v in s:
                mask |= 1 << v
            sample.append((mask, s))
        cur = sample_penalty(adj, n, sample)
        edges = [(u, v) for u in range(n) for v in range(u + 1, n)]
        no_improve = 0
        t0 = time.time()
        while time_left() > 0 and (time.time() - t0) < time_slice:
            u, v = random.choice(edges)
            adj[u] ^= 1 << v
            adj[v] ^= 1 << u
            new = sample_penalty(adj, n, sample)
            if new < cur:
                cur = new
                no_improve = 0
                if cur == 0 and time_left() > 25:
                    if full_verify(adj, n):
                        return adj
                    else:
                        adj[u] ^= 1 << v
                        adj[v] ^= 1 << u
                        for _ in range(n):
                            a, b = random.choice(edges)
                            adj[a] ^= 1 << b
                            adj[b] ^= 1 << a
                        cur = sample_penalty(adj, n, sample)
            else:
                adj[u] ^= 1 << v
                adj[v] ^= 1 << u
                no_improve += 1
                if no_improve > 200:
                    for _ in range(max(3, n // 2)):
                        a, b = random.choice(edges)
                        adj[a] ^= 1 << b
                        adj[b] ^= 1 << a
                    cur = sample_penalty(adj, n, sample)
                    no_improve = 0
        return None

    p_cycle = [0.5, 0.4, 0.6, 0.45, 0.55]
    for n in (21, 20):
        if time_left() < 5:
            break
        max_restarts = 300
        for r in range(max_restarts):
            if time_left() < 3:
                break
            p = p_cycle[r % len(p_cycle)]
            ts = min(3.0, max(0.5, time_left() / max(1, max_restarts - r)))
            adj = try_local_search(n, p, ts)
            if adj is not None:
                return {'n': n, 'edges': [[u, v] for u in range(n) for v in range(u + 1, n) if (adj[u] >> v) & 1]}
    # Guaranteed-valid fallback
    return {'n': 5, 'edges': []}
