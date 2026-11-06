import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # Leave 35s margin before 150s kill

    def has_tri(adj, u, v):
        return bool(adj[u] & adj[v])

    def exact_alpha(adj, n):
        """Exact maximum independent set via branch and bound."""
        best = [0]

        def rec(avail, cnt):
            if not avail:
                return
            if cnt + len(avail) <= best[0]:
                return
            # Pick vertex with max degree in induced subgraph
            v = max(avail, key=lambda x: len(adj[x] & avail))
            # Include v
            new_avail = avail - {v} - adj[v]
            rec(new_avail, cnt + 1)
            if cnt + 1 > best[0]:
                best[0] = cnt + 1
            # Exclude v
            rec(avail - {v}, cnt)

        rec(set(range(n)), 0)
        return best[0]

    def greedy_is(adj, n, avail):
        """Greedy maximal independent set from avail."""
        result = []
        while avail:
            v = min(avail, key=lambda x: len(adj[x] & avail))
            result.append(v)
            avail = avail - {v} - adj[v]
        return result

    def try_build(n, seed):
        random.seed(seed)
        adj = [set() for _ in range(n)]
        all_edges = [(i, j) for i in range(n) for j in range(i + 1, n)]
        random.shuffle(all_edges)

        # Phase 1: Random maximal triangle-free
        for u, v in all_edges:
            if not has_tri(adj, u, v):
                adj[u].add(v)
                adj[v].add(u)

        # Phase 2: Iteratively find large IS and add edges
        for _ in range(300):
            if time.time() > deadline:
                return None
            a = exact_alpha(adj, n)
            if a < 10:
                return adj
            is_set = greedy_is(adj, n, set(range(n)))
            if len(is_set) < 10:
                break
            added = False
            for i in range(len(is_set)):
                for j in range(i + 1, len(is_set)):
                    u, v = is_set[i], is_set[j]
                    if v not in adj[u] and not has_tri(adj, u, v):
                        adj[u].add(v)
                        adj[v].add(u)
                        added = True
            if not added:
                break
        return None

    # Try n=40 with many seeds
    for s in range(2000):
        if time.time() > deadline:
            break
        res = try_build(40, s)
        if res is not None:
            edges = [[i, j] for i in range(40) for j in sorted(res[i]) if i < j]
            return {'n': 40, 'edges': edges}

    # Try n=39
    for s in range(2000):
        if time.time() > deadline:
            break
        res = try_build(39, s)
        if res is not None:
            edges = [[i, j] for i in range(39) for j in sorted(res[i]) if i < j]
            return {'n': 39, 'edges': edges}

    # Fallback: C5 blow-up with m=8 gives n=40, but alpha = 8*2 = 16? No.
    # C5 blow-up: groups G0..G4, edges between Gi and Gj if |i-j| mod 5 in {1,4}
    # alpha of C5 blow-up = max sum of non-adjacent groups.
    # Non-adjacent pairs in C5: {0,2}, {0,3}, {1,3}, {1,4}, {2,4}
    # For C5, independence number of blow-up = 2*m (take two non-adjacent groups)
    # So for alpha < 10, need 2*m < 10, m <= 4, n <= 20.
    # Better fallback: Use a known small valid graph.
    # Actually, let's just return the C5 blow-up with m=4, n=20, alpha=8
    n = 20
    adj = [set() for _ in range(n)]
    for g1 in range(5):
        for g2 in range(g1 + 1, 5):
            if (g2 - g1) % 5 in (1, 4):
                for u in range(g1 * 4, (g1 + 1) * 4):
                    for v in range(g2 * 4, (g2 + 1) * 4):
                        adj[u].add(v)
                        adj[v].add(u)
    edges = [[i, j] for i in range(n) for j in sorted(adj[i]) if i < j]
    return {'n': n, 'edges': edges}


if __name__ == '__main__':
    res = construct()
    print(res['n'])
