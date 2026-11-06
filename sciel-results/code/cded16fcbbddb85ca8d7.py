import time
import random
import sys

def construct():
    start_time = time.time()
    deadline = start_time + 130.0  # Leave 20s for safety

    def has_triangle(adj, u, v):
        """Check if adding edge (u,v) creates a triangle using bitsets."""
        return (adj[u] & adj[v]) != 0

    def is_triangle_free(adj, n):
        """Verify no triangles in the graph."""
        for u in range(n):
            for v in range(u + 1, n):
                if (adj[u] >> v) & 1:
                    if (adj[u] & adj[v]) != 0:
                        return False
        return True

    def max_independent_set_size(adj, n):
        """
        Exact branch and bound for maximum independent set size.
        adj: list of int bitsets, adj[i] has bit j set if (i,j) is an edge.
        n: number of vertices.
        Returns the size of the maximum independent set.
        """
        best = [0]

        # Order vertices by degree (descending) for better pruning
        degrees = [bin(adj[i]).count('1') for i in range(n)]
        order = sorted(range(n), key=lambda x: degrees[x], reverse=True)

        def search(candidates, current_size, depth):
            nonlocal best
            if not candidates:
                return
            if current_size + bin(candidates).count('1') <= best[0]:
                return

            # Pick vertex with highest degree in candidates
            v = -1
            for u in order:
                if (candidates >> u) & 1:
                    v = u
                    break

            if v == -1:
                return

            # Branch 1: Include v
            new_cands = candidates & ~(1 << v) & ~adj[v]
            search(new_cands, current_size + 1, depth + 1)
            if current_size + 1 > best[0]:
                best[0] = current_size + 1

            # Branch 2: Exclude v
            new_cands = candidates & ~(1 << v)
            search(new_cands, current_size, depth + 1)

        search((1 << n) - 1, 0, 0)
        return best[0]

    def find_independent_set_of_size(adj, n, target):
        """Find an independent set of exactly 'target' size, or None."""
        def search(candidates, current_set, needed):
            if needed == 0:
                return list(current_set)
            if bin(candidates).count('1') < needed:
                return None

            # Pick vertex with min degree in candidates
            v = -1
            min_deg = n + 1
            for u in range(n):
                if (candidates >> u) & 1:
                    deg = bin(adj[u] & candidates).count('1')
                    if deg < min_deg:
                        min_deg = deg
                        v = u

            if v == -1:
                return None

            # Try including v
            new_cands = candidates & ~(1 << v) & ~adj[v]
            res = search(new_cands, current_set + [v], needed - 1)
            if res is not None:
                return res

            # Try excluding v
            new_cands = candidates & ~(1 << v)
            return search(new_cands, current_set, needed)

        return search((1 << n) - 1, [], target)

    def try_add_edge(adj, n, u, v):
        """Try to add edge (u,v) if it doesn't create a triangle. Returns True if added."""
        if u >= n or v >= n or u == v:
            return False
        if (adj[u] >> v) & 1:
            return True  # already exists
        if has_triangle(adj, u, v):
            return False
        adj[u] |= (1 << v)
        adj[v] |= (1 << u)
        return True

    def try_remove_edge(adj, n, u, v):
        """Remove edge (u,v). Returns True if it existed."""
        if u >= n or v >= n or u == v:
            return False
        if not ((adj[u] >> v) & 1):
            return False
        adj[u] &= ~(1 << v)
        adj[v] &= ~(1 << u)
        return True

    def local_search(adj, n, max_iterations=100):
        """
        Local search to reduce independence number below 10.
        Uses edge additions within independent sets and edge swaps.
        Returns (success, adj) where success means alpha < 10.
        """
        for iteration in range(max_iterations):
            if time.time() > deadline:
                return False, adj

            alpha = max_independent_set_size(adj, n)
            if alpha < 10:
                return True, adj

            # Find a large independent set
            is_set = find_independent_set_of_size(adj, n, min(10, alpha))
            if is_set is None:
                break

            improved = False

            # Strategy 1: Add edges within the independent set
            for i in range(len(is_set)):
                for j in range(i + 1, len(is_set)):
                    u = is_set[i]
                    v = is_set[j]
                    if not ((adj[u] >> v) & 1):
                        if not has_triangle(adj, u, v):
                            adj[u] |= (1 << v)
                            adj[v] |= (1 << u)
                            improved = True

            # Strategy 2: Edge swaps - remove an edge and add two new ones
            if not improved:
                edges_list = []
                for u in range(n):
                    for v in range(u + 1, n):
                        if (adj[u] >> v) & 1:
                            edges_list.append((u, v))

                rng = random.Random(hash((n, iteration)) % (2**31))
                rng.shuffle(edges_list)

                for a, b in edges_list[:50]:
                    if time.time() > deadline:
                        break

                    orig_adj_a = adj[a]
                    orig_adj_b = adj[b]
                    try_remove_edge(adj, n, a, b)

                    added_edges = []
                    for w in is_set:
                        if w != a and w != b:
                            if not ((adj[a] >> w) & 1) and not has_triangle(adj, a, w):
                                adj[a] |= (1 << w)
                                adj[w] |= (1 << a)
                                added_edges.append((a, w))
                            if not ((adj[b] >> w) & 1) and not has_triangle(adj, b, w):
                                adj[b] |= (1 << w)
                                adj[w] |= (1 << b)
                                added_edges.append((b, w))

                    if len(added_edges) >= 2:
                        improved = True
                        break
                    else:
                        adj[a] = orig_adj_a
                        adj[b] = orig_adj_b
                        for x, y in added_edges:
                            adj[x] &= ~(1 << y)
                            adj[y] &= ~(1 << x)

            if not improved:
                break

        return False, adj

    def clebsch_graph():
        """Return the Clebsch graph (16 vertices, triangle-free, alpha=5)."""
        n = 16
        adj = [0] * n
        for i in range(n):
            for j in range(i + 1, n):
                xor = i ^ j
                if bin(xor).count('1') == 2:
                    adj[i] |= (1 << j)
                    adj[j] |= (1 << i)
        return adj, n

    def c5_blowup(k):
        """
        Return a C5 blow-up with k vertices per part.
        n = 5*k vertices, triangle-free, alpha = 2k.
        """
        n = 5 * k
        adj = [0] * n
        for g1 in range(5):
            for g2 in [(g1 + 1) % 5, (g1 + 4) % 5]:
                if g1 < g2:
                    for u in range(g1 * k, (g1 + 1) * k):
                        for v in range(g2 * k, (g2 + 1) * k):
                            adj[u] |= (1 << v)
                            adj[v] |= (1 << u)
        return adj, n

    def random_triangle_free(n, seed):
        """Generate a random triangle-free graph on n vertices."""
        rng = random.Random(seed)
        adj = [0] * n
        edges = [(i, j) for i in range(n) for j in range(i + 1, n)]
        rng.shuffle(edges)
        for u, v in edges:
            if not has_triangle(adj, u, v):
                adj[u] |= (1 << v)
                adj[v] |= (1 << u)
        return adj, n

    def try_construct_from_base(base_adj, base_n, target_n, max_time=30.0):
        """
        Start from a base graph and try to reach target_n vertices with alpha < 10.
        Uses vertex addition and local search.
        """
        start_t = time.time()
        adj = list(base_adj)
        n = base_n

        if n >= target_n:
            alpha = max_independent_set_size(adj, n)
            if alpha < 10:
                return adj, n

        while n < target_n and time.time() - start_t < max_time:
            best_added = False
            rng = random.Random(hash((n, time.time())) % (2**31))

            for attempt in range(20):
                if time.time() - start_t > max_time:
                    break

                S = []
                candidates = (1 << n) - 1
                max_conn = min(n, 15)
                for _ in range(max_conn):
                    if not candidates:
                        break
                    bits = []
                    c = candidates
                    while c:
                        if c & 1:
                            bits.append(len(bits))
                        c >>= 1
                    if not bits:
                        break
                    idx = rng.choice(bits)
                    v = -1
                    c = candidates
                    count = 0
                    while c:
                        if c & 1:
                            if count == idx:
                                v = count
                                break
                            count += 1
                        c >>= 1

                    if v == -1:
                        break

                    ok = True
                    for s in S:
                        if (adj[v] >> s) & 1:
                            ok = False
                            break
                    if ok:
                        S.append(v)
                        candidates &= ~(1 << v)

                if not S:
                    continue

                new_adj = adj + [0]
                new_n = n + 1
                for s in S:
                    new_adj[n] |= (1 << s)
                    new_adj[s] |= (1 << n)

                tf = True
                for u in range(new_n):
                    for v in range(u + 1, new_n):
                        if (new_adj[u] >> v) & 1:
                            if (new_adj[u] & new_adj[v]) != 0:
                                tf = False
                                break
                    if not tf:
                        break

                if not tf:
                    continue

                alpha = max_independent_set_size(new_adj, new_n)
                if alpha < 10:
                    adj = new_adj
                    n = new_n
                    best_added = True
                    break

            if not best_added:
                success, adj = local_search(adj, n, max_iterations=20)
                if success:
                    continue
                else:
                    break

        return adj, n

    # Try various constructions
    best_result = None
    best_n = 0

    # 1. Try C5 blow-ups with k=4 (n=20, alpha=8)
    for k in [4, 5, 6]:
        if time.time() > deadline:
            break
        adj, n = c5_blowup(k)
        alpha = max_independent_set_size(adj, n)
        if alpha < 10:
            if n > best_n:
                best_n = n
                best_result = list(adj)

        adj2, n2 = try_construct_from_base(adj, n, 40, max_time=20.0)
        if n2 > best_n:
            alpha2 = max_independent_set_size(adj2, n2)
            if alpha2 < 10:
                best_n = n2
                best_result = list(adj2)

    # 2. Try Clebsch graph (n=16, alpha=5)
    if time.time() < deadline:
        adj, n = clebsch_graph()
        alpha = max_independent_set_size(adj, n)
        if alpha < 10:
            if n > best_n:
                best_n = n
                best_result = list(adj)

        adj2, n2 = try_construct_from_base(adj, n, 40, max_time=20.0)
        if n2 > best_n:
            alpha2 = max_independent_set_size(adj2, n2)
            if alpha2 < 10:
                best_n = n2
                best_result = list(adj2)

    # 3. Try random triangle-free graphs
    for seed in range(20):
        if time.time() > deadline:
            break
        for n_try in [38, 39, 40]:
            if time.time() > deadline:
                break
            adj, n = random_triangle_free(n_try, seed)
            alpha = max_independent_set_size(adj, n)
            if alpha < 10:
                if n > best_n:
                    best_n = n
                    best_result = list(adj)
            else:
                success, adj2 = local_search(adj, n, max_iterations=50)
                if success:
                    alpha2 = max_independent_set_size(adj2, n)
                    if alpha2 < 10 and n > best_n:
                        best_n = n
                        best_result = list(adj2)

    # Final fallback: C5 blow-up with k=4
    if best_result is None:
        n = 20
        adj = [0] * n
        for g1 in range(5):
            for g2 in range(g1 + 1, 5):
                if (g2 - g1) % 5 in [1, 4]:
                    for u in range(g1 * 4, (g1 + 1) * 4):
                        for v in range(g2 * 4, (g2 + 1) * 4):
                            adj[u] |= (1 << v)
                            adj[v] |= (1 << u)
        best_result = adj
        best_n = n

    # Convert to edge list
    edges = []
    for i in range(best_n):
        for j in range(i + 1, best_n):
            if (best_result[i] >> j) & 1:
                edges.append([i, j])

    return {'n': best_n, 'edges': edges}

if __name__ == '__main__':
    res = construct()
    print(res['n'])
