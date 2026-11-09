"""Combined program: PG(2,5) polarity core + fast bitmask local search for the 91st edge."""
import time
import random
import math

def construct():
    t0 = time.time()
    deadline = t0 + 115.0
    n = 31
    target_edges = 91
    q = 5

    # --- Build PG(2,5) polarity graph (90 edges, C4-free) ---
    def gf_inv(a, mod=5):
        return pow(a, mod - 2, mod)

    points = []
    seen = set()
    for x in range(q):
        for y in range(q):
            for z in range(q):
                if x == 0 and y == 0 and z == 0:
                    continue
                if x != 0:
                    inv_x = gf_inv(x)
                    key = (1, (y * inv_x) % q, (z * inv_x) % q)
                elif y != 0:
                    inv_y = gf_inv(y)
                    key = (0, 1, (z * inv_y) % q)
                else:
                    key = (0, 0, 1)
                if key not in seen:
                    seen.add(key)
                    points.append(key)

    assert len(points) == 31

    def adjacent(i, j):
        x1, y1, z1 = points[i]
        x2, y2, z2 = points[j]
        val = (x1 * x2 + y1 * y2 - z1 * z2) % q
        return val == 0

    edges_core = []
    for i in range(n):
        for j in range(i + 1, n):
            if adjacent(i, j):
                edges_core.append([i, j])

    if len(edges_core) != 90:
        # Fallback: greedy construction
        return _greedy_construct()

    # Build bitmask adjacency for core
    adj = [0] * n
    for u, v in edges_core:
        adj[u] |= (1 << v)
        adj[v] |= (1 << u)

    # Count C4s using bitmask
    def count_c4(adj_list):
        total = 0
        for i in range(n):
            for j in range(i + 1, n):
                k = (adj_list[i] & adj_list[j]).bit_count()
                if k > 1:
                    total += k * (k - 1) // 2
        return total // 2

    # Verify core is C4-free
    core_c4 = count_c4(adj)
    if core_c4 != 0:
        # Something wrong, fallback
        return _greedy_construct()

    # Find the best single edge to add to minimize total C4s
    # Adding edge (u,v) creates C4s = number of length-2 paths between neighbors
    # i.e., for each a in N(u), b in N(v), if a-b is an edge, that's a C4
    # More precisely: C4s added = sum over pairs (a,b) where a~u, b~v, a~b
    # But we need to count distinct C4s. Each C4 containing the new edge (u,v)
    # corresponds to a path u-a-b-v where a,b are distinct and a~b.
    # Number of such paths = for each a in N(u), count neighbors of a that are in N(v)
    # But this counts each C4 twice? No, each C4 u-a-b-v is unique for each (a,b) pair.
    
    best_edge = None
    best_add = float('inf')
    
    for u in range(n):
        for v in range(u + 1, n):
            if (adj[u] >> v) & 1:
                continue
            # Count C4s that would be created by adding (u,v)
            add = 0
            # For each common "path of length 2" from u to v
            # u-a-b-v where a in N(u), b in N(v), a~b
            # We can compute: for each a in N(u), count |N(a) & N(v)|
            # But we must exclude a==v and b==u cases, which aren't in N since no self-loops
            for a in range(n):
                if (adj[u] >> a) & 1:
                    # a is neighbor of u
                    common = (adj[a] & adj[v]).bit_count()
                    add += common
            # Each C4 is counted once per (a,b) pair, which is correct
            # But wait: a C4 u-a-b-v is counted when we pick a, and b is in N(a) & N(v)
            # This is correct, each C4 corresponds to exactly one (a,b) pair
            if add < best_add:
                best_add = add
                best_edge = [u, v]
            if add == 0:
                break
        if best_add == 0:
            break

    if best_edge is None:
        best_edge = [0, 1]

    final_edges = edges_core + [best_edge]
    
    # Verify
    assert len(final_edges) == 91
    edge_set = set()
    for e in final_edges:
        key = (min(e[0], e[1]), max(e[0], e[1]))
        if key in edge_set:
            return _greedy_construct()
        edge_set.add(key)

    # Now do local search to potentially improve: swap the added edge with another
    # or do general 1-swap local search on the full 91-edge graph
    if time.time() < deadline - 5:
        time_left = deadline - time.time() - 5
        best_adj, best_edges = local_search(adj, final_edges, 123, time_left)
        return [list(e) for e in best_edges]

    return [list(e) for e in final_edges]


def local_search(adj, edges, seed, time_limit):
    """Fast bitmask local search: 1-swap moves."""
    random.seed(seed)
    n = 31
    current_adj = adj[:]
    current_edges = edges[:]
    edge_set = set((min(u, v), max(u, v)) for u, v in current_edges)
    
    def count_c4(adj_list):
        total = 0
        for i in range(n):
            for j in range(i + 1, n):
                k = (adj_list[i] & adj_list[j]).bit_count()
                if k > 1:
                    total += k * (k - 1) // 2
        return total // 2

    current_c4 = count_c4(current_adj)
    best_adj = current_adj[:]
    best_edges = current_edges[:]
    best_c4 = current_c4

    t_start = time.time()
    T = 1.0
    cooling_rate = 0.9999

    while time.time() - t_start < time_limit:
        idx = random.randint(0, len(current_edges) - 1)
        u, v = current_edges[idx]
        u, v = min(u, v), max(u, v)

        # Pick random new edge
        while True:
            x = random.randint(0, n - 1)
            y = random.randint(0, n - 1)
            if x == y:
                continue
            if x > y:
                x, y = y, x
            if (x, y) not in edge_set:
                break

        if (u, v) == (x, y):
            continue

        # Remove (u,v), add (x,y)
        current_adj[u] &= ~(1 << v)
        current_adj[v] &= ~(1 << u)
        edge_set.remove((u, v))

        current_adj[x] |= (1 << y)
        current_adj[y] |= (1 << x)
        edge_set.add((x, y))

        current_edges[idx] = (x, y)

        new_c4 = count_c4(current_adj)
        delta = new_c4 - current_c4

        accept = False
        if delta < 0:
            accept = True
        else:
            if T > 0.01:
                if random.random() < math.exp(-delta / T):
                    accept = True

        if accept:
            current_c4 = new_c4
            if new_c4 < best_c4:
                best_c4 = new_c4
                best_adj = current_adj[:]
                best_edges = current_edges[:]
            T *= cooling_rate
        else:
            # Revert
            current_adj[u] |= (1 << v)
            current_adj[v] |= (1 << u)
            edge_set.add((u, v))

            current_adj[x] &= ~(1 << y)
            current_adj[y] &= ~(1 << x)
            edge_set.remove((x, y))

            current_edges[idx] = (u, v)

    return best_adj, best_edges


def _greedy_construct():
    """Fallback greedy construction."""
    import random
    random.seed(42)
    n = 31
    target_edges = 91

    def count_c4(adj_list):
        total = 0
        for i in range(n):
            for j in range(i + 1, n):
                k = (adj_list[i] & adj_list[j]).bit_count()
                if k > 1:
                    total += k * (k - 1) // 2
        return total // 2

    adj = [0] * n
    edges = []
    all_edges = [(i, j) for i in range(n) for j in range(i + 1, n)]
    random.shuffle(all_edges)

    for u, v in all_edges:
        if len(edges) >= target_edges:
            break
        cost = 0
        for a in range(n):
            if (adj[u] >> a) & 1:
                common = (adj[a] & adj[v]).bit_count()
                cost += common
        if cost <= 1:
            adj[u] |= (1 << v)
            adj[v] |= (1 << u)
            edges.append([u, v])

    while len(edges) < target_edges:
        best_e = None
        best_cost = float('inf')
        for u, v in all_edges:
            if (adj[u] >> v) & 1:
                continue
            cost = 0
            for a in range(n):
                if (adj[u] >> a) & 1:
                    common = (adj[a] & adj[v]).bit_count()
                    cost += common
            if cost < best_cost:
                best_cost = cost
                best_e = (u, v)
        if best_e:
            u, v = best_e
            adj[u] |= (1 << v)
            adj[v] |= (1 << u)
            edges.append([u, v])
        else:
            break

    return edges
