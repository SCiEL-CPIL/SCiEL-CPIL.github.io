import time
import random
import sys
from collections import deque

def construct(k, g):
    start_time = time.time()
    deadline = start_time + 115.0  # 5s margin before 120s kill

    # Moore bound for k-regular graph with girth g (odd g=2r+1)
    # n >= 1 + k * sum_{i=0}^{r-1} (k-1)^i
    r = (g - 1) // 2
    moore_bound = 1 + k * sum((k-1)**i for i in range(r))

    def seed_construct(k, g):
        """Lazebnik-Ustimenko-Woldar graph construction."""
        q = k
        d = max(3, g - 5 if (g - 5) % 2 else g - 4)
        if q == 4:
            table = [[0, 0, 0, 0], [0, 1, 2, 3], [0, 2, 3, 1], [0, 3, 1, 2]]
            add, sub, mul = (lambda x, y: x ^ y), (lambda x, y: x ^ y), (lambda x, y: table[x][y])
        else:
            add, sub, mul = (lambda x, y: (x + y) % q), (lambda x, y: (x - y) % q), (lambda x, y: x * y % q)
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
                    y[c] = add(x[c], mul(y[a], x[b])) if side == 0 else sub(x[c], mul(x[a], y[b]))
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

    def build_adj(edges, n):
        adj = [[] for _ in range(n)]
        for u, v in edges:
            adj[u].append(v)
            adj[v].append(u)
        return adj

    def exact_girth_check(n, adj, g):
        """Exact girth check using BFS from every vertex with non-backtracking constraint."""
        half_g = g // 2
        for s in range(n):
            dist = [-1] * n
            parent = [-1] * n
            dist[s] = 0
            q = deque([s])
            while q:
                u = q.popleft()
                if dist[u] >= half_g:
                    continue
                for v in adj[u]:
                    if dist[v] == -1:
                        dist[v] = dist[u] + 1
                        parent[v] = u
                        q.append(v)
                    else:
                        if v != parent[u]:
                            cycle_len = dist[u] + dist[v] + 1
                            if cycle_len < g:
                                return False
        return True

    def fast_girth_check(n, adj, g, sample_size=20):
        """Faster girth check using random sampling of vertices."""
        half_g = g // 2
        if n <= sample_size:
            sources = list(range(n))
        else:
            sources = random.sample(range(n), sample_size)
        for s in sources:
            dist = [-1] * n
            parent = [-1] * n
            dist[s] = 0
            q = deque([s])
            while q:
                u = q.popleft()
                if dist[u] >= half_g:
                    continue
                for v in adj[u]:
                    if dist[v] == -1:
                        dist[v] = dist[u] + 1
                        parent[v] = u
                        q.append(v)
                    else:
                        if v != parent[u]:
                            cycle_len = dist[u] + dist[v] + 1
                            if cycle_len < g:
                                return False
        return True

    def generate_simple_regular(n, k):
        """Generate a simple k-regular graph using configuration model with retries."""
        for attempt in range(200):
            stubs = []
            for i in range(n):
                for _ in range(k):
                    stubs.append(i)
            random.shuffle(stubs)
            edges = []
            edge_set = set()
            valid = True
            for i in range(0, n*k, 2):
                u, v = stubs[i], stubs[i+1]
                if u == v:
                    valid = False
                    break
                key = (min(u,v), max(u,v))
                if key in edge_set:
                    valid = False
                    break
                edge_set.add(key)
                edges.append((u, v))
            if valid:
                return edges
        return None

    def find_short_cycle(n, adj, g, max_search_time=0.5):
        """Find a cycle of length < g using BFS. Returns cycle or None."""
        start_t = time.time()
        half_g = g // 2
        for s in range(n):
            if time.time() - start_t > max_search_time:
                return None
            dist = [-1] * n
            parent = [-1] * n
            dist[s] = 0
            q = deque([s])
            while q:
                u = q.popleft()
                if dist[u] >= half_g:
                    continue
                for v in adj[u]:
                    if dist[v] == -1:
                        dist[v] = dist[u] + 1
                        parent[v] = u
                        q.append(v)
                    else:
                        if v != parent[u] and dist[u] + dist[v] + 1 < g:
                            # Reconstruct cycle
                            path_u = []
                            a = u
                            while a != -1:
                                path_u.append(a)
                                a = parent[a]
                            path_u.reverse()
                            path_v = []
                            b = v
                            while b != -1:
                                path_v.append(b)
                                b = parent[b]
                            path_v.reverse()
                            cycle = path_u + path_v[-2::-1]
                            clean = []
                            for x in cycle:
                                if not clean or clean[-1] != x:
                                    clean.append(x)
                            if len(clean) >= 3:
                                return clean
        return None

    def repair_short_cycles(n, adj, g, max_iters=2000, time_budget=10.0):
        """Iteratively find short cycles and rewire edges to break them."""
        start_t = time.time()
        edge_set = set()
        for u in range(n):
            for v in adj[u]:
                if u < v:
                    edge_set.add((u, v))

        for it in range(max_iters):
            if time.time() - start_t > time_budget:
                return False

            cycle = find_short_cycle(n, adj, g, max_search_time=0.3)
            if cycle is None:
                return True

            # Find a valid edge on the cycle
            u, v = None, None
            key = None
            for i in range(len(cycle)-1):
                a, b = cycle[i], cycle[i+1]
                k2 = (min(a,b), max(a,b))
                if k2 in edge_set:
                    u, v = a, b
                    key = k2
                    break
            if u is None:
                return False

            # Remove edge (u,v)
            edge_set.remove(key)
            adj[u].remove(v)
            adj[v].remove(u)

            # Find candidates for rewiring
            adj_u_set = set(adj[u])
            adj_v_set = set(adj[v])
            candidates_u = [x for x in range(n) if x != u and x not in adj_u_set and x != v]
            candidates_v = [y for y in range(n) if y != v and y not in adj_v_set and y != u]

            if not candidates_u or not candidates_v:
                edge_set.add(key)
                adj[u].append(v)
                adj[v].append(u)
                return False

            # Try random rewiring with multiple attempts
            success = False
            for attempt in range(50):
                x = random.choice(candidates_u)
                y = random.choice(candidates_v)
                if x == y:
                    continue
                if x in adj_u_set or y in adj_v_set:
                    continue
                if (min(u,x), max(u,x)) in edge_set or (min(v,y), max(v,y)) in edge_set:
                    continue
                adj[u].append(x)
                adj[v].append(y)
                edge_set.add((min(u,x), max(u,x)))
                edge_set.add((min(v,y), max(v,y)))
                success = True
                break

            if not success:
                for attempt in range(50):
                    x = random.choice(candidates_u)
                    y = random.choice(candidates_v)
                    if x == y:
                        continue
                    if y in adj_u_set or x in adj_v_set:
                        continue
                    if (min(u,y), max(u,y)) in edge_set or (min(v,x), max(v,x)) in edge_set:
                        continue
                    adj[u].append(y)
                    adj[v].append(x)
                    edge_set.add((min(u,y), max(u,y)))
                    edge_set.add((min(v,x), max(v,x)))
                    success = True
                    break

            if not success:
                edge_set.add(key)
                adj[u].append(v)
                adj[v].append(u)
                return False

        return False

    def psl2q_cayley(k, g):
        """
        Try to build a k-regular Cayley graph on PSL(2,q) for appropriate q.
        For k=5, q=13 gives |PSL(2,13)|=1092 vertices.
        We use involutions as generators and check girth.
        """
        if k != 5:
            return None

        # For k=5, try q=13 (PSL(2,13) has 1092 elements)
        # We need exactly 5 involutions as generators
        q = 13

        # Build PSL(2,q) elements
        psl_elements = []
        seen = set()
        for a in range(q):
            for b in range(q):
                for c in range(q):
                    if a != 0:
                        d = (1 + b * c) * pow(a, -1, q) % q
                        mat = (a, b, c, d)
                        neg = (-a % q, -b % q, -c % q, -d % q)
                        canon = mat if mat < neg else neg
                        if canon not in seen:
                            seen.add(canon)
                            psl_elements.append(canon)
                    else:
                        if b == 0:
                            continue
                        c_val = (-1 * pow(b, -1, q)) % q
                        if c_val != c:
                            continue
                        for d in range(q):
                            mat = (a, b, c, d)
                            neg = (-a % q, -b % q, -c % q, -d % q)
                            canon = mat if mat < neg else neg
                            if canon not in seen:
                                seen.add(canon)
                                psl_elements.append(canon)

        n_vertices = len(psl_elements)
        idx_map = {e: i for i, e in enumerate(psl_elements)}
        identity = (1, 0, 0, 1)

        def mat_mult(u, s):
            ua, ub, uc, ud = u
            sa, sb, sc, sd = s
            return ((ua * sa + ub * sc) % q,
                    (ua * sb + ub * sd) % q,
                    (uc * sa + ud * sc) % q,
                    (uc * sb + ud * sd) % q)

        # Find involutions
        involutions = []
        for e in psl_elements:
            a, b, c, d = e
            if (a + d) % q == 0 and (b, c) != (0, 0):
                m2 = mat_mult(e, e)
                if m2 == identity:
                    involutions.append(e)

        if len(involutions) < k:
            return None

        # Precompute multiplication of each element by each involution
        # For efficiency, precompute neighbor indices
        inv_idx = {e: i for i, e in enumerate(involutions)}

        # We need to pick k involutions as generators
        # Try the first k involutions
        generators = involutions[:k]

        # Build adjacency
        adj = [[] for _ in range(n_vertices)]
        edge_set = set()

        for i, elem in enumerate(psl_elements):
            for gen in generators:
                prod = mat_mult(elem, gen)
                if prod in idx_map:
                    j = idx_map[prod]
                    if j > i:
                        key = (i, j)
                        if key not in edge_set:
                            edge_set.add(key)
                            adj[i].append(j)
                            adj[j].append(i)

        # Verify it's k-regular
        for i in range(n_vertices):
            if len(adj[i]) != k:
                return None

        # Check girth
        if exact_girth_check(n_vertices, adj, g):
            edges = list(edge_set)
            return n_vertices, edges

        # Try different subsets of involutions
        # Sample a few random subsets
        for trial in range(5):
            if time.time() > deadline - 20:
                return None
            random.shuffle(involutions)
            generators = involutions[:k]
            adj = [[] for _ in range(n_vertices)]
            edge_set = set()
            for i, elem in enumerate(psl_elements):
                for gen in generators:
                    prod = mat_mult(elem, gen)
                    if prod in idx_map:
                        j = idx_map[prod]
                        if j > i:
                            key = (i, j)
                            if key not in edge_set:
                                edge_set.add(key)
                                adj[i].append(j)
                                adj[j].append(i)
            valid = True
            for i in range(n_vertices):
                if len(adj[i]) != k:
                    valid = False
                    break
            if not valid:
                continue
            if exact_girth_check(n_vertices, adj, g):
                edges = list(edge_set)
                return n_vertices, edges

        return None

    # Search strategy:
    # 1. Use seed as baseline
    # 2. Try PSL(2,q) Cayley graph approach
    # 3. Try to find smaller graphs by generating random regular graphs and repairing
    best_n = None
    best_edges = None

    # Try PSL approach first (may give better n than seed)
    psl_result = psl2q_cayley(k, g)
    if psl_result is not None:
        best_n, best_edges = psl_result

    # Verify seed is valid and compare
    seed_n, seed_edges = seed_construct(k, g)
    seed_adj = build_adj(seed_edges, seed_n)
    if exact_girth_check(seed_n, seed_adj, g):
        if best_n is None or seed_n < best_n:
            best_n = seed_n
            best_edges = seed_edges

    # If we have a valid result, try to improve
    if best_n is not None:
        # Try smaller n values from Moore bound upward
        target_n = best_n - 1
        min_n = max(moore_bound, best_n - 200)

        for n in range(min_n, target_n + 1):
            if time.time() > deadline - 25:
                break

            # Number of trials per n value
            trials = 5

            for _ in range(trials):
                if time.time() > deadline - 25:
                    break

                edges = generate_simple_regular(n, k)
                if edges is None:
                    continue

                adj = build_adj(edges, n)

                # Quick check with sampling
                if fast_girth_check(n, adj, g, sample_size=30):
                    # Full exact check
                    if exact_girth_check(n, adj, g):
                        return n, edges
                else:
                    # Try to repair short cycles
                    if repair_short_cycles(n, adj, g, max_iters=500, time_budget=2.0):
                        if exact_girth_check(n, adj, g):
                            return n, edges

    # Return best found
    if best_n is not None:
        return best_n, best_edges

    # Fallback: return seed even if not verified (shouldn't happen)
    return seed_n, seed_edges
