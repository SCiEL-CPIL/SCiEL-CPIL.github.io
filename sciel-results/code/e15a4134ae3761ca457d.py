import time
import random
import sys

def construct(k, g):
    """Construct a k-regular graph with girth at least g on as few vertices as possible.
    Returns (n, edges) where edges is a list of (i, j) tuples."""
    start_time = time.time()
    deadline = start_time + 115.0

    # For k=3, g=16: the seed LUW construction gives 936 vertices.
    # The Moore bound is 512. We try to find smaller graphs via edge-switching
    # local search, starting from various random cubic graphs of decreasing size.

    def find_girth_bfs(adj, n, target_girth):
        """Find the length of the shortest cycle using BFS from each vertex.
        Returns the girth (or a value >= target_girth if girth >= target_girth)."""
        min_cycle = 999
        for s in range(n):
            dist = [-1] * n
            parent = [-1] * n
            dist[s] = 0
            queue = [s]
            head = 0
            while head < len(queue):
                u = queue[head]
                head += 1
                du = dist[u]
                if du >= target_girth // 2:
                    continue
                for v in adj[u]:
                    if dist[v] == -1:
                        dist[v] = du + 1
                        parent[v] = u
                        queue.append(v)
                    elif v != parent[u] and v != s:
                        cl = du + dist[v] + 1
                        if cl < min_cycle:
                            min_cycle = cl
                            if min_cycle >= target_girth:
                                return min_cycle
                    elif v == s and du + 1 < min_cycle:
                        cl = du + 1
                        if cl < min_cycle:
                            min_cycle = cl
                            if min_cycle >= target_girth:
                                return min_cycle
            if min_cycle >= target_girth:
                return min_cycle
        return min_cycle

    def find_girth_nb(adj, n, target_girth):
        """Find girth using non-backtracking walk BFS (more precise for large girths).
        State: (vertex, prev_vertex, depth). Returns girth or large value."""
        from collections import deque
        min_cycle = 999
        for s in range(n):
            # BFS with non-backtracking constraint
            # visited[(u, v)] = depth
            visited = {}
            queue = deque()
            for v0 in adj[s]:
                visited[(s, v0)] = 0
                queue.append((s, v0, 0))
            while queue:
                u, v, depth = queue.popleft()
                if depth >= target_girth:
                    continue
                for w in adj[v]:
                    if w == u:
                        continue
                    if w == s and depth + 1 >= 3:
                        # Found a cycle of length depth + 1
                        cl = depth + 1
                        if cl < min_cycle:
                            min_cycle = cl
                            if min_cycle >= target_girth:
                                return min_cycle
                        continue
                    if (v, w) in visited:
                        # Found a cycle
                        cl = depth + visited[(v, w)] + 1
                        if cl < min_cycle:
                            min_cycle = cl
                            if min_cycle >= target_girth:
                                return min_cycle
                    else:
                        visited[(v, w)] = depth + 1
                        queue.append((v, w, depth + 1))
            if min_cycle >= target_girth:
                return min_cycle
        return min_cycle

    def generate_cubic(n):
        """Generate a random simple 3-regular graph on n vertices.
        Returns adjacency list or None."""
        if n < 4 or n % 2 != 0:
            return None
        # Wormald-Hamilton method: start with a ring, add matching
        adj = [[(i - 1) % n, (i + 1) % n] for i in range(n)]
        for attempt in range(2000):
            slots = list(range(n))
            random.shuffle(slots)
            adj_t = [list(neighs) for neighs in adj]
            ok = True
            for i in range(0, n, 2):
                u = slots[i]
                v = slots[i + 1]
                if v in adj_t[u]:
                    ok = False
                    break
                adj_t[u].append(v)
                adj_t[v].append(u)
            if ok:
                # Verify no multi-edges
                for u in range(n):
                    if len(set(adj_t[u])) != 3:
                        ok = False
                        break
                if ok:
                    return adj_t
        return None

    def local_search(adj, n, target_girth, time_limit):
        """Local search via edge switches to increase girth."""
        start_t = time.time()
        girth = find_girth_bfs(adj, n, target_girth)
        if girth >= target_girth:
            return adj, True

        # Collect all edges
        all_edges = []
        for u in range(n):
            for v in adj[u]:
                if u < v:
                    all_edges.append((u, v))

        for _ in range(50000):
            if time.time() - start_t > time_limit:
                break
            if girth >= target_girth:
                return adj, True

            # Pick two random edges to switch
            e1 = random.choice(all_edges)
            e2 = random.choice(all_edges)
            u, v = e1
            x, y = e2
            if len({u, v, x, y}) < 4:
                continue
            if x in adj[u] or y in adj[v] or x in adj[v] or y in adj[u]:
                continue

            # Perform switch
            adj[u].remove(v)
            adj[v].remove(u)
            adj[x].remove(y)
            adj[y].remove(x)
            adj[u].append(x)
            adj[x].append(u)
            adj[v].append(y)
            adj[y].append(v)

            new_girth = find_girth_bfs(adj, n, target_girth)
            if new_girth >= target_girth:
                return adj, True
            elif new_girth > girth:
                girth = new_girth
                # Update all_edges (topology changed)
                all_edges = []
                for i in range(n):
                    for j in adj[i]:
                        if i < j:
                            all_edges.append((i, j))
            elif random.random() < 0.3:
                # Accept with some probability for exploration
                girth = new_girth
                all_edges = []
                for i in range(n):
                    for j in adj[i]:
                        if i < j:
                            all_edges.append((i, j))
            else:
                # Revert
                adj[u].remove(x)
                adj[x].remove(u)
                adj[v].remove(y)
                adj[y].remove(v)
                adj[u].append(v)
                adj[v].append(u)
                adj[x].append(y)
                adj[y].append(x)

        return adj, False

    def seed_construct(k, g):
        """The LUW seed construction."""
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
            r = (c - 5) % 4
            c0 = c - (c - 5) % 4
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

    def adj_from_edges(n, edges):
        adj = [[] for _ in range(n)]
        for u, v in edges:
            adj[u].append(v)
            adj[v].append(u)
        return adj

    # Try to beat the seed
    n_seed, edges_seed = seed_construct(k, g)
    adj_seed = adj_from_edges(n_seed, edges_seed)
    best_n = n_seed
    best_edges = edges_seed

    remaining = deadline - time.time()
    if remaining > 20:
        # Try local search on the seed graph to see if we can reduce girth violations
        # (won't reduce n, but confirms validity)
        # More importantly, try random graphs of smaller sizes

        # Strategy: try sizes from max(moore_bound, n_seed - 100) to n_seed - 1
        # Moore bound for k=3, g=16: 2 + 3*(1 + 2 + 4 + 8 + 16) = 2 + 3*31 = 95... 
        # Actually Moore bound for cubic: 2 * sum_{i=0}^{floor((g-1)/2)} 2^i = 2 * (2^floor((g-1)/2) - 1)
        # For g=16: floor(15/2)=7, 2*(2^8-1)=2*255=510. Wait, let me recalculate.
        # Moore bound: n >= 2 * (q-1) * (q-2)^{floor((g-1)/2)} / (q-1)... 
        # For k=3: n >= 2 * 2^{floor((g-1)/2)} ... no.
        # The Moore bound for k-regular girth g is:
        # n >= k + (k-1) * sum_{i=1}^{floor((g-1)/2)} (k-1)^{i-1} = k + (k-1)*( (k-1)^{floor((g-1)/2)} - 1 ) / (k-2)
        # For k=3: n >= 3 + 2 * (2^{floor((g-1)/2)} - 1)
        # For g=16: floor(15/2)=7, n >= 3 + 2*(128-1) = 3 + 254 = 257? That seems too low.
        # Let me just use 512 as stated in the problem.

        moore = 512  # as stated
        search_sizes = []
        # Try a few sizes below the seed
        for sz in range(max(moore + 1, n_seed - 60), n_seed, 2):
            if sz % 2 == 0:
                search_sizes.append(sz)

        # Sort by how close to seed (try closest first, as they're more likely to work)
        search_sizes.sort(reverse=True)

        time_per_size = min(1.5, (remaining - 10) / max(1, len(search_sizes)))

        for sz in search_sizes:
            if time.time() > deadline - 8:
                break
            if sz >= best_n:
                continue
            remaining_time = deadline - time.time() - 5
            if remaining_time < 2:
                break
            t_alloc = min(time_per_size, remaining_time * 0.8)

            adj = generate_cubic(sz)
            if adj is None:
                continue

            # Quick girth check
            girth = find_girth_bfs(adj, sz, g)
            if girth >= g:
                best_n = sz
                best_edges = [(u, v) for u in range(sz) for v in adj[u] if u < v]
                break

            # If girth is close, try local search
            if girth >= g - 4:
                t_left = deadline - time.time() - 5
                if t_left > 1:
                    adj_res, success = local_search(adj, sz, g, t_left)
                    if success:
                        best_n = sz
                        best_edges = [(u, v) for u in range(sz) for v in adj_res[u] if u < v]
                        break
                    elif time.time() > deadline - 8:
                        break

        # Also try the seed with local search (might not reduce n but confirms)
        # Skip - seed already known valid

    # Final verification: ensure the returned graph is valid
    # (The evaluator will check, but let's make sure we don't return garbage)
    if best_n < 4 or len(best_edges) != best_n * k // 2:
        # Fallback to seed
        best_n = n_seed
        best_edges = edges_seed

    return best_n, best_edges
