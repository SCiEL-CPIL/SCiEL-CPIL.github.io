import time
import random
import math
from collections import deque
from heapq import heappush, heappop

# EVOLVE-BLOCK-START

def construct(k, g):
    start_time = time.time()
    deadline = start_time + 110.0

    def remaining():
        return deadline - time.time()

    def is_prime(n):
        if n < 2:
            return False
        if n == 2:
            return True
        if n % 2 == 0:
            return False
        for i in range(3, int(math.isqrt(n)) + 1, 2):
            if n % i == 0:
                return False
        return True

    def is_prime_power(n):
        if n < 2:
            return False
        if is_prime(n):
            return True
        for p in range(2, int(math.isqrt(n)) + 1):
            if n % p == 0:
                t = n
                while t % p == 0:
                    t //= p
                return t == 1
        return False

    def modinv(a, m):
        if m <= 1:
            return None
        try:
            return pow(a, -1, m)
        except (ValueError, ZeroDivisionError):
            return None

    def moore_bound(k, g):
        if k <= 1:
            return 0
        if g == 1:
            return 1
        r = g // 2
        if g % 2 == 1:
            s = 1
            term = k
            for _ in range(r - 1):
                s += term
                term *= (k - 1)
            return s
        else:
            s = 0
            term = 1
            for _ in range(r):
                s += term
                term *= (k - 1)
            return 2 * s

    n_min = moore_bound(k, g)

    def check_girth_nb(adj, n, g):
        for u in range(n):
            dist = {}
            queue = deque()
            for v in adj[u]:
                dist[(v, u)] = 1
                queue.append((v, u))
            while queue:
                x, prev = queue.popleft()
                d = dist[(x, prev)]
                if d >= g - 1:
                    continue
                for y in adj[x]:
                    if y == prev:
                        continue
                    if y == u:
                        if d + 1 < g:
                            return False
                        continue
                    key = (y, x)
                    if key not in dist:
                        dist[key] = d + 1
                        queue.append((y, x))
        return True

    def is_connected(adj, n):
        visited = [False] * n
        visited[0] = True
        queue = deque([0])
        count = 0
        while queue:
            u = queue.popleft()
            count += 1
            for v in adj[u]:
                if not visited[v]:
                    visited[v] = True
                    queue.append(v)
        return count == n

    def validate(adj, n, k, g):
        for i in range(n):
            if len(adj[i]) != k:
                return False
        if not is_connected(adj, n):
            return False
        if not check_girth_nb(adj, n, g):
            return False
        edges = []
        for i in range(n):
            for j in adj[i]:
                if i < j:
                    edges.append((i, j))
        return (n, edges)

    def greedy_embed(k, g, time_budget):
        """Greedy embedding of universal cover into finite set using union-find.
        Build BFS tree from root, at depth r = g//2, merge vertices to close
        the tree into a graph while maintaining girth >= g."""
        r = g // 2
        # Number of vertices at each depth in infinite k-tree
        # depth 0: 1, depth 1: k, depth d>=2: k*(k-1)^(d-1)
        target_n = None
        # Moore bound gives minimum; try to find a valid graph near it
        n_min = moore_bound(k, g)
        # For even g, Moore bound = 2*sum_{i=0}^{r-1}(k-1)^i
        # For odd g, Moore bound = 1 + k*sum_{i=0}^{r-1}(k-1)^i
        # We'll try to construct at exactly the Moore bound or slightly above
        
        # Union-Find
        parent = []
        rank = []
        
        def uf_init(n):
            nonlocal parent, rank
            parent = list(range(n))
            rank = [0] * n
        
        def uf_find(x):
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x
        
        def uf_union(x, y):
            rx, ry = uf_find(x), uf_find(y)
            if rx == ry:
                return False
            if rank[rx] < rank[ry]:
                rx, ry = ry, rx
            parent[ry] = rx
            if rank[rx] == rank[ry]:
                rank[rx] += 1
            return True
        
        # Build BFS tree structure
        # Each node: (parent_idx, depth)
        # We label nodes by their position in BFS order
        # depth 0: node 0
        # depth 1: nodes 1..k (children of 0)
        # depth d: nodes from offset[d] to offset[d+1]-1
        
        # Compute offsets
        offsets = [0]
        for d in range(1, r + 1):
            if d == 1:
                offsets.append(offsets[-1] + k)
            else:
                offsets.append(offsets[-1] + k * (k - 1) ** (d - 1))
        
        total_tree_nodes = offsets[r]
        
        # Try to construct with n = total_tree_nodes (all leaves merged)
        # Actually for girth g, we need to merge leaves at depth r
        # Number of leaves at depth r:
        if r == 0:
            n_leaves = 1
        elif r == 1:
            n_leaves = k
        else:
            n_leaves = k * (k - 1) ** (r - 1)
        
        # The graph will have n = offsets[r-1] + n_merge vertices
        # where n_merge is how many equivalence classes the leaves form
        # For a k-regular graph with girth g, each leaf at depth r must be
        # connected to k-1 other leaves (since one edge goes to its parent at depth r-1)
        # So n_merge must satisfy: n_merge * (k-1) / 2 is integer and
        # the leaf graph must be (k-1)-regular with girth >= g - 2r + 1
        
        # Actually, let's think differently. The standard approach:
        # n = offsets[r-1] + m where m is the number of "boundary" vertices
        # Each internal vertex at depth < r has k children (except root has k)
        # Each boundary vertex at depth r has k-1 edges to other boundary vertices
        # So we need a (k-1)-regular graph on m vertices with girth >= g - 2*(r-1)
        
        # For simplicity, try m = k*(k-1)^(r-1) (no merging) first,
        # then try smaller m values
        
        # Let's try the Moore bound construction directly
        # For even g=2r: n = 2*sum_{i=0}^{r-1}(k-1)^i
        # For odd g=2r+1: n = 1 + k*sum_{i=0}^{r-1}(k-1)^i
        
        if g % 2 == 0:
            # Even girth: bipartite
            # n = 2 * (1 + (k-1) + (k-1)^2 + ... + (k-1)^(r-1))
            n_target = 2 * sum((k - 1) ** i for i in range(r))
        else:
            # Odd girth
            # n = 1 + k * (1 + (k-1) + (k-1)^2 + ... + (k-1)^(r-1))
            n_target = 1 + k * sum((k - 1) ** i for i in range(r))
        
        # Try to construct at n_target
        if n_target < n_min:
            n_target = n_min
        
        # Build the tree and merge leaves
        # For now, use a simple greedy approach:
        # 1. Build BFS tree with n_target vertices
        # 2. Assign edges greedily to satisfy regularity and girth
        
        # This is complex, so let's use a different approach:
        # Randomly generate (k-1)-regular graphs on the leaf set and check girth
        
        if r <= 1:
            return None
        
        # Number of leaves in the tree
        if g % 2 == 0:
            # Bipartite: two parts of size sum_{i=0}^{r-1}(k-1)^i
            part_size = sum((k - 1) ** i for i in range(r))
            # Part A: depths 0, 2, 4, ..., 2*(r-1) if r is even... 
            # Actually for g=2r, depths 0..r-1 on one side, r..2r-1 on other?
            # No, for girth 2r, BFS from any vertex reaches depth r-1 without cycles
            # Let's just try random (k-1)-regular graphs
            
            # Simpler: try random k-regular graphs and check
            pass
        
        # For now, return None to indicate this approach needs more work
        # The PGL approach is more reliable for small parameters
        return None

    def luw_construct(k, g):
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

    n_luw, edges_luw = luw_construct(k, g)
    best_n = [n_luw]
    best_result = (n_luw, edges_luw)

    # Try greedy embedding if we have time
    if time.time() < deadline - 5.0:
        greedy_result = greedy_embed(k, g, deadline - time.time() - 3.0)
        if greedy_result and greedy_result[0] < best_n[0]:
            best_n[0] = greedy_result[0]
            best_result = greedy_result

    def action_on_point(x, a, b, c, d, q):
        if x == q:
            if c == 0:
                return q
            inv_c = modinv(c, q)
            if inv_c is None:
                return None
            return (a * inv_c) % q
        num = (a * x + b) % q
        den = (c * x + d) % q
        if den == 0:
            return q
        inv_den = modinv(den, q)
        if inv_den is None:
            return None
        return (num * inv_den) % q

    def inv_matrix(a, b, c, d, q):
        det = (a * d - b * c) % q
        det_inv = modinv(det, q)
        if det_inv is None:
            return None
        return (d * det_inv % q, -b * det_inv % q, -c * det_inv % q, a * det_inv % q)

    def is_derangement(a, b, c, d, q):
        if c == 0:
            return False
        disc = ((d - a) ** 2 + 4 * b * c) % q
        if disc == 0:
            return False
        return pow(disc, (q - 1) // 2, q) != 1

    def build_from_gens(gens, q, k, g):
        n = q + 1
        adj = [set() for _ in range(n)]
        for i in range(n):
            x = i if i < q else q
            for gen in gens:
                y = action_on_point(x, gen[0], gen[1], gen[2], gen[3], q)
                if y is None or y == i:
                    return None
                j = y if y < q else q
                adj[i].add(j)
                adj[j].add(i)
        for i in range(n):
            if len(adj[i]) != k:
                return None
        adj_list = [sorted(adj[i]) for i in range(n)]
        return validate(adj_list, n, k, g)

    def random_search_pgl(q, k, g, budget):
        n = q + 1
        if n < n_min or n >= best_n[0]:
            return None
        if q % 2 == 0 or not is_prime_power(q):
            return None

        def random_gen():
            while True:
                a = random.randrange(q)
                b = random.randrange(q)
                c = random.randrange(q)
                d = random.randrange(q)
                if (a * d - b * c) % q != 0:
                    return (a, b, c, d)

        def random_gens():
            out = []
            for _ in range(k):
                m = random_gen()
                if random.random() < 0.5:
                    mi = inv_matrix(*m, q)
                    if mi is not None:
                        out.extend([m, mi])
                        if len(out) >= k:
                            return out[:k]
                else:
                    out.append(m)
            return out

        seen = set()
        while time.time() < budget:
            gens = random_gens()
            key = tuple(gens)
            if key in seen:
                continue
            seen.add(key)
            if any((a * d - b * c) % q == 0 for (a, b, c, d) in gens):
                continue
            if not all(is_derangement(*gen, q) for gen in gens):
                continue
            res = build_from_gens(gens, q, k, g)
            if res:
                return res
        return None

    q_cursor = max(3, n_min - 1)
    while q_cursor + 1 < best_n[0] and time.time() < deadline - 2.0:
        # Allocate more time for smaller q (more likely to succeed if solution exists)
        # Actually, smaller q means denser graph, harder to get large girth.
        # Larger q means sparser, easier to get large girth, but n is larger.
        # We want to find ANY q < best_n - 1.
        remaining_time = deadline - time.time() - 2.0
        if remaining_time < 1.0:
            break
        budget = min(remaining_time, time.time() + 0.5)
        res = random_search_pgl(q_cursor, k, g, budget)
        if res and res[0] < best_n[0]:
            best_n[0] = res[0]
            best_result = res
        q_cursor += 1

    return best_result

# EVOLVE-BLOCK-END
