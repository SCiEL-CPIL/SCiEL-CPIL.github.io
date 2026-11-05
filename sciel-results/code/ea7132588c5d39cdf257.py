import time
import random
import math
from collections import deque
from itertools import product

# EVOLVE-BLOCK-START

def construct(k, g):
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin before 120s limit

    def remaining():
        return deadline - time.time()

    # ---------- arithmetic helpers ----------
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

    def is_quadratic_residue(a, q):
        a = a % q
        if a == 0:
            return True
        if not is_prime_power(q):
            return True
        p = None
        for cand in range(2, int(math.isqrt(q)) + 1):
            if q % cand == 0:
                t = q
                while t % cand == 0:
                    t //= cand
                if t == 1:
                    p = cand
                    break
        if p is None:
            p = q
        a_p = a % p
        if a_p == 0:
            return True
        return pow(a_p, (p - 1) // 2, p) == 1

    def modinv(a, m):
        if m <= 1:
            return None
        a = a % m
        if a == 0:
            return None
        # Extended Euclidean for general modulus
        def egcd(a, b):
            if b == 0:
                return (a, 1, 0)
            g, x, y = egcd(b, a % b)
            return (g, y, x - (a // b) * y)
        g, x, _ = egcd(a, m)
        if g != 1:
            return None
        return x % m

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

    # ---------- graph validation ----------
    def get_girth_fast(adj, n, limit=None):
        """Returns the girth of the graph using BFS from each vertex.
        If limit is given, stops early if girth >= limit (returns limit)."""
        min_girth = float('inf')
        if limit is not None:
            min_girth = limit
        for u in range(n):
            dist = [-1] * n
            parent = [-1] * n
            dist[u] = 0
            queue = deque([u])
            while queue:
                x = queue.popleft()
                if dist[x] + 1 >= min_girth:
                    continue
                for y in adj[x]:
                    if dist[y] == -1:
                        dist[y] = dist[x] + 1
                        parent[y] = x
                        queue.append(y)
                    elif y != parent[x] and parent[y] != x:
                        cycle_len = dist[x] + dist[y] + 1
                        if cycle_len < min_girth:
                            min_girth = cycle_len
                            if limit is not None and min_girth >= limit:
                                return min_girth
        return min_girth if min_girth != float('inf') else 0

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
                return None
        if not is_connected(adj, n):
            return None
        girth = get_girth_fast(adj, n, limit=g)
        if girth < g:
            return None
        edges = []
        for i in range(n):
            for j in adj[i]:
                if i < j:
                    edges.append((i, j))
        return (n, edges)

    # ---------- Lazebnik-Ustimenko-Woldar baseline ----------
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

    # ---------- PGL(2, q) machinery ----------
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
        a, b, c, d = a % q, b % q, c % q, d % q
        if c == 0:
            return False
        disc = ((d - a) ** 2 + 4 * b * c) % q
        if disc == 0:
            return False
        if is_quadratic_residue(disc, q):
            return False
        return True

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
        return validate(adj, n, k, g)

    def random_search_pgl(q, k, g, budget):
        n = q + 1
        if n < n_min:
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
            attempts = 0
            while len(out) < k and attempts < k * 20:
                attempts += 1
                m = random_gen()
                mi = inv_matrix(*m, q)
                if mi is None:
                    continue
                if not is_derangement(*m, q):
                    continue
                # Check involution
                a2 = (m[0]*m[0] + m[1]*m[2]) % q
                b2 = (m[1]*(m[0]+m[3])) % q
                c2 = (m[2]*(m[0]+m[3])) % q
                d2 = (m[3]*m[3] + m[1]*m[2]) % q
                is_involution = (a2 == 1 and b2 == 0 and c2 == 0 and d2 == 1)
                
                if is_involution or random.random() < 0.4:
                    out.append(m)
                    if mi != m and len(out) < k:
                        out.append(mi)
            while len(out) < k:
                m = random_gen()
                if is_derangement(*m, q):
                    out.append(m)
            return out[:k]

        seen = set()
        while time.time() < budget:
            gens = random_gens()
            key = tuple(gens)
            if key in seen:
                continue
            seen.add(key)
            if not all(is_derangement(*gen, q) for gen in gens):
                continue
            res = build_from_gens(gens, q, k, g)
            if res:
                return res
        return None

    # ---------- Simulated Annealing for small n ----------
    def sa_construct(n, k, g, budget):
        if n < n_min:
            return None
        if n > 150: # SA is slow for large n
            return None
            
        # Generate random k-regular graph
        # Use configuration model
        def random_k_regular(n, k):
            if (n * k) % 2 != 0:
                return None
            stubs = []
            for i in range(n):
                for _ in range(k):
                    stubs.append(i)
            random.shuffle(stubs)
            adj = [set() for _ in range(n)]
            valid = True
            for i in range(0, len(stubs), 2):
                u, v = stubs[i], stubs[i+1]
                if u == v:
                    valid = False
                    break
                if v in adj[u]:
                    valid = False
                    break
                adj[u].add(v)
                adj[v].add(u)
            if not valid:
                return None
            # Check connectivity
            if not is_connected(adj, n):
                return None
            return adj

        adj = None
        attempts = 0
        while attempts < 100 and time.time() < budget - 5.0:
            attempts += 1
            adj = random_k_regular(n, k)
            if adj:
                break
        if not adj:
            return None
            
        current_girth = get_girth_fast(adj, n, limit=g)
        if current_girth >= g:
            return validate(adj, n, k, g)
            
        best_girth = current_girth
        best_adj = [set(s) for s in adj]
        
        T = 1.0
        cooling = 0.995
        
        while time.time() < budget:
            # Pick random edges to swap
            # (u, v) and (x, y) -> (u, x) and (v, y)
            # Ensure u!=x, v!=y, u!=y, v!=x to avoid loops
            # And (u,x) not in adj, (v,y) not in adj
            
            u = random.randrange(n)
            v = random.choice(list(adj[u]))
            
            x = random.randrange(n)
            if x == u or x == v:
                continue
            if x not in adj[u]: # Ensure (u,x) is not an edge
                y = random.choice(list(adj[x]))
                if y == u or y == v:
                    continue
                if y not in adj[v]: # Ensure (v,y) is not an edge
                    # Check for multi-edges
                    if x in adj[v] or y in adj[u]:
                        continue
                    
                    # Perform swap
                    adj[u].remove(v)
                    adj[v].remove(u)
                    adj[x].remove(y)
                    adj[y].remove(x)
                    
                    adj[u].add(x)
                    adj[x].add(u)
                    adj[v].add(y)
                    adj[y].add(v)
                    
                    new_girth = get_girth_fast(adj, n, limit=g)
                    
                    if new_girth >= g:
                        return validate(adj, n, k, g)
                    
                    if new_girth > best_girth:
                        best_girth = new_girth
                        best_adj = [set(s) for s in adj]
                    else:
                        # Metropolis
                        delta = best_girth - new_girth
                        if delta > 0 and random.random() > math.exp(-delta / T):
                            # Revert
                            adj[u].remove(x)
                            adj[x].remove(u)
                            adj[v].remove(y)
                            adj[y].remove(v)
                            
                            adj[u].add(v)
                            adj[v].add(u)
                            adj[x].add(y)
                            adj[y].add(x)
                            
                    T *= cooling
            if T < 0.01:
                T = 0.01
                
        return None

    # ---------- BFS-Tree Pruning Strategy ----------
    def bfs_tree_prune(k, g, budget):
        """
        Construct a k-regular graph with girth >= g using BFS-tree pruning.
        Build a tree of non-backtracking walks of length floor(g/2) from a root,
        then merge leaves to satisfy k-regularity.
        """
        r = g // 2  # radius: max distance from root without creating short cycles
        
        # Generate the tree structure: all non-backtracking walks of length <= r
        # Node (depth, parent, index) where index distinguishes children
        # For depth 0: root
        # For depth d: each node at depth d-1 has (k if d==1 else k-1) children
        
        # Build tree as list of (parent, child_index, depth)
        # We'll use a compact representation
        # tree_nodes: list of (parent_idx, child_idx, depth)
        # parent_idx = -1 for root
        
        tree_nodes = [(-1, 0, 0)]  # root
        node_count = 1
        nodes_at_depth = {0: 1}
        
        for d in range(1, r + 1):
            num_children = k if d == 1 else (k - 1)
            nodes_at_depth[d] = nodes_at_depth[d-1] * num_children
            parent_idx = 0
            for p in range(nodes_at_depth[d-1]):
                for c in range(num_children):
                    tree_nodes.append((p, c, d))
                    parent_idx += 1
            node_count += nodes_at_depth[d]
        
        # Total nodes in tree
        total_tree_nodes = sum(nodes_at_depth.values())
        
        # The number of leaves (at depth r) is what we need to merge
        # For even g: girth = 2*r, so we need to ensure no two leaves at depth r
        #   that are "close" in the tree get merged in a way that creates short cycles
        # For odd g: girth = 2*r+1, similar logic
        
        # Strategy: Assign each leaf a "target" vertex in [0, n-1]
        # We need to figure out what n should be.
        # The Moore bound gives the minimum possible n.
        # Let's try n = n_min first, then increase.
        
        # For now, let's try a random assignment approach:
        # 1. Try n from n_min upward
        # 2. For each n, randomly assign leaves to vertices
        # 3. Check if the resulting graph is k-regular and has girth >= g
        
        # Actually, let's use a more systematic approach:
        # The tree has total_tree_nodes nodes.
        # Internal nodes (depth < r) must map to distinct vertices.
        # Leaves (depth = r) can map to any vertex, but we need to ensure
        # that the resulting graph is k-regular.
        
        # Number of internal nodes = total_tree_nodes - nodes_at_depth[r]
        internal_count = total_tree_nodes - nodes_at_depth[r]
        leaf_count = nodes_at_depth[r]
        
        # For the graph to be k-regular with n vertices, we need n*k/2 edges.
        # The tree gives us total_tree_nodes - 1 edges (each non-root node has one parent edge).
        # Additional edges come from leaf-to-vertex connections.
        
        # Let's try a different approach: use the tree to define a partial graph,
        # then complete it to k-regular.
        
        # For small n, we can try random completions.
        
        if time.time() > budget:
            return None
        
        # Try n from n_min to min(n_min + 50, 200)
        for n_try in range(n_min, min(n_min + 50, 200)):
            if time.time() > budget:
                break
            if n_try < internal_count:
                continue  # Need at least internal_count vertices
            
            # Try several random assignments
            for attempt in range(20):
                if time.time() > budget:
                    break
                
                # Assign internal nodes to distinct vertices
                vertices = list(range(n_try))
                random.shuffle(vertices)
                
                # Map tree nodes to graph vertices
                node_to_vertex = {}
                for i, (parent, child, depth) in enumerate(tree_nodes):
                    if depth < r:
                        node_to_vertex[i] = vertices[i]
                    else:
                        # Leaves: assign randomly (will be refined)
                        node_to_vertex[i] = random.choice(vertices)
                
                # Build adjacency from tree edges
                adj = [set() for _ in range(n_try)]
                for i, (parent, child, depth) in enumerate(tree_nodes):
                    if parent == -1:
                        continue
                    u = node_to_vertex[i]
                    v = node_to_vertex[parent]
                    if u != v:
                        adj[u].add(v)
                        adj[v].add(u)
                
                # Check current degrees
                valid = True
                for v in range(n_try):
                    if len(adj[v]) > k:
                        valid = False
                        break
                
                if not valid:
                    continue
                
                # Complete the graph to k-regular by adding edges
                # This is a k-regular graph completion problem
                # For simplicity, try random edge additions
                success = False
                for _ in range(1000):
                    # Find vertices with degree < k
                    underfull = [v for v in range(n_try) if len(adj[v]) < k]
                    if not underfull:
                        break
                    
                    # Try to add an edge between two underfull vertices
                    u = random.choice(underfull)
                    candidates = [v for v in underfull if v != u and v not in adj[u]]
                    if not candidates:
                        continue
                    v = random.choice(candidates)
                    
                    # Check if adding this edge creates a short cycle
                    # Quick check: if dist(u,v) < g-1 in current graph, skip
                    # For now, just add and check girth at the end
                    adj[u].add(v)
                    adj[v].add(u)
                
                # Check if k-regular
                if all(len(adj[v]) == k for v in range(n_try)):
                    # Check girth
                    girth = get_girth_fast(adj, n_try, limit=g)
                    if girth >= g:
                        return validate(adj, n_try, k, g)
        
        return None

    # ---------- Cayley graph on Z_n with 5 generators ----------
    def cayley_zn_search(k, g, budget):
        """
        Search for a k-regular Cayley graph on Z_n with girth >= g.
        For k=5: vertex i connected to i±d1, i±d2, i+d5 (or similar asymmetric set).
        Girth condition: no non-trivial integer linear combination of generators
        with |coefficients| summing to < g equals 0 mod n.
        """
        if k != 5:
            return None
        
        r = (g - 1) // 2  # max half-length of a non-backtracking walk
        
        # For 5-regular on Z_n, we need 5 "directions" (generators).
        # Symmetric: i ± d1, i ± d2, i + d3 where d3 != -d3 mod n
        # or: i + d1, i + d2, i + d3, i + d4, i + d5 (all distinct, none is inverse of another unless we want loops)
        # Actually for 5-regular simple graph on Z_n, we need 5 distinct nonzero elements
        # such that no element is the negative of another (or if it is, we handle it).
        # Standard: pick 5 elements d1..d5 where -di != dj for all i,j, and all distinct.
        # Then neighbors of i are i+di for i=1..5.
        # But we need degree exactly 5, so each di gives one neighbor.
        # The graph is undirected if we also include -di, but that would give degree 10.
        # So for undirected 5-regular: we need pairs {di, -di} plus one more.
        # Actually: 5 is odd. So we can have 2 pairs {d1,-d1}, {d2,-d2} and one self-inverse d3 where 2*d3=0 mod n.
        # Or: just use directed generators and make the graph directed... but we need undirected.
        
        # For undirected 5-regular Cayley on Z_n:
        # We need a symmetric generating set S with |S|=5, S=S^{-1}, 0 not in S.
        # Since 5 is odd, at least one element s in S satisfies s = -s, i.e., 2s=0 mod n.
        # This requires n even.
        # So: S = {a, -a, b, -b, c} where 2c = 0 mod n, i.e., c = n/2.
        
        # Girth condition: the shortest non-trivial relation sum of generators = 0 mod n
        # with no immediate backtracking.
        
        # Let's enumerate: for each n from n_min upward, try all (a, b) pairs.
        # c = n/2 (only valid if n even)
        
        def check_girth_cayley(n, a, b, c, g):
            """Check if Cayley graph on Z_n with generators {a,-a,b,-b,c} has girth >= g."""
            # BFS from 0 to find shortest cycle
            # State: (current_vertex, previous_generator) to avoid backtracking
            # For girth g, we need no cycle of length < g.
            # A cycle of length L means a sequence of L generators g1..gL with sum = 0 mod n,
            # and no consecutive pair is (gi, -gi).
            
            # BFS in the non-backtracking walk space
            # State: (vertex, last_gen) where last_gen is the generator used to get here
            # Start: from 0, take each generator
            # We want to find if we can return to 0 with a non-backtracking walk of length < g
            
            visited = set()  # (vertex, last_gen_index)
            queue = deque()
            
            gens = [a, -a % n, b, -b % n, c]
            # Remove duplicates
            seen_gens = set()
            unique_gens = []
            for gen in gens:
                gen = gen % n
                if gen != 0 and gen not in seen_gens:
                    seen_gens.add(gen)
                    unique_gens.append(gen)
            
            if len(unique_gens) != 5:
                return False
            
            # Map each gen to its index for backtracking check
            gen_to_idx = {gen: i for i, gen in enumerate(unique_gens)}
            # For backtracking: gen i and gen j are inverses if (gen_i + gen_j) % n == 0
            inverse = {}
            for i, gen in enumerate(unique_gens):
                inv = (-gen) % n
                if inv in gen_to_idx:
                    inverse[i] = gen_to_idx[inv]
                else:
                    inverse[i] = None  # self-inverse (c = n/2)
            
            # BFS: state is (vertex, last_gen_idx)
            # Start: from 0, take each generator
            for i, gen in enumerate(unique_gens):
                v = gen % n
                state = (v, i)
                if state not in visited:
                    visited.add(state)
                    queue.append((v, i, 1))  # length 1
            
            while queue:
                v, last_i, length = queue.popleft()
                if length >= g:
                    continue
                for j, gen in enumerate(unique_gens):
                    if j == inverse.get(last_i, -1):
                        continue  # backtracking
                    new_v = (v + gen) % n
                    new_state = (new_v, j)
                    if new_v == 0 and length + 1 >= 2:
                        # Found a cycle of length length+1
                        if length + 1 < g:
                            return False
                        else:
                            continue  # this path is OK, don't expand
                    if new_state not in visited:
                        visited.add(new_state)
                        queue.append((new_v, j, length + 1))
            
            return True
        
        # Try n from n_min upward
        max_n = min(n_min + 500, 2000)
        for n in range(n_min, max_n + 1):
            if time.time() > budget:
                break
            if n % 2 != 0:
                continue  # Need even n for odd degree Cayley on Z_n
            
            c = n // 2
            # Try different (a, b) pairs
            # a and b should be "small" to avoid creating short cycles
            # a should not be 0, c, or -c
            # b should not be 0, a, -a, c, -c
            
            found = False
            for a in range(1, n // 2):
                if a == c:
                    continue
                if time.time() > budget:
                    break
                for b in range(1, n // 2):
                    if b == a or b == c or b == n - a or b == n - c:
                        continue
                    # Check that all 5 generators are distinct
                    gens = {a, n - a, b, n - b, c}
                    if len(gens) != 5:
                        continue
                    # Also need: a != b, a != -b, a != c, etc.
                    if a == b or a == n - b or a == c or a == n - c:
                        continue
                    if b == c or b == n - c:
                        continue
                    
                    if check_girth_cayley(n, a, b, c, g):
                        # Build the graph
                        adj = [set() for _ in range(n)]
                        for i in range(n):
                            for gen in [a, n - a, b, n - b, c]:
                                j = (i + gen) % n
                                adj[i].add(j)
                                adj[j].add(i)
                        # Validate
                        result = validate(adj, n, k, g)
                        if result:
                            return result
                        found = True
                        break
                if found:
                    break
            if found:
                break
        
        return None

    # ---------- main ----------
    # 1. deterministic baseline
    n_luw, edges_luw = luw_construct(k, g)
    best_n = [n_luw]
    best_result = (n_luw, edges_luw)

    # 2. Try Cayley graph on Z_n (deterministic scan)
    cayley_budget = min(40.0, deadline - 15.0 - time.time())
    if cayley_budget > 5.0:
        res = cayley_zn_search(k, g, cayley_budget)
        if res and res[0] < best_n[0]:
            best_n[0] = res[0]
            best_result = res

    # 3. Try BFS-tree pruning for smaller n
    bfs_budget = min(20.0, deadline - 10.0 - time.time())
    if bfs_budget > 5.0:
        res = bfs_tree_prune(k, g, bfs_budget)
        if res and res[0] < best_n[0]:
            best_n[0] = res[0]
            best_result = res

    # 4. Search for smaller n using PGL and SA
    n_cursor = n_min
    while n_cursor < best_n[0] and time.time() < deadline - 10.0:
        
        # Check if n-1 is a prime power for PGL
        q = n_cursor - 1
        if q >= 3 and is_prime_power(q):
            # Try PGL with short budget
            pgl_budget = min(5.0, deadline - 10.0 - time.time())
            res = random_search_pgl(q, k, g, pgl_budget)
            if res:
                best_n[0] = res[0]
                best_result = res
                n_cursor += 1
                continue
        
        # Try Simulated Annealing
        if n_cursor <= 150:
            sa_budget = min(10.0, deadline - 10.0 - time.time())
            res = sa_construct(n_cursor, k, g, sa_budget)
            if res:
                best_n[0] = res[0]
                best_result = res
                n_cursor += 1
                continue
        
        n_cursor += 1

    return best_result

# EVOLVE-BLOCK-END
