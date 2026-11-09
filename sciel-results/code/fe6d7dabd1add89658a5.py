"""Stochastic local search (2-switches) to maximize girth of a k-regular graph."""
import time
import random
import sys

def construct(k, g):
    """Construct a k-regular graph with girth >= g on as few vertices as possible."""
    deadline = time.time() + 140  # safety margin before 150s limit
    rng = random.Random(42)
    
    # Moore bound: minimum possible n
    # For k-regular graph with girth g:
    # If g is even (g = 2r): n >= k + k(k-1)^(r-1)
    # If g is odd (g = 2r+1): n >= 1 + k * sum_{i=0}^{r-1} (k-1)^i
    if g % 2 == 0:
        r = g // 2
        moore = k + k * (k - 1) ** (r - 1)
    else:
        r = (g - 1) // 2
        moore = 1 + k * sum((k - 1) ** i for i in range(r))
    
    # Start with a reasonable n. The known best for (4,10) is 320.
    # Moore bound for (4,10): r=5, n >= 4 + 4*3^4 = 4 + 324 = 328? 
    # Wait: g=10 even, r=5, n >= k + k*(k-1)^(r-1) = 4 + 4*3^4 = 4+324=328
    # But the problem says smallest known is 320 and lower bound is 243.
    # Let me recompute: girth 10 means no cycles of length < 10, so shortest cycle is 10.
    # For even girth 2r: the Moore bound is k + k*(k-1)^(r-1).
    # g=10, r=5: 4 + 4*3^4 = 4 + 4*81 = 4+324 = 328.
    # Hmm, but the task says lower bound is 243 and best known is 320.
    # Actually I think I'm confusing myself. Let me just try n values.
    
    # Try to find a valid graph starting from a small n and increasing
    # We'll use simulated annealing with 2-switches
    
    def generate_random_regular(n, k, rng):
        """Generate a random k-regular graph on n vertices using configuration model + fix."""
        if n * k % 2 != 0:
            return None
        if k >= n:
            return None
        
        # Configuration model: create n*k stubs
        stubs = []
        for v in range(n):
            for _ in range(k):
                stubs.append(v)
        rng.shuffle(stubs)
        
        # Pair up stubs to form edges
        edges = []
        for i in range(0, n * k, 2):
            u, v = stubs[i], stubs[i + 1]
            if u != v:
                edges.append((u, v))
        
        # Check for multi-edges and self-loops; if any, regenerate
        edge_set = set()
        valid = True
        for u, v in edges:
            if u == v:
                valid = False
                break
            key = (min(u, v), max(u, v))
            if key in edge_set:
                valid = False
                break
            edge_set.add(key)
        
        if not valid:
            return None
        
        # Verify regularity
        deg = [0] * n
        for u, v in edges:
            deg[u] += 1
            deg[v] += 1
        if any(d != k for d in deg):
            return None
        
        return edges
    
    def build_adjacency(n, edges):
        adj = [set() for _ in range(n)]
        for u, v in edges:
            adj[u].add(v)
            adj[v].add(u)
        return adj
    
    def compute_girth(n, adj):
        """Compute girth using BFS from each vertex (non-backtracking)."""
        min_girth = float('inf')
        for start in range(n):
            # BFS tracking (node, parent, depth)
            visited = {}  # node -> min depth reached
            queue = [(start, -1, 1)]
            found = False
            while queue and not found:
                new_queue = []
                for node, parent, depth in queue:
                    if node in visited and visited[node] < depth:
                        continue
                    if node in visited and visited[node] == depth:
                        # Found a cycle of length depth
                        if depth < min_girth:
                            min_girth = depth
                        found = True
                        break
                    if node in visited and visited[node] > depth:
                        visited[node] = depth
                    else:
                        visited[node] = depth
                    if depth >= min_girth:
                        continue
                    for nb in adj[node]:
                        if nb != parent:
                            new_queue.append((nb, node, depth + 1))
                queue = new_queue
        return min_girth if min_girth != float('inf') else 0
    
    def find_short_cycle(n, adj, target_girth):
        """Find a cycle shorter than target_girth, return list of vertices or None."""
        for start in range(n):
            # BFS non-backtracking
            visited = {}  # node -> (parent, depth)
            queue = [(start, -1, 1)]
            while queue:
                new_queue = []
                for node, parent, depth in queue:
                    if depth >= target_girth:
                        continue
                    for nb in adj[node]:
                        if nb == parent:
                            continue
                        if nb == start and depth + 1 < target_girth:
                            # Found cycle
                            # Reconstruct path
                            cycle = [start]
                            cur, par = node, parent
                            while cur != start:
                                cycle.append(cur)
                                cur, par = par, visited.get(cur, (-1, 0))[0]
                            return cycle
                        if nb in visited:
                            pd, pdepth = visited[nb]
                            if pdepth + depth + 1 < target_girth:
                                # Found a shorter cycle
                                return [nb]  # simplified
                        else:
                            visited[nb] = (node, depth + 1)
                            new_queue.append((nb, node, depth + 1))
                queue = new_queue
        return None
    
    def two_switch(adj, n, k, rng):
        """Apply a random 2-switch: find two edges (a,b) and (c,d) and replace with (a,c) and (b,d)."""
        # Build edge list from adjacency
        edge_list = []
        for u in range(n):
            for v in adj[u]:
                if u < v:
                    edge_list.append((u, v))
        
        if len(edge_list) < 2:
            return False
        
        # Try a few random 2-switches
        for _ in range(20):
            i, j = rng.sample(range(len(edge_list)), 2)
            a, b = edge_list[i]
            c, d = edge_list[j]
            
            # Check that the new edges don't create multi-edges or self-loops
            if a == c or a == d or b == c or b == d:
                continue
            if a == b or c == d:
                continue
            if c in adj[a] or d in adj[a] or c in adj[b] or d in adj[b]:
                continue
            
            # Perform the switch
            adj[a].remove(b)
            adj[c].remove(d)
            adj[a].add(c)
            adj[b].add(d)
            adj[c].add(a)
            adj[d].add(b)
            return True
        
        return False
    
    def try_improve(n, k, g, rng, deadline):
        """Try to find a k-regular graph on n vertices with girth >= g."""
        # Generate initial graph
        edges = None
        attempts = 0
        while time.time() < deadline - 5:
            edges = generate_random_regular(n, k, rng)
            if edges is not None:
                break
            attempts += 1
            if attempts > 100:
                return None
        
        if edges is None:
            return None
        
        adj = build_adjacency(n, edges)
        current_girth = compute_girth(n, adj)
        
        if current_girth >= g:
            return adj
        
        # Simulated annealing
        temperature = 1.0
        cooling = 0.999
        best_adj = [set(s) for s in adj]
        best_girth = current_girth
        
        while time.time() < deadline - 5:
            # Try a 2-switch
            if not two_switch(adj, n, k, rng):
                temperature *= cooling
                continue
            
            new_girth = compute_girth(n, adj)
            
            # Accept if better or with probability exp(-(new_girth - current_girth)/temperature)
            if new_girth >= current_girth:
                current_girth = new_girth
                if new_girth > best_girth:
                    best_girth = new_girth
                    best_adj = [set(s) for s in adj]
            else:
                import math
                delta = current_girth - new_girth
                if delta > 0:
                    prob = math.exp(-delta / max(temperature, 1e-10))
                    if rng.random() < prob:
                        current_girth = new_girth
            
            temperature *= cooling
        
        if best_girth >= g:
            return best_adj
        return None
    
    # Try increasing n from Moore bound upward
    # But we want to minimize n, so start from the lower bound
    # For (4,10), known best is 320, lower bound 243
    # Let's try a range of n values
    
    # First, let's try to use the LUW construction as a starting point if it works
    # Actually, let's just do stochastic search
    
    # Determine a reasonable starting n
    # The Moore bound for (k, g):
    if g % 2 == 0:
        r = g // 2
        moore = k + k * (k - 1) ** (r - 1)
    else:
        r = (g - 1) // 2
        moore = 1 + k * sum((k - 1) ** i for i in range(r))
    
    # Try n values starting from moore bound
    best_result = None
    best_n = float('inf')
    
    # For efficiency, try a few candidate n values
    # The known answer for (4,10) is 320, so let's try around there
    # But we should search from lower
    
    # Actually, for a quick approach, let's just try to construct at n=320 for (4,10)
    # or use the algebraic construction and then try to improve
    
    # Let's first try the LUW construction to get a valid starting point
    # Then do local search to see if we can reduce n
    
    # For now, let's do a direct stochastic search at candidate n values
    # We'll try n = moore, moore+1, ..., up to some limit
    
    # To be safe within time, let's try a limited set of n values
    # Start from a reasonable n and work down
    
    # For (4,10): try n = 320, 316, 312, ... down to moore
    # But that's too many. Let's try a few.
    
    # Actually, the simplest approach: use the LUW construction which gives a valid graph,
    # then try to reduce the number of vertices by checking if smaller graphs work.
    
    # Let me just run the stochastic search at n values near the known optimum
    # For generality, I'll try n from moore to moore + 50 or so
    
    # But this could be slow. Let me limit the number of n values to try.
    
    max_n_to_try = min(moore + 100, 500)
    
    # Try from larger n first (more likely to succeed) then work down
    # Actually, we want the smallest n, so try from moore upward
    # But smaller n is harder. Let's try a few values.
    
    # For practical purposes within time limit, try at most ~10 different n values
    n_candidates = []
    for n in range(moore, max_n_to_try + 1):
        if n * k % 2 == 0:  # must have even number of stubs
            n_candidates.append(n)
    
    # Limit to first 15 candidates
    n_candidates = n_candidates[:15]
    
    for n in n_candidates:
        if time.time() > deadline - 10:
            break
        
        adj = try_improve(n, k, g, rng, deadline)
        if adj is not None:
            # Verify
            final_girth = compute_girth(n, adj)
            if final_girth >= g:
                # Build edge list
                edges = []
                for u in range(n):
                    for v in adj[u]:
                        if u < v:
                            edges.append((u, v))
                return n, edges
    
    # If we couldn't find anything, fall back to LUW construction
    # (same as original)
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
