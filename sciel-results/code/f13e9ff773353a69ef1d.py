import time
import random
import sys
from collections import deque

def construct(k, g):
    start_time = time.time()
    deadline = start_time + 145.0

    # Moore bound for k=3, g=16
    if k == 3:
        if g % 2 == 0:
            moore_bound = 2 * sum((k - 1) ** i for i in range(g // 2))
        else:
            moore_bound = (k - 1) * sum((k - 1) ** i for i in range((g - 1) // 2))
    else:
        moore_bound = 1

    def greedy_nonbacktracking_construction(n, g, seed):
        """
        Build a 3-regular graph with high girth using greedy non-backtracking path extension.
        
        Strategy:
        1. Start with a path of length g (no cycles possible yet).
        2. Greedily extend paths using non-backtracking constraint.
        3. When a path can't extend without creating a short cycle, 
           close it to a neighbor at maximum distance.
        4. Use a priority queue to always extend the longest available path.
        
        This avoids random sampling and instead directly constructs long paths.
        """
        rng = random.Random(seed)
        
        # adj[v] = list of neighbors
        adj = [[] for _ in range(n)]
        # Track which vertices have degree < k (still need connections)
        # deg[v] = current degree
        deg = [0] * n
        
        # We'll build the graph by adding edges one at a time.
        # Key insight: to maximize girth, we want to avoid short cycles.
        # Strategy: build a spanning tree first (BFS tree), then add remaining edges
        # in a way that maximizes the distance between connected vertices.
        
        # Phase 1: Build a BFS tree (spanning tree)
        # This gives us n-1 edges with no cycles.
        parent = [-1] * n
        dist = [0] * n
        order = [0]
        visited = [False] * n
        visited[0] = True
        queue = deque([0])
        
        while queue:
            u = queue.popleft()
            # Find unvisited neighbors in random order
            neighbors = list(range(n))
            rng.shuffle(neighbors)
            added = 0
            for v in neighbors:
                if not visited[v]:
                    visited[v] = True
                    parent[v] = u
                    dist[v] = dist[u] + 1
                    order.append(v)
                    queue.append(v)
                    adj[u].append(v)
                    adj[v].append(u)
                    deg[u] += 1
                    deg[v] += 1
                    added += 1
                    if added >= 1:  # Just one child per node in tree phase for simplicity
                        break
            # Actually for a proper spanning tree, each node (except root) gets one parent
            # and can have multiple children. Let's redo this.
        
        # Reset - let's use a cleaner approach
        adj = [[] for _ in range(n)]
        deg = [0] * n
        
        # Phase 1: Build a random spanning tree
        # Use a randomized BFS/DFS
        parent = [-1] * n
        visited = [False] * n
        visited[0] = True
        order = [0]
        queue = deque([0])
        
        while queue:
            u = queue.popleft()
            # Shuffle potential children
            candidates = [v for v in range(n) if not visited[v]]
            rng.shuffle(candidates)
            # Add up to (k-1) children to u (since u already has 1 edge to parent, except root)
            max_children = k - 1 if parent[u] != -1 else k
            for v in candidates[:max_children]:
                visited[v] = True
                parent[v] = u
                order.append(v)
                adj[u].append(v)
                adj[v].append(u)
                deg[u] += 1
                deg[v] += 1
                queue.append(v)
        
        # Now we have a spanning tree with n-1 edges.
        # We need k*n/2 - (n-1) more edges to make it k-regular.
        # For k=3: need 3n/2 - n + 1 = n/2 + 1 more edges.
        
        # Phase 2: Add remaining edges greedily to maximize girth.
        # For each pair of vertices (u,v) not already connected, compute the
        # tree distance. We want to add edges between vertices that are
        # far apart in the tree (to avoid short cycles).
        
        # Compute tree distances using LCA or BFS from each vertex.
        # For efficiency with n up to ~1000, we can do BFS from each vertex.
        
        # Actually, let's think differently.
        # The girth of the final graph is determined by the shortest cycle.
        # A cycle in the final graph = a path in the tree + the added edge.
        # So the girth >= min over all added edges (u,v) of (tree_dist(u,v) + 1).
        # Wait, that's not quite right. A cycle could use multiple added edges.
        # But the shortest cycle will typically use at most 2 added edges (otherwise
        # it would be longer).
        
        # More precisely: any cycle uses some added edges. If it uses exactly one
        # added edge (u,v), its length is tree_dist(u,v) + 1.
        # If it uses two added edges (u,v) and (w,x), the cycle length is at least
        # tree_dist(u,w) + 1 + tree_dist(v,x) + 1 or similar combinations.
        
        # To maximize girth, we should add edges between vertices that are far apart
        # in the tree. Let's sort all possible edges by tree distance (descending)
        # and greedily add them while maintaining k-regularity.
        
        # Compute tree distances. For n up to 1000, BFS from each vertex is O(n^2).
        # That's 10^6 operations, fine.
        
        # Actually, let's use a smarter approach.
        # We need to add exactly (k*n/2 - (n-1)) = (3n/2 - n + 1) = n/2 + 1 edges for k=3.
        # These edges must complete the degree requirements.
        
        # Let's identify which vertices still need edges:
        # In the tree, deg[v] = (number of children) + (1 if not root else 0)
        # We need deg[v] = k = 3 for all v.
        # So vertices with deg[v] < 3 need (3 - deg[v]) more edges.
        
        # Greedy approach: repeatedly pick the pair (u,v) with:
        # - deg[u] < k and deg[v] < k
        # - (u,v) not already an edge
        # - tree_dist(u,v) is maximized
        # Add the edge.
        
        # For efficiency, precompute tree distances.
        # BFS from each vertex.
        tree_dist = [[0] * n for _ in range(n)]
        for s in range(n):
            d = [-1] * n
            d[s] = 0
            q = deque([s])
            while q:
                u = q.popleft()
                for v in adj[u]:
                    if d[v] == -1:
                        d[v] = d[u] + 1
                        q.append(v)
            tree_dist[s] = d
        
        # Build list of vertices that need more edges
        need_edges = {v: k - deg[v] for v in range(n)}
        
        # Total edges to add
        total_to_add = sum(need_edges.values()) // 2
        
        # For efficiency, let's use a different strategy:
        # Instead of O(n^2) per edge addition, let's be smarter.
        
        # Alternative: pair up "leaves" (vertices with most missing edges) 
        # with vertices that are far from them in the tree.
        
        # Actually, for n up to ~1000 and ~500 edges to add, 
        # even O(n^2) per edge is 500 * 10^6 = 5 * 10^8, too slow.
        
        # Let's use a heap-based approach or just random sampling with rejection.
        
        # Simpler approach: random sampling with girth check at the end.
        # But that's what the current program does!
        
        # Let me try yet another approach: 
        # Use the "cage" construction method based on voltage graphs or 
        # algebraic constructions.
        
        # Actually, let's try a different structural approach:
        # Build the graph as a "ladder" or "circular ladder" variant.
        
        return None  # This approach is getting complicated; let's try something else

    def build_high_girth_random(n, g, seed):
        """
        Build a random 3-regular graph optimized for high girth.
        
        Key insight: instead of uniform random pairing, use a "far apart" 
        pairing strategy. When pairing stubs, prefer to connect vertices
        that are far apart in some ordering.
        
        This is a structural change: instead of uniform random, we bias
        the random process to avoid short cycles.
        """
        rng = random.Random(seed)
        
        if n < 4 or n % 2 != 0:
            return None
        
        # Create stubs
        stubs = []
        for i in range(n):
            for _ in range(k):
                stubs.append(i)
        
        # Instead of uniform shuffle, use a "spread out" shuffle.
        # Idea: pair stubs from "opposite" halves of the vertex range.
        # This biases against short cycles.
        
        # Strategy: divide vertices into two halves.
        # First, pair within each half (creating local structure).
        # Then, pair across halves (creating long-range connections).
        
        # Actually, a better approach: use a "random but spread" pairing.
        # For each vertex i, prefer to connect to vertices in range 
        # [i + n/4, i + 3n/4] (mod n) to create long-range edges.
        
        # Let's try: random permutation but with a "spread" bias.
        # Pair stubs such that the two vertices are far apart in cyclic order.
        
        # Simple approach: create two lists of stubs, shuffle each, pair across.
        half = n // 2
        stubs_first = []
        stubs_second = []
        for i in range(n):
            for _ in range(k):
                if i < half:
                    stubs_first.append(i)
                else:
                    stubs_second.append(i)
        
        # This doesn't work well because we need exactly k*n/2 pairs.
        
        # Let's try a different structural approach:
        # Build a "random" 3-regular graph but use rejection sampling
        # with an incremental girth check.
        
        # Actually, let's use a completely different method:
        # The "BFS tree + far edges" approach, but done efficiently.
        
        # Phase 1: Build a random spanning tree
        adj = [[] for _ in range(n)]
        deg = [0] * n
        parent = [-1] * n
        visited = [False] * n
        visited[0] = True
        order = [0]
        queue = deque([0])
        
        while queue:
            u = queue.popleft()
            candidates = [v for v in range(n) if not visited[v]]
            rng.shuffle(candidates)
            max_children = k - 1 if parent[u] != -1 else k
            for v in candidates[:max_children]:
                visited[v] = True
                parent[v] = u
                order.append(v)
                adj[u].append(v)
                adj[v].append(u)
                deg[u] += 1
                deg[v] += 1
                queue.append(v)
        
        # Phase 2: Add remaining edges greedily, preferring far-apart pairs.
        # Use a max-heap of (tree_dist, u, v) candidates.
        # But computing all tree distances is O(n^2).
        
        # For n up to 1000, O(n^2) = 10^6, which is fine.
        # Let's compute tree distances.
        tree_dist = [[0] * n for _ in range(n)]
        for s in range(n):
            d = [-1] * n
            d[s] = 0
            q = deque([s])
            while q:
                u = q.popleft()
                for v in adj[u]:
                    if d[v] == -1:
                        d[v] = d[u] + 1
                        q.append(v)
            for i in range(n):
                tree_dist[s][i] = d[i]
        
        # Now greedily add edges.
        # Vertices that need more edges:
        need = [k - deg[v] for v in range(n)]
        
        # Build a list of candidate edges sorted by tree distance (descending).
        # For efficiency, don't sort all n^2/2 pairs. Instead, use random sampling
        # but bias toward far pairs.
        
        # Actually, let's just try random pairs but reject if tree_dist < g/2.
        # This is a simple rejection sampling that biases toward long edges.
        
        edge_set = set()
        for u in range(n):
            for v in adj[u]:
                edge_set.add((min(u, v), max(u, v)))
        
        min_dist = g  # We want tree_dist + 1 >= g, so tree_dist >= g-1
        
        added = 0
        total_needed = sum(need) // 2
        max_attempts = total_needed * 200  # Allow many retries
        
        for attempt in range(max_attempts):
            if time.time() > deadline - 10:
                return None
            if added >= total_needed:
                break
            
            u = rng.randint(0, n - 1)
            v = rng.randint(0, n - 1)
            if u == v:
                continue
            if need[u] <= 0 or need[v] <= 0:
                continue
            key = (min(u, v), max(u, v))
            if key in edge_set:
                continue
            # Check tree distance
            if tree_dist[u][v] < min_dist - 1:
                continue
            
            # Add edge
            adj[u].append(v)
            adj[v].append(u)
            deg[u] += 1
            deg[v] += 1
            need[u] -= 1
            need[v] -= 1
            edge_set.add(key)
            added += 1
        
        if added < total_needed:
            # Fallback: add remaining edges randomly (may create short cycles)
            for attempt in range(max_attempts):
                if added >= total_needed:
                    break
                u = rng.randint(0, n - 1)
                v = rng.randint(0, n - 1)
                if u == v or need[u] <= 0 or need[v] <= 0:
                    continue
                key = (min(u, v), max(u, v))
                if key in edge_set:
                    continue
                adj[u].append(v)
                adj[v].append(u)
                deg[u] += 1
                deg[v] += 1
                need[u] -= 1
                need[v] -= 1
                edge_set.add(key)
                added += 1
        
        if added < total_needed:
            return None
        
        # Verify 3-regular
        for i in range(n):
            if len(adj[i]) != k:
                return None
        
        return adj

    def check_girth_exact(adj, n, g):
        """Exact girth check using BFS from each vertex."""
        half_g = g // 2
        for start in range(n):
            dist = [-1] * n
            dist[start] = 0
            queue = deque([start])
            while queue:
                u = queue.popleft()
                if dist[u] >= half_g:
                    continue
                for v in adj[u]:
                    if dist[v] == -1:
                        dist[v] = dist[u] + 1
                        queue.append(v)
                    elif v != start and dist[u] + dist[v] + 1 < g:
                        return False
        return True

    def two_switch(adj, n, rng, g):
        """
        Perform a 2-switch that improves girth.
        Pick two edges, swap endpoints to increase the shortest cycle length.
        """
        edges = []
        edge_set = set()
        for u in range(n):
            for v in adj[u]:
                if u < v:
                    edges.append((u, v))
                    edge_set.add((u, v))
        
        if len(edges) < 2:
            return False
        
        for _ in range(500):
            i = rng.randint(0, len(edges) - 1)
            j = rng.randint(0, len(edges) - 1)
            if i == j:
                continue
            a, b = edges[i]
            c, d = edges[j]
            
            if len({a, b, c, d}) < 4:
                continue
            
            # Try both swap options
            for new_e1, new_e2 in [((a, c), (b, d)), ((a, d), (b, c))]:
                if (min(new_e1[0], new_e1[1]), max(new_e1[0], new_e1[1])) in edge_set:
                    continue
                if (min(new_e2[0], new_e2[1]), max(new_e2[0], new_e2[1])) in edge_set:
                    continue
                
                # Apply switch
                adj[a].remove(b)
                adj[b].remove(a)
                adj[c].remove(d)
                adj[d].remove(c)
                
                adj[a].append(new_e1[1])
                adj[new_e1[1]].append(a)
                adj[b].append(new_e2[1])
                adj[new_e2[1]].append(b)
                
                # Update edge_set
                edge_set.discard((min(a, b), max(a, b)))
                edge_set.discard((min(c, d), max(c, d)))
                edge_set.add((min(new_e1[0], new_e1[1]), max(new_e1[0], new_e1[1])))
                edge_set.add((min(new_e2[0], new_e2[1]), max(new_e2[0], new_e2[1])))
                
                # Update edges list
                edges[i] = new_e1
                edges[j] = new_e2
                
                return True
        
        return False

    # Main strategy
    best_n = float('inf')
    best_edges = None
    
    # Try sizes from Moore bound upward
    # For k=3, g=16: Moore bound = 254
    # Known best: 936
    # Try: 256, 300, 400, 500, 600, 700, 800, 900
    
    # The "far edge" construction should work better for smaller n
    # because it explicitly avoids short cycles during construction.
    
    candidate_ns = []
    for n in range(256, 1000, 16):
        if n % 2 == 0:
            candidate_ns.append(n)
    
    for n in candidate_ns:
        if time.time() > deadline - 15:
            break
        
        found = False
        num_attempts = max(5, int((deadline - time.time() - 10) / max(0.01, (1000 - 256) / len(candidate_ns))))
        num_attempts = min(num_attempts, 50)
        
        for attempt in range(num_attempts):
            if time.time() > deadline - 10:
                break
            
            seed = n * 1000 + attempt * 7 + 42
            adj = build_high_girth_random(n, g, seed)
            if adj is None:
                continue
            
            if check_girth_exact(adj, n, g):
                edges = []
                for u in range(n):
                    for v in adj[u]:
                        if u < v:
                            edges.append((u, v))
                
                best_n = n
                best_edges = edges
                found = True
                break
        
        if found:
            # Found a valid graph at this n. 
            # Since we're going in increasing order, this is the smallest.
            # But let's try a few more seeds at this n to see if we can 
            # confirm, then try to go smaller? No, we started at 256.
            # Let's try a couple more at this n for robustness, then stop.
            break
    
    # If we found one, try to improve by doing 2-switches to potentially
    # allow a smaller n? No, n is fixed. 
    # But we could try 2-switches to verify/improve the girth margin.
    
    if best_edges is not None:
        # Try to see if we can find a valid graph at a slightly smaller n
        # by trying the same construction with more attempts.
        # Actually, we already tried from 256 upward. If we found one at n,
        # we know nothing smaller worked with our attempt count.
        # Let's just return what we have.
        
        # Optional: try 2-switches to potentially find a graph with 
        # girth > g (not needed for score, but good for robustness).
        pass
    
    # Fallback if nothing found: try larger n with random approach
    if best_edges is None:
        for n in range(1000, 2000, 50):
            if time.time() > deadline - 5:
                break
            for attempt in range(10):
                if time.time() > deadline - 2:
                    break
                seed = n * 1000 + attempt * 13 + 99
                adj = build_high_girth_random(n, g, seed)
                if adj is None:
                    continue
                if check_girth_exact(adj, n, g):
                    edges = []
                    for u in range(n):
                        for v in adj[u]:
                            if u < v:
                                edges.append((u, v))
                    best_n = n
                    best_edges = edges
                    break
            if best_edges:
                break
    
    # Ultimate fallback: LUW construction (known valid, but large)
    if best_edges is None:
        def build_luw_fallback(d):
            q = 3
            add = lambda x, y: (x + y) % q
            sub = lambda x, y: (x - y) % q
            mul = lambda x, y: (x * y) % q
            
            rule = {}
            if d >= 2:
                rule[2] = (1, 1)
            if d >= 3:
                rule[3] = (2, 1)
            if d >= 4:
                rule[4] = (1, 2)
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
        
        best_n, best_edges = build_luw_fallback(11)
    
    return best_n, best_edges
