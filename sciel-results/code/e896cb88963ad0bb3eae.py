import time
import random
from collections import deque
import sys

# EVOLVE-BLOCK-START

def construct(k, g):
    start_time = time.time()
    deadline = start_time + 110.0  # 10s safety margin for 120s limit
    
    # Moore bound calculation for reference
    # M = 1 + k * sum((k-1)^i for i in range(0, g//2 - 1))
    # For k=4, g=10: 1 + 4 * (1 + 3 + 9 + 27) = 1 + 4*40 = 161? 
    # Wait, Moore bound for even girth g=2r is: 1 + k * sum_{i=0}^{r-2} (k-1)^i
    # g=10 -> r=5. Sum i=0..3: 1 + 3 + 9 + 27 = 40.
    # M = 1 + 4*40 = 161.
    # However, the prompt says "best lower bound is 243".
    # Let's re-read the prompt carefully.
    # "The smallest known such graph has 320 vertices... best lower bound is 243."
    # The Moore bound is a theoretical lower bound. For (4,10), the Moore bound is actually 161.
    # Wait, the standard Moore bound formula for k-regular graph with girth g is:
    # If g is even, g=2r: n >= 1 + k * sum_{i=0}^{r-2} (k-1)^i
    # If g is odd, g=2r+1: n >= 1 + k * sum_{i=0}^{r-1} (k-1)^i
    # For k=4, g=10 (r=5): 1 + 4 * (1 + 3 + 9 + 27) = 1 + 160 = 161.
    # Why does the prompt say 243? 
    # Ah, the prompt says "best lower bound is 243". This might be a stronger lower bound derived from other constraints, or I am misinterpreting "Moore bound" in the prompt context.
    # Actually, for k=4, g=10, the Moore bound is indeed 161.
    # Let's check the LUW graph size.
    # LUW(4, 10) usually yields a graph with ~400 vertices.
    # The prompt says "n must lie between the Moore bound and max_vertices".
    # I will try to construct graphs starting from a reasonable lower bound.
    # Let's try to find a graph with n close to the theoretical minimum.
    # I will iterate n from a lower estimate upwards.
    
    # A safe lower bound to start searching. 
    # If I start too low, the greedy search will fail.
    # Let's start from 200 and go up.
    
    def is_valid_graph(n, edges, k, g):
        """Check if the graph is k-regular, simple, and has girth >= g."""
        if len(edges) != n * k // 2:
            return False
        
        # Check degrees
        deg = [0] * n
        adj = [[] for _ in range(n)]
        for u, v in edges:
            if u == v: return False
            if u > v: u, v = v, u
            # Check for duplicate edges roughly (assumed unique by construction)
            deg[u] += 1
            deg[v] += 1
            adj[u].append(v)
            adj[v].append(u)
        
        for d in deg:
            if d != k:
                return False
        
        # Check girth using BFS from each vertex
        # We need to ensure no cycle of length < g.
        # This is equivalent to: for every edge (u,v), the distance between u and v 
        # in the graph WITHOUT that edge is >= g-1.
        # Or simpler: BFS from every node, check if any node is reached at dist < g/2 
        # via a path that doesn't immediately backtrack?
        # Actually, the standard way to check girth >= g is:
        # For every vertex u, the number of vertices within distance < g/2 is bounded.
        # But a direct check is: 
        # For every edge (u, v), remove it, find dist(u, v). If dist < g-1, fail.
        # This is O(E * (N+E)) which is 200 * 2000 = 400k ops. Fast enough.
        
        # Optimization: We can just BFS from every node and check if we encounter a "back edge" 
        # that creates a short cycle.
        # Simpler: For each node u, BFS. If we find a node v at distance d < g/2 
        # that is already visited at distance d' < g/2 and the path doesn't share edges... 
        # Actually, the most robust simple check for small n:
        # For each edge (u, v), temporarily remove it. BFS from u to v. 
        # If dist < g-1, then there is a cycle of length dist+1 < g.
        
        for u, v in edges:
            # BFS from u to v avoiding the direct edge (u,v)
            visited = {u: 0}
            queue = deque([(u, 0)])
            found = False
            while queue:
                curr, dist = queue.popleft()
                if curr == v:
                    if dist < g - 1:
                        return False
                    else:
                        found = True
                    break
                if dist >= g - 1:
                    break
                for nb in adj[curr]:
                    if nb == u and curr == v: continue # This logic is tricky with undirected
                    # We just need to avoid using the edge (u,v) directly.
                    # Since it's undirected, we avoid going from u to v and v to u directly.
                    if (curr == u and nb == v) or (curr == v and nb == u):
                        continue
                    if nb not in visited:
                        visited[nb] = dist + 1
                        queue.append((nb, dist + 1))
            if not found:
                # Graph is disconnected? A regular graph with girth >= g should be connected if n is small enough?
                # Actually, if it's disconnected, the girth is infinite (or defined per component).
                # But a "cage" implies a single connected component usually.
                # The prompt says "The graph must be simple... every vertex degree k... no cycle shorter than g".
                # It doesn't explicitly demand connectivity, but "cage problem" usually implies connected.
                # However, if it's disconnected, we can just check each component.
                # But for the greedy construction, we will likely end up with one component.
                # If BFS doesn't find v, it means u and v are in different components.
                # In that case, there is no cycle containing (u,v) in this component? 
                # No, (u,v) is an edge, so u and v are connected. 
                # So BFS MUST find v. 
                # If it doesn't, there's a bug in my BFS or the graph structure.
                # Wait, if I remove (u,v), they might become disconnected.
                # If they are disconnected, then there is NO cycle containing (u,v). 
                # So the girth condition is satisfied for this edge.
                pass
        
        return True

    def greedy_construct(n, k, g, seed_val):
        random.seed(seed_val)
        deg = [0] * n
        adj = [set() for _ in range(n)]
        edges = []
        
        # Pre-compute distances? No, dynamic.
        # We need to check if adding (u,v) creates a short cycle.
        # This is equivalent to: dist(u,v) in current graph >= g-1.
        # We can maintain a distance matrix or compute on the fly.
        # For n~200-300, computing BFS on the fly for each candidate edge is expensive.
        # Number of candidate edges to check: O(n^2).
        # BFS cost: O(n+k*n).
        # Total: O(n^3). 300^3 = 27,000,000. Doable in Python? Maybe tight.
        
        # Optimization: 
        # Instead of checking every pair, we can be smarter.
        # Or, we can use a simpler heuristic: 
        # Just try to add edges randomly, and check girth at the end? 
        # No, that will fail often.
        
        # Let's try a different approach: 
        # Build the graph by adding edges one by one.
        # For each step, pick a random pair (u,v) where deg[u]<k, deg[v]<k, and u!=v.
        # Check if dist(u,v) >= g-1.
        # To check dist(u,v) quickly, we can run BFS from u.
        
        # To speed up, we can limit the BFS depth to g-2.
        
        def check_dist_ge(u, v, limit):
            if u == v: return 0
            visited = {u: 0}
            queue = deque([(u, 0)])
            while queue:
                curr, dist = queue.popleft()
                if dist >= limit:
                    break
                for nb in adj[curr]:
                    if nb not in visited:
                        visited[nb] = dist + 1
                        if nb == v:
                            return dist + 1
                        if dist + 1 < limit:
                            queue.append((nb, dist + 1))
            return -1 # Not found within limit
        
        # We want dist(u,v) >= g-1.
        # So if check_dist_ge(u, v, g-2) returns -1, then dist >= g-1.
        # If it returns d < g-1, then dist < g-1, so we CANNOT add the edge.
        
        target_edges = n * k // 2
        current_edges = 0
        
        # List of available vertices sorted by degree?
        # We'll just iterate.
        
        # To avoid getting stuck, we can use a "conflict" check.
        
        # Heuristic: Try to add edges.
        # If we get stuck (no valid edges to add), we might need to backtrack or restart.
        # Given the time limit, simple restarts with different seeds are best.
        
        while current_edges < target_edges:
            progress = False
            # Shuffle vertices to avoid bias
            verts = list(range(n))
            random.shuffle(verts)
            
            for u in verts:
                if deg[u] >= k:
                    continue
                # Find candidates for v
                # Candidates must have deg < k
                # And dist(u,v) >= g-1
                
                # To make it faster, we can pre-filter candidates by degree
                candidates = [v for v in range(n) if v != u and deg[v] < k and v not in adj[u]]
                
                # Shuffle candidates
                random.shuffle(candidates)
                
                for v in candidates:
                    # Check distance
                    # We need dist(u,v) >= g-1
                    # BFS from u, stop at depth g-2. If v is not reached, dist >= g-1.
                    
                    # Optimization: If we have already added many edges, the graph is dense.
                    # Distances are small.
                    
                    # Let's run BFS
                    visited = {u: 0}
                    queue = deque([(u, 0)])
                    found_short = False
                    while queue:
                        curr, dist = queue.popleft()
                        if dist >= g - 1:
                            break
                        for nb in adj[curr]:
                            if nb not in visited:
                                visited[nb] = dist + 1
                                if nb == v:
                                    found_short = True
                                    break
                                if dist + 1 < g - 1:
                                    queue.append((nb, dist + 1))
                        if found_short:
                            break
                    
                    if not found_short:
                        # Safe to add
                        adj[u].add(v)
                        adj[v].add(u)
                        deg[u] += 1
                        deg[v] += 1
                        edges.append((u, v))
                        current_edges += 1
                        progress = True
                        if current_edges == target_edges:
                            break
                if current_edges == target_edges:
                    break
            
            if not progress:
                # Stuck!
                return None

        # Verify final graph
        if is_valid_graph(n, edges, k, g):
            return n, edges
        else:
            return None

    # Try different sizes and seeds
    # Start from a reasonable lower bound.
    # Moore bound for (4,10) is 161.
    # Known best is 320.
    # Let's try to beat 320.
    # We'll iterate n from 200 to 320.
    
    best_result = None
    
    # We can try to construct a graph of size n.
    # If it fails, try n+1, etc.
    
    # To be safe and fast, we will try a few sizes.
    # If we find a valid graph, we return it.
    # We want the SMALLEST n.
    
    # Let's try n starting from 200.
    # If we can't find one in reasonable time, we stop.
    
    # Note: The greedy algorithm might fail to find a graph even if one exists.
    # So we should try multiple seeds for each n.
    
    # Time budget per n: ~5 seconds.
    # Total time: 110s.
    # So we can try ~20 configurations.
    
    # Let's try n = 200, 201, ...
    # Actually, it's better to try a specific range.
    
    # If we fail, we fall back to the LUW graph?
    # The LUW graph is valid and has n=432.
    # If we can't find anything better, we should return the LUW graph to ensure a valid score.
    
    def luw_fallback(k, g):
        # Re-implement the seed logic quickly for fallback
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

    # Try greedy search
    # We will try to find a graph with n < 432.
    # Let's try n in range(200, 432, 1) but limited by time.
    
    # To improve success rate, we can use a slightly randomized greedy approach.
    # The current greedy is deterministic given the seed.
    
    # Let's try a few seeds for a few n values.
    
    # Strategy:
    # 1. Try n=200 with 5 seeds.
    # 2. If fail, try n=210 with 5 seeds.
    # ...
    
    # Actually, the gap between 200 and 432 is large.
    # It's unlikely to find one at 200.
    # Let's start closer to the known best 320.
    # Try 300, 310, 320.
    
    # Let's try to find ANY valid graph smaller than 432.
    
    # I will iterate n from 250 to 431.
    # For each n, try 2 seeds.
    # If found, return.
    
    # To save time, I will skip n if time is running out.
    
    found = False
    for n in range(250, 432):
        if time.time() > deadline - 5:
            break
        
        # Try 2 seeds for this n
        for seed in [12345 + n, 54321 + n]:
            if time.time() > deadline - 2:
                break
            
            res = greedy_construct(n, k, g, seed)
            if res is not None:
                found = True
                # We found a valid graph with this n.
                # Since we iterate n upwards, this is the smallest n we found so far.
                # But we should check if we can find a smaller one?
                # The loop goes upwards. So the first one found is the smallest in this sequence.
                # However, we skipped 200-249.
                # If we want to be thorough, we should check lower bounds.
                # But 250 is a reasonable start.
                # Let's return this.
                return res
    
    # If not found, return the fallback LUW graph
    return luw_fallback(k, g)

# EVOLVE-BLOCK-END
