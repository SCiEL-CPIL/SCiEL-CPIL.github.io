import random
import time
import math
import sys

def construct(k, g):
    # Fallback to LUW construction if search fails
    def luw_fallback(k, g):
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

    if k != 5 or g != 9:
        # Only optimize for the specific task instance, fallback otherwise
        return luw_fallback(k, g)

    start_time = time.time()
    deadline = start_time + 110  # 110s to be safe
    
    # Moore bound for 5-regular, girth 9 is 428.
    # n must be even.
    min_n = 428
    max_n = 1124 # Just below LUW's 1125
    
    best_n, best_edges = None, None
    
    # We will try increasing n
    # Since finding a solution for small n is hard, we might start from a reasonable guess
    # or just iterate.
    
    # Helper: Generate random 5-regular graph
    def generate_random_5_reg(n):
        # Configuration model
        # 5n stubs
        # Randomly pair them
        # Ensure no loops/multi-edges by retrying or swapping
        # Simpler: Start with a path/cycle and add edges?
        # Or use a known generator.
        
        # Simple approach:
        # 1. Create a random permutation of [0..5n-1]
        # 2. Pair (2i, 2i+1)
        # 3. Check for loops/multi-edges
        # 4. If invalid, swap endpoints of some pairs
        # This is slow.
        
        # Better:
        # Start with a 5-regular graph structure, e.g., disjoint union of cycles? No, must be connected?
        # Task doesn't require connectedness explicitly, but "graph" usually implies it or at least simple.
        # "The graph must be simple... every vertex must have degree exactly k".
        # It doesn't explicitly say connected. But cage problems usually imply connected.
        # If disconnected, we can take the smallest component? No, n is the number of vertices.
        # If we return a disconnected graph, the girth is the min girth of components.
        # We can just ensure the whole graph has girth >= 9.
        
        # Let's use a simple random regular graph generator using NetworkX logic (manual)
        # Or just use the "swap" method on a valid 5-regular graph.
        # We need a starting valid 5-regular graph.
        # For n=428, it's hard to generate one quickly without a library.
        
        # Let's try to construct a 5-regular graph using a specific pattern.
        # Or, use the LUW graph and try to "shrink" it? No.
        
        # Let's implement a random 5-regular graph generator.
        # 1. Create list of 5*n stubs: [0]*5 + [1]*5 ...
        # 2. Shuffle
        # 3. Pair consecutive stubs
        # 4. Check validity
        # 5. If invalid, perform swaps to fix loops/multi-edges.
        
        stubs = []
        for i in range(n):
            stubs.extend([i] * 5)
        random.shuffle(stubs)
        edges = []
        for i in range(0, 5*n, 2):
            u, v = stubs[i], stubs[i+1]
            if u == v:
                # Loop, need to fix
                pass
            else:
                edges.append((u, v))
        
        # This raw generation often has loops/multi-edges.
        # We will use a local search to fix the graph and increase girth.
        # Represent graph as adjacency list or matrix.
        # For speed, use a set of edges.
        edge_set = set()
        for u, v in edges:
            if u != v:
                edge_set.add((min(u,v), max(u,v)))
        
        # Check degree
        deg = [0]*n
        for u, v in edge_set:
            deg[u] += 1
            deg[v] += 1
            
        # If degrees are not 5, this is bad.
        # The configuration model produces exactly 5-regular multigraphs.
        # We need to remove loops and multi-edges.
        # This is complex to do robustly in short time.
        
        # Alternative: Use a known 5-regular graph and perturb.
        # But we don't have a library.
        
        # Let's try a different approach:
        # Use the LUW graph (1125 vertices) and try to find a smaller subgraph?
        # No, that won't work easily.
        
        # Let's stick to the Configuration Model but with a repair phase.
        # Actually, for n ~ 500-1000, generating a simple 5-regular graph is feasible.
        
        def gen_simple_5_reg(n):
            while True:
                stubs = []
                for i in range(n):
                    stubs.extend([i] * 5)
                random.shuffle(stubs)
                edges = []
                valid = True
                seen = set()
                for i in range(0, 5*n, 2):
                    u, v = stubs[i], stubs[i+1]
                    if u == v:
                        valid = False
                        break
                    if u > v: u, v = v, u
                    if (u, v) in seen:
                        valid = False
                        break
                    seen.add((u, v))
                    edges.append((u, v))
                if valid:
                    return edges
        
        # This might take many iterations for large n if not careful, but 5-regular is sparse.
        # Probability of loop is ~ 1/n. Probability of multi-edge is low.
        # For n=500, prob of any loop in 1250 edges is 1250 * (1/500) = 2.5.
        # So we expect to try a few times.
        
        # Girth check
        def get_girth(n, edges):
            adj = [[] for _ in range(n)]
            for u, v in edges:
                adj[u].append(v)
                adj[v].append(u)
                
            min_girth = n + 1
            for start in range(n):
                # BFS from start
                dist = [-1] * n
                parent = [-1] * n
                dist[start] = 0
                queue = [start]
                head = 0
                while head < len(queue):
                    u = queue[head]
                    head += 1
                    for v in adj[u]:
                        if dist[v] == -1:
                            dist[v] = dist[u] + 1
                            parent[v] = u
                            queue.append(v)
                        else:
                            # Found a cycle
                            # Cycle length = dist[u] + dist[v] + 1
                            # This is not necessarily the shortest cycle through u,v
                            # But it's a cycle.
                            # To get exact girth, we need to be careful.
                            # Standard BFS finds shortest paths.
                            # If we encounter a visited node v from u, and v is not parent of u,
                            # we have a cycle of length dist[u] + dist[v] + 1.
                            # This is the length of the cycle formed by the two shortest paths from start.
                            # It might not be the global girth, but it's a lower bound on the girth?
                            # No, it's a specific cycle length.
                            # If we do this for all start nodes, the minimum such length is the girth.
                            if v != parent[u] and u != parent[v]:
                                cycle_len = dist[u] + dist[v] + 1
                                if cycle_len < min_girth:
                                    min_girth = cycle_len
                if min_girth == 3: # Early exit
                    return 3
            return min_girth if min_girth != n + 1 else -1 # -1 if no cycle (tree, impossible for 5-reg)
            
        # Improvement: BFS from each node is O(n*m).
        # For n=1000, m=2500, 2.5M ops.
        # 1000 BFS runs = 2.5B ops. Too slow in Python.
        
        # We need a faster girth check or fewer checks.
        # We only need to check if girth < 9.
        # We can stop BFS at depth 4.
        # If we find a cycle of length < 9, we return it.
        # If we explore all nodes to depth 4 without finding a cycle < 9, 
        # does that mean girth >= 9?
        # No. A cycle of length 9 might exist.
        # But if girth < 9, there is a cycle of length 3, 4, 5, 6, 7, or 8.
        # Such a cycle has a node with eccentricity <= 4 within the cycle.
        # So if we do BFS from every node to depth 4, and don't find any back-edge to a node at dist < 4,
        # then there is no cycle of length < 9?
        # Yes. If a cycle of length L < 9 exists, pick a node v on it.
        # The distance from v to any other node on the cycle is <= L/2 < 4.5.
        # So all nodes on the cycle are within distance 4 of v.
        # BFS from v to depth 4 will see all nodes on the cycle.
        # It will detect the cycle.
        
        def has_short_cycle(n, edges, max_girth=8):
            adj = [[] for _ in range(n)]
            for u, v in edges:
                adj[u].append(v)
                adj[v].append(u)
                
            for start in range(n):
                # BFS to depth 4
                dist = [-1] * n
                parent = [-1] * n
                dist[start] = 0
                queue = [start]
                head = 0
                while head < len(queue):
                    u = queue[head]
                    head += 1
                    if dist[u] >= 4:
                        continue
                    for v in adj[u]:
                        if dist[v] == -1:
                            dist[v] = dist[u] + 1
                            parent[v] = u
                            queue.append(v)
                        else:
                            if v != parent[u] and u != parent[v]:
                                cycle_len = dist[u] + dist[v] + 1
                                if cycle_len <= max_girth:
                                    return True, cycle_len
            return False, -1

        # Local Search
        def local_search(n, edges, time_limit):
            end_time = time.time() + time_limit
            edge_set = set((min(u,v), max(u,v)) for u,v in edges)
            adj = [[] for _ in range(n)]
            for u, v in edge_set:
                adj[u].append(v)
                adj[v].append(u)
                
            has_short, _ = has_short_cycle(n, edge_set, 8)
            if not has_short:
                return edge_set
                
            current_edges = list(edge_set)
            # Randomize order
            random.shuffle(current_edges)
            
            while time.time() < end_time:
                # Pick a random edge to remove?
                # Or pick a short cycle and break it.
                # Finding the cycle is expensive.
                # Let's just do random edge swaps.
                # Pick two random edges (u,v) and (x,y)
                # Check if swap (u,x) and (v,y) is valid (no loops, no multi-edges)
                # If valid, apply swap.
                # Check if girth improved (or just check if short cycle exists).
                
                # To be faster, we can bias towards edges in short cycles.
                # But finding them is the bottleneck.
                # Let's stick to random swaps.
                
                e1 = random.choice(current_edges)
                e2 = random.choice(current_edges)
                if e1 == e2: continue
                
                u, v = e1
                x, y = e2
                
                # Try swap 1: (u,x) and (v,y)
                # Check validity
                if u != x and u != y and v != x and v != y:
                    if (min(u,x), max(u,x)) not in edge_set and (min(v,y), max(v,y)) not in edge_set:
                        # Apply
                        edge_set.remove(e1)
                        edge_set.remove(e2)
                        new_e1 = (min(u,x), max(u,x))
                        new_e2 = (min(v,y), max(v,y))
                        edge_set.add(new_e1)
                        edge_set.add(new_e2)
                        
                        # Update adj
                        adj[u].remove(x) if x in adj[u] else None # This is wrong, we didn't add it yet
                        # Rebuilding adj is expensive.
                        # Let's just check girth on the new edge_set.
                        
                        has_short, _ = has_short_cycle(n, edge_set, 8)
                        if not has_short:
                            return edge_set
                            
                        # Accept if girth didn't decrease?
                        # We don't track girth easily.
                        # Let's just accept always? No, random walk might get stuck.
                        # Let's accept if it breaks the specific cycle?
                        # We don't know the cycle.
                        # Let's use a simple acceptance: always accept, but restart if stuck?
                        # Or, keep a history.
                        
                        # For simplicity, let's just accept and hope.
                        # To improve, we can check if the new graph has a short cycle.
                        # If yes, and the old one did, we can't tell if it's better.
                        # Let's just do a random walk.
                        
                        current_edges = list(edge_set)
                        # Shuffle occasionally
                        if random.random() < 0.1:
                            random.shuffle(current_edges)
                            
            return edge_set
            
    # Main loop
    # Try n from 428 to 1124
    # We will allocate time proportional to n?
    # Or just iterate quickly.
    
    # Given the difficulty, let's try a few specific n values.
    # 500, 600, 800, 1000.
    # If we find one, return it.
    
    for n in range(428, 1125, 2):
        if time.time() > deadline - 10:
            break
            
        # Time budget per n
        # If n is small, search harder.
        t_budget = 5.0 if n < 600 else 2.0
        
        edges = gen_simple_5_reg(n)
        if edges is None:
            continue
            
        result = local_search(n, edges, t_budget)
        
        if result:
            has_short, _ = has_short_cycle(n, result, 8)
            if not has_short:
                return n, list(result)
                
    # Fallback
    return luw_fallback(5, 9)

