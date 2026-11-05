import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin
    
    # Helper to check if adding edge (u, v) creates a C5
    # Graph is represented as a list of sets for neighbors
    def creates_c5(graph, u, v):
        # We check if there is a path of length 4 between u and v
        # Path: u - a - b - c - v
        # a in N(u), c in N(v)
        # b is a common neighbor of a and c? No, a-b and b-c are edges.
        # So b is in N(a) and b is in N(c).
        # We need to check if there exists a in N(u), c in N(v) such that N(a) and N(c) have a common neighbor b.
        # And b != u, b != v, b != a, b != c.
        
        nu = graph[u]
        nv = graph[v]
        
        # Optimization: If u and v are already connected, we don't add it.
        # We assume we are checking potential addition.
        
        # For each a in nu:
        #   For each c in nv:
        #     if a == c: continue
        #     if a == v or c == u: continue
        #     Check if N(a) and N(c) have a common neighbor b
        #     b != u, b != v, b != a, b != c
        #     Also a-b-c must be a simple path, so a != c (checked), b != a, b != c.
        #     Also u-a-b-c-v must be simple, so u != c (checked by c in nv, u not in nv if simple graph? u and v not adjacent), 
        #     u != b (b in N(a), if b==u then u-a-u? no, simple graph no self loops. u in N(a) means a-u edge. 
        #     If b==u, then a-u edge and u-c edge? No, b is the middle.
        #     Let's just check intersections.
        
        for a in nu:
            if a == v: continue
            for c in nv:
                if c == u: continue
                if a == c: continue
                # Check if N(a) and N(c) have a common neighbor b
                # b must not be u, v, a, c
                common = graph[a] & graph[c]
                # Remove invalid candidates
                common -= {u, v, a, c}
                if common:
                    return True
        return False

    def has_independent_set_size_9(graph, n):
        # Greedy approximation first. If it finds 9, return True.
        # If not, do a more thorough check if time permits?
        # For n=33, a branch and bound is possible.
        # But let's try a randomized greedy multiple times.
        # If we can't find 9, it's likely alpha < 9.
        # Given we start from alpha=8 and add edges, alpha should stay low.
        
        # Simple exact check for small n?
        # We can use a bitset based maximal independent set search.
        
        # For now, use a heuristic: 
        # Try to build an independent set greedily.
        # If max size found < 9, return False.
        
        # To be safe, if n is small, we can do a more exact check.
        # But 33 is small enough for a quick B&B?
        # Let's stick to greedy + random restarts for speed.
        
        vertices = list(range(n))
        max_size = 0
        
        # Try a few random orderings
        for _ in range(20):
            random.shuffle(vertices)
            current_set = []
            used = set()
            for v in vertices:
                if v in used: continue
                # Check if v is independent from current_set
                independent = True
                for u in current_set:
                    if u in graph[v]:
                        independent = False
                        break
                if independent:
                    current_set.append(v)
                    used.add(v)
                    # Mark neighbors as used? No, they can't be in set, but others can.
                    # Actually, if v is in set, its neighbors cannot.
                    # We can just check against current_set.
            if len(current_set) > max_size:
                max_size = len(current_set)
                if max_size >= 9:
                    return True
        
        # If greedy didn't find 9, assume alpha < 9?
        # Risky. But for this specific task, the graph is dense.
        # Let's add a stronger check: 
        # If n <= 33, we can do a simple backtracking to find if alpha >= 9.
        
        def find_independent_set(vertices, current_set, graph, n, target=9):
            if len(current_set) >= target:
                return True
            if not vertices:
                return False
            
            # Pick a vertex v
            v = vertices[0]
            rest = vertices[1:]
            
            # Branch 1: Include v
            # Remove v and its neighbors from rest
            new_rest = [u for u in rest if u not in graph[v]]
            if find_independent_set(new_rest, current_set + [v], graph, n, target):
                return True
            
            # Branch 2: Exclude v
            if find_independent_set(rest, current_set, graph, n, target):
                return True
            
            return False

        # This B&B can be exponential. 
        # Let's use it only if the greedy failed and we are close to deadline?
        # Or use it always but with a node limit?
        # For now, let's trust the greedy for the search loop, 
        # and do a final verification with a more robust method if we find a candidate.
        
        return False

    def is_valid(graph, n):
        # Check C5
        # We can check if the graph contains a C5.
        # Since we maintain it, we can just check the last added edge?
        # But for final check, we do a full check.
        
        # Full C5 check:
        # For each edge (u, v), temporarily remove it and check if u, v are connected by path of length 4.
        # Or just use the creates_c5 logic for all edges?
        # Actually, we can just check if any 5-cycle exists.
        
        # Let's use a simple check:
        # For each u, for each v in N(u), for each w in N(v) \ {u}, for each x in N(w) \ {u, v}, for each y in N(x) \ {u, v, w}:
        #   if u in N(y): return True
        
        # This is O(n^5) worst case.
        # For n=33, this is too slow for a full check in every iteration.
        # So we rely on incremental checks during search.
        
        # For the final result, we MUST be sure.
        # We will do a full check on the final graph.
        
        return True # Placeholder, actual check in main loop

    # Initialize graph
    n = 33
    graph = [set() for _ in range(n)]
    
    # Start with 8 K4s
    for b in range(8):
        base = b * 4
        for i in range(4):
            for j in range(i + 1, 4):
                u = base + i
                v = base + j
                graph[u].add(v)
                graph[v].add(u)
    
    # Vertex 32 is isolated initially
    # We need to connect it to reduce alpha from 9 to < 9.
    
    best_graph = None
    
    while time.time() < deadline:
        # Randomize connections of vertex 32
        # Connect to random vertices in each block?
        # Or just random vertices overall.
        
        # Reset vertex 32 connections
        graph[32] = set()
        for u in range(32):
            graph[u].discard(32)
        
        # Connect 32 to random vertices
        # To reduce alpha, we should connect to many vertices.
        # But avoid creating C5.
        
        # Try to connect 32 to a random subset of vertices
        # If alpha >= 9, add more connections.
        # If C5, remove connections.
        
        # Strategy: Start with no connections. Alpha = 9.
        # Add connections until alpha < 9, ensuring no C5.
        
        # Shuffle vertices
        vertices = list(range(32))
        random.shuffle(vertices)
        
        for v in vertices:
            # Try to add edge (32, v)
            if not creates_c5(graph, 32, v):
                graph[32].add(v)
                graph[v].add(32)
            
            # Check alpha
            if not has_independent_set_size_9(graph, n):
                # Valid candidate found!
                # We should verify it more thoroughly before accepting
                # But for speed, accept and break?
                # Let's accept and break to ensure we return a valid graph.
                best_graph = [set(s) for s in graph]
                break
        
        if best_graph is not None:
            break
    
    if best_graph is None:
        # Fallback to 32 vertices
        n = 32
        graph = [set() for _ in range(n)]
        for b in range(8):
            base = b * 4
            for i in range(4):
                for j in range(i + 1, 4):
                    u = base + i
                    v = base + j
                    graph[u].add(v)
                    graph[v].add(u)
        best_graph = graph

    # Final verification and conversion to edge list
    # We must ensure the final graph is valid.
    # Perform a strict check on best_graph
    
    n = len(best_graph)
    
    # Strict C5 Check
    def strict_has_c5(g, n):
        # Check for any 5-cycle
        # Use the O(n^5) loop but optimized with sets
        for u in range(n):
            nu = g[u]
            for v in nu:
                if v <= u: continue
                nv = g[v]
                for w in nv:
                    if w == u: continue
                    nw = g[w]
                    for x in nw:
                        if x == u or x == v: continue
                        nx = g[x]
                        for y in nx:
                            if y == u or y == v or y == w: continue
                            if u in g[y]:
                                # Found u-v-w-x-y-u
                                return True
        return False

    # Strict Alpha Check (B&B with limit)
    def strict_alpha_at_least_9(g, n):
        # Use a more robust B&B
        # State: (available_vertices, current_size)
        # We want to know if we can reach size 9
        
        # Sort vertices by degree (descending) to prune?
        # Or just simple B&B
        
        import sys
        sys.setrecursionlimit(10000)
        
        def dfs(available, size):
            if size >= 9:
                return True
            if not available:
                return False
            
            # Heuristic: pick vertex with min degree in available?
            # Or just first one.
            v = available[0]
            rest = available[1:]
            
            # Include v
            new_available = [u for u in rest if u not in g[v]]
            if dfs(new_available, size + 1):
                return True
            
            # Exclude v
            if dfs(rest, size):
                return True
            
            return False
        
        return dfs(list(range(n)), 0)

    if strict_has_c5(best_graph, n) or strict_alpha_at_least_9(best_graph, n):
        # Invalid, fallback to 32
        n = 32
        best_graph = [set() for _ in range(n)]
        for b in range(8):
            base = b * 4
            for i in range(4):
                for j in range(i + 1, 4):
                    u = base + i
                    v = base + j
                    best_graph[u].add(v)
                    best_graph[v].add(u)

    edges = []
    for u in range(n):
        for v in best_graph[u]:
            if u < v:
                edges.append([u, v])
                
    return {'n': n, 'edges': edges}
