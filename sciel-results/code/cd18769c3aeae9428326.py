"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time

def construct():
    from random import Random
    import heapq

    def instance(splits, seed, palette=5):
        rng = Random(seed)
        faces = [[0, 1, 2, 3], [4, 5, 6, 7], [0, 1, 5, 4], [1, 2, 6, 5], [2, 3, 7, 6], [3, 0, 4, 7]]
        edges = {(min(a, b), max(a, b)) for f in faces for a, b in zip(f, f[1:] + f[:1])}
        n = 8
        for _ in range(splits):
            face = faces.pop(rng.randrange(len(faces)))
            inner = list(range(n, n + 4))
            n += 4
            for i in range(4):
                a, b = face[i], face[(i + 1) % 4]
                edges.add((min(a, inner[i]), max(a, inner[i])))
                edges.add((min(inner[i], inner[(i + 1) % 4]), max(inner[i], inner[(i + 1) % 4])))
                faces.append([a, b, inner[(i + 1) % 4], inner[i]])
            faces.append(inner)
        lists = [sorted(rng.sample(range(palette), 3)) for _ in range(n)]
        return n, sorted(edges), lists

    def solve_list_coloring(n, edges, lists, time_limit=10.0):
        # Build adjacency list
        adj = [[] for _ in range(n)]
        for u, v in edges:
            adj[u].append(v)
            adj[v].append(u)
        
        # Order vertices by degree (descending) for better greedy/backtracking performance
        order = sorted(range(n), key=lambda x: -len(adj[x]))
        # Map original vertex index to position in order for fast lookup
        pos_in_order = [0] * n
        for i, v in enumerate(order):
            pos_in_order[v] = i
        
        # We will assign colours in the order defined by 'order'
        # colour[i] is the colour assigned to vertex order[i]
        colour = [None] * n
        # For each vertex, we track which colours are forbidden by already-coloured neighbours
        # forbidden[i] is a set of colours that are taken by neighbours of order[i] that have been coloured
        
        # We use a priority queue to pick the next vertex to colour? 
        # Actually, the static order by degree is usually good for list coloring.
        # Let's stick to the static order for simplicity and speed, but use backtracking if stuck.
        
        # To speed up backtracking, we can use a simple DFS.
        # However, for n ~ 400, a naive DFS might be slow if the graph is dense or constraints are tight.
        # But these are sparse planar graphs.
        
        # Let's implement a recursive backtracking solver with memoization or just simple recursion.
        # Given the small size (n <= 8 + 4*100 = 408), and low degree (avg ~4), recursion depth is manageable.
        
        import sys
        sys.setrecursionlimit(10000)
        
        start_time = time.time()
        
        # Pre-compute neighbours in order of their position in 'order' to quickly check if coloured
        # Actually, we just need to check if colour[neighbor] is not None.
        
        def backtrack(idx):
            if idx == n:
                return True
            if time.time() - start_time > time_limit:
                return False
            
            v = order[idx]
            # Collect colours used by coloured neighbours
            used = set()
            for u in adj[v]:
                if colour[u] is not None:
                    used.add(colour[u])
            
            for c in lists[v]:
                if c not in used:
                    colour[v] = c
                    if backtrack(idx + 1):
                        return True
                    colour[v] = None
            
            return False
        
        if backtrack(0):
            # Convert colour array (indexed by original vertex) back to list
            # Wait, my colour array is indexed by original vertex index in the code above?
            # No, in the code above:
            # colour = [None] * n
            # v = order[idx]
            # colour[v] = c
            # So colour is indexed by original vertex.
            # This is correct.
            return [int(c) for c in colour]
        else:
            return None

    def greedy(n, edges, lists):
        # Fallback greedy solver
        adj = [set() for _ in range(n)]
        for u, v in edges:
            adj[u].add(v)
            adj[v].add(u)
        colour = [None] * n
        for v in sorted(range(n), key=lambda u: -len(adj[u])):
            taken = {colour[u] for u in adj[v]}
            free = [c for c in lists[v] if c not in taken]
            colour[v] = free[0] if free else lists[v][0]
        return colour

    out = []
    total_start = time.time()
    deadline = 140.0  # Leave 10s margin for 150s limit
    
    for splits, seed in ((20, 701), (40, 702), (60, 703), (100, 704)):
        if time.time() > deadline:
            # If we run out of time, we must return something valid.
            # We'll fill remaining with greedy solutions quickly.
            for s, sd in ((20, 701), (40, 702), (60, 703), (100, 704))[len(out):]:
                n, edges, lists = instance(s, sd)
                out.append(greedy(n, edges, lists))
            break
        
        n, edges, lists = instance(splits, seed)
        
        # Try backtracking first
        solution = solve_list_coloring(n, edges, lists, time_limit=5.0)
        
        if solution is not None:
            out.append(solution)
        else:
            # Fallback to greedy
            out.append(greedy(n, edges, lists))
            
    # Ensure all are Python ints
    final_out = []
    for sol in out:
        final_out.append([int(c) for c in sol])
        
    return final_out
# EVOLVE-BLOCK-END
