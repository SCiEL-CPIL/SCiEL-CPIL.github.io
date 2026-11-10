import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 100.0 # Safety margin

    def random_graph(n, p, seed):
        rng = random.Random(seed)
        adj = [set() for _ in range(n)]
        for u in range(n):
            for v in range(u + 1, n):
                if rng.random() < p:
                    adj[u].add(v)
                    adj[v].add(u)
        return adj

    def min_vertex_cover(adj):
        n = len(adj)
        # Copy adj to avoid modifying original if we were to reuse, but here it's local
        # We represent the graph as a list of sets.
        # To make it mutable and efficient, we'll work with a copy.
        
        # Helper to check if time is up
        def time_up():
            return time.time() > deadline

        # We use a recursive function that returns (min_size, set_of_vertices)
        # To avoid passing huge sets around, we can reconstruct the path or pass the current selection.
        # Given n<=40, passing sets is fine.
        
        best = [n] # Upper bound
        best_set = [set(range(n))] # Initial worst case: all vertices
        
        def solve(adj_copy, current_set):
            if time_up():
                return
            
            # Check if graph is empty
            has_edge = False
            for u in range(len(adj_copy)):
                if adj_copy[u]:
                    has_edge = True
                    break
            if not has_edge:
                size = len(current_set)
                if size < best[0]:
                    best[0] = size
                    best_set[0] = set(current_set)
                return

            # Find vertex with max degree
            max_deg = -1
            v = -1
            for u in range(len(adj_copy)):
                deg = len(adj_copy[u])
                if deg > max_deg:
                    max_deg = deg
                    v = u
            
            if max_deg == 0:
                # No edges left
                size = len(current_set)
                if size < best[0]:
                    best[0] = size
                    best_set[0] = set(current_set)
                return

            # Branch 1: Include v
            new_set1 = set(current_set)
            new_set1.add(v)
            
            # Create new adj without v and its neighbors? 
            # No, if we include v, we only remove v and edges incident to v.
            # The neighbors remain in the graph, but edges to v are gone.
            
            adj1 = [set(s) for s in adj_copy]
            # Remove v and all edges connected to v
            for u in list(adj1[v]):
                adj1[u].remove(v)
            adj1[v] = set()
            
            solve(adj1, new_set1)

            # Branch 2: Exclude v
            # If v is excluded, all its neighbors MUST be included.
            new_set2 = set(current_set)
            neighbors = list(adj_copy[v])
            new_set2.update(neighbors)
            
            # Remove v and all neighbors and all edges incident to them
            adj2 = [set(s) for s in adj_copy]
            to_remove = {v} | set(neighbors)
            for u in to_remove:
                for w in list(adj2[u]):
                    adj2[w].remove(u)
                adj2[u] = set()
            
            # Pruning: if current size + len(neighbors) >= best[0], we can skip?
            # Actually, we already added them to new_set2.
            if len(new_set2) < best[0]:
                solve(adj2, new_set2)

        solve(adj, set())
        return best_set[0]

    out = []
    for n, p, seed in ((30, 0.3, 201), (35, 0.25, 202), (40, 0.2, 203)):
        if time.time() > deadline:
            # Fallback to a simple heuristic if we run out of time
            # Just return all vertices with degree > 0 to be safe? Or empty?
            # The task requires meeting every maximal clique. 
            # If we return all vertices, it's always valid.
            # But we want to minimize size.
            # If time is up, we should return the best we have.
            # Since we process sequentially, if one takes too long, we might fail the others.
            # Let's try to be fast.
            pass
            
        adj = random_graph(n, p, seed)
        cover = min_vertex_cover(adj)
        out.append(sorted(cover))
        
    return out
