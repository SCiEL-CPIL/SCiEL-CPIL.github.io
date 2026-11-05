import time
import random
import numpy as np

def construct():
    start_time = time.time()
    deadline = 280.0  # 300s limit, 20s buffer

    def time_left():
        return deadline - time.time()

    # 1. Construct the seed Paley graph on 37 vertices
    q = 37
    residues = {(x * x) % q for x in range(1, q)}
    n_init = q
    # Adjacency list for fast access
    # graph[i] is a set of neighbors of i
    graph = [set() for _ in range(n_init)]
    edges_init = []
    for i in range(n_init):
        for j in range(i + 1, n_init):
            if (j - i) % q in residues:
                graph[i].add(j)
                graph[j].add(i)
                edges_init.append((i, j))
    
    current_n = n_init
    current_edges = edges_init
    current_graph = graph
    
    # Helper: Check if a subset of vertices contains a clique of size k
    def has_clique(vertices, G, k=4):
        # vertices is a list of indices
        # Returns a clique if found, else None
        if len(vertices) < k:
            return None
        # Simple backtracking
        def backtrack(current_clique, candidates):
            if len(current_clique) == k:
                return current_clique
            for i, v in enumerate(candidates):
                # Check if v is connected to all in current_clique
                if all(v in G[u] for u in current_clique):
                    res = backtrack(current_clique + [v], candidates[i+1:])
                    if res:
                        return res
            return None
        
        # Use list for candidates
        return backtrack([], vertices)

    # Helper: Check if a subset of vertices contains an independent set of size k
    # Independent set in G is a clique in G_complement
    def has_independent_set(vertices, G, k=4):
        if len(vertices) < k:
            return None
        def backtrack(current_set, candidates):
            if len(current_set) == k:
                return current_set
            for i, v in enumerate(candidates):
                # Check if v is NOT connected to all in current_set
                if all(v not in G[u] for u in current_set):
                    res = backtrack(current_set + [v], candidates[i+1:])
                    if res:
                        return res
            return None
        return backtrack([], vertices)

    # Helper: Check if adding a vertex with neighbor set A is valid
    # A is a set of indices in current_graph
    def is_valid_extension(A):
        # A must not contain a K4
        clique = has_clique(list(A), current_graph, 4)
        if clique:
            return False, ("clique", clique)
        
        # V\A must not contain an independent set of size 4
        non_A = [v for v in range(current_n) if v not in A]
        ind_set = has_independent_set(non_A, current_graph, 4)
        if ind_set:
            return False, ("independent", ind_set)
            
        return True, None

    # Helper: Try to find a valid extension using local search
    def find_extension():
        if time.time() > deadline:
            return None
            
        # We represent the neighbor set as a boolean list or set
        # Random restarts
        for restart in range(500):
            if time.time() > deadline:
                return None
                
            # Random initialization
            A = set(random.sample(range(current_n), random.randint(0, current_n)))
            
            # Local search loop
            for step in range(1000):
                if time.time() > deadline:
                    return None
                
                valid, violation = is_valid_extension(A)
                if valid:
                    return A
                
                # If violation, try to fix it
                if violation[0] == "clique":
                    # We have a K4 in A. We must remove at least one vertex from this K4 from A (make it non-neighbor)
                    # Or add edges? No, A is the set of neighbors.
                    # If we remove v from A, it becomes a non-neighbor.
                    # This might create an independent set issue, but let's try.
                    v = random.choice(violation[1])
                    A.remove(v)
                else: # independent
                    # We have an independent set of size 4 in V\A.
                    # This means these 4 vertices are not in A, and they form an independent set.
                    # To fix, we must make at least one of them a neighbor (add to A).
                    v = random.choice(violation[1])
                    A.add(v)
                    
                # Also try random flips to escape local minima
                if random.random() < 0.2:
                    v = random.choice(range(current_n))
                    if v in A:
                        A.remove(v)
                    else:
                        A.add(v)
                        
        return None

    # Attempt to extend the graph up to 42 vertices
    max_target = 42
    while current_n < max_target:
        if time.time() > deadline:
            break
            
        extension = find_extension()
        if extension is not None:
            # Add the new vertex
            new_v = current_n
            # Update graph
            current_graph.append(set(extension))
            for u in extension:
                current_graph[u].add(new_v)
            
            # Update edge list
            for u in extension:
                current_edges.append((min(u, new_v), max(u, new_v)))
                
            current_n += 1
        else:
            # Could not extend further
            break

    return current_n, current_edges
