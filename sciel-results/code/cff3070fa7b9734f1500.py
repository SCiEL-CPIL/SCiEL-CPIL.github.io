import random
import time
import itertools

def construct():
    start_time = time.time()
    deadline = start_time + 110  # 10s safety margin
    
    n = 18
    all_edges = [(i, j) for i in range(n) for j in range(i+1, n)]
    
    best_score = 19 # Max possible is 18
    best_graph = set() # Store edges as frozensets or tuples
    
    def get_adjacency(graph_edges):
        adj = [set() for _ in range(n)]
        for u, v in graph_edges:
            adj[u].add(v)
            adj[v].add(u)
        return adj

    def is_k4_free(graph_edges):
        # Used for validation, but we build incrementally
        pass

    def can_add_edge(adj, u, v):
        # Check if adding (u,v) creates a K4
        # Common neighbors
        common = adj[u] & adj[v]
        # Check if any edge exists in common
        for w in common:
            if w in adj[v]: # w is neighbor of v, check if w is neighbor of another common neighbor?
                # Wait, K4 requires w and x to be connected.
                # If there is any edge (w, x) with w, x in common, then K4 is formed.
                # So we just need to check if the subgraph induced by 'common' has any edge.
                pass
        
        # Faster check:
        # Iterate w in common. If w has any neighbor in common, return False.
        for w in common:
            if adj[w] & common:
                return False
        return True

    def calc_score(graph_edges):
        adj = get_adjacency(graph_edges)
        # Check subsets of size k from 18 down to 1
        # We want the largest k such that there exists a subset of size k with no triangle.
        
        # Precompute triangle masks?
        # For n=18, we can just iterate combinations.
        
        # To speed up, we can check if a specific subset is triangle free.
        def is_triangle_free(vertices):
            # vertices is a set of ints
            # Check all triples
            v_list = list(vertices)
            for i in range(len(v_list)):
                for j in range(i+1, len(v_list)):
                    u, v = v_list[i], v_list[j]
                    if v in adj[u]:
                        # Check common neighbors in vertices
                        common = adj[u] & adj[v] & vertices
                        if common:
                            return False
            return True

        # Search for max k
        # We know score is at least 1.
        # Start from 18 down.
        for k in range(18, 0, -1):
            found = False
            for subset in itertools.combinations(range(n), k):
                if is_triangle_free(set(subset)):
                    found = True
                    break
            if found:
                return k
        return 0

    # Random search
    trials = 0
    while time.time() < deadline:
        trials += 1
        # Generate a random maximal K4-free graph
        graph_edges = set()
        adj = [set() for _ in range(n)]
        
        # Random order of edges
        random_edges = all_edges[:]
        random.shuffle(random_edges)
        
        for u, v in random_edges:
            if can_add_edge(adj, u, v):
                graph_edges.add((u, v))
                adj[u].add(v)
                adj[v].add(u)
                
        # Evaluate
        score = calc_score(graph_edges)
        
        if score < best_score:
            best_score = score
            best_graph = graph_edges
            # If we reach a very low score, we might stop early?
            # Theoretical min is likely around 8-10.
            if best_score <= 8:
                break

    return [[u, v] for u, v in best_graph]

construct()
