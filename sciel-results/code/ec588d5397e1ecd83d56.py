import random
import itertools

def construct():
    n = 31
    target_edges = 91
    
    # We will try to build a C4-free graph with 90 edges.
    # Then add one edge that minimizes the number of 4-cycles created.
    
    def count_c4(edges):
        # This is for verification, not used in the main loop for speed
        # But we need to return the graph, the score is calculated by evaluator.
        # We just need to return the edges.
        pass

    best_graph = None
    best_score = float('inf')
    
    # Try a few random restarts to find a good C4-free graph with 90 edges
    for seed in range(10):
        random.seed(seed)
        adj = [set() for _ in range(n)]
        edges = set()
        
        # Generate all possible edges
        all_edges = list(itertools.combinations(range(n), 2))
        random.shuffle(all_edges)
        
        edge_count = 0
        for u, v in all_edges:
            if edge_count >= 90:
                break
            # Check if adding (u, v) creates a C4
            # A C4 is created if u and v have >= 2 common neighbors
            common = adj[u].intersection(adj[v])
            if len(common) < 2:
                # Safe to add
                adj[u].add(v)
                adj[v].add(u)
                edges.add((u, v))
                edge_count += 1
        
        if edge_count < 90:
            continue
            
        # We have a C4-free graph with 90 edges.
        # Now find the non-edge (u, v) that minimizes the number of common neighbors.
        # Adding (u, v) will create binom(k, 2) C4s, where k is common neighbors.
        
        min_common = float('inf')
        best_pair = None
        
        for u in range(n):
            for v in range(u + 1, n):
                if (u, v) in edges or (v, u) in edges:
                    continue
                common = len(adj[u].intersection(adj[v]))
                if common < min_common:
                    min_common = common
                    best_pair = (u, v)
        
        if best_pair is None:
            continue
            
        # The number of C4s will be binom(min_common, 2)
        score = min_common * (min_common - 1) // 2
        
        if score < best_score:
            best_score = score
            # Add the best pair to the edges
            u, v = best_pair
            final_edges = list(edges)
            final_edges.append([u, v])
            best_graph = final_edges
            
            # If we found a graph with 1 C4, that's optimal (since ex=90, we must have >=1)
            if best_score == 1:
                break
                
    if best_graph is None:
        # Fallback: just return first 91 edges if something went wrong
        pairs = [[i, j] for i in range(n) for j in range(i + 1, n)]
        return pairs[:91]
        
    return best_graph
