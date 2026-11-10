import random
import time

def construct():
    d = 7
    n_vertices = 2 ** d
    all_edges = []
    edge_to_idx = {}
    
    # Generate all edges
    for x in range(n_vertices):
        for i in range(d):
            y = x ^ (1 << i)
            if x < y:
                e = (x, y)
                edge_to_idx[e] = len(all_edges)
                all_edges.append(e)
                
    num_edges = len(all_edges)
    
    # Generate all squares (C4s)
    # A square is defined by two distinct bits i, j and a vertex x 
    # such that x has 0 at i and 0 at j.
    squares = [] # List of list of edge indices
    edge_to_squares = [[] for _ in range(num_edges)]
    
    for i in range(d):
        for j in range(i + 1, d):
            for x in range(n_vertices):
                if (x & (1 << i)) == 0 and (x & (1 << j)) == 0:
                    v0 = x
                    v1 = x ^ (1 << i)
                    v2 = x ^ (1 << j)
                    v3 = x ^ (1 << i) ^ (1 << j)
                    
                    # Edges: (v0, v1), (v1, v3), (v3, v2), (v2, v0)
                    e0 = (min(v0, v1), max(v0, v1))
                    e1 = (min(v1, v3), max(v1, v3))
                    e2 = (min(v3, v2), max(v3, v2))
                    e3 = (min(v2, v0), max(v2, v0))
                    
                    idxs = [edge_to_idx[e] for e in [e0, e1, e2, e3]]
                    squares.append(idxs)
                    for idx in idxs:
                        edge_to_squares[idx].append(len(squares) - 1)
                        
    num_squares = len(squares)
    
    best_size = 0
    best_set = set()
    
    start_time = time.time()
    time_limit = 100
    
    for _ in range(500): # More iterations possible as removal is fast
        if time.time() - start_time > time_limit:
            break
            
        # State for this run
        # removed_edges: set of edge indices
        removed = set()
        # remaining_edges_in_square: count of edges in square still present
        rem_count = [4] * num_squares
        # alive_squares: set of square indices that are currently C4s (all 4 edges present)
        alive_squares = set(range(num_squares))
        
        # threat[e]: number of alive squares containing edge e
        threat = [0] * num_edges
        
        # Initialize threat
        for s in range(num_squares):
            for e in squares[s]:
                threat[e] += 1
                
        # Greedy removal
        while alive_squares:
            # Find edge with max threat
            # To speed up, we can just scan, or use a heap.
            # Scanning 448 edges is fast.
            max_threat = -1
            best_edge = -1
            candidates = []
            for e in range(num_edges):
                if e not in removed:
                    if threat[e] > max_threat:
                        max_threat = threat[e]
                        best_edge = e
                        candidates = [e]
                    elif threat[e] == max_threat and max_threat > 0:
                        candidates.append(e)
                        
            if best_edge == -1:
                break
                
            # Random tie-break
            e_to_remove = random.choice(candidates)
            
            removed.add(e_to_remove)
            
            # Update threats and alive squares
            for s in edge_to_squares[e_to_remove]:
                if s in alive_squares:
                    rem_count[s] -= 1
                    if rem_count[s] < 4:
                        # Square s is no longer a C4
                        alive_squares.remove(s)
                        # Decrease threat of the other 3 edges in s
                        for e in squares[s]:
                            if e != e_to_remove and e not in removed:
                                threat[e] -= 1
                    # If rem_count[s] >= 4, it stays alive, but threat of e_to_remove is irrelevant now.
                    # Wait, if s is still alive, it means the other 3 edges are still there.
                    # The threat of the other 3 edges should NOT decrease because s is still an alive C4.
                    # The threat of e_to_remove is no longer used.
                    
            # Note: If s was alive, and we removed e_to_remove, rem_count becomes 3.
            # So s is removed from alive_squares.
            # The other 3 edges lose this "threat" from s.
            # So the logic above is correct.
            
        current_size = num_edges - len(removed)
        if current_size > best_size:
            best_size = current_size
            best_set = set(range(num_edges)) - removed
            
    # Convert indices back to edges
    return [[all_edges[u], all_edges[v]] if False else [all_edges[e][0], all_edges[e][1]] for e in best_set]
