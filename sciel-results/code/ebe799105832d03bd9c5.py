"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 85.0
    n = 14
    m = 35
    
    # 1. Initialize values: Geometric progression helps keep products distinct but bounded.
    #    We want distinct integers <= 10000.
    #    Start with a base set.
    values = [int(x) for x in [2, 3, 5, 8, 13, 21, 34, 55, 89, 144, 233, 377, 610, 987]]
    # Ensure distinct and positive
    if len(set(values)) != n or any(v <= 0 or v > 10000 for v in values):
        values = list(range(1, n+1))

    # 2. Initialize edges: 
    #    A star graph centered at the vertex with the largest value (index 13) minimizes 
    #    the number of distinct sums/products if the other values are "small" relative to the center?
    #    Actually, sum = center + leaf. If center is huge, sums are distinct.
    #    Product = center * leaf. If center is huge, products are distinct.
    #    Wait, we want to MINIMIZE distinct counts.
    #    If we pick edges among small values, sums/products might collide.
    #    Let's try a "clique" of the first k vertices. 
    #    C(k, 2) = 35 => k ~ 9.4. 
    #    If we take a clique of 9 vertices (36 edges) and remove 1, we have 35 edges.
    #    If we take a clique of 8 vertices (28 edges) and add 7 edges to a 9th vertex, 
    #    the sums involving the 9th vertex might be distinct from the internal clique sums.
    
    # Let's try a specific structure: 
    # Vertices 0..7 form a clique (28 edges).
    # Vertex 8 connects to 0..6 (7 edges). Total 35.
    # Vertex 9..13 are isolated? No, all vertices must be part of the graph? 
    # "edges on their indices" implies the graph is defined on the 14 indices.
    # Isolated vertices are allowed as long as there are exactly 35 edges.
    # However, usually "graph on their indices" implies the vertices are the indices.
    # The objective is max(distinct sums, distinct products) OVER THE EDGES.
    # Isolated vertices contribute nothing to the sets.
    # So we can effectively ignore vertices 9-13 in the edge selection if we want.
    # But wait, if we ignore them, do we need to use them? No.
    # "Small sums and products along selected graph edges".
    
    # Strategy:
    # Use a subset of vertices that creates many collisions.
    # If values are small, sums collide.
    # Let's try to make values such that a+b = c+d frequently.
    # Or just use small values.
    # Let's try values = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14]
    # And a dense subgraph on the first few.
    
    def get_edges_clique_plus_star():
        # Clique on 0..7 (8 vertices) -> 28 edges
        edges = [list(e) for e in combinations(range(8), 2)]
        # Add edges from 8 to 0..6 (7 edges)
        for i in range(7):
            edges.append([8, i])
        return edges

    edges = get_edges_clique_plus_star()
    
    def objective(vals, eds):
        sums = set()
        prods = set()
        for u, v in eds:
            sums.add(vals[u] + vals[v])
            prods.add(vals[u] * vals[v])
        return max(len(sums), len(prods))
    
    best_score = objective(values, edges)
    best_vals = values[:]
    best_edges = edges[:]
    
    # Local Search: Perturb values and edges
    # Time limit check
    while time.time() < deadline:
        # Randomly perturb a value
        if random.random() < 0.5:
            new_vals = best_vals[:]
            idx = random.randint(0, n-1)
            # Change value to something nearby or random small
            if random.random() < 0.5:
                delta = random.randint(-10, 10)
                new_val = best_vals[idx] + delta
            else:
                new_val = random.randint(1, 20)
            
            if 1 <= new_val <= 10000 and new_val not in [v for i, v in enumerate(best_vals) if i != idx]:
                new_vals[idx] = new_val
                score = objective(new_vals, best_edges)
                if score < best_score:
                    best_score = score
                    best_vals = new_vals
                    best_edges = best_edges[:]
        else:
            # Perturb edges: swap an edge
            new_edges = [e[:] for e in best_edges]
            # Remove a random edge
            e_idx = random.randint(0, len(new_edges)-1)
            removed = new_edges.pop(e_idx)
            # Add a random new edge
            while True:
                u = random.randint(0, n-1)
                v = random.randint(0, n-1)
                if u == v: continue
                if [u, v] in new_edges or [v, u] in new_edges: continue
                break
            new_edges.append([u, v])
            
            score = objective(best_vals, new_edges)
            if score < best_score:
                best_score = score
                best_edges = new_edges
                best_vals = best_vals[:]
                
    # Final sanity check and type conversion
    # Ensure values are distinct positive ints
    if len(set(best_vals)) != n:
        # Fallback to simple distinct
        best_vals = list(range(1, n+1))
        
    # Ensure edges are valid
    valid_edges = []
    seen = set()
    for e in best_edges:
        u, v = e
        if u == v: continue
        key = (min(u,v), max(u,v))
        if key in seen: continue
        seen.add(key)
        valid_edges.append([int(u), int(v)])
        
    if len(valid_edges) != m:
        # Fallback to a valid edge set
        # Generate a valid set of 35 edges
        valid_edges = [list(e) for e in combinations(range(n), 2)]
        # Shuffle and take first 35? Or just first 35.
        random.shuffle(valid_edges)
        valid_edges = valid_edges[:m]
        
    return {
        'values': [int(v) for v in best_vals],
        'edges': valid_edges
    }
# EVOLVE-BLOCK-END
