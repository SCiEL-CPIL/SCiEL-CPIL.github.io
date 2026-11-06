"""Fewest four-cycles in graphs with 31 vertices and 91 edges."""
# EVOLVE-BLOCK-START
import time
import numpy as np

def construct():
    start_time = time.time()
    DEADLINE = 118.0  # Safety margin
    
    n = 31
    q = 5
    target_edges = 91
    
    # Generate points of PG(2, q)
    points = []
    # Points at infinity: [x:1:0] for x in 0..q-1, and [1:0:0]
    for x in range(q):
        points.append((x, 1, 0))
    points.append((1, 0, 0))
    # Finite points: [x:y:1] for x, y in 0..q-1
    for x in range(q):
        for y in range(q):
            points.append((x, y, 1))
            
    if len(points) != n:
        # Fallback to simple random graph if something goes wrong with coordinates
        import random
        random.seed(42)
        edges = []
        while len(edges) < target_edges:
            u = random.randint(0, n-1)
            v = random.randint(0, n-1)
            if u == v: continue
            if u > v: u, v = v, u
            if (u, v) not in edges:
                edges.append((u, v))
        return [[u, v] for u, v in edges]

    # Build adjacency
    neighbors = [set() for _ in range(n)]
    edges = []
    
    for i in range(n):
        for j in range(i+1, n):
            p1 = points[i]
            p2 = points[j]
            # Polarity condition: x1*x2 + y1*y2 - z1*z2 == 0 mod q
            if (p1[0]*p2[0] + p1[1]*p2[1] - p1[2]*p2[2]) % q == 0:
                neighbors[i].add(j)
                neighbors[j].add(i)
                edges.append((i, j))
                
    current_edges = len(edges)
    
    # If we don't have exactly 90 edges, we might need to adjust.
    # But theory predicts 90.
    if current_edges > target_edges:
        # Remove edges greedily? Unlikely to happen.
        pass
    elif current_edges < target_edges:
        # Add edges with min cost
        pass

    # Find non-edges
    non_edges = []
    for i in range(n):
        for j in range(i+1, n):
            if j not in neighbors[i]:
                non_edges.append((i, j))
                
    # We need to add (target_edges - current_edges) edges.
    # Usually this is 1.
    edges_to_add = target_edges - current_edges
    
    added_edges = []
    
    # To be safe, if we need to add more than 1, we add them one by one,
    # updating neighbors after each addition to ensure correctness for subsequent additions.
    # However, for this specific task, edges_to_add should be 1.
    
    for _ in range(edges_to_add):
        if time.time() - start_time > DEADLINE:
            break
        best_edge = None
        min_cost = float('inf')
        
        # Recalculate non-edges if we added some?
        # If edges_to_add is 1, non_edges is static.
        # If > 1, we should remove added edges from non_edges list.
        # For simplicity and given the specific case (1 edge), we use the static list.
        # If we need to add multiple, we'd need to be careful.
        
        for u, v in non_edges:
            # Check if this edge was already added in this loop (if edges_to_add > 1)
            if (u, v) in added_edges:
                continue
                
            # Cost = number of 4-cycles that would be formed by adding edge (u,v)
            # A 4-cycle through (u,v) is u-a-b-v-u where a~u, b~v, a~b, all distinct
            cost = 0
            for b in neighbors[v]:
                if b == u:
                    continue
                for a in neighbors[u]:
                    if a == v or a == b:
                        continue
                    if b in neighbors[a]:
                        cost += 1
            
            if cost < min_cost:
                min_cost = cost
                best_edge = (u, v)
                
        if best_edge:
            u, v = best_edge
            edges.append((u, v))
            neighbors[u].add(v)
            neighbors[v].add(u)
            added_edges.append((u, v))
        else:
            break
            
    return [[u, v] for u, v in edges]
# EVOLVE-BLOCK-END
