import random
import time

def construct():
    n = 31
    target_edges = 91
    deadline = time.time() + 120.0
    
    # 1. Construct the base quadratic polarity graph of PG(2,5)
    # Vertices: 0..30
    # Points of PG(2,5)
    points = []
    # z=1
    for x in range(5):
        for y in range(5):
            points.append((x, y, 1))
    # z=0, x=1
    for y in range(5):
        points.append((1, y, 0))
    # z=0, x=0, y=1
    points.append((0, 1, 0))
    
    # Map point to index
    point_to_idx = {p: i for i, p in enumerate(points)}
    
    # Quadratic form Q(x,y,z) = x^2 + y^2 + z^2 mod 5
    def q_form(p):
        x, y, z = p
        return (x*x + y*y + z*z) % 5
    
    # Bilinear form B(u,v) = 2 * dot(u,v) mod 5
    # u ~ v if dot(u,v) == 0 and u != v
    edges = set()
    for i in range(n):
        for j in range(i+1, n):
            u = points[i]
            v = points[j]
            dot = (u[0]*v[0] + u[1]*v[1] + u[2]*v[2]) % 5
            if dot == 0:
                edges.add((i, j))
                
    # This graph has 80 edges and is C4-free.
    # We need to add 11 edges.
    
    # Data structure for fast neighbor lookup
    neighbors = [set() for _ in range(n)]
    for u, v in edges:
        neighbors[u].add(v)
        neighbors[v].add(u)
        
    def count_c4s(edges_set, neighbors_list):
        total = 0
        for i in range(n):
            for j in range(i+1, n):
                common = len(neighbors_list[i].intersection(neighbors_list[j]))
                if common >= 2:
                    total += common * (common - 1) // 2
        return total // 2 # Each C4 counted twice
        
    current_c4s = count_c4s(edges, neighbors)
    
    # Greedy addition of 11 edges
    all_pairs = [(i, j) for i in range(n) for j in range(i+1, n)]
    non_edges = [p for p in all_pairs if p not in edges]
    
    for _ in range(11):
        best_edge = None
        min_common = float('inf')
        for u, v in non_edges:
            common = len(neighbors[u].intersection(neighbors[v]))
            if common < min_common:
                min_common = common
                best_edge = (u, v)
        
        if best_edge is None:
            break
            
        u, v = best_edge
        edges.add((u, v))
        neighbors[u].add(v)
        neighbors[v].add(u)
        non_edges.remove((u, v))
        
    current_c4s = count_c4s(edges, neighbors)
    
    # Local Search
    # Precompute all non-edges for random selection
    non_edges_list = list(non_edges)
    
    while time.time() < deadline:
        # Random swap
        if not non_edges_list:
            break
            
        idx_add = random.randint(0, len(non_edges_list) - 1)
        u, v = non_edges_list[idx_add]
        
        # Pick random edge to remove
        edge_list = list(edges)
        idx_rem = random.randint(0, len(edge_list) - 1)
        a, b = edge_list[idx_rem]
        
        # Skip if adding edge is same as removing
        if (u == a and v == b) or (u == b and v == a):
            continue
            
        # Tentative change
        # Remove (a,b)
        edges.remove((a, b))
        neighbors[a].remove(b)
        neighbors[b].remove(a)
        
        # Add (u,v)
        edges.add((u, v))
        neighbors[u].add(v)
        neighbors[v].add(u)
        
        new_c4s = count_c4s(edges, neighbors)
        
        if new_c4s < current_c4s:
            current_c4s = new_c4s
            # Update non_edges_list
            non_edges_list.remove((u, v))
            non_edges_list.append((a, b))
        else:
            # Revert
            edges.remove((u, v))
            neighbors[u].remove(v)
            neighbors[v].remove(u)
            
            edges.add((a, b))
            neighbors[a].add(b)
            neighbors[b].add(a)
            
    # Return list of edges
    return [list(e) for e in edges]
