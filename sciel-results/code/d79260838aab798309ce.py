import time
import sys
import random
from itertools import combinations

def construct():
    n = 15
    start_time = time.time()
    deadline = start_time + 85

    # Deterministic search over disjoint union of complete bipartite components K_{a, b}
    # where a != b.
    # We partition 15 vertices into pairs (a_i, b_i).
    # Sum(a_i + b_i) = 15.
    # Maximize Sum(a_i * b_i).
    
    best_edges_count = 0
    best_partition = []
    
    # Since n=15 is small, we can iterate through possible component sizes.
    # Components are defined by (a, b) with a+b = size, a != b.
    # Possible sizes: 2..15.
    # We need a set of components that sum to 15.
    
    # Recursive search for partitions
    def search_partitions(remaining, current_components):
        nonlocal best_edges_count, best_partition
        
        # Pruning: even if we use the max possible edges for remaining, can we beat best?
        # Max edges for size S is floor(S^2/4) if S is even (but a!=b, so (S/2-1)(S/2+1) = S^2/4 - 1)
        # If S is odd, floor(S^2/4).
        
        if remaining == 0:
            count = sum(a*b for a,b in current_components)
            if count > best_edges_count:
                best_edges_count = count
                best_partition = current_components[:]
            return
        
        # Try all possible component sizes for the next component
        # Min size 2 (1,1 is a=b invalid? K_1,1 has deg 1,1. Path length 3? No, K_1,1 has no path of length 3.
        # But K_1,1 is a component. Is it valid?
        # "no simple three-edge path". K_1,1 has no such path. So it is valid.
        # But a=1, b=1 => a=b.
        # Does a=b matter for K_1,1?
        # The constraint is on paths of length 3.
        # If a component has < 4 vertices, it has no path of length 3.
        # So K_1,1, K_1,2, K_2,2 are valid *locally* if they don't form paths of length 3.
        # K_2,2 has 4 vertices. Path 1-3-2-4. Endpoints 1,4. deg(1)=2, deg(4)=2. Invalid.
        # So K_2,2 is invalid.
        # K_1,1 is valid (no path len 3).
        # K_1,2 is valid (no path len 3).
        # K_1,3 is valid (no path len 3).
        # K_2,3: Path 1(L)-3(R)-2(L)-4(R). deg(1)=3, deg(4)=2. OK.
        # So a != b is sufficient for a,b >= 2?
        # If a=1, b=1, no path len 3. Valid.
        # If a=1, b=2, no path len 3. Valid.
        # If a=1, b=3, no path len 3. Valid.
        # If a=2, b=2, path exists. deg equal. Invalid.
        # So we must avoid a=b if a+b >= 4.
        
        max_size = remaining
        min_size = 2
        
        for size in range(min_size, max_size + 1):
            # Generate all (a, b) for this size
            for a in range(1, size):
                b = size - a
                if a > b: continue # symmetry
                # Check validity for this component
                if a == b and size >= 4:
                    continue
                # Estimate max possible edges for rest
                # This is a weak bound, but n=15 is small.
                search_partitions(remaining - size, current_components + [(a, b)])

    search_partitions(15, [])
    
    # Construct the graph from the best partition
    edges = set()
    vertex_id = 0
    for a, b in best_partition:
        # Component vertices: vertex_id ... vertex_id+a-1 (Part A)
        # vertex_id+a ... vertex_id+a+b-1 (Part B)
        for u in range(vertex_id, vertex_id + a):
            for v in range(vertex_id + a, vertex_id + a + b):
                edges.add((u, v))
        vertex_id += a + b
    
    current_edges = list(edges)
    
    # Local improvement: try to add edges
    # (This is likely to fail for K_7,8, but we keep it as per "investigate improvements")
    
    def is_valid(edge_list):
        if len(edge_list) == 0:
            return True
        adj = {i: set() for i in range(n)}
        deg = [0] * n
        for u, v in edge_list:
            adj[u].add(v)
            adj[v].add(u)
            deg[u] += 1
            deg[v] += 1

        for u, v in edge_list:
            for s in adj[u]:
                if s == v:
                    continue
                for t in adj[v]:
                    if t == u or t == s:
                        continue
                    if deg[s] == deg[t]:
                        return False
        return True

    all_edges = set()
    for i in range(n):
        for j in range(i + 1, n):
            all_edges.add((i, j))
            
    missing_edges = list(all_edges - edges)
    
    improved = True
    while improved:
        improved = False
        if time.time() > deadline:
            break
        for e in missing_edges:
            if e in edges:
                continue
            next_edges = current_edges + [e]
            if is_valid(next_edges):
                current_edges = next_edges
                edges.add(e)
                improved = True

    final_edges = [[int(u), int(v)] for u, v in current_edges]
    return final_edges
