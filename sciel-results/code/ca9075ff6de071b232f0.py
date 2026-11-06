"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 120  # safety margin
    
    # Try to find a triangle-free graph on up to 40 vertices with alpha < 10
    # Start with And(9) on 25 vertices as seed, then try to extend
    
    best_n = 26
    best_edges = [[i, j] for i in range(26) for j in range(i + 1, 26) if (j - i) % 26 % 3 == 1]
    
    # Try random local search to find larger graphs
    for target_n in range(27, 41):
        if time.time() > deadline - 10:
            break
        
        # Try multiple random restarts
        for attempt in range(50):
            if time.time() > deadline - 10:
                break
            
            # Start with a random graph and optimize
            # Use a simple approach: start with dense random, remove triangles, check alpha
            
            # Actually, let's use a different strategy:
            # Start from And(9) and try to add vertices with appropriate connections
            # This is complex, so let's try a simpler random approach
            
            # Generate random graph
            edges = set()
            for i in range(target_n):
                for j in range(i + 1, target_n):
                    if random.random() < 0.3:
                        edges.add((i, j))
            
            # Remove triangles by removing edges
            changed = True
            while changed:
                changed = False
                edge_list = list(edges)
                for (u, v) in edge_list:
                    # Check if removing this edge helps
                    # Find a triangle containing this edge
                    for w in range(target_n):
                        if w != u and w != v:
                            if (min(u,w), max(u,w)) in edges and (min(v,w), max(v,w)) in edges:
                                # Triangle u-v-w, remove edge (u,v)
                                edges.discard((u, v))
                                changed = True
                                break
                    if time.time() > deadline - 10:
                        break
                if time.time() > deadline - 10:
                    break
            
            if time.time() > deadline - 10:
                break
            
            # Now check independence number using greedy upper bound
            # Build adjacency
            adj = [set() for _ in range(target_n)]
            for (u, v) in edges:
                adj[u].add(v)
                adj[v].add(u)
            
            # Try to find independent set of size 10
            # Use greedy: repeatedly pick vertex with fewest neighbors in remaining set
            remaining = set(range(target_n))
            indep = []
            while remaining:
                # Pick vertex with minimum degree in remaining
                min_deg = float('inf')
                chosen = None
                for v in remaining:
                    deg = len(adj[v] & remaining)
                    if deg < min_deg:
                        min_deg = deg
                        chosen = v
                indep.append(chosen)
                remaining.discard(chosen)
                for u in adj[chosen]:
                    remaining.discard(u)
            
            if len(indep) < 10:
                # Found a valid graph
                best_n = target_n
                best_edges = [[u, v] for (u, v) in edges if u < v]
                break
        
        if best_n == target_n:
            continue
    
    return {'n': best_n, 'edges': best_edges}
# EVOLVE-BLOCK-END
