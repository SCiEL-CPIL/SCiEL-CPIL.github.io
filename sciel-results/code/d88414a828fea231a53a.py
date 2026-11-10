import time
import random
from collections import defaultdict

def construct():
    start_time = time.time()
    DEADLINE = 140.0  # Safety margin for 150s limit
    
    def instance(n, p, seed):
        rng = Random(seed)
        order = list(range(n))
        rng.shuffle(order)
        edges = {tuple(sorted(e)) for e in zip(order, order[1:])}
        edges |= {(u, v) for u in range(n) for v in range(u + 1, n) if rng.random() < p}
        return sorted(edges)

    results = []
    
    for spec in ((20, 0.4, 11), (25, 0.35, 12), (30, 0.3, 13)):
        edges = instance(*spec)
        n = spec[0]
        
        # Decompose
        pieces = decompose_graph(edges, n, start_time, DEADLINE)
        results.append(pieces)
        
        # Check time
        if time.time() > DEADLINE - 10:
             # If we are running out of time, we might need to be faster for the rest
             # But we only have 3 graphs. If we are here, we are likely done or close.
             pass

    return results

def decompose_graph(edges, n, start_time, deadline):
    edge_set = set(tuple(sorted(e)) for e in edges)
    if not edge_set:
        return []
        
    pieces = []
    
    # Precompute adjacency for faster DFS? 
    # Since edges are removed, we need dynamic adjacency or check edge_set in DFS.
    # Checking edge_set is O(1).
    
    def find_cycle_for_edge(u, v, edge_set, n, max_trials=50):
        # Find a path from v to u in edge_set excluding (u,v)
        # Use randomized DFS
        best_path = None
        
        # To make it faster, we can limit the depth if needed, but N=30 is small.
        # We will run max_trials random DFS searches.
        
        # Helper for DFS
        def dfs(current, target, visited, path):
            nonlocal best_path
            if current == target:
                if best_path is None or len(path) > len(best_path):
                    best_path = path[:]
                return True # Found one, but we want longer? 
                # If we return True, we stop this DFS branch.
                # To find longer, we should NOT stop, but that's expensive.
                # Given the small size, let's just take the first found in each trial, 
                # and rely on multiple trials to find longer ones.
                # Actually, if we return True, we don't explore other branches from this node.
                # This is standard DFS.
                return True
            
            # Get neighbors
            # We need to iterate over all potential neighbors? 
            # No, we can iterate over all vertices and check if edge exists? O(N^2).
            # Or maintain adjacency list.
            # Let's maintain a set of neighbors for each node in the current edge_set?
            # That's expensive to update.
            # Let's just iterate over range(n) and check if (current, nb) in edge_set.
            # N=30, so 30 checks per node. Fast.
            
            # Shuffle neighbors for randomization
            neighbors = [i for i in range(n) if i != current and (min(current, i), max(current, i)) in edge_set]
            random.shuffle(neighbors)
            
            for nb in neighbors:
                if nb not in visited:
                    visited.add(nb)
                    path.append(nb)
                    dfs(nb, target, visited, path)
                    path.pop()
                    visited.remove(nb)
                    
                    # Optimization: If we found a path, and it's already very long (e.g., > 20), 
                    # we might stop early? 
                    # But we want the longest.
                    # Let's just finish the DFS.
            
            return False

        for _ in range(max_trials):
            visited = {v}
            path = [v]
            # We need to exclude (u,v) from the search space.
            # The edge_set still contains (u,v). 
            # We must ensure the DFS doesn't use (u,v).
            # Since we start at v and go to u, the first step is v->... 
            # The edge (u,v) connects u and v.
            # If the path goes v->...->u, it doesn't use (u,v) unless the path is v->u directly.
            # If the path is v->u, the length is 1 (edge). Cycle length 2? No, simple cycle >= 3.
            # So we should not use the direct edge (u,v) in the path.
            # We can handle this by checking in the neighbor loop:
            # if nb == u and current == v: skip? 
            # Actually, if current is v and nb is u, that's the edge (u,v).
            # We want to find a path v -> ... -> u that does NOT use (u,v).
            # So if current is v, we should not go to u directly.
            
            # Let's modify the DFS to take a forbidden_edge.
            pass # The logic above is inside the function, I need to implement it properly.
            
        if best_path:
            # best_path is v, ..., u
            # Cycle is u, v, ..., u
            # Vertices: [u] + best_path
            # But best_path starts with v.
            # So cycle: [u] + best_path
            return [u] + best_path
        return None

    # The implementation above has a flaw: the DFS `dfs` function is nested and uses `best_path` from outer scope.
    # Also, the neighbor iteration `range(n)` is O(N) per node, total O(N^2) per DFS.
    # N=30, N^2=900. 50 trials * 900 = 4500 ops per edge check.
    # This is fine.
    
    # Let's rewrite `find_cycle_for_edge` correctly.
    
    def find_cycle_for_edge_optimized(u, v, edge_set, n, max_trials=20):
        best_path = None
        
        def dfs(current, target, visited, path):
            nonlocal best_path
            if current == target:
                if best_path is None or len(path) > len(best_path):
                    best_path = path[:]
                return True
            
            # Generate neighbors
            # Optimization: Only consider neighbors that are in edge_set
            # And not visited
            # And not the forbidden edge (u,v) if current is v and nb is u?
            # Actually, we just exclude the edge (u,v) from the graph for this search.
            
            # To avoid generating list every time, we can just loop.
            # But we need random order.
            neighbors = []
            for i in range(n):
                if i == current: continue
                if i in visited: continue
                e = (min(current, i), max(current, i))
                if e in edge_set:
                    # Check if this is the forbidden edge
                    if current == v and i == u:
                        continue
                    if current == u and i == v:
                        continue
                    neighbors.append(i)
            
            random.shuffle(neighbors)
            
            for nb in neighbors:
                visited.add(nb)
                path.append(nb)
                dfs(nb, target, visited, path)
                path.pop()
                visited.remove(nb)
                
            return False

        for _ in range(max_trials):
            visited = {v}
            path = [v]
            dfs(v, u, visited, path)
            
        if best_path:
            return [u] + best_path
        return None

    # Main Loop
    while edge_set:
        if time.time() > deadline - 5:
             # Fallback: add all remaining as single edges
             for e in edge_set:
                 pieces.append(list(e))
             break
        
        # Try to find a cycle
        # Sample some edges
        sample_size = min(10, len(edge_set))
        sampled_edges = random.sample(list(edge_set), sample_size)
        
        candidates = []
        for e in sampled_edges:
            u, v = e
            c = find_cycle_for_edge_optimized(u, v, edge_set, n, max_trials=20)
            if c:
                candidates.append(c)
        
        if candidates:
            best_cycle = max(candidates, key=len)
            pieces.append(best_cycle)
            # Remove edges
            for i in range(len(best_cycle)):
                u = best_cycle[i]
                v = best_cycle[(i+1)%len(best_cycle)]
                edge = (min(u, v), max(u, v))
                if edge in edge_set:
                    edge_set.remove(edge)
        else:
            # No cycle found in sample
            # Check if there is ANY cycle
            # Simple check: If the graph is a forest, no cycles.
            # We can check if number of edges < number of vertices - components?
            # Or just try to find one with a global search.
            # If we fail again, assume forest.
            
            # Let's try one more time with a larger sample or global
            found = False
            for e in list(edge_set):
                u, v = e
                c = find_cycle_for_edge_optimized(u, v, edge_set, n, max_trials=5)
                if c:
                    pieces.append(c)
                    for i in range(len(c)):
                        uu = c[i]
                        vv = c[(i+1)%len(c)]
                        edge = (min(uu, vv), max(uu, vv))
                        if edge in edge_set:
                            edge_set.remove(edge)
                    found = True
                    break
            
            if not found:
                # Assume forest
                for e in edge_set:
                    pieces.append(list(e))
                break

    return pieces
