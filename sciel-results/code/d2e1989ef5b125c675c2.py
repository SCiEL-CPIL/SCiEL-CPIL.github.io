import time
from random import Random
import sys

def instance(n, p, seed):
    rng = Random(seed)
    order = list(range(n))
    rng.shuffle(order)
    edges = {tuple(sorted(e)) for e in zip(order, order[1:])}
    edges |= {(u, v) for u in range(n) for v in range(u + 1, n) if rng.random() < p}
    return sorted(edges)

def decompose_paths(edges, deadline):
    """
    Decompose edges into edge-disjoint simple paths, minimizing path count.
    Uses randomized greedy initialization followed by local search merging.
    """
    if not edges:
        return []
    
    # Normalize edges to (min, max) tuples
    edge_set = {tuple(sorted(e)) for e in edges}
    if not edge_set:
        return []
        
    # Build adjacency list
    adj = {}
    for u, v in edge_set:
        if u not in adj: adj[u] = []
        if v not in adj: adj[v] = []
        adj[u].append(v)
        adj[v].append(u)
        
    # Sort neighbors for deterministic behavior in base greedy
    for v in adj:
        adj[v].sort()
        
    best_paths = None
    best_count = float('inf')
    
    # Multiple randomized restarts
    # Number of restarts depends on available time
    # Each restart is O(E^2) or so, which is very fast for E~100
    max_restarts = 5000 
    rng = Random(42)
    
    start_time = time.time()
    
    for restart in range(max_restarts):
        if time.time() > deadline - 2:
            break
            
        # 1. Greedy Initialization
        # We want to construct long paths.
        # Strategy: Pick a random start, then greedily extend.
        # To avoid bias, we can shuffle the order in which we pick starting points
        # or shuffle the neighbor order.
        
        remaining_edges = set(edge_set)
        paths = []
        
        # Create a list of vertices with remaining edges
        # We will repeatedly pick a vertex, start a path, and extend it as far as possible.
        
        # To make it randomized, we can shuffle the vertices or the neighbor lists.
        # Let's shuffle the neighbor lists for this restart.
        adj_rand = {v: list(neighs) for v, neighs in adj.items()}
        for v in adj_rand:
            rng.shuffle(adj_rand[v])
            
        # Also, we can shuffle the order in which we process "active" vertices.
        # A simple way: just iterate through a shuffled list of all vertices.
        all_vertices = list(adj.keys())
        rng.shuffle(all_vertices)
        
        # We need to ensure we cover all edges.
        # Greedy: 
        # While remaining_edges is not empty:
        #   Find a vertex with remaining edges.
        #   Start a path from it.
        #   Extend the path greedily.
        
        # To improve quality, we can prioritize starting paths at vertices 
        # that have high degree in the remaining graph? 
        # Or just random.
        
        # Let's try a specific heuristic: 
        # Sort vertices by degree in remaining graph (descending) and pick the first one?
        # This tends to create long paths.
        
        # But we want diversity. Let's use a mix.
        
        # Simple Greedy Loop
        local_rem = set(remaining_edges)
        
        # Build local adj for speed
        local_adj = {v: [] for v in adj}
        for u, v in local_rem:
            local_adj[u].append(v)
            local_adj[v].append(u)
            
        # We will pick a start vertex, then extend.
        # To ensure we don't get stuck with many short paths, 
        # we can use a "longest path" heuristic during extension:
        # At each step, choose the neighbor that leads to the longest possible continuation?
        # That's expensive. 
        # Instead, just pick a random neighbor from the shuffled list.
        
        # To restart the process effectively, we can just run the greedy 
        # with the shuffled adj_rand.
        
        # We need to pick start vertices.
        # If we just pick any vertex with remaining edges, we might pick a leaf 
        # and create a short path.
        # Better: Pick a vertex with maximum degree in local_rem?
        
        while local_rem:
            # Find a good start vertex
            # Candidates are vertices in local_rem
            candidates = set()
            for u, v in local_rem:
                candidates.add(u)
                candidates.add(v)
                
            # Pick the one with highest degree in local_rem
            # This encourages longer paths
            start = max(candidates, key=lambda x: len(local_adj[x]))
            
            path = [start]
            visited = {start}
            current = start
            
            while True:
                # Find next vertex
                next_v = None
                # Use the shuffled adjacency for this restart
                # We need to filter by local_rem and not visited
                # local_adj contains ALL neighbors, not just remaining ones.
                # We must check if edge is in local_rem.
                
                # Optimization: Keep a set of remaining neighbors for current?
                # For small graphs, linear scan is fine.
                
                found_next = False
                for nb in adj_rand[current]:
                    key = (min(current, nb), max(current, nb))
                    if key in local_rem and nb not in visited:
                        next_v = nb
                        found_next = True
                        break
                        
                if not found_next:
                    break
                
                key = (min(current, next_v), max(current, next_v))
                local_rem.discard(key)
                visited.add(next_v)
                path.append(next_v)
                current = next_v
                
                # Update local_adj? 
                # We don't strictly need to update local_adj for the *current* path extension 
                # if we check `key in local_rem`. 
                # But for the *next* start vertex selection, we need accurate degrees.
                # Updating local_adj for every edge removal is O(E). 
                # Since we do this for every edge, it's O(E^2) per restart.
                # E is ~100. E^2 is 10,000. 5000 restarts is 50M ops. Might be too slow in Python.
                
                # Let's optimize: 
                # Instead of updating local_adj, we can just recompute degrees for start selection?
                # Or just pick a random candidate?
                
                # Let's stick to random candidate for speed, but weighted by degree if possible.
                # Actually, the `max` call above requires accurate `local_adj` counts.
                # Let's update `local_adj` counts.
                local_adj[current].remove(next_v)
                local_adj[next_v].remove(current)
                
            paths.append(path)
            
        # Now we have a set of paths. 
        # 2. Local Search: Merge paths if possible.
        # We can try to merge any two paths P_i, P_j if they form a simple path.
        # We do this repeatedly until no more merges are possible.
        
        changed = True
        while changed:
            changed = False
            # Try to merge every pair
            # To make it faster, we can just check a few random pairs or all pairs.
            # Number of paths is small (e.g., 10-20). Pairs are ~200.
            # Checking simplicity is O(L^2) or O(L) with a set.
            
            n_paths = len(paths)
            for i in range(n_paths):
                if changed: break # Restart loop if a merge happened
                for j in range(i + 1, n_paths):
                    if changed: break
                    
                    P1 = paths[i]
                    P2 = paths[j]
                    
                    # Try 4 combinations of directions
                    # 1. P1 + P2
                    # 2. P1 + P2_reversed
                    # 3. P1_reversed + P2
                    # 4. P1_reversed + P2_reversed
                    
                    merged = None
                    for r1 in [False, True]:
                        for r2 in [False, True]:
                            p1 = P1[::-1] if r1 else P1
                            p2 = P2[::-1] if r2 else P2
                            
                            # Check if they connect
                            # End of p1 must be start of p2
                            if p1[-1] == p2[0]:
                                # Check if union is simple
                                # We know P1 and P2 are simple internally.
                                # We just need to check if P1 and P2 share any vertices.
                                # If they share a vertex, the union is not a simple path.
                                set1 = set(p1)
                                set2 = set(p2)
                                # They share at least the connection vertex p1[-1] == p2[0]
                                # So intersection must be exactly {p1[-1]}
                                if len(set1 & set2) == 1:
                                    merged = p1 + p2[1:]
                                    break
                        if merged: break
                    if merged: break
                    
                    if merged:
                        # Replace paths[i] and paths[j] with merged
                        paths[i] = merged
                        paths.pop(j)
                        changed = True
                        break
                        
        count = len(paths)
        if count < best_count:
            best_count = count
            best_paths = paths
            
    # Verify best_paths
    if best_paths is None:
        # Fallback: each edge as a path
        return [[list(e) for e in edge_set]]
        
    return best_paths

def construct():
    """Return three lists of paths, one per fixed instance graph."""
    deadline = time.time() + 120  # Safety margin
    
    results = []
    for spec in ((20, 0.3, 1), (25, 0.3, 2), (30, 0.25, 3)):
        edges = instance(*spec)
        # Allocate time proportionally or fixed
        # 120s total. 3 instances. 40s each.
        # But the first instance is smaller, might finish faster.
        # Let's just use the global deadline.
        paths = decompose_paths(edges, deadline)
        results.append(paths)
        
    return results

if __name__ == "__main__":
    # Quick test
    res = construct()
    total = sum(len(r) for r in res)
    print(f"Total paths: {total}")
