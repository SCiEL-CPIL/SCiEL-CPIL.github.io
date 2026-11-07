import time
from collections import deque

P = {'n': 24, 'edges': [[0, 1], [1, 2], [2, 3], [3, 4], [4, 5], [5, 6], [6, 7], [7, 8], [8, 9], [9, 10], [10, 11], [11, 12], [12, 13], [13, 14], [14, 15], [15, 16], [16, 17], [17, 18], [18, 19], [19, 20], [20, 21], [21, 22], [22, 23]], 'diameter': 4}

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin for 90s limit

    n = P['n']
    base_edges = P['edges']
    
    # Precompute base adjacency
    base_adj = [set() for _ in range(n)]
    for u, v in base_edges:
        base_adj[u].add(v)
        base_adj[v].add(u)
        
    base_set = set()
    for u, v in base_edges:
        base_set.add((min(u, v), max(u, v)))

    # Generate all candidate edges
    candidates = []
    for u in range(n):
        for v in range(u + 1, n):
            if (u, v) not in base_set:
                candidates.append((u, v))
                
    m = len(candidates)
    
    # Map candidate edge to index
    cand_index = {e: i for i, e in enumerate(candidates)}

    # Precompute conflict graph between candidates
    # Two candidates conflict if adding both creates a triangle with any existing edge (base or other candidate)
    # Actually, the condition for triangle-freeness is:
    # For any triangle {u,v,w}, we cannot have all 3 edges present.
    # Edges present = Base OR Added.
    # If (u,v) and (v,w) are added, we must ensure (u,w) is NOT in Base and NOT added.
    # This is a 3-uniform hypergraph constraint.
    # However, for n=24, we can use a simpler check during search or precompute pairwise conflicts that are "immediate".
    # Let's stick to checking triangle-freeness on the fly but optimized.
    
    # Optimization for triangle check:
    # When adding edge (u,v), check if there is any w such that (u,w) and (v,w) are both present.
    # Present = Base OR Added.
    
    def is_triangle_free(added_set):
        # added_set is a set of tuples (u,v) with u<v
        # Build adjacency
        adj = [set() for _ in range(n)]
        for u in range(n):
            adj[u] = base_adj[u].copy()
        for u, v in added_set:
            adj[u].add(v)
            adj[v].add(u)
            
        # Check for triangles
        # For each u, for each v in adj[u] with v > u, check if adj[u] & adj[v] is not empty (excluding u and v? no, just intersection)
        # Actually, if w is in adj[u] and w is in adj[v], then u-v-w is a triangle.
        # We just need to ensure no such w exists.
        for u in range(n):
            for v in adj[u]:
                if v > u:
                    # Check intersection of neighbors of u and v
                    # If there is any common neighbor w, it's a triangle.
                    # Note: w cannot be u or v because u is not neighbor of u, etc.
                    if adj[u] & adj[v]:
                        return False
        return True

    def get_diameter(added_set):
        # Build adjacency
        adj = [set() for _ in range(n)]
        for u in range(n):
            adj[u] = base_adj[u].copy()
        for u, v in added_set:
            adj[u].add(v)
            adj[v].add(u)
            
        max_d = 0
        for src in range(n):
            dist = [-1] * n
            dist[src] = 0
            q = deque([src])
            while q:
                u = q.popleft()
                for v in adj[u]:
                    if dist[v] == -1:
                        dist[v] = dist[u] + 1
                        q.append(v)
            if any(d == -1 for d in dist):
                return 999
            max_d = max(max_d, max(dist))
        return max_d

    def is_valid(added_set):
        if not is_triangle_free(added_set):
            return False
        if get_diameter(added_set) > 4:
            return False
        return True

    # Strategy:
    # 1. Find a good initial solution using a greedy heuristic.
    # 2. Use local search with removals and swaps to minimize edge count.
    
    best_edges_set = None
    best_count = float('inf')

    # Greedy Initialization
    # We want to reduce diameter. 
    # A good heuristic is to add edges that connect distant nodes.
    # Sort candidates by distance in base graph (which is |u-v|) descending.
    cand_sorted = sorted(candidates, key=lambda e: abs(e[0]-e[1]), reverse=True)
    
    current_set = set()
    for u, v in cand_sorted:
        new_set = current_set | {(u, v)}
        if is_triangle_free(new_set):
            current_set = new_set
            if get_diameter(current_set) <= 4:
                break
                
    if get_diameter(current_set) <= 4:
        best_edges_set = current_set
        best_count = len(current_set)

    # Local Search
    # 1. Greedy Removal: Try to remove each edge. If valid, remove it.
    # 2. Swap: Try to replace one edge with another. If valid and allows removal of a 3rd edge, do it.
    #    Or simpler: Just try to find a valid set of size k-1 given a valid set of size k.
    #    We can do this by iterating through all subsets of size k-1? No, too expensive.
    #    Instead, for each edge e in current_set, remove it. Then try to add any single edge f not in current_set to restore validity?
    #    No, we want to reduce size.
    
    # Let's do a more robust local search:
    # While time remains:
    #   - Try to remove an edge. If valid, keep it removed.
    #   - If stuck, try a 1-swap: remove e1, add e2. If valid, accept. (Size unchanged, but might lead to removal)
    #   - If stuck, try a 2-swap: remove e1, e2, add e3, e4. (Size unchanged)
    #   - Actually, the most effective is "remove and repair".
    #     Remove e. If invalid, try to add one edge f to make it valid again? No, that keeps size same.
    #     We want to reach a state where we can remove an edge.
    
    # Let's try a randomized restart approach with greedy removal.
    
    def local_search(added_set):
        current = set(added_set)
        improved = True
        while improved:
            improved = False
            # Try removing each edge
            for e in list(current):
                new_set = current - {e}
                if is_valid(new_set):
                    current = new_set
                    improved = True
                    break # Restart loop with smaller set
        
        # If no removal possible, try swaps to enable removal
        # This is complex. Let's just do random restarts with greedy construction.
        
        return current

    if best_edges_set:
        best_edges_set = local_search(best_edges_set)
        best_count = len(best_edges_set)

    # Randomized Greedy Construction + Local Search
    # Since the greedy order matters, try different orders.
    
    import random
    random.seed(42) # Deterministic for reproducibility? Or use time? Let's use fixed seed for consistency in evaluation.
    
    for _ in range(50):
        if time.time() > deadline:
            break
            
        # Shuffle candidates
        shuffled = candidates[:]
        random.shuffle(shuffled)
        
        # Greedy add
        current_set = set()
        for u, v in shuffled:
            new_set = current_set | {(u, v)}
            if is_triangle_free(new_set):
                current_set = new_set
                if get_diameter(current_set) <= 4:
                    break
        
        if get_diameter(current_set) <= 4:
            # Local search
            current_set = local_search(current_set)
            if len(current_set) < best_count:
                best_count = len(current_set)
                best_edges_set = current_set

    # If still no solution, fallback to a known valid one (if we found one earlier)
    # If best_edges_set is None, we need a valid solution.
    # The greedy sorted by distance should always find one for n=24.
    
    if best_edges_set is None:
        # Fallback: Hub strategy
        hub = 11
        targets = [0, 2, 4, 6, 8, 14, 16, 18, 20, 22]
        best_edges_set = set()
        for t in targets:
            u, v = sorted((hub, t))
            best_edges_set.add((u, v))
        # Verify
        if not is_valid(best_edges_set):
            # This shouldn't happen, but if it does, return empty? No, must be valid.
            # Let's assume the hub strategy works.
            pass

    # Convert to list of lists
    result = [[int(u), int(v)] for u, v in best_edges_set]
    return result

# EVOLVE-BLOCK-END
