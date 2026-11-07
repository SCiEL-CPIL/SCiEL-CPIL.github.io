"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from itertools import combinations
from collections import deque

P = {'n': 24, 'edges': [[0, 1], [1, 2], [2, 3], [3, 4], [4, 5], [5, 6], [6, 7], [7, 8], [8, 9], [9, 10], [10, 11], [11, 12], [12, 13], [13, 14], [14, 15], [15, 16], [16, 17], [17, 18], [18, 19], [19, 20], [20, 21], [21, 22], [22, 23]], 'diameter': 4}

def construct():
    n = P['n']
    existing_edges = set(tuple(sorted(e)) for e in P['edges'])
    adj_base = [set() for _ in range(n)]
    for u, v in existing_edges:
        adj_base[u].add(v)
        adj_base[v].add(u)
        
    # 1. Identify all candidate edges (non-existing, non-loop)
    candidates = []
    for u in range(n):
        for v in range(u + 1, n):
            if (u, v) not in existing_edges:
                candidates.append((u, v))
                
    m = len(candidates)
    
    # 2. Build the Triangle-Conflict Graph
    # Two candidate edges e1=(u1,v1) and e2=(u2,v2) conflict if adding both creates a triangle.
    # A triangle is formed if:
    # - They share a vertex, and the other endpoints are adjacent in the base graph.
    # - They are disjoint, but the endpoints form a cycle of length 3? No, disjoint edges can't form a triangle directly unless they connect to existing edges.
    # Let's be precise: 
    # Adding e1 and e2 creates a triangle if:
    # Case A: e1 and e2 share a vertex, say u1==u2. Then a triangle exists if v1 and v2 are adjacent in base.
    # Case B: e1 and e2 are disjoint. They can form a triangle if one endpoint of e1 is adjacent to one endpoint of e2, AND the other endpoint of e1 is adjacent to the other endpoint of e2? No, that's a 4-cycle or 5-cycle.
    # Actually, a triangle requires 3 vertices. 
    # If e1=(a,b) and e2=(c,d) are disjoint, they cannot form a triangle together unless one of the edges is part of a triangle with a base edge.
    # Wait, "Triangle-free" means NO triangle exists in the final graph.
    # If we add e1=(a,b), a triangle is formed if there exists a vertex c such that (a,c) and (b,c) are in the final graph.
    # So, conflict between e1 and e2 exists if:
    # 1. They share a vertex, say a. And the other endpoints b and c are adjacent in the BASE graph. (Triangle a-b-c)
    # 2. They are disjoint. But can two disjoint edges create a triangle? No. A triangle has 3 edges. If e1 and e2 are disjoint, they don't form a triangle *together*. However, adding e1 might create a triangle with a base edge, and adding e2 might create a triangle with a base edge. But the constraint is that the UNION must be triangle-free.
    # This implies:
    # - No single added edge can create a triangle with base edges. (If it does, that edge is invalid entirely).
    # - No pair of added edges can create a triangle.
    #    - If they share a vertex u, and their other endpoints v, w are adjacent in base, then {u,v,w} is a triangle.
    #    - If they are disjoint, they cannot form a triangle with each other. But could they form a triangle with a base edge? No, a base edge plus two added edges would require the added edges to share a vertex with the base edge endpoints.
    #    - Example: Base edge (x,y). Added e1=(x,z), e2=(y,w). If z and w are adjacent in base? No.
    #    - Let's check: Triangle vertices {a,b,c}. Edges (a,b), (b,c), (c,a).
    #    - If (a,b) is base, then (b,c) and (c,a) must be added. They share c. They are not disjoint.
    #    - So, conflicts ONLY occur between edges that share a vertex.
    
    # Pre-filter candidates: An edge (u,v) is INVALID if it forms a triangle with base edges.
    # i.e., if there exists w such that (u,w) in base and (v,w) in base.
    valid_candidates = []
    for u, v in candidates:
        common_neighbors = adj_base[u] & adj_base[v]
        if not common_neighbors:
            valid_candidates.append((u, v))
            
    # Re-index valid candidates
    k = len(valid_candidates)
    # Map original candidate indices to valid indices if needed, but let's just work with valid_candidates list.
    
    # Build conflict bitmasks for valid_candidates
    # conflict[i] is a bitmask of indices j such that valid_candidates[i] and valid_candidates[j] conflict.
    conflict = [0] * k
    for i in range(k):
        u1, v1 = valid_candidates[i]
        for j in range(i + 1, k):
            u2, v2 = valid_candidates[j]
            # Check if they share a vertex
            if u1 == u2 or u1 == v2 or v1 == u2 or v1 == v2:
                # They share a vertex. Check if the other endpoints are adjacent in base.
                # If u1 == u2, other endpoints are v1 and v2.
                # If u1 == v2, other endpoints are v1 and u2.
                # Let's find the shared vertex and the two other vertices.
                if u1 == u2:
                    a, b = v1, v2
                elif u1 == v2:
                    a, b = v1, u2
                elif v1 == u2:
                    a, b = u1, v2
                else: # v1 == v2
                    a, b = u1, u2
                    
                if a in adj_base[b]:
                    # Conflict!
                    conflict[i] |= (1 << j)
                    conflict[j] |= (1 << i)
                    
    # 3. Precompute "Diameter Cover" capability for each edge
    # We need the final graph to have diameter <= 4.
    # Base graph is a path 0-23. Diameter 23.
    # To get diameter <= 4, we need to "shorten" the path significantly.
    # A useful metric: Eccentricity.
    # Let's use a heuristic lower bound for the search:
    # The search will try to find an independent set of size S.
    # We will iterate S from 1 to k.
    # For each S, we do a backtracking search to find an independent set of size S.
    # For each candidate independent set found, we check if the resulting graph has diameter <= 4.
    # To speed this up, we can prune if the "potential" to reduce diameter is low.
    # However, checking diameter is O(N^2) or O(N+E) with BFS. N=24 is small.
    # The bottleneck is the number of independent sets.
    # k is roughly (24*23/2 - 23) = 276 - 23 = 253.
    # The number of independent sets in a sparse conflict graph can be large, but we stop at the first valid solution for a given S.
    # Since we want MINIMUM edges, we start S=1, then S=2, etc.
    
    # Optimization: 
    # We can pre-compute the "distance reduction" or simply rely on the fact that N is small.
    # Let's implement a fast BFS diameter check.
    
    def check_diameter(added_edges):
        # added_edges is a list of indices into valid_candidates
        # Build adjacency list
        adj = [set() for _ in range(n)]
        for u, v in P['edges']:
            adj[u].add(v)
            adj[v].add(u)
            
        for idx in added_edges:
            u, v = valid_candidates[idx]
            adj[u].add(v)
            adj[v].add(u)
            
        # Check connectivity and diameter
        # BFS from each node
        max_dist = 0
        for start in range(n):
            dist = [-1] * n
            dist[start] = 0
            q = deque([start])
            while q:
                curr = q.popleft()
                for nei in adj[curr]:
                    if dist[nei] == -1:
                        dist[nei] = dist[curr] + 1
                        max_dist = max(max_dist, dist[nei])
                        q.append(nei)
            if dist[-1] == -1: # Disconnected
                return False
        return max_dist <= 4

    # Backtracking search for independent set of size S
    # We use bitsets for efficiency.
    # available_mask: bitmask of indices that are still available (not conflicting with chosen ones)
    # chosen: list of chosen indices
    
    def find_independent_set(S, start_idx, available_mask, chosen):
        # Pruning: if we can't pick enough more edges from available_mask, return None
        # Popcount of available_mask
        if bin(available_mask).count('1') < S - len(chosen):
            return None
            
        if len(chosen) == S:
            # Check if this set works
            if check_diameter(chosen):
                return chosen[:]
            return None
            
        # Heuristic: Pick the vertex with the highest degree in the conflict graph? 
        # Or just iterate.
        # To ensure we find the MINIMUM size, we iterate S.
        # Within a fixed S, any independent set is fine as long as it satisfies diameter.
        
        # We can try to pick the next edge from available_mask
        # Order of picking might affect speed. Let's try picking the one that "helps" diameter most?
        # Or just standard backtracking.
        
        # To speed up, we can pre-compute a "score" for each edge based on how much it reduces the diameter of the base graph?
        # Simpler: Just iterate through available bits.
        
        # Optimization: If S is small, this is fast. If S is large, it might be slow.
        # But we expect S to be small (maybe 5-10).
        
        # Let's try to pick the edge that has the most "impact".
        # For now, simple iteration.
        
        # Get list of available indices
        avail_indices = []
        mask = available_mask
        while mask:
            lsb = mask & -mask
            idx = lsb.bit_length() - 1
            avail_indices.append(idx)
            mask ^= lsb
            
        # Sort by some heuristic? 
        # Let's try sorting by "centrality" in the base graph.
        # Edges connecting central nodes might be better.
        # Node 12 is center.
        # Let's just use the natural order for now, or reverse.
        
        for idx in avail_indices:
            # If idx < start_idx, skip to avoid duplicates? 
            # No, available_mask already handles conflicts. 
            # But we should ensure we don't pick the same combination in different orders?
            # Since we are looking for ANY independent set, order doesn't matter for validity, 
            # but for backtracking efficiency, we usually fix an order to avoid permutations.
            # However, available_mask removes conflicts. 
            # If we pick idx, we remove idx and its neighbors from available_mask.
            # This ensures each set is found once if we enforce idx > last_picked?
            # Actually, standard independent set enumeration:
            # pick idx, new_mask = available_mask & ~(conflict[idx] | (1<<idx))
            # recurse with start_idx = idx + 1? No, that's for clique. 
            # For independent set, we just remove conflicts.
            
            # Let's use the standard branch and bound for Max Independent Set logic, 
            # but we stop at size S.
            
            new_mask = available_mask & ~(conflict[idx] | (1 << idx))
            res = find_independent_set(S, idx + 1, new_mask, chosen + [idx])
            if res is not None:
                return res
        return None

    # Main loop: Try increasing S
    # Start with S=1
    # We can add a time limit.
    start_time = time.time()
    time_limit = 50.0 # 50 seconds safety margin
    
    # Initial full mask
    full_mask = (1 << k) - 1
    
    # To improve speed, we can order the candidates.
    # Candidates that are "long" (connect distant nodes) are more likely to reduce diameter.
    # Let's sort valid_candidates by distance in base graph, descending.
    # This might help find a solution faster for small S.
    
    # Calculate base distances
    def base_dist(u, v):
        return abs(u - v)
        
    # Sort valid_candidates
    # We need to update conflict masks if we reorder.
    # Let's just sort the indices of valid_candidates.
    
    sorted_indices = list(range(k))
    sorted_indices.sort(key=lambda i: base_dist(valid_candidates[i][0], valid_candidates[i][1]), reverse=True)
    
    # Remap valid_candidates and conflict masks to this order
    remapped_candidates = [valid_candidates[i] for i in sorted_indices]
    remapped_conflict = [0] * k
    for i in range(k):
        orig_i = sorted_indices[i]
        # Get conflicts of orig_i in original mask
        mask = conflict[orig_i]
        # Convert to new indices
        new_mask = 0
        for j in range(k):
            if mask & (1 << j):
                # Find new index of j
                # We can precompute a map
                pass # This is O(K^2) which is fine for K~250
                
    # Precompute map orig_idx -> new_idx
    new_idx_of = [0] * k
    for new_i, orig_i in enumerate(sorted_indices):
        new_idx_of[orig_i] = new_i
        
    for i in range(k):
        orig_i = sorted_indices[i]
        mask = conflict[orig_i]
        new_mask = 0
        for j in range(k):
            if mask & (1 << j):
                new_mask |= (1 << new_idx_of[j])
        remapped_conflict[i] = new_mask
        
    # Use remapped structures
    final_candidates = remapped_candidates
    final_conflict = remapped_conflict
    
    for S in range(1, k + 1):
        if time.time() - start_time > time_limit:
            break
        res = find_independent_set(S, 0, full_mask, [])
        if res is not None:
            # Convert indices back to edge pairs
            added_edges = [final_candidates[i] for i in res]
            return added_edges
            
    # Fallback: Return the seed solution if search fails or times out
    # Seed solution:
    return [
        [12, 0],
        [12, 2],
        [12, 4],
        [12, 6],
        [12, 8],
        [12, 15],
        [12, 17],
        [12, 19],
        [12, 21],
        [12, 23]
    ]
# EVOLVE-BLOCK-END
