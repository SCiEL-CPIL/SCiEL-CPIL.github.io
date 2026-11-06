import itertools
import random
import sys

def construct():
    """
    Constructs a C4-free graph on up to 43 vertices with independence number <= 10.
    Starts with the polarity graph of PG(2,5) and greedily adds vertices.
    """
    q = 5
    
    # 1. Generate base graph (Polarity graph of PG(2,5))
    points = []
    for y in range(q):
        for z in range(q):
            points.append((1, y, z))
    for z in range(q):
        points.append((0, 1, z))
    points.append((0, 0, 1))
    
    n_base = len(points)
    point_to_idx = {p: i for i, p in enumerate(points)}
    
    edges = []
    # Adjacency list
    adj = [set() for _ in range(n_base)]
    
    for i in range(n_base):
        for j in range(i + 1, n_base):
            u = points[i]
            v = points[j]
            dot = (u[0] * v[0] + u[1] * v[1] + u[2] * v[2]) % q
            if dot == 0:
                edges.append([i, j])
                adj[i].add(j)
                adj[j].add(i)
                
    current_n = n_base
    current_adj = [set(s) for s in adj] # Copy
    
    # Helper: Check if a set S is C4-compatible in the current graph
    # S is C4-compatible if for all u != v in S, N(u) and N(v) are disjoint.
    def is_c4_compatible(S):
        S_list = list(S)
        for i in range(len(S_list)):
            u = S_list[i]
            for j in range(i+1, len(S_list)):
                v = S_list[j]
                if current_adj[u].intersection(current_adj[v]):
                    return False
        return True

    # Helper: Find Maximum Independent Set size in induced subgraph on vertices V_set
    # Optimized Branch and Bound using bitsets for speed
    def max_independent_set_size(V_set):
        if not V_set:
            return 0
        
        # Map vertices to 0..m-1 for bitset operations
        V_list = list(V_set)
        m = len(V_list)
        if m == 0:
            return 0
        
        # Build adjacency bitmasks for the subgraph
        # adj_mask[i] is a bitmask of neighbors of V_list[i] within V_set
        index_map = {v: i for i, v in enumerate(V_list)}
        adj_masks = [0] * m
        for i, u in enumerate(V_list):
            mask = 0
            for v in current_adj[u]:
                if v in index_map:
                    mask |= (1 << index_map[v])
            adj_masks[i] = mask
            
        best = 0
        
        def dfs(candidates, count):
            nonlocal best
            if not candidates:
                if count > best:
                    best = count
                return
            if count + candidates.bit_length() <= best:
                return
            
            # Pick a vertex with high degree in the candidate set for better pruning
            # Simple heuristic: pick the one with max degree in candidates
            # Or just pick the lowest set bit for determinism/speed
            # Let's use a greedy coloring bound or just simple bit manipulation
            
            # Find a vertex with max degree in the induced subgraph of candidates
            # This is expensive to compute exactly every step.
            # Let's just pick the first available vertex (lowest bit) or use a simple heuristic.
            # For small m, simple is fine.
            
            # Greedy lower bound update?
            # Let's just branch on the vertex with the highest degree in the candidate set
            # to prune "Include" branch more effectively?
            # Actually, branching on high degree vertices reduces the candidate set more.
            
            # Calculate degrees in candidate set
            # deg[i] = (adj_masks[i] & candidates).bit_count()
            
            # To save time, just pick a vertex. 
            # A good heuristic is to pick the vertex with the highest degree.
            max_deg = -1
            pivot = -1
            # Iterate through set bits
            c = candidates
            while c:
                i = (c & -c).bit_length() - 1
                c &= c - 1
                deg = (adj_masks[i] & candidates).bit_count()
                if deg > max_deg:
                    max_deg = deg
                    pivot = i
            
            if pivot == -1:
                return
                
            # Option 1: Include pivot
            # Remove pivot and its neighbors from candidates
            new_candidates = candidates & ~(adj_masks[pivot] | (1 << pivot))
            dfs(new_candidates, count + 1)
            
            # Option 2: Exclude pivot
            dfs(candidates & ~(1 << pivot), count)

        # Start with all vertices
        all_mask = (1 << m) - 1
        dfs(all_mask, 0)
        return best

    # Helper: Try to add a new vertex with neighbor set S
    def try_add_vertex(S):
        nonlocal current_n, current_adj, edges
        if not is_c4_compatible(S):
            return False
        
        # Check independence constraint
        # The new graph has vertices 0..current_n
        # New vertex is current_n
        # Independent sets containing new vertex: {current_n} U I, where I is independent in G[V \ S]
        # So we need alpha(G[V \ S]) <= 9
        V_set = set(range(current_n)) - set(S)
        if max_independent_set_size(V_set) > 9:
            return False
            
        # Valid! Add the vertex
        new_v = current_n
        current_adj.append(set(S))
        for u in S:
            current_adj[u].add(new_v)
            edges.append([u, new_v])
        current_n += 1
        return True

    # Helper: Find a Maximum Independent Set in the Conflict Graph
    # Conflict Graph C: u~v if N(u) cap N(v) != empty
    def find_max_independent_set_in_conflict_graph():
        n = current_n
        # Build conflict graph adjacency
        conflict_adj = [set() for _ in range(n)]
        for i in range(n):
            for j in range(i + 1, n):
                if current_adj[i].intersection(current_adj[j]):
                    conflict_adj[i].add(j)
                    conflict_adj[j].add(i)
        
        # Use a simple backtracking to find a large independent set
        # We want to maximize the size of S
        best_set = set()
        
        def dfs(remaining, current_set):
            nonlocal best_set
            if not remaining:
                if len(current_set) > len(best_set):
                    best_set = set(current_set)
                return
            if len(current_set) + len(remaining) <= len(best_set):
                return
            
            # Pick a vertex
            # Heuristic: pick vertex with max degree in conflict graph restricted to remaining
            # to prune "Include" branch?
            # Actually, for Independent Set, we want to pick a vertex that allows us to keep many others?
            # Standard: pick vertex, branch on Include (remove neighbors) and Exclude.
            
            # To speed up, we can use a greedy coloring bound on the remaining graph's complement?
            # Or just simple B&B.
            # Since n <= 43, this is small.
            
            # Pick first vertex
            v = next(iter(remaining))
            
            # Option 1: Include v
            # Remove v and its conflict-neighbors from remaining
            new_remaining = remaining - conflict_adj[v] - {v}
            dfs(new_remaining, current_set | {v})
            
            # Option 2: Exclude v
            dfs(remaining - {v}, current_set)

        dfs(set(range(n)), set())
        return best_set

    # Greedy addition loop
    for _ in range(43 - current_n):
        added = False
        
        # Strategy: Find the Maximum Independent Set in the Conflict Graph
        # This gives the largest possible S that is C4-compatible.
        S_candidate = find_max_independent_set_in_conflict_graph()
        
        # It's possible that the MIS is not the one that minimizes alpha(V\S).
        # But it's the best first guess.
        if try_add_vertex(S_candidate):
            added = True
        else:
            # If the MIS fails, maybe a smaller IS works?
            # Unlikely, but we can try to find another IS.
            # For now, we stop if the MIS fails.
            # This is a limitation, but often the MIS is the only viable option.
            pass
            
        if not added:
            break

    return {'n': current_n, 'edges': edges}
