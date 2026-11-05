import time
import random
import sys

def has_c5_after_adding_edge(adj, u, v):
    """
    Checks if adding edge (u, v) to the graph with adjacency sets 'adj'
    creates a 5-cycle.
    """
    nu = adj[u]
    nv = adj[v]
    
    # We look for a path u-a-b-c-v where a in N(u), c in N(v), b in N(a) intersect N(c)
    # and u,v,a,b,c are distinct.
    # Since (u,v) is not yet an edge, v not in nu and u not in nv.
    # a is in nu, so a != u.
    # c is in nv, so c != v.
    # We need a != c.
    # b must be in N(a) and N(c).
    # b must not be u, v, a, c.
    
    for a in nu:
        if a == v: # Should not happen if graph is simple and (u,v) not added
            continue
        for c in nv:
            if a == c:
                continue
            if c == u: # Should not happen
                continue
            
            # Find b in N(a) intersect N(c)
            # N(a) and N(c) are sets
            common = adj[a] & adj[c]
            # Remove u, v, a, c from common
            # We can just check if any element in common is not in {u, v, a, c}
            # Since n is small, iterating is fine
            for b in common:
                if b != u and b != v and b != a and b != c:
                    return True
    return False

def has_independent_set_9(adj, n):
    """
    Returns True if the graph has an independent set of size 9.
    Uses backtracking with pruning.
    """
    # Convert adj to a list of sets for easy intersection if needed, 
    # but adj is already list of sets.
    
    # Precompute degrees for initial sorting? 
    # We will sort dynamically in the recursion.
    
    def greedy_coloring_upper_bound(candidates):
        """
        Returns an upper bound on the size of the maximum independent set
        in the subgraph induced by 'candidates' using greedy coloring.
        """
        if not candidates:
            return 0
        
        # Sort vertices by degree in the subgraph induced by candidates
        # Higher degree first is a common heuristic for coloring
        sub_degrees = {v: len(adj[v] & candidates) for v in candidates}
        order = sorted(candidates, key=lambda x: sub_degrees[x], reverse=True)
        
        colors = {}
        max_color = -1
        
        for v in order:
            # Get colors of neighbors of v that are already colored
            used_colors = set()
            for u in adj[v]:
                if u in colors:
                    used_colors.add(colors[u])
            
            # Find the smallest available color
            c = 0
            while c in used_colors:
                c += 1
            colors[v] = c
            if c > max_color:
                max_color = c
        
        return max_color + 1

    def dfs(current_size, candidates):
        if current_size == 9:
            return True
        
        if current_size + len(candidates) < 9:
            return False
        
        # Pruning with greedy coloring
        ub = greedy_coloring_upper_bound(candidates)
        if current_size + ub < 9:
            return False
        
        if not candidates:
            return False
            
        # Select a vertex to branch on
        # Heuristic: vertex with minimum degree in the subgraph induced by candidates
        # This tends to branch less
        min_deg = float('inf')
        v = None
        for x in candidates:
            d = len(adj[x] & candidates)
            if d < min_deg:
                min_deg = d
                v = x
                
        rest = candidates - {v}
        
        # Option 1: Include v
        # New candidates are neighbors of v removed from rest
        new_candidates = rest - adj[v]
        if dfs(current_size + 1, new_candidates):
            return True
            
        # Option 2: Exclude v
        if dfs(current_size, rest):
            return True
            
        return False

    # Start with all vertices
    all_vertices = set(range(n))
    return dfs(0, all_vertices)

def construct():
    start_time = time.time()
    deadline = start_time + 145  # 5s margin
    
    # Try n = 35, 34, 33, 32
    for n in [35, 34, 33, 32]:
        if time.time() > deadline:
            break
            
        # Number of attempts depends on n
        # For n=35, we might need more attempts if it's hard
        max_attempts = 50
        
        for attempt in range(max_attempts):
            if time.time() > deadline:
                break
                
            random.seed(attempt)
            adj = [set() for _ in range(n)]
            
            # Generate all possible edges
            edges = [(i, j) for i in range(n) for j in range(i + 1, n)]
            random.shuffle(edges)
            
            # Greedy add edges
            for u, v in edges:
                if time.time() > deadline:
                    break
                if v in adj[u]:
                    continue
                if has_c5_after_adding_edge(adj, u, v):
                    continue
                adj[u].add(v)
                adj[v].add(u)
                
            # Check if valid: no C5 (guaranteed by construction) and alpha < 9
            # We need to check if alpha >= 9. If it does, it's invalid.
            if not has_independent_set_9(adj, n):
                # Valid graph found
                edge_list = []
                for i in range(n):
                    for j in adj[i]:
                        if i < j:
                            edge_list.append([i, j])
                return {'n': n, 'edges': edge_list}
                
    # Fallback to n=32 if nothing found
    # The seed program returns a specific n=32 graph.
    # We can construct a simple n=32 graph: 8 disjoint K4s.
    # K4 is C5-free.
    # Independent set in disjoint union is sum of independent sets.
    # alpha(K4) = 1.
    # alpha(8 * K4) = 8.
    # 8 < 9. So this is valid.
    n = 32
    edges = []
    for block_idx in range(8):
        start = block_idx * 4
        verts = [start, start + 1, start + 2, start + 3]
        for i in range(len(verts)):
            for j in range(i + 1, len(verts)):
                edges.append([verts[i], verts[j]])
    return {'n': n, 'edges': edges}
