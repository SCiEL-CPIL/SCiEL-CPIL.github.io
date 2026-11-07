import time
import sys

# EVOLVE-BLOCK-START
P = {'n': 14, 'edges': [[0, 2], [0, 3], [0, 4], [0, 5], [0, 6], [0, 10], [0, 12], [1, 4], [1, 9], [1, 11], [2, 3], [2, 5], [2, 6], [2, 7], [2, 12], [2, 13], [3, 4], [3, 5], [3, 8], [3, 11], [3, 12], [3, 13], [4, 5], [4, 7], [4, 13], [5, 8], [5, 9], [5, 10], [5, 12], [5, 13], [6, 8], [6, 11], [6, 13], [7, 12], [8, 9], [8, 13], [9, 11], [9, 12], [11, 13]]}

def construct():
    n = P['n']
    edges = P['edges']
    
    # Build adjacency matrix
    adj = [[False] * n for _ in range(n)]
    for u, v in edges:
        adj[u][v] = True
        adj[v][u] = True
        
    # Build complement adjacency matrix
    comp_adj = [[False] * n for _ in range(n)]
    for i in range(n):
        for j in range(n):
            if i != j and not adj[i][j]:
                comp_adj[i][j] = True

    # Helper: Check if a set of vertices is a clique in adj
    def is_clique(vertices, mat):
        for i in range(len(vertices)):
            for j in range(i + 1, len(vertices)):
                if not mat[vertices[i]][vertices[j]]:
                    return False
        return True

    # Helper: Check if a set of vertices is a valid part (clique in G or clique in Gbar)
    def is_valid_part(vertices):
        if not vertices:
            return True
        return is_clique(vertices, adj) or is_clique(vertices, comp_adj)

    # Sort vertices by degree in G (descending) to prune earlier
    degrees = [sum(1 for j in range(n) if adj[i][j]) for i in range(n)]
    sorted_vertices = sorted(range(n), key=lambda x: degrees[x], reverse=True)
    
    # Map original index to sorted index for easier lookup if needed, 
    # but we just process in sorted order.
    
    best_parts = n  # Upper bound
    best_solution = None
    
    # Precompute valid extensions? 
    # For a given part P and vertex v, check if P U {v} is valid.
    
    def dfs(idx, parts):
        nonlocal best_parts, best_solution
        
        # Pruning
        if len(parts) >= best_parts:
            return
            
        if idx == n:
            best_parts = len(parts)
            best_solution = [list(p) for p in parts]
            return
            
        # Lower bound: remaining vertices / max possible part size
        # Max part size is at most n, but usually smaller.
        # A simple bound: remaining vertices must be covered by at least 
        # ceil(remaining / max_clique_or_indep_size) parts.
        # Let's just use remaining vertices count / 1 as a weak bound, 
        # or calculate max size of clique/independent set in remaining graph.
        # For n=14, this is fast enough without complex bounds.
        
        v = sorted_vertices[idx]
        
        # Try adding v to each existing part
        for i in range(len(parts)):
            new_part = parts[i] + [v]
            if is_valid_part(new_part):
                # Create new list of parts
                new_parts = [list(p) for p in parts]
                new_parts[i] = new_part
                dfs(idx + 1, new_parts)
                
        # Try creating a new part with v
        # Symmetry breaking: 
        # We only create a new part if we haven't already created a new part 
        # in this recursion level? 
        # Actually, since parts are unlabeled, the order of parts in the list 
        # is arbitrary. 
        # However, by always appending new parts, we distinguish them by creation time.
        # Is there a symmetry we are missing?
        # If we have parts [A, B], and we add v to A -> [A+v, B].
        # If we add v to B -> [A, B+v].
        # These are distinct states in our search tree.
        # But [A+v, B] is equivalent to [B, A+v].
        # Since we always append, the "new" part is always at the end.
        # So [A, B, {v}] is the only way to get a 3rd part.
        # The symmetry is that [A, B] is same as [B, A].
        # Our DFS explores [A, B] and [B, A] separately?
        # No, because we process vertices in a fixed order.
        # v0 goes to Part 0.
        # v1 goes to Part 0 or Part 1.
        # If v1 goes to Part 1, Part 1 is created.
        # The state is defined by the sets.
        # If we swap Part 0 and Part 1, the history is different, but the current sets are swapped.
        # To avoid this, we can enforce that parts are in a canonical order (e.g., lexicographic).
        # But checking canonical order at each step is expensive.
        # Given n=14, the factor of symmetry is at most 14! which is huge, 
        # but the branching factor is low.
        # Let's try without canonical ordering first. If it's slow, add it.
        
        # Add v to a new part
        new_parts = [list(p) for p in parts]
        new_parts.append([v])
        dfs(idx + 1, new_parts)

    dfs(0, [])
    
    # Convert best_solution to list of lists of ints
    # Ensure types are Python int
    if best_solution:
        result = []
        for part in best_solution:
            result.append([int(x) for x in part])
        return result
    else:
        # Fallback (should not happen)
        return [[i] for i in range(n)]

# EVOLVE-BLOCK-END
