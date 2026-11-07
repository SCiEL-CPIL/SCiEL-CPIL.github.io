"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import random
import time
from itertools import combinations, permutations

P = {'n': 21, 'edges': [[0, 1], [0, 2], [0, 5], [0, 16], [0, 19], [0, 20], [1, 2], [1, 3], [1, 6], [1, 17], [1, 20], [2, 3], [2, 4], [2, 7], [2, 18], [3, 4], [3, 5], [3, 8], [3, 19], [4, 5], [4, 6], [4, 9], [4, 20], [5, 6], [5, 7], [5, 10], [6, 7], [6, 8], [6, 11], [7, 8], [7, 9], [7, 12], [8, 9], [8, 10], [8, 13], [9, 10], [9, 11], [9, 14], [10, 11], [10, 12], [10, 15], [11, 12], [11, 13], [11, 16], [12, 13], [12, 14], [12, 17], [13, 14], [13, 15], [13, 18], [14, 15], [14, 16], [14, 19], [15, 16], [15, 17], [15, 20], [16, 17], [16, 18], [17, 18], [17, 19], [18, 19], [18, 20], [19, 20]]}

def construct():
    n = P['n']
    edges = P['edges']
    
    # Build adjacency and edge mapping
    adj = {v: [] for v in range(n)}
    edge_index = {}
    for i, (u, v) in enumerate(edges):
        adj[u].append(v)
        adj[v].append(u)
        edge_index[(min(u, v), max(u, v))] = i
        
    # Sort neighbors for deterministic processing
    for v in range(n):
        adj[v].sort()
        
    # Precompute edge list for each vertex, distinguishing < v and > v
    # We will process vertices 0 to 20
    # For vertex v, edges to u < v are fixed by the time we reach v.
    # Edges to u > v are variables.
    
    # State:
    # color[i] = 0, 1, or 2 for edge i. -1 if unassigned.
    colors = [-1] * len(edges)
    # counts[v][c] = number of edges of color c incident to v that are assigned.
    counts = [[0, 0, 0] for _ in range(n)]
    
    # Helper to get edge index
    def get_edge_idx(u, v):
        return edge_index[(min(u, v), max(u, v))]
        
    # Precompute for each vertex v, the list of neighbors > v
    future_neighbors = {v: [] for v in range(n)}
    for v in range(n):
        for u in adj[v]:
            if u > v:
                future_neighbors[v].append(u)
                
    # Precompute for each vertex v, the list of neighbors < v
    past_neighbors = {v: [] for v in range(n)}
    for v in range(n):
        for u in adj[v]:
            if u < v:
                past_neighbors[v].append(u)
                
    def is_feasible(v):
        """Check if the current assignment allows a valid completion for vertices >= v."""
        # We have assigned colors for edges incident to vertices < v.
        # For any vertex u >= v, we need to ensure that the number of 
        # unassigned edges is sufficient to meet the remaining degree requirements.
        
        for u in range(v, n):
            # Total edges incident to u is 6.
            # Assigned edges are those to neighbors < u.
            # Unassigned edges are those to neighbors > u.
            # Note: Edges to neighbors == u don't exist.
            # Edges to neighbors < u are assigned.
            # Edges to neighbors > u are unassigned.
            
            assigned_count = 0
            unassigned_count = 0
            for w in adj[u]:
                if w < u:
                    # This edge is assigned
                    e_idx = get_edge_idx(u, w)
                    # It must be assigned
                    if colors[e_idx] == -1:
                        return False # Should not happen
                    assigned_count += 1
                elif w > u:
                    unassigned_count += 1
            
            # Check counts for each color
            for c in range(3):
                current_count = counts[u][c]
                if current_count > 2:
                    return False
                remaining_needed = 2 - current_count
                if remaining_needed < 0:
                    return False
                # We have unassigned_count edges left to assign colors for u.
                # We need to place remaining_needed edges of color c.
                # Obviously remaining_needed <= unassigned_count
                if remaining_needed > unassigned_count:
                    return False
            # Additional check: The sum of remaining_needed for all colors must equal unassigned_count
            # Sum(2 - counts[u][c]) = 6 - sum(counts[u][c])
            # sum(counts[u][c]) is the number of assigned edges incident to u.
            # This equals assigned_count.
            # So sum(remaining_needed) = 6 - assigned_count = unassigned_count.
            # This identity holds automatically if counts are consistent.
            
        return True

    def check_hamiltonian():
        """Check if the color classes form Hamiltonian cycles."""
        # Build subgraphs for each color
        for c in range(3):
            # Check degree 2 for all vertices
            for v in range(n):
                if counts[v][c] != 2:
                    return False
            
            # Check connectivity: traverse the cycle
            # Start at 0
            visited = set()
            current = 0
            visited.add(0)
            for _ in range(n):
                # Find the two neighbors of current in color c
                neighbors = []
                for u in adj[current]:
                    e_idx = get_edge_idx(current, u)
                    if colors[e_idx] == c:
                        neighbors.append(u)
                if len(neighbors) != 2:
                    return False
                
                # Move to the next unvisited neighbor
                next_node = None
                for nb in neighbors:
                    if nb not in visited:
                        next_node = nb
                        break
                
                if next_node is None:
                    # If both are visited, we must be at the last node and the other must be 0
                    if len(visited) == n:
                        if 0 in neighbors:
                            return True
                        else:
                            return False
                    else:
                        return False
                
                current = next_node
                visited.add(current)
                
            # If we looped back to 0 before visiting all nodes, it's not Hamiltonian
            if len(visited) != n:
                return False
                
        return True

    def backtrack(v):
        if v == n:
            return check_hamiltonian()
        
        # Get edges to future neighbors
        var_edges = []
        for u in future_neighbors[v]:
            var_edges.append((u, get_edge_idx(v, u)))
            
        # Calculate required colors for v
        req = [0, 0, 0]
        for c in range(3):
            req[c] = 2 - counts[v][c]
            
        # Generate all permutations of colors for var_edges that satisfy req
        # var_edges length is L. We need to assign colors such that count of c is req[c].
        # We can generate all unique permutations of the multiset of colors.
        
        # To avoid generating too many, we can use a recursive helper or itertools.permutations
        # with set filtering, but L is small (max 6).
        
        # Create a list of required colors
        color_pool = []
        for c in range(3):
            color_pool.extend([c] * req[c])
            
        # If len(color_pool) != len(var_edges), it's infeasible (should be caught by is_feasible)
        if len(color_pool) != len(var_edges):
            return False
            
        # Generate unique permutations
        # Since L is small, we can use a set of tuples
        seen_assignments = set()
        
        # Sort var_edges for consistent ordering
        var_edges.sort()
        
        for perm in permutations(color_pool):
            if perm in seen_assignments:
                continue
            seen_assignments.add(perm)
            
            # Check if this assignment is valid for the future vertices
            # We need to simulate the assignment and check feasibility for u > v
            
            # Save old state
            old_colors = {}
            old_counts = {}
            
            valid = True
            for i, (u, e_idx) in enumerate(var_edges):
                c = perm[i]
                # Assign
                old_colors[e_idx] = colors[e_idx]
                colors[e_idx] = c
                
                # Update counts for v and u
                # v is already processed in terms of req, but we update counts for consistency
                # Actually, we don't need to update counts[v] as we are done with v
                # But we must update counts[u]
                
                # Check if counts[u][c] exceeds 2
                if counts[u][c] >= 2:
                    valid = False
                    break
                counts[u][c] += 1
                
            if not valid:
                # Rollback
                for e_idx, old_c in old_colors.items():
                    colors[e_idx] = old_c
                # Note: counts[u] was modified. We need to rollback those too.
                # This is getting messy. Let's use a cleaner rollback mechanism.
                
                # Re-implementing with explicit rollback lists
                pass # This block is replaced by the cleaner version below

        # Cleaner Backtracking Step
        assignments = []
        # Generate unique permutations of color_pool
        # To optimize, we can just iterate through permutations and use a set
        # But 6! is 720, which is fine.
        
        # We need to be careful with rollback.
        
        # Let's use a stack of changes
        changes = []
        
        for perm in permutations(color_pool):
            # Skip duplicates
            # We can't easily skip duplicates in permutations generator without a set
            # So we use the set check above
            
            # Apply changes
            valid = True
            local_changes = []
            
            for i, (u, e_idx) in enumerate(var_edges):
                c = perm[i]
                old_c = colors[e_idx]
                local_changes.append((e_idx, old_c))
                colors[e_idx] = c
                
                # Update counts[u]
                if counts[u][c] >= 2:
                    valid = False
                    break
                counts[u][c] += 1
                local_changes.append((u, c, 1)) # marker for count update
                
            if valid:
                # Check feasibility for future vertices
                if is_feasible(v + 1):
                    if backtrack(v + 1):
                        return True
            
            # Rollback
            # We need to reverse the changes
            # The local_changes list is mixed. Let's separate them.
            
        # Re-doing the loop with proper rollback structure
        
        for perm in permutations(color_pool):
            # Unique check
            if perm in seen_assignments:
                continue
            seen_assignments.add(perm)
            
            # Apply
            edge_changes = []
            count_changes = []
            valid = True
            
            for i, (u, e_idx) in enumerate(var_edges):
                c = perm[i]
                edge_changes.append((e_idx, colors[e_idx]))
                colors[e_idx] = c
                
                count_changes.append((u, c))
                if counts[u][c] >= 2:
                    valid = False
                    break
                counts[u][c] += 1
            
            if valid and is_feasible(v + 1):
                if backtrack(v + 1):
                    return True
            
            # Rollback
            for u, c in reversed(count_changes):
                counts[u][c] -= 1
            for e_idx, old_c in reversed(edge_changes):
                colors[e_idx] = old_c
                
        return False

    # Start backtracking
    # Initialize counts to 0
    if backtrack(0):
        # Extract cycles
        cycles = []
        for c in range(3):
            # Reconstruct cycle for color c
            # Start at 0
            cycle = [0]
            current = 0
            visited = {0}
            for _ in range(n - 1):
                # Find next node
                next_node = None
                for u in adj[current]:
                    e_idx = get_edge_idx(current, u)
                    if colors[e_idx] == c and u not in visited:
                        next_node = u
                        break
                if next_node is None:
                    # Should not happen if check_hamiltonian passed
                    break
                cycle.append(next_node)
                visited.add(next_node)
                current = next_node
            cycles.append(cycle)
            
        # Convert to int lists
        result = [[int(v) for v in cycle] for cycle in cycles]
        return result
    
    # Fallback if backtracking fails (should not happen for this graph)
    return [list(range(n))]

# EVOLVE-BLOCK-END
