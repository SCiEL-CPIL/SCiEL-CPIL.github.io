import time
import sys

# EVOLVE-BLOCK-START
P = {'n': 30, 'edges': [[0, 4], [0, 7], [0, 9], [0, 10], [0, 11], [0, 16], [0, 17], [0, 18], [0, 19], [0, 23], [0, 24], [0, 26], [0, 27], [0, 28], [1, 2], [1, 4], [1, 6], [1, 8], [1, 10], [1, 11], [1, 12], [1, 14], [1, 15], [1, 16], [1, 17], [1, 21], [1, 24], [1, 25], [1, 28], [1, 29], [2, 6], [2, 7], [2, 9], [2, 12], [2, 13], [2, 14], [2, 19], [2, 22], [2, 23], [2, 26], [2, 27], [2, 29], [3, 5], [3, 7], [3, 10], [3, 11], [3, 16], [3, 17], [3, 19], [3, 20], [3, 22], [3, 23], [3, 24], [3, 28], [3, 29], [4, 5], [4, 8], [4, 10], [4, 12], [4, 14], [4, 16], [4, 17], [4, 18], [4, 20], [4, 21], [4, 28], [5, 6], [5, 7], [5, 10], [5, 15], [5, 17], [5, 18], [5, 20], [5, 21], [5, 22], [5, 23], [5, 29], [6, 7], [6, 11], [6, 12], [6, 14], [6, 17], [6, 18], [6, 19], [6, 20], [6, 21], [6, 24], [6, 25], [6, 27], [7, 8], [7, 14], [7, 15], [7, 16], [7, 17], [7, 18], [7, 19], [7, 21], [7, 24], [7, 25], [7, 26], [7, 28], [8, 9], [8, 10], [8, 11], [8, 12], [8, 14], [8, 18], [8, 19], [8, 20], [8, 23], [8, 24], [8, 25], [8, 27], [8, 28], [9, 10], [9, 12], [9, 17], [9, 18], [9, 19], [9, 20], [9, 21], [9, 23], [9, 24], [9, 28], [9, 29], [10, 11], [10, 12], [10, 14], [10, 17], [10, 18], [10, 22], [10, 24], [10, 27], [10, 29], [11, 14], [11, 16], [11, 20], [11, 24], [11, 25], [11, 27], [11, 28], [11, 29], [12, 14], [12, 17], [12, 19], [12, 20], [12, 26], [12, 29], [13, 14], [13, 15], [13, 19], [13, 22], [13, 23], [13, 25], [13, 26], [13, 27], [13, 28], [13, 29], [14, 15], [14, 24], [14, 26], [14, 27], [14, 28], [15, 19], [15, 24], [15, 25], [15, 29], [16, 18], [16, 20], [16, 21], [16, 27], [16, 28], [17, 18], [17, 19], [17, 21], [17, 23], [17, 24], [17, 25], [17, 27], [17, 28], [17, 29], [18, 19], [18, 21], [18, 24], [18, 25], [18, 27], [18, 29], [19, 23], [19, 24], [19, 26], [19, 27], [19, 29], [20, 21], [20, 22], [20, 23], [20, 24], [20, 26], [20, 29], [21, 23], [21, 25], [21, 26], [21, 28], [21, 29], [22, 23], [22, 24], [22, 25], [24, 29], [25, 26], [26, 29], [27, 29]]}

def construct():
    start_time = time.time()
    deadline = start_time + 80.0  # Safety margin for 90s limit
    
    n = P['n']
    # Build adjacency matrix for fast lookups
    # adj[u][v] = True if edge (u,v) exists in remaining graph
    adj = [[False] * n for _ in range(n)]
    edges = P['edges']
    for u, v in edges:
        adj[u][v] = True
        adj[v][u] = True
        
    # List of current edges to iterate
    # We will maintain a set of edges for O(1) removal and iteration
    remaining_edges = set()
    for u, v in edges:
        if u < v:
            remaining_edges.add((u, v))
        else:
            remaining_edges.add((v, u))
            
    bicliques = []
    
    def expand_biclique(u, v, adj, remaining_edges):
        """
        Expand the biclique starting with edge (u, v).
        Returns (L, R) where L and R are sets of vertices.
        """
        L = {u}
        R = {v}
        
        # Precompute neighbors for speed? 
        # For n=30, direct loop is fine.
        
        changed = True
        while changed:
            changed = False
            
            # Expand L: find vertices w not in L or R such that w is connected to all in R
            # w must be in the graph (degree > 0 in remaining)
            # Actually, w just needs to be connected to all in R via remaining edges.
            new_L_candidates = set()
            for w in range(n):
                if w in L or w in R:
                    continue
                # Check if w is connected to all in R
                is_connected = True
                for r in R:
                    if not adj[w][r]:
                        is_connected = False
                        break
                if is_connected:
                    # Ensure w is actually part of the remaining graph (has at least one edge)
                    # It's safer to just check if it's connected to R. If R is non-empty, 
                    # and w is connected to all in R, w is "active".
                    new_L_candidates.add(w)
            
            if new_L_candidates:
                L.update(new_L_candidates)
                changed = True
            
            # Expand R: find vertices z not in L or R such that z is connected to all in L
            new_R_candidates = set()
            for z in range(n):
                if z in L or z in R:
                    continue
                is_connected = True
                for l in L:
                    if not adj[z][l]:
                        is_connected = False
                        break
                if is_connected:
                    new_R_candidates.add(z)
            
            if new_R_candidates:
                R.update(new_R_candidates)
                changed = True
        
        return L, R

    def find_max_biclique(adj, remaining_edges):
        """
        Find the largest biclique in the remaining graph.
        """
        best_L = set()
        best_R = set()
        max_size = 0
        
        # To improve heuristic, sort edges by sum of degrees of endpoints
        # Higher degree endpoints are more likely to be in large bicliques
        edge_list = list(remaining_edges)
        
        # Calculate degrees based on remaining_edges
        deg = [0] * n
        for u, v in edge_list:
            deg[u] += 1
            deg[v] += 1
            
        # Sort edges by (deg[u] + deg[v]) descending
        edge_list.sort(key=lambda e: deg[e[0]] + deg[e[1]], reverse=True)
        
        for u, v in edge_list:
            if not adj[u][v]:
                continue
            L, R = expand_biclique(u, v, adj, remaining_edges)
            size = len(L) * len(R)
            if size > max_size:
                max_size = size
                best_L = L
                best_R = R
                # Early termination if we find a very large one?
                # For n=30, max possible is 15*15=225. 
                # If we find > 50, it's likely good enough for greedy step.
                # But let's check all to be safe, it's fast.
        
        return best_L, best_R

    while remaining_edges:
        if time.time() > deadline:
            break
        
        L, R = find_max_biclique(adj, remaining_edges)
        
        if not L or not R:
            # Fallback: pick any edge and make it a biclique
            u, v = next(iter(remaining_edges))
            L = {u}
            R = {v}
        
        # Add to result
        # Convert sets to sorted lists for consistent output (not strictly required but good)
        bicliques.append([sorted(L), sorted(R)])
        
        # Remove edges of this biclique from remaining graph
        # An edge (u, v) is in the biclique if u in L and v in R (or vice versa)
        # Since L and R are disjoint, we just check cross pairs.
        edges_to_remove = []
        for u, v in remaining_edges:
            if (u in L and v in R) or (u in R and v in L):
                edges_to_remove.append((u, v))
                # Update adjacency matrix
                # Note: adj is symmetric
                # We need to be careful: if we remove (u,v), adj[u][v] becomes False.
                # This affects future expansions.
                
        for u, v in edges_to_remove:
            remaining_edges.remove((u, v))
            adj[u][v] = False
            adj[v][u] = False

    # If we timed out, we might have remaining edges.
    # We must cover ALL edges.
    # So if we broke early, we need to cover the rest with single edges.
    while remaining_edges:
        if time.time() > deadline - 1.0: # Last second safety
             # Force finish with single edges
             u, v = next(iter(remaining_edges))
             bicliques.append([[u], [v]])
             remaining_edges.remove((u, v))
             adj[u][v] = False
             adj[v][u] = False
             continue
        # Just use single edges for safety if greedy failed or timed out
        u, v = next(iter(remaining_edges))
        bicliques.append([[u], [v]])
        remaining_edges.remove((u, v))
        adj[u][v] = False
        adj[v][u] = False

    # Final validation and cleanup
    # Ensure types are Python int and lists
    final_result = []
    for b in bicliques:
        left = [int(x) for x in b[0]]
        right = [int(x) for x in b[1]]
        if left and right:
            final_result.append([left, right])
            
    return final_result
# EVOLVE-BLOCK-END
