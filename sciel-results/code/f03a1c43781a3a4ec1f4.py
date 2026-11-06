"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import itertools

def _is_3_colorable(adj, n):
    """
    Check if graph is 3-colorable using backtracking with bitmasks and smart ordering.
    adj[i] is a bitmask of neighbors of vertex i.
    """
    # Order vertices by degree (highest first) for better pruning
    order = sorted(range(n), key=lambda x: bin(adj[x]).count('1'), reverse=True)
    
    # Precompute for each position in the order, which colors are forbidden based on already-colored neighbors
    color = [-1] * n
    
    def backtrack(idx):
        if idx == n:
            return True
        v = order[idx]
        
        # Determine which colors are used by already-colored neighbors
        used = 0
        v_mask = adj[v]
        for nb in range(n):
            if v_mask & (1 << nb):
                if color[nb] != -1:
                    used |= (1 << color[nb])
        
        # Try each color not in used
        for c in range(3):
            if not (used & (1 << c)):
                color[v] = c
                if backtrack(idx + 1):
                    return True
                color[v] = -1
        return False
    
    return backtrack(0)

def _count_edges(adj, n):
    """Count edges in adjacency bitmask representation."""
    count = 0
    for i in range(n):
        count += bin(adj[i]).count('1')
    return count // 2

def _is_triangle_free(adj, n, u, v):
    """Check if adding edge (u,v) creates a triangle."""
    # Check if u and v share any common neighbor
    common = adj[u] & adj[v]
    return common == 0

def construct():
    n = 16
    deadline = time.time() + 140  # Safety margin before 150s kill
    
    # All possible edges
    all_edges = [(i, j) for i in range(n) for j in range(i+1, n)]
    num_edges = len(all_edges)
    
    # Start with a known 4-chromatic triangle-free graph: the Grötzsch graph (11 vertices, 20 edges)
    # Plus 5 isolated vertices
    adj = [0] * n
    def add_edge(u, v):
        adj[u] |= (1 << v)
        adj[v] |= (1 << u)
    
    # Build Grötzsch graph on vertices 0-10
    # Outer 5-cycle: 0-1-2-3-4-0
    for i in range(5):
        add_edge(i, (i+1) % 5)
    # Inner 5-cycle: 5-6-7-8-9-5
    for i in range(5):
        add_edge(5+i, 5+(i+1) % 5)
    # Connect outer to inner: i connected to 5+i and 5+(i-1)%5
    for i in range(5):
        add_edge(i, 5+i)
        add_edge(i, 5+(i-1) % 5)
    # Connect vertex 10 to all inner vertices
    for i in range(5):
        add_edge(10, 5+i)
    
    best_adj = adj[:]
    best_edges = _count_edges(adj, n)
    
    # Local search: try to add edges while maintaining triangle-free and 4-chromatic
    # Strategy: repeatedly try to add the "easiest" edge (one that creates fewest conflicts)
    # and check if the graph remains 4-chromatic
    
    adj_current = best_adj[:]
    count = best_edges
    
    # Precompute for each non-edge, the number of common neighbors (conflicts)
    # An edge (u,v) can be added if u and v have no common neighbors
    
    for iteration in range(500):
        if time.time() > deadline:
            break
        
        # Find all addable edges (triangle-free)
        addable = []
        for u, v in all_edges:
            if adj_current[u] & (1 << v):
                continue  # Edge already exists
            if adj_current[u] & adj_current[v]:
                continue  # Would create triangle
            # Count how many edges would become non-addable if we add (u,v)
            # (i.e., how many vertices w are common neighbors of u or v that are also adjacent to the other)
            # Actually, adding (u,v) makes it so that for any w adjacent to u, w can no longer connect to v
            # and for any w adjacent to v, w can no longer connect to u
            # The "cost" is roughly deg(u) + deg(v) - 2*common(u,v) but common is 0 since triangle-free
            cost = bin(adj_current[u]).count('1') + bin(adj_current[v]).count('1')
            addable.append((cost, u, v))
        
        if not addable:
            break
        
        # Sort by cost (prefer edges that "sacrifice" fewer future possibilities)
        addable.sort()
        
        found = False
        for cost, u, v in addable:
            if time.time() > deadline:
                break
            
            # Try adding this edge
            adj_current[u] |= (1 << v)
            adj_current[v] |= (1 << u)
            count += 1
            
            # Check if still 4-chromatic (not 3-colorable)
            if not _is_3_colorable(adj_current, n):
                # Still 4-chromatic! Update best
                if count > best_edges:
                    best_edges = count
                    best_adj = adj_current[:]
                found = True
                break
            else:
                # Revert
                adj_current[u] &= ~(1 << v)
                adj_current[v] &= ~(1 << u)
                count -= 1
        
        if not found:
            # No single edge addition works; try random restarts with perturbation
            # Remove a random edge from current best and try again
            if best_edges > 0:
                edge_list = []
                for i in range(n):
                    for j in range(i+1, n):
                        if best_adj[i] & (1 << j):
                            edge_list.append((i, j))
                
                if edge_list:
                    # Remove a random edge to create space
                    u, v = random.choice(edge_list)
                    adj_current = best_adj[:]
                    adj_current[u] &= ~(1 << v)
                    adj_current[v] &= ~(1 << u)
                    count = best_edges - 1
                    
                    # Now try to add different edges
                    for _ in range(20):
                        if time.time() > deadline:
                            break
                        
                        # Find addable edges
                        addable2 = []
                        for uu, vv in all_edges:
                            if adj_current[uu] & (1 << vv):
                                continue
                            if adj_current[uu] & adj_current[vv]:
                                continue
                            cost2 = bin(adj_current[uu]).count('1') + bin(adj_current[vv]).count('1')
                            addable2.append((cost2, uu, vv))
                        
                        if not addable2:
                            break
                        
                        addable2.sort()
                        
                        # Try a few random ones
                        random.shuffle(addable2)
                        for cost2, uu, vv in addable2[:5]:
                            adj_current[uu] |= (1 << vv)
                            adj_current[vv] |= (1 << uu)
                            count += 1
                            
                            if not _is_3_colorable(adj_current, n):
                                if count > best_edges:
                                    best_edges = count
                                    best_adj = adj_current[:]
                                break
                            else:
                                adj_current[uu] &= ~(1 << vv)
                                adj_current[vv] &= ~(1 << uu)
                                count -= 1
                        
                        if count > best_edges:
                            break
    
    # Convert best_adj back to edge list
    result = []
    for i in range(n):
        for j in range(i+1, n):
            if best_adj[i] & (1 << j):
                result.append([i, j])
    
    return result
# EVOLVE-BLOCK-END
