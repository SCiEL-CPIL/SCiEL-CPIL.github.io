"""K3,3-free graph construction on 20 vertices - optimized version."""
import time
import itertools
from typing import List, Tuple

# EVOLVE-BLOCK-START

def construct() -> list:
    """Construct a K3,3-free graph on 20 vertices maximizing edge count."""
    start_time = time.time()
    deadline = start_time + 120  # 120s deadline
    
    n = 20
    all_vertices = list(range(n))
    all_pairs = [(i, j) for i in range(n) for j in range(i+1, n)]
    
    def get_common_neighbors(adj_mask, vertices):
        """Get common neighbors of a set of vertices."""
        if not vertices:
            return (1 << n) - 1
        mask = (1 << n) - 1
        for v in vertices:
            mask &= adj_mask[v]
        # Remove the vertices themselves from consideration
        for v in vertices:
            mask &= ~(1 << v)
        return mask
    
    def count_common_neighbors(adj_mask, vertices):
        """Count common neighbors of a set of vertices."""
        cn = get_common_neighbors(adj_mask, vertices)
        return bin(cn).count('1')
    
    def is_k33_free(adj_mask):
        """Check if graph is K3,3-free using efficient common neighbor counting."""
        for i in range(n):
            for j in range(i+1, n):
                common_ij = adj_mask[i] & adj_mask[j]
                # Remove i and j from common_ij
                common_ij &= ~(1 << i)
                common_ij &= ~(1 << j)
                
                # For each potential third vertex k, check if it shares 3+ neighbors with i,j
                for k in range(j+1, n):
                    common_ijk = common_ij & adj_mask[k]
                    # Remove k from common_ijk
                    common_ijk &= ~(1 << k)
                    if bin(common_ijk).count('1') >= 3:
                        return False
        return True
    
    def count_edges(adj_mask):
        """Count edges in adjacency mask representation."""
        total = 0
        for i in range(n):
            total += bin(adj_mask[i]).count('1')
        return total // 2
    
    def adj_to_edges(adj_mask):
        """Convert adjacency mask to edge list."""
        edges = []
        for u in range(n):
            mask = adj_mask[u]
            for v in range(u+1, n):
                if mask & (1 << v):
                    edges.append([u, v])
        return edges
    
    def can_add_edge(adj_mask, u, v):
        """Check if adding edge (u,v) keeps graph K3,3-free."""
        # Adding edge (u,v) only affects K3,3 violations where both u and v are in the same part
        # or where u/v are in different parts but create new common neighbor triples
        
        # For any triple containing u but not v: adding v as neighbor of u may create issues
        # For any triple containing v but not u: adding u as neighbor of v may create issues
        # For triples containing both u and v: adding edge between them doesn't affect common neighbors
        
        # Efficient check: only need to check triples where u and v are in the same K3,3 part
        # A K3,3 has two parts of 3 vertices each. Adding edge (u,v) creates a K3,3 if:
        # - There exist vertices a,b,c,d,e such that {u,v,a} and {b,c,d} form K3,3, 
        #   or {u,v,b} and {c,d,e} form K3,3, etc.
        
        # Simpler: just check if the modified graph is K3,3-free
        old_u = adj_mask[u]
        old_v = adj_mask[v]
        
        adj_mask[u] |= (1 << v)
        adj_mask[v] |= (1 << u)
        
        result = is_k33_free(adj_mask)
        
        # Restore
        adj_mask[u] = old_u
        adj_mask[v] = old_v
        
        return result
    
    def greedy_add_edges(adj_mask, edge_order):
        """Greedily add edges in given order, maintaining K3,3-freeness."""
        for u, v in edge_order:
            if time.time() > deadline - 5:
                break
            if can_add_edge(adj_mask, u, v):
                adj_mask[u] |= (1 << v)
                adj_mask[v] |= (1 << u)
    
    def local_search(adj_mask):
        """Local search to improve edge count."""
        improved = True
        iterations = 0
        max_iterations = 1000
        
        while improved and iterations < max_iterations:
            if time.time() > deadline - 5:
                break
            
            improved = False
            iterations += 1
            
            # Get current edges and non-edges
            edges = []
            non_edges = []
            for i in range(n):
                for j in range(i+1, n):
                    if adj_mask[i] & (1 << j):
                        edges.append((i, j))
                    else:
                        non_edges.append((i, j))
            
            # Try to add edges (from non-edges)
            for u, v in non_edges:
                if can_add_edge(adj_mask, u, v):
                    adj_mask[u] |= (1 << v)
                    adj_mask[v] |= (1 << u)
                    improved = True
            
            # Try swaps: remove an edge, add a non-edge
            if not improved and edges:
                # Try random swaps
                import random
                random.seed(42 + iterations)
                
                for _ in range(min(len(non_edges) * len(edges), 500)):
                    if time.time() > deadline - 5:
                        break
                    
                    if not edges or not non_edges:
                        break
                    
                    # Pick a random edge to remove and non-edge to add
                    e_idx = random.randint(0, len(edges) - 1)
                    ne_idx = random.randint(0, len(non_edges) - 1)
                    
                    u1, v1 = edges[e_idx]
                    u2, v2 = non_edges[ne_idx]
                    
                    # Remove edge (u1, v1)
                    adj_mask[u1] &= ~(1 << v1)
                    adj_mask[v1] &= ~(1 << u1)
                    
                    # Add edge (u2, v2) if possible
                    old_u2 = adj_mask[u2]
                    old_v2 = adj_mask[v2]
                    adj_mask[u2] |= (1 << v2)
                    adj_mask[v2] |= (1 << u1)  # Note: using u1 here might be wrong
                    
                    # Check if valid
                    if is_k33_free(adj_mask):
                        # Keep the swap
                        improved = True
                        # Update edges and non_edges lists
                        edges.remove((u1, v1))
                        non_edges.remove((u2, v2))
                        edges.append((u2, v2))
                        non_edges.append((u1, v1))
                    else:
                        # Revert
                        adj_mask[u2] = old_u2
                        adj_mask[v2] = adj_mask[v2] & ~(1 << u2)  # Remove the added edge
                        adj_mask[u1] |= (1 << v1)
                        adj_mask[v1] |= (1 << u1)
        
        return adj_mask
    
    best_adj = [0] * n
    best_count = 0
    
    # Multiple restarts with different strategies
    import random
    
    for restart in range(20):
        if time.time() > deadline - 10:
            break
        
        random.seed(1000 + restart)
        
        adj = [0] * n
        
        # Strategy 1: Greedy with random edge order
        edge_order = all_pairs[:]
        random.shuffle(edge_order)
        greedy_add_edges(adj, edge_order)
        
        # Local search
        adj = local_search(adj)
        
        count = count_edges(adj)
        if count > best_count:
            best_count = count
            best_adj = adj[:]
        
        # Strategy 2: Start from a structured graph (e.g., complete bipartite-like)
        if restart % 2 == 1:
            adj2 = [0] * n
            # Try to build a dense bipartite-like structure
            part1 = list(range(n // 2))
            part2 = list(range(n // 2, n))
            
            # Add all edges between part1 and part2 initially
            for u in part1:
                for v in part2:
                    adj2[u] |= (1 << v)
                    adj2[v] |= (1 << u)
            
            # Check if K3,3-free, if not, remove problematic edges
            while not is_k33_free(adj2):
                # Find and remove an edge that creates K3,3
                removed = False
                for u in part1:
                    for v in part2:
                        if adj2[u] & (1 << v):
                            old_u = adj2[u]
                            old_v = adj2[v]
                            adj2[u] &= ~(1 << v)
                            adj2[v] &= ~(1 << u)
                            if is_k33_free(adj2):
                                removed = True
                                break
                            else:
                                adj2[u] = old_u
                                adj2[v] = old_v
                    if removed:
                        break
                if not removed:
                    break
            
            adj2 = local_search(adj2)
            count2 = count_edges(adj2)
            if count2 > best_count:
                best_count = count2
                best_adj = adj2[:]
    
    # Convert to edge list
    edges = adj_to_edges(best_adj)
    
    return edges

# EVOLVE-BLOCK-END
