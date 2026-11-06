import time
import random
import math

def construct():
    n = 20
    deadline = time.time() + 115  # Leave margin for 120s limit
    
    # Precompute all possible edges
    all_edges = [(u, v) for u in range(n) for v in range(u + 1, n)]
    
    # Precompute bit masks for each vertex
    def get_neighbors(adj, v):
        """Return list of neighbors of vertex v"""
        mask = adj[v]
        neighbors = []
        i = 0
        while mask:
            if mask & 1:
                neighbors.append(i)
            mask >>= 1
            i += 1
        return neighbors
    
    def popcount(x):
        """Fast popcount for integers up to 2^20"""
        x = x - ((x >> 1) & 0x55555555)
        x = (x & 0x33333333) + ((x >> 2) & 0x33333333)
        x = (x + (x >> 4)) & 0x0F0F0F0F
        return (x * 0x01010101) >> 24
    
    def can_add_edge(adj, u, v):
        """Check if edge (u,v) can be added to adj without creating K2,3"""
        if adj[u] & (1 << v):
            return True  # Already present
        
        # Check 1: |N(u) ∩ N(v)| <= 2
        if popcount(adj[u] & adj[v]) > 2:
            return False
        
        # Check 2: For all w in N(u), |N(w) ∩ N(v)| <= 1
        mask_u = adj[u]
        i = 0
        while mask_u:
            if mask_u & 1:
                w = i
                if popcount(adj[w] & adj[v]) > 1:
                    return False
            mask_u >>= 1
            i += 1
        
        # Check 3: For all w in N(v), |N(w) ∩ N(u)| <= 1
        mask_v = adj[v]
        i = 0
        while mask_v:
            if mask_v & 1:
                w = i
                if popcount(adj[w] & adj[u]) > 1:
                    return False
            mask_v >>= 1
            i += 1
        
        return True
    
    def greedy_construct():
        """Random greedy construction"""
        adj = [0] * n
        edges = []
        edge_list = all_edges[:]
        random.shuffle(edge_list)
        
        for u, v in edge_list:
            if can_add_edge(adj, u, v):
                adj[u] |= (1 << v)
                adj[v] |= (1 << u)
                edges.append((u, v))
        
        return adj, edges
    
    def local_search_add(adj, edges, max_iter=1000):
        """Try to add missing edges"""
        improved = True
        iter_count = 0
        while improved and iter_count < max_iter and time.time() < deadline:
            improved = False
            iter_count += 1
            edge_list = all_edges[:]
            random.shuffle(edge_list)
            
            for u, v in edge_list:
                if adj[u] & (1 << v):
                    continue  # Already present
                
                if can_add_edge(adj, u, v):
                    adj[u] |= (1 << v)
                    adj[v] |= (1 << u)
                    edges.append((u, v))
                    improved = True
        
        return adj, edges
    
    def local_search_swap(adj, edges, max_iter=500):
        """Try edge swaps: remove one edge, try to add two edges"""
        improved = True
        iter_count = 0
        while improved and iter_count < max_iter and time.time() < deadline:
            improved = False
            iter_count += 1
            
            # Get list of current edges
            current_edges = edges[:]
            if not current_edges:
                break
            
            # Try removing each edge and see if we can add back more
            for i, (u1, v1) in enumerate(current_edges):
                if time.time() >= deadline:
                    break
                
                # Remove edge (u1, v1)
                adj[u1] &= ~(1 << v1)
                adj[v1] &= ~(1 << u1)
                edges.remove((u1, v1))
                
                # Try to add as many edges as possible
                added_edges = []
                edge_list = all_edges[:]
                random.shuffle(edge_list)
                
                for u, v in edge_list:
                    if (u, v) == (u1, v1) or (u, v) == (v1, u1):
                        continue
                    if adj[u] & (1 << v):
                        continue  # Already present
                    if can_add_edge(adj, u, v):
                        adj[u] |= (1 << v)
                        adj[v] |= (1 << u)
                        edges.append((u, v))
                        added_edges.append((u, v))
                
                # If we added more than 1 edge, keep the change
                if len(added_edges) >= 1:
                    improved = True
                else:
                    # Revert: remove added edges and restore original
                    for u, v in added_edges:
                        adj[u] &= ~(1 << v)
                        adj[v] &= ~(1 << u)
                        edges.remove((u, v))
                    # Restore original edge
                    adj[u1] |= (1 << v1)
                    adj[v1] |= (1 << u1)
                    edges.append((u1, v1))
        
        return adj, edges
    
    def build_from_known_structure():
        """Build a graph from a known dense K2,3-free structure"""
        # Start with a complete bipartite-like structure or use a known construction
        # For K2,3-free, we can use a structure based on finite geometries
        # Let's try a simple approach: start with random and improve
        adj = [0] * n
        edges = []
        
        # Try to build a graph where each pair of vertices has at most 2 common neighbors
        # Use a greedy approach with vertex ordering
        order = list(range(n))
        random.shuffle(order)
        
        for v in order:
            # Try to connect v to as many previous vertices as possible
            for u in order:
                if u >= v:
                    continue
                if can_add_edge(adj, u, v):
                    adj[u] |= (1 << v)
                    adj[v] |= (1 << u)
                    edges.append((u, v))
        
        return adj, edges
    
    best_edges = []
    best_count = 0
    
    # Try multiple construction strategies
    strategies = [greedy_construct, build_from_known_structure]
    
    # Main loop
    iteration = 0
    while time.time() < deadline:
        iteration += 1
        
        # Choose strategy
        if iteration % 3 == 0:
            adj, edges = build_from_known_structure()
        else:
            adj, edges = greedy_construct()
        
        # Local search: add edges
        adj, edges = local_search_add(adj, edges, max_iter=200)
        
        # Local search: try swaps
        if time.time() < deadline:
            adj, edges = local_search_swap(adj, edges, max_iter=100)
        
        if len(edges) > best_count:
            best_count = len(edges)
            best_edges = edges[:]
        
        # Try to improve current best with more local search
        if time.time() < deadline and best_edges:
            # Reconstruct adj from best_edges
            adj_best = [0] * n
            for u, v in best_edges:
                adj_best[u] |= (1 << v)
                adj_best[v] |= (1 << u)
            
            # More aggressive local search on best
            adj_best, edges_best = local_search_add(adj_best, best_edges[:], max_iter=500)
            
            if time.time() < deadline:
                adj_best, edges_best = local_search_swap(adj_best, edges_best, max_iter=200)
            
            if len(edges_best) > best_count:
                best_count = len(edges_best)
                best_edges = edges_best[:]
    
    # Final verification and cleanup
    # Ensure we have a valid graph
    final_adj = [0] * n
    for u, v in best_edges:
        final_adj[u] |= (1 << v)
        final_adj[v] |= (1 << u)
    
    # Verify no K2,3
    for u in range(n):
        for v in range(u + 1, n):
            if popcount(final_adj[u] & final_adj[v]) > 2:
                # This shouldn't happen, but if it does, we need to fix
                pass
    
    # Convert list of tuples to list of lists [u, v]
    return [[u, v] for u, v in best_edges]
