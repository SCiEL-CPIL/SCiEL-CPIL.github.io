# EVOLVE-BLOCK-START
import time
import random
import numpy as np

def construct():
    """
    Construct a K2,3-free graph on 20 vertices maximizing edges.
    Strategy: random initialization -> repair violations -> greedy edge addition.
    Multiple restarts with deadline guard.
    """
    start_time = time.time()
    deadline = start_time + 110  # safety margin before 120s limit
    
    n = 20
    best_edges = []
    best_count = 0
    
    def is_k23_free(adj):
        """Check if adjacency matrix has no K2,3 (no pair with >=3 common neighbors)."""
        # Common neighbors of u,v = sum over w of adj[u,w]*adj[v,w]
        # For u!=v, this is (adj @ adj)[u,v] since adj[u,u]=0
        common = adj @ adj
        for u in range(n):
            for v in range(u + 1, n):
                if common[u, v] >= 3:
                    return False
        return True
    
    def count_common_neighbors(adj, u, v):
        """Count common neighbors of u and v."""
        return int(np.sum(adj[u] * adj[v]))
    
    def can_add_edge(adj, u, v):
        """Check if adding edge (u,v) would create a K2,3."""
        # For each common neighbor w of u and v, check if w and some other vertex
        # would then have >=3 common neighbors.
        # Actually, adding (u,v) affects common neighbor counts for pairs involving u or v.
        # A K2,3 is created if there exist two vertices x,y with >=3 common neighbors
        # after the addition. The new edge (u,v) can only increase common neighbor
        # counts for pairs where one is u or v, or where both u and v are common neighbors.
        
        # For pair (u, x) where x != v: common neighbors increase by 1 if adj[x,v]=1
        # For pair (v, x) where x != u: common neighbors increase by 1 if adj[x,u]=1
        # For pair (x, y) where x,y != u,v: common neighbors increase by 1 if both adj[x,u]=1 and adj[y,u]=1... no
        # Actually: common neighbors of (x,y) include u if adj[x,u]=1 and adj[y,u]=1, and v if adj[x,v]=1 and adj[y,v]=1.
        # Adding edge (u,v) doesn't change adj[x,u] or adj[x,v], so it doesn't change common neighbors of (x,y) for x,y != u,v.
        
        # So we only need to check pairs involving u or v.
        
        # Check pairs (u, x) for x != v, x != u
        for x in range(n):
            if x == u or x == v:
                continue
            if adj[x, v]:  # x is a common neighbor of u and v after adding? No, x is neighbor of v
                # After adding (u,v), common neighbors of (u,x) = old common neighbors + (1 if adj[x,v] else 0)
                # Wait, common neighbors of u and x: vertices w such that adj[u,w]=1 and adj[x,w]=1.
                # Adding edge (u,v) makes adj[u,v]=1. So if adj[x,v]=1, then v becomes a common neighbor of u and x.
                new_common = count_common_neighbors(adj, u, x) + (1 if adj[x, v] else 0)
                if new_common >= 3:
                    return False
        
        # Check pairs (v, x) for x != u, x != v
        for x in range(n):
            if x == u or x == v:
                continue
            if adj[x, u]:  # x is neighbor of u
                new_common = count_common_neighbors(adj, v, x) + (1 if adj[x, u] else 0)
                if new_common >= 3:
                    return False
        
        return True
    
    def repair(adj):
        """Remove edges to eliminate K2,3 violations."""
        changed = True
        while changed:
            changed = False
            # Find a violation
            common = adj @ adj
            for u in range(n):
                if changed:
                    break
                for v in range(u + 1, n):
                    if common[u, v] >= 3:
                        # Remove an edge incident to a common neighbor to break the violation
                        # Find common neighbors
                        common_neighbors = [w for w in range(n) if adj[u, w] and adj[v, w]]
                        # Remove edge (u, w) for one of the common neighbors
                        # Choose the one whose removal causes least damage
                        best_w = None
                        best_loss = float('inf')
                        for w in common_neighbors:
                            # If we remove (u,w), how many other violations might be fixed?
                            # Just remove the first one for simplicity
                            best_w = w
                            break
                        if best_w is not None:
                            adj[u, best_w] = 0
                            adj[best_w, u] = 0
                            changed = True
                            break
        return adj
    
    def greedy_add(adj):
        """Try to add edges greedily."""
        added = True
        while added:
            added = False
            # Try all non-edges
            for u in range(n):
                if added:
                    break
                for v in range(u + 1, n):
                    if not adj[u, v]:
                        if can_add_edge(adj, u, v):
                            adj[u, v] = 1
                            adj[v, u] = 1
                            added = True
                            break
        return adj
    
    def edges_from_adj(adj):
        return [[u, v] for u in range(n) for v in range(u + 1, n) if adj[u, v]]
    
    def count_edges(adj):
        return int(np.sum(adj) // 2)
    
    # Multiple restarts
    num_restarts = 50
    for restart in range(num_restarts):
        if time.time() > deadline:
            break
        
        # Random initialization: random graph with some density
        # Start with a random graph, then repair
        # Density between 0.1 and 0.4
        density = 0.1 + 0.3 * random.random()
        adj = np.zeros((n, n), dtype=int)
        for u in range(n):
            for v in range(u + 1, n):
                if random.random() < density:
                    adj[u, v] = 1
                    adj[v, u] = 1
        
        # Repair
        adj = repair(adj)
        
        # Greedy add
        adj = greedy_add(adj)
        
        # Verify
        if is_k23_free(adj):
            cnt = count_edges(adj)
            if cnt > best_count:
                best_count = cnt
                best_edges = edges_from_adj(adj)
        else:
            # Shouldn't happen, but if it does, repair more aggressively
            adj = repair(adj)
            adj = greedy_add(adj)
            if is_k23_free(adj):
                cnt = count_edges(adj)
                if cnt > best_count:
                    best_count = cnt
                    best_edges = edges_from_adj(adj)
        
        # Try a different initialization: start from a known structure
        if restart % 2 == 0:
            # Start from a random regular-like graph
            adj = np.zeros((n, n), dtype=int)
            # Random permutation to break symmetry
            perm = list(range(n))
            random.shuffle(perm)
            # Add edges based on distance in permutation
            for i in range(n):
                for j in range(i + 1, n):
                    dist = abs(perm.index(i) - perm.index(j))
                    if dist <= 3 and random.random() < 0.7:
                        adj[i, j] = 1
                        adj[j, i] = 1
            adj = repair(adj)
            adj = greedy_add(adj)
            if is_k23_free(adj):
                cnt = count_edges(adj)
                if cnt > best_count:
                    best_count = cnt
                    best_edges = edges_from_adj(adj)
    
    # Final verification
    if best_edges:
        adj_check = np.zeros((n, n), dtype=int)
        for e in best_edges:
            adj_check[e[0], e[1]] = 1
            adj_check[e[1], e[0]] = 1
        if is_k23_free(adj_check):
            return best_edges
    
    # Fallback: return the known valid construction
    edges = []
    for k in range(5):
        edges += [[4 * k + a, 4 * k + b] for a in range(4) for b in range(a + 1, 4)]
    return edges
# EVOLVE-BLOCK-END
