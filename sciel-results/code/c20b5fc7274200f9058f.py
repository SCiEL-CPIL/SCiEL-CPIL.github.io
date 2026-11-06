import time
import random
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 145  # Safety margin for 150s limit
    
    n = 20
    
    # Precompute all pairs
    all_pairs = [(i, j) for i in range(n) for j in range(i+1, n)]
    pair_index = {}
    for idx, (u, v) in enumerate(all_pairs):
        pair_index[(u, v)] = idx
    
    def is_valid_addition(adj, u, v):
        """
        Checks if adding edge (u, v) violates the K3,3-free condition.
        Condition: For any triple T of vertices, |N(T)| <= 2.
        Adding edge (u,v) creates a violation iff there exist a,b != u,v
        such that v is in N(a)∩N(b) and |N(u)∩N(a)∩N(b)| >= 2,
        OR u is in N(a)∩N(b) and |N(v)∩N(a)∩N(b)| >= 2.
        """
        if adj[u] & (1 << v):
            return False
            
        # Check triples containing u: {u, a, b}
        # If v is in N(a) and N(b), then v is added to N({u, a, b}).
        # We need old |N({u, a, b})| <= 1.
        # N({u, a, b}) = N(u) & N(a) & N(b)
        
        for a in range(n):
            if a == u or a == v:
                continue
            for b in range(a + 1, n):
                if b == u or b == v:
                    continue
                
                # Check if v is a common neighbor of a and b
                if (adj[a] & (1 << v)) and (adj[b] & (1 << v)):
                    count = (adj[u] & adj[a] & adj[b]).bit_count()
                    if count > 1:
                        return False
                        
                # Check if u is a common neighbor of a and b
                # This corresponds to triples {v, a, b} where u is added to N({v, a, b})
                if (adj[a] & (1 << u)) and (adj[b] & (1 << u)):
                    count = (adj[v] & adj[a] & adj[b]).bit_count()
                    if count > 1:
                        return False
                        
        return True

    def count_edges(adj):
        return sum(x.bit_count() for x in adj) // 2

    def greedy_construct(random_seed, prefer_dense=False):
        random.seed(random_seed)
        adj = [0] * n
        
        pairs = all_pairs[:]
        random.shuffle(pairs)
        
        if prefer_dense:
            # Sort by some heuristic to prefer denser connections early
            # This is a simple heuristic: try to connect vertices that already have edges
            pass
            
        for u, v in pairs:
            if time.time() > deadline:
                break
            if is_valid_addition(adj, u, v):
                adj[u] |= (1 << v)
                adj[v] |= (1 << u)
                
        return adj

    def local_search_add(adj, random_seed=None):
        if random_seed is not None:
            random.seed(random_seed)
        current_adj = adj[:]
        
        # Try to add edges in random order
        pairs = [(i, j) for i in range(n) for j in range(i+1, n) if not (current_adj[i] & (1 << j))]
        random.shuffle(pairs)
        
        for u, v in pairs:
            if time.time() > deadline:
                break
            if is_valid_addition(current_adj, u, v):
                current_adj[u] |= (1 << v)
                current_adj[v] |= (1 << u)
                
        return current_adj

    def local_search_swap(adj, random_seed=None):
        """
        Try to improve by swapping edges: remove (u1,v1), add (u2,v2) if it increases density
        while maintaining K3,3-free property. Actually, we just want to add edges, so this is
        more about finding better addition orders. Let's do a simpler 2-opt style:
        Remove an edge, then try to add two edges.
        """
        if random_seed is not None:
            random.seed(random_seed)
        current_adj = adj[:]
        current_edges = count_edges(current_adj)
        
        # Generate random swaps
        existing_edges = [(i, j) for i in range(n) for j in range(i+1, n) if current_adj[i] & (1 << j)]
        missing_edges = [(i, j) for i in range(n) for j in range(i+1, n) if not (current_adj[i] & (1 << j))]
        
        random.shuffle(existing_edges)
        random.shuffle(missing_edges)
        
        # Try to remove one edge and add two others
        for u1, v1 in existing_edges[:50]:
            if time.time() > deadline:
                break
            
            # Temporarily remove edge (u1, v1)
            current_adj[u1] &= ~(1 << v1)
            current_adj[v1] &= ~(1 << u1)
            
            improved = False
            # Try to add two missing edges
            for u2, v2 in missing_edges[:100]:
                if time.time() > deadline:
                    break
                if is_valid_addition(current_adj, u2, v2):
                    current_adj[u2] |= (1 << v2)
                    current_adj[v2] |= (1 << u2)
                    improved = True
                    break
            
            if not improved:
                # Restore edge (u1, v1)
                current_adj[u1] |= (1 << v1)
                current_adj[v1] |= (1 << u1)
        
        return current_adj

    best_adj = [0] * n
    best_edges = 0
    
    # Phase 1: Generate many greedy solutions
    num_restarts = 500
    for i in range(num_restarts):
        if time.time() > deadline - 10:
            break
        
        adj = greedy_construct(i)
        edges = count_edges(adj)
        
        # Local search improvement
        adj = local_search_add(adj, random_seed=i)
        edges = count_edges(adj)
        
        if edges > best_edges:
            best_edges = edges
            best_adj = adj[:]
    
    # Phase 2: More intensive local search on the best solution
    if time.time() < deadline - 5:
        # Try multiple rounds of local search with different seeds
        for seed in range(100):
            if time.time() > deadline - 2:
                break
            adj = local_search_add(best_adj, random_seed=seed + 1000)
            edges = count_edges(adj)
            if edges > best_edges:
                best_edges = edges
                best_adj = adj[:]
                
    # Phase 3: Swap-based local search
    if time.time() < deadline - 2:
        adj = local_search_swap(best_adj, random_seed=2000)
        edges = count_edges(adj)
        if edges > best_edges:
            best_edges = edges
            best_adj = adj[:]
            
        # Final greedy add
        adj = local_search_add(best_adj, random_seed=3000)
        edges = count_edges(adj)
        if edges > best_edges:
            best_edges = edges
            best_adj = adj[:]
            
    # Convert bitmask to list of edges
    result_edges = []
    for i in range(n):
        for j in range(i+1, n):
            if best_adj[i] & (1 << j):
                result_edges.append([i, j])
                
    return result_edges

if __name__ == "__main__":
    edges = construct()
    print(f"Found {len(edges)} edges")
