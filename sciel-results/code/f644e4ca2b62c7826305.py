import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 115  # Safety margin for 120s limit
    
    n = 20
    
    # adj[u] is a bitmask of neighbors of u
    # bit i is set if edge (u, i) exists
    
    def is_valid_addition(adj, u, v):
        """
        Checks if adding edge (u, v) violates the K3,3-free condition.
        Condition: For any triple T, |N(T)| <= 2.
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
                    # v is in N(a) ∩ N(b)
                    # So adding u-v adds v to N({u, a, b})
                    # We must ensure current |N({u, a, b})| <= 1
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

    def greedy_construct(random_seed):
        random.seed(random_seed)
        adj = [0] * n
        
        pairs = [(i, j) for i in range(n) for j in range(i+1, n)]
        random.shuffle(pairs)
        
        for u, v in pairs:
            if time.time() > deadline:
                break
            if is_valid_addition(adj, u, v):
                adj[u] |= (1 << v)
                adj[v] |= (1 << u)
                
        return adj

    def local_search(adj, max_steps=500):
        current_adj = adj[:]
        current_edges = count_edges(current_adj)
        
        # Try to add edges in random order
        pairs = [(i, j) for i in range(n) for j in range(i+1, n) if not (current_adj[i] & (1 << j))]
        random.shuffle(pairs)
        
        for u, v in pairs:
            if time.time() > deadline:
                break
            if is_valid_addition(current_adj, u, v):
                current_adj[u] |= (1 << v)
                current_adj[v] |= (1 << u)
                current_edges += 1
                
        return current_adj, current_edges

    best_adj = [0] * n
    best_edges = 0
    
    # Generate initial candidates
    # We can run many more restarts now due to speedup
    num_restarts = 2000 
    for i in range(num_restarts):
        if time.time() > deadline:
            break
        
        adj = greedy_construct(i)
        edges = count_edges(adj)
        
        # Local search improvement
        adj, edges = local_search(adj)
        
        if edges > best_edges:
            best_edges = edges
            best_adj = adj[:]
            
    # If we have time, try to improve the best solution further
    if time.time() < deadline - 5:
        adj, edges = local_search(best_adj)
        if edges > best_edges:
            best_edges = edges
            best_adj = adj
            
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
