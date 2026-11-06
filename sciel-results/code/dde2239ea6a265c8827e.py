import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 115  # Safety margin for 120s limit
    
    n = 20
    all_edges = [(i, j) for i in range(n) for j in range(i+1, n)]
    
    def can_add(adj, u, v):
        """Check if adding edge (u,v) violates K3,3-free condition."""
        if adj[u] & (1 << v):
            return False
            
        # Check triples {u, a, b} where a, b are neighbors of v
        # v becomes a common neighbor of {u, a, b}
        # Need |N(u) \cap N(a) \cap N(b)| <= 1
        
        # Get neighbors of v (excluding u, v)
        # We only care about a, b that are neighbors of v
        mask_v = adj[v]
        # Iterate over pairs of neighbors of v
        # To speed up, we can iterate over set bits
        neighbors_v = []
        m = mask_v
        while m:
            lsb = m & -m
            idx = lsb.bit_length() - 1
            if idx != u and idx != v:
                neighbors_v.append(idx)
            m ^= lsb
            
        for i in range(len(neighbors_v)):
            a = neighbors_v[i]
            for j in range(i+1, len(neighbors_v)):
                b = neighbors_v[j]
                common = adj[u] & adj[a] & adj[b]
                if common.bit_count() > 1:
                    return False
                    
        # Check triples {v, a, b} where a, b are neighbors of u
        # u becomes a common neighbor of {v, a, b}
        # Need |N(v) \cap N(a) \cap N(b)| <= 1
        
        mask_u = adj[u]
        neighbors_u = []
        m = mask_u
        while m:
            lsb = m & -m
            idx = lsb.bit_length() - 1
            if idx != v and idx != u:
                neighbors_u.append(idx)
            m ^= lsb
            
        for i in range(len(neighbors_u)):
            a = neighbors_u[i]
            for j in range(i+1, len(neighbors_u)):
                b = neighbors_u[j]
                common = adj[v] & adj[a] & adj[b]
                if common.bit_count() > 1:
                    return False
                    
        return True

    def get_edge_count(adj):
        return sum(x.bit_count() for x in adj) // 2

    def greedy_fill(adj):
        """Greedily add as many edges as possible."""
        current_edges = get_edge_count(adj)
        missing = [(i, j) for i in range(n) for j in range(i+1, n) if not (adj[i] & (1 << j))]
        
        best_adj = adj[:]
        best_edges = current_edges
        
        # Try a few random orders
        for _ in range(3):
            temp_adj = adj[:]
            temp_missing = missing[:]
            random.shuffle(temp_missing)
            
            for u, v in temp_missing:
                if can_add(temp_adj, u, v):
                    temp_adj[u] |= (1 << v)
                    temp_adj[v] |= (1 << u)
            
            temp_edges = get_edge_count(temp_adj)
            if temp_edges > best_edges:
                best_edges = temp_edges
                best_adj = temp_adj[:]
                
        return best_adj, best_edges

    def local_search(adj, max_time):
        ls_start = time.time()
        current_edges = get_edge_count(adj)
        best_adj = adj[:]
        best_edges = current_edges
        
        # Initial greedy fill
        adj, current_edges = greedy_fill(adj)
        best_adj = adj[:]
        best_edges = current_edges
        
        missing = [(i, j) for i in range(n) for j in range(i+1, n) if not (adj[i] & (1 << j))]
        
        no_improve = 0
        max_no_improve = 100
        
        while time.time() - ls_start < max_time:
            if time.time() > deadline - 2:
                break
                
            if not missing:
                break
                
            # Randomly pick a missing edge
            idx = random.randint(0, len(missing) - 1)
            u, v = missing[idx]
            
            if can_add(adj, u, v):
                adj[u] |= (1 << v)
                adj[v] |= (1 << u)
                current_edges += 1
                missing.pop(idx)
                no_improve = 0
                if current_edges > best_edges:
                    best_edges = current_edges
                    best_adj = adj[:]
            else:
                no_improve += 1
                if no_improve >= max_no_improve:
                    # Kick: remove random edges
                    kick_size = random.choice([1, 2, 3])
                    existing = [(i, j) for i in range(n) for j in range(i+1, n) if (adj[i] & (1 << j))]
                    random.shuffle(existing)
                    for a, b in existing[:kick_size]:
                        adj[a] &= ~(1 << b)
                        adj[b] &= ~(1 << a)
                        current_edges -= 1
                    
                    # Greedy fill
                    adj, current_edges = greedy_fill(adj)
                    
                    if current_edges > best_edges:
                        best_edges = current_edges
                        best_adj = adj[:]
                        no_improve = 0
                    else:
                        # Revert to best if stuck
                        adj = best_adj[:]
                        current_edges = best_edges
                        missing = [(i, j) for i in range(n) for j in range(i+1, n) if not (adj[i] & (1 << j))]
                        no_improve = 0
                        
                    missing = [(i, j) for i in range(n) for j in range(i+1, n) if not (adj[i] & (1 << j))]
                    
        return best_adj, best_edges

    best_adj = [0] * n
    best_edges = 0
    
    # Multiple restarts
    for i in range(2000):
        if time.time() > deadline - 5:
            break
            
        random.seed(i * 123 + 456)
        adj = [0] * n
        
        # Strategy: Random dense start
        # Start with a random bipartite graph?
        # Or just random edges?
        # Let's try starting with a random set of edges
        num_initial = random.randint(20, 60)
        pairs = all_edges[:]
        random.shuffle(pairs)
        
        for u, v in pairs[:num_initial]:
            if can_add(adj, u, v):
                adj[u] |= (1 << v)
                adj[v] |= (1 << u)
        
        # Local search
        ls_time = min(1.0, max(0.2, (deadline - time.time()) * 0.2))
        adj, edges = local_search(adj, ls_time)
        
        if edges > best_edges:
            best_edges = edges
            best_adj = adj[:]
            
    # Final polish
    if time.time() < deadline - 1:
        best_adj, final_edges = greedy_fill(best_adj)
        if final_edges > best_edges:
            best_edges = final_edges
            
    result = []
    for i in range(n):
        for j in range(i+1, n):
            if best_adj[i] & (1 << j):
                result.append([i, j])
                
    return result

if __name__ == "__main__":
    edges = construct()
    print(f"Found {len(edges)} edges")
