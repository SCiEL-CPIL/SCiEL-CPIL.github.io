import time
import random
import math

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin before 120s limit
    
    n = 20
    
    def popcount(x):
        return bin(x).count('1')
        
    # Precompute bit masks for vertices
    # adj[i] is a bitmask where bit j is 1 if edge (i,j) exists
    
    def can_add(adj, u, v):
        """Check if adding edge (u,v) keeps graph K2,3-free."""
        if (adj[u] >> v) & 1:
            return False
        
        # 1. Pair (u,v) common neighbors must be <= 2
        common_uv = adj[u] & adj[v]
        if popcount(common_uv) > 2:
            return False
        
        # 2. For all w in N(v) \ {u}: pair (u,w) gains v as common neighbor
        #    Need popcount(adj[u] & adj[w]) <= 1
        #    Note: v is not in adj[u] yet.
        mask_w = adj[v]
        while mask_w:
            lsb = mask_w & (-mask_w)
            w = lsb.bit_length() - 1
            mask_w ^= lsb
            if w == u:
                continue
            if popcount(adj[u] & adj[w]) > 1:
                return False
        
        # 3. For all w in N(u) \ {v}: pair (v,w) gains u as common neighbor
        #    Need popcount(adj[v] & adj[w]) <= 1
        mask_w = adj[u]
        while mask_w:
            lsb = mask_w & (-mask_w)
            w = lsb.bit_length() - 1
            mask_w ^= lsb
            if w == v:
                continue
            if popcount(adj[v] & adj[w]) > 1:
                return False
        
        return True
    
    def add_edge(adj, u, v):
        adj[u] |= (1 << v)
        adj[v] |= (1 << u)
        
    def remove_edge(adj, u, v):
        adj[u] &= ~(1 << v)
        adj[v] &= ~(1 << u)
        
    def edge_count(adj):
        total = 0
        for i in range(n):
            total += popcount(adj[i])
        return total // 2
        
    def greedy_add(adj):
        """Greedily add any valid edge until none remain."""
        improved = True
        while improved:
            improved = False
            for u in range(n):
                for v in range(u + 1, n):
                    if not ((adj[u] >> v) & 1):
                        if can_add(adj, u, v):
                            add_edge(adj, u, v)
                            improved = True
        return adj

    def simulated_annealing(adj_init, max_steps=50000):
        """Optimize a graph using simulated annealing."""
        adj = list(adj_init)
        best_adj = list(adj)
        best_count = edge_count(adj)
        current_count = best_count
        
        T = 10.0
        T_min = 0.1
        cooling = 0.9999
        step = 0
        stuck_counter = 0
        
        while step < max_steps:
            if time.time() > deadline:
                break
            
            # Pick a random move: add or remove an edge
            u = random.randint(0, n - 1)
            v = random.randint(0, n - 1)
            if u == v:
                continue
            if u > v:
                u, v = v, u
            
            has_edge = (adj[u] >> v) & 1
            
            if has_edge:
                # Try removing edge
                remove_edge(adj, u, v)
                new_count = current_count - 1
                delta = -1
                # Accept with probability
                if random.random() < math.exp(delta / (T + 0.01)):
                    current_count = new_count
                    if current_count > best_count:
                        best_count = current_count
                        best_adj = list(adj)
                        stuck_counter = 0
                    else:
                        stuck_counter += 1
                else:
                    add_edge(adj, u, v)
            else:
                # Try adding edge
                if can_add(adj, u, v):
                    add_edge(adj, u, v)
                    current_count += 1
                    if current_count > best_count:
                        best_count = current_count
                        best_adj = list(adj)
                        stuck_counter = 0
                    else:
                        stuck_counter += 1
                else:
                    stuck_counter += 1
            
            step += 1
            T *= cooling
            if T < T_min:
                T = T_min
                
            # If stuck for too long, restart temperature or break
            if stuck_counter > 5000:
                # Reset temperature to kick out of local minimum
                T = 5.0
                stuck_counter = 0
                # Or break and rely on multi-start
                # break 

        return best_adj, best_count

    # Multi-start strategy
    best_adj = None
    best_count = 0
    
    # 1. Deterministic Greedy Start
    adj0 = [0] * n
    adj0 = greedy_add(adj0)
    c0 = edge_count(adj0)
    if c0 > best_count:
        best_count = c0
        best_adj = list(adj0)
        
    # 2. Randomized Starts
    num_starts = 5
    for seed in range(num_starts):
        if time.time() > deadline - 10:
            break
            
        random.seed(seed * 999 + 123)
        
        # Initialize with random edges, then greedy repair/add
        adj_greedy = [0] * n
        # Add some random edges to break symmetry
        for _ in range(20):
            u = random.randint(0, n-1)
            v = random.randint(0, n-1)
            if u != v and not ((adj_greedy[u] >> v) & 1):
                if can_add(adj_greedy, u, v):
                    add_edge(adj_greedy, u, v)
        
        adj_greedy = greedy_add(adj_greedy)
        c_greedy = edge_count(adj_greedy)
        
        if c_greedy > best_count:
            best_count = c_greedy
            best_adj = list(adj_greedy)
            
        # Run SA
        adj_sa, c_sa = simulated_annealing(adj_greedy, max_steps=30000)
        if c_sa > best_count:
            best_count = c_sa
            best_adj = list(adj_sa)
            
        # Also run SA on the current global best to refine it
        if time.time() < deadline - 15:
            adj_sa2, c_sa2 = simulated_annealing(best_adj, max_steps=20000)
            if c_sa2 > best_count:
                best_count = c_sa2
                best_adj = list(adj_sa2)
                
    # Final Polish
    if best_adj is not None and time.time() < deadline - 5:
        adj_final, c_final = simulated_annealing(best_adj, max_steps=50000)
        if c_final > best_count:
            best_count = c_final
            best_adj = list(adj_final)
            
    # Convert to edge list
    edges = []
    for u in range(n):
        for v in range(u + 1, n):
            if (best_adj[u] >> v) & 1:
                edges.append([u, v])
                
    return edges
