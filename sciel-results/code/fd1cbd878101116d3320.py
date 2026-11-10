import time
import random
import math

def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin before 120s limit
    
    n = 20
    
    # Precompute bit masks for vertices
    def popcount(x):
        return bin(x).count('1')
        
    def can_add(adj, u, v):
        # Check if adding edge (u,v) keeps graph K2,3-free
        if (adj[u] >> v) & 1:
            return False
            
        # 1. Pair (u,v) common neighbors must be <= 2
        common_uv = adj[u] & adj[v]
        if popcount(common_uv) > 2:
            return False
            
        # 2. For all w in N(v), pair (u,w) gains v as common neighbor.
        #    So current common(u,w) must be <= 1.
        w = adj[v]
        while w:
            lsb = w & (-w)
            w ^= lsb
            wi = lsb.bit_length() - 1
            if popcount(adj[u] & adj[wi]) > 1:
                return False
                
        # 3. For all w in N(u), pair (v,w) gains u as common neighbor.
        #    So current common(v,w) must be <= 1.
        w = adj[u]
        while w:
            lsb = w & (-w)
            w ^= lsb
            wi = lsb.bit_length() - 1
            if popcount(adj[v] & adj[wi]) > 1:
                return False
                
        return True

    def add_edge(adj, u, v):
        adj[u] |= (1 << v)
        adj[v] |= (1 << u)
        
    def remove_edge(adj, u, v):
        adj[u] &= ~(1 << v)
        adj[v] &= ~(1 << u)
        
    def count_edges(adj):
        return sum(popcount(adj[i]) for i in range(n)) // 2
        
    def adj_to_edges(adj):
        edges = []
        for u in range(n):
            v_mask = adj[u]
            while v_mask:
                lsb = v_mask & (-v_mask)
                v_mask ^= lsb
                v = lsb.bit_length() - 1
                if u < v:
                    edges.append([u, v])
        return edges

    def greedy_build(adj):
        # Randomly add edges while maintaining K2,3-free property
        # Use a shuffled list of all possible edges
        pairs = [(u, v) for u in range(n) for v in range(u + 1, n)]
        random.shuffle(pairs)
        
        for u, v in pairs:
            if time.time() > deadline:
                break
            if can_add(adj, u, v):
                add_edge(adj, u, v)

    def local_search_adds(adj):
        # Try to add any possible edge repeatedly until none are left
        improved = True
        while improved and time.time() < deadline:
            improved = False
            pairs = [(u, v) for u in range(n) for v in range(u + 1, n) if not ((adj[u] >> v) & 1)]
            random.shuffle(pairs)
            for u, v in pairs:
                if can_add(adj, u, v):
                    add_edge(adj, u, v)
                    improved = True
        return adj

    def simulated_annealing(adj, max_iters=50000, initial_temp=1.0, cooling=0.999):
        current_adj = adj[:]
        current_edges = count_edges(current_adj)
        best_adj = current_adj[:]
        best_edges = current_edges
        temp = initial_temp
        
        # Precompute list of all edges for random selection
        all_pairs = [(u, v) for u in range(n) for v in range(u + 1, n)]
        
        for it in range(max_iters):
            if time.time() > deadline:
                break
                
            move_type = random.random()
            
            if move_type < 0.5:
                # Try to add a random missing edge
                # Pick a random pair
                u, v = random.choice(all_pairs)
                if (current_adj[u] >> v) & 1:
                    continue
                if can_add(current_adj, u, v):
                    add_edge(current_adj, u, v)
                    current_edges += 1
                    if current_edges > best_edges:
                        best_edges = current_edges
                        best_adj = current_adj[:]
                        
            elif move_type < 0.8:
                # Try to remove a random edge (to escape local optima)
                # Pick a random existing edge
                # Efficient way: pick random u, then random v from adj[u]
                u = random.randint(0, n - 1)
                mask = current_adj[u]
                if mask == 0:
                    continue
                # Pick a random set bit
                # Use random selection from bits
                # Or just pick a random v and check if it's a neighbor
                # Since n=20, we can just pick a random v and check
                v = random.randint(0, n - 1)
                if u == v or not ((current_adj[u] >> v) & 1):
                    continue
                
                remove_edge(current_adj, u, v)
                new_edges = current_edges - 1
                delta = -1 # Energy increases by 1 (we maximize edges, so removing is bad)
                # Accept if rand < exp(-delta/T) = exp(1/T)? 
                # Standard SA: minimize Energy. Here Maximize Edges. 
                # Let Energy = -Edges. 
                # Removing edge: Energy increases by 1. 
                # Accept if rand < exp(-1/T).
                if random.random() < math.exp(-1.0 / temp):
                    current_edges = new_edges
                else:
                    add_edge(current_adj, u, v)
                    
            else:
                # Swap: remove one edge, add another
                # Pick a random existing edge to remove
                u_rem = random.randint(0, n - 1)
                mask = current_adj[u_rem]
                if mask == 0:
                    continue
                v_rem = random.randint(0, n - 1)
                if u_rem == v_rem or not ((current_adj[u_rem] >> v_rem) & 1):
                    continue
                
                # Pick a random missing edge to add
                u_add = random.randint(0, n - 1)
                v_add = random.randint(0, n - 1)
                if u_add == v_add or (u_add == u_rem and v_add == v_rem) or (u_add == v_rem and v_add == u_rem):
                    continue
                if (current_adj[u_add] >> v_add) & 1:
                    continue
                    
                # Remove the first edge
                remove_edge(current_adj, u_rem, v_rem)
                
                # Check if we can add the second edge
                if can_add(current_adj, u_add, v_add):
                    add_edge(current_adj, u_add, v_add)
                    # Edge count is same. Accept to explore structure.
                    # We can use a small probability to revert if we want to bias towards "stable" structures,
                    # but usually we accept swaps to move in the landscape.
                    # Let's accept with probability 1.0 to ensure exploration.
                    # Or accept with 0.5 to allow some noise.
                    pass 
                else:
                    # Revert
                    add_edge(current_adj, u_rem, v_rem)
                    
            temp *= cooling
            if temp < 0.01:
                temp = 0.01
                
        return best_adj, best_edges

    # Multi-start approach
    best_adj = [0] * n
    best_edges = 0
    
    # Run multiple restarts
    num_starts = 100 # Increase starts to cover more of the search space
    for s in range(num_starts):
        if time.time() > deadline:
            break
            
        adj = [0] * n
        greedy_build(adj)
        adj = local_search_adds(adj)
        
        # Apply simulated annealing
        sa_adj, sa_edges = simulated_annealing(adj)
        
        if sa_edges > best_edges:
            best_edges = sa_edges
            best_adj = sa_adj[:]
            
    # Final polish: try to add any remaining edges
    best_adj = local_search_adds(best_adj)
    
    return adj_to_edges(best_adj)
