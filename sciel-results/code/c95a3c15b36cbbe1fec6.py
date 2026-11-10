import time
import random
import math

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin
    
    n = 20
    # Bitmask representation
    # adj[i] is an integer where bit j is 1 if edge (i,j) exists
    adj = [0] * n
    
    def popcount(x):
        return bin(x).count('1')
        
    def can_add(u, v):
        # Check if adding edge (u,v) keeps graph K2,3-free
        if (adj[u] >> v) & 1:
            return False
            
        # 1. Pair (u,v) common neighbors must be <= 2
        common_uv = adj[u] & adj[v]
        if popcount(common_uv) > 2:
            return False
            
        # 2. For all w in N(v), pair (u,w) gains v as common neighbor.
        #    So current common(u,w) must be <= 1.
        #    w iterates over set bits in adj[v]
        neighbors_v = adj[v]
        while neighbors_v:
            lsb = neighbors_v & -neighbors_v
            w = lsb.bit_length() - 1
            neighbors_v ^= lsb
            if w == u:
                continue
            # Check common neighbors of u and w
            # common(u,w) = adj[u] & adj[w]
            # If this count is > 1, adding v makes it >= 2? 
            # Wait. If count is 1, new count is 2. OK.
            # If count is 2, new count is 3. BAD.
            # So we need count <= 1.
            if popcount(adj[u] & adj[w]) > 1:
                return False
                
        # 3. For all w in N(u), pair (v,w) gains u as common neighbor.
        #    So current common(v,w) must be <= 1.
        neighbors_u = adj[u]
        while neighbors_u:
            lsb = neighbors_u & -neighbors_u
            w = lsb.bit_length() - 1
            neighbors_u ^= lsb
            if w == v:
                continue
            if popcount(adj[v] & adj[w]) > 1:
                return False
                
        return True

    def add_edge(u, v):
        adj[u] |= (1 << v)
        adj[v] |= (1 << u)
        
    def remove_edge(u, v):
        adj[u] &= ~(1 << v)
        adj[v] &= ~(1 << u)
        
    def edge_count():
        # Sum of degrees / 2
        total_deg = 0
        for i in range(n):
            total_deg += popcount(adj[i])
        return total_deg // 2

    # Initialize with a greedy maximal K2,3-free graph
    # Random order of edges
    all_edges = [(u, v) for u in range(n) for v in range(u + 1, n)]
    random.shuffle(all_edges)
    
    for u, v in all_edges:
        if can_add(u, v):
            add_edge(u, v)
            
    best_adj = adj[:]
    best_count = edge_count()
    
    # Simulated Annealing
    T = 10.0
    cooling = 0.999
    min_T = 0.1
    
    # Precompute list of all possible edges for random selection
    # We need to pick random non-edges and random edges
    # For efficiency, we can just pick random u, v
    
    while time.time() < deadline:
        if T < min_T:
            T = min_T
            
        current_count = edge_count()
        
        # Choose move type
        r = random.random()
        
        if r < 0.6:
            # Try to add an edge
            u = random.randint(0, n - 1)
            v = random.randint(0, n - 1)
            if u == v:
                continue
            if (adj[u] >> v) & 1:
                continue # Already exists
            
            if can_add(u, v):
                add_edge(u, v)
                new_count = edge_count()
                # Always accept if better (which it is, +1)
                if new_count > best_count:
                    best_count = new_count
                    best_adj = adj[:]
            else:
                # Reject, no change
                pass
                
        elif r < 0.9:
            # Try to remove an edge
            # Pick a random edge from current graph
            # To pick random edge, pick random u, then random neighbor v
            u = random.randint(0, n - 1)
            neighbors = adj[u]
            if neighbors == 0:
                continue
            # Pick random set bit
            # Get list of set bits or use random shift
            # Efficient way: pick random number of shifts? No, biased.
            # Just get list of neighbors for u
            neigh_list = []
            temp = neighbors
            while temp:
                lsb = temp & -temp
                w = lsb.bit_length() - 1
                neigh_list.append(w)
                temp ^= lsb
            
            if not neigh_list:
                continue
                
            v = random.choice(neigh_list)
            
            remove_edge(u, v)
            new_count = edge_count()
            delta = current_count - new_count # delta is 1 (energy increase)
            # Accept with prob exp(-delta / T)
            if random.random() < math.exp(-delta / T):
                # Accepted
                if new_count > best_count:
                    best_count = new_count
                    best_adj = adj[:]
            else:
                # Rejected, undo
                add_edge(u, v)
                
        else:
            # Try a swap: remove (u1, v1), add (u2, v2)
            # This helps escape local optima where no single add is valid
            # but a swap is.
            
            # Pick edge to remove
            u1 = random.randint(0, n - 1)
            neighbors1 = adj[u1]
            if neighbors1 == 0:
                continue
            neigh_list1 = []
            temp = neighbors1
            while temp:
                lsb = temp & -temp
                w = lsb.bit_length() - 1
                neigh_list1.append(w)
                temp ^= lsb
            if not neigh_list1:
                continue
            v1 = random.choice(neigh_list1)
            
            # Pick edge to add (must be non-edge)
            u2 = random.randint(0, n - 1)
            v2 = random.randint(0, n - 1)
            if u2 == v2:
                continue
            if (adj[u2] >> v2) & 1:
                continue
            
            # Check if swap is valid
            # Temporarily remove (u1, v1)
            remove_edge(u1, v1)
            
            if can_add(u2, v2):
                add_edge(u2, v2)
                new_count = edge_count()
                # Edge count is same as current_count
                # Accept with prob exp(0/T) = 1? 
                # Usually swaps are accepted if they lead to better future states.
                # In SA, if energy is same, accept with prob 1 (or high).
                # Let's accept it to explore.
                if new_count > best_count:
                    best_count = new_count
                    best_adj = adj[:]
            else:
                # Invalid, undo add (not done yet) and undo remove
                add_edge(u1, v1)
                
        T *= cooling

    # Convert best_adj back to list of edges
    edges = []
    for u in range(n):
        for v in range(u + 1, n):
            if (best_adj[u] >> v) & 1:
                edges.append([u, v])
                
    return edges
