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
    
    # Fast popcount using Python's int.bit_count() (available in 3.10+)
    # Fallback for older versions
    if hasattr(int, 'bit_count'):
        def popcount(x):
            return x.bit_count()
    else:
        # Lookup table for bytes
        _pc = [0] * 256
        for i in range(256):
            _pc[i] = bin(i).count('1')
        def popcount(x):
            c = 0
            while x:
                c += _pc[x & 0xFF]
                x >>= 8
            return c
        
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
        neighbors_v = adj[v]
        while neighbors_v:
            lsb = neighbors_v & -neighbors_v
            w = lsb.bit_length() - 1
            neighbors_v ^= lsb
            if w == u:
                continue
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

    def get_neighbors(u):
        """Return list of neighbors of u."""
        result = []
        temp = adj[u]
        while temp:
            lsb = temp & -temp
            w = lsb.bit_length() - 1
            result.append(w)
            temp ^= lsb
        return result

    def greedy_init():
        """Generate a random greedy K2,3-free graph."""
        nonlocal adj
        adj = [0] * n
        all_edges = [(u, v) for u in range(n) for v in range(u + 1, n)]
        random.shuffle(all_edges)
        
        for u, v in all_edges:
            if can_add(u, v):
                add_edge(u, v)
        return edge_count()

    # Multiple greedy restarts to find a good starting point
    best_adj = [0] * n
    best_count = 0
    
    num_greedy_restarts = 20
    for _ in range(num_greedy_restarts):
        cnt = greedy_init()
        if cnt > best_count:
            best_count = cnt
            best_adj = adj[:]
    
    # Simulated Annealing with restarts
    T = 10.0
    cooling = 0.998
    min_T = 0.05
    stagnation = 0
    max_stagnation = 3000
    
    while time.time() < deadline:
        if T < min_T:
            # Restart with higher temperature
            T = 5.0
            adj = best_adj[:]
            stagnation = 0
            continue
            
        current_count = edge_count()
        
        # Choose move type with adjusted probabilities
        r = random.random()
        
        if r < 0.50:
            # Try to add an edge
            u = random.randint(0, n - 1)
            v = random.randint(0, n - 1)
            if u == v:
                continue
            if (adj[u] >> v) & 1:
                continue
            
            if can_add(u, v):
                add_edge(u, v)
                new_count = edge_count()
                if new_count > best_count:
                    best_count = new_count
                    best_adj = adj[:]
                    stagnation = 0
                else:
                    stagnation += 1
            else:
                stagnation += 1
                
        elif r < 0.80:
            # Try to remove an edge
            u = random.randint(0, n - 1)
            neighbors = adj[u]
            if neighbors == 0:
                continue
            neigh_list = get_neighbors(u)
            if not neigh_list:
                continue
            v = random.choice(neigh_list)
            
            remove_edge(u, v)
            new_count = edge_count()
            delta = current_count - new_count  # delta is 1
            if random.random() < math.exp(-delta / T):
                if new_count > best_count:
                    best_count = new_count
                    best_adj = adj[:]
                    stagnation = 0
                else:
                    stagnation += 1
            else:
                add_edge(u, v)
                stagnation += 1
                
        else:
            # Try a swap: remove (u1, v1), add (u2, v2)
            u1 = random.randint(0, n - 1)
            neighbors1 = adj[u1]
            if neighbors1 == 0:
                continue
            neigh_list1 = get_neighbors(u1)
            if not neigh_list1:
                continue
            v1 = random.choice(neigh_list1)
            
            u2 = random.randint(0, n - 1)
            v2 = random.randint(0, n - 1)
            if u2 == v2:
                continue
            if (adj[u2] >> v2) & 1:
                continue
            
            remove_edge(u1, v1)
            
            if can_add(u2, v2):
                add_edge(u2, v2)
                new_count = edge_count()
                if new_count > best_count:
                    best_count = new_count
                    best_adj = adj[:]
                    stagnation = 0
                else:
                    stagnation += 1
            else:
                add_edge(u1, v1)
                stagnation += 1
                
        T *= cooling
        
        # If stuck too long, restart
        if stagnation > max_stagnation:
            T = 5.0
            adj = best_adj[:]
            stagnation = 0

    # Final local-search polish pass
    adj = best_adj[:]
    improved = True
    while improved and time.time() < deadline:
        improved = False
        # Try all single-edge additions
        for u in range(n):
            for v in range(u + 1, n):
                if (adj[u] >> v) & 1:
                    continue
                if can_add(u, v):
                    add_edge(u, v)
                    improved = True
                    if edge_count() > best_count:
                        best_count = edge_count()
                        best_adj = adj[:]
        
        # Try all swaps more efficiently
        if time.time() < deadline - 5:  # Leave time for finishing
            for u1 in range(n):
                if time.time() >= deadline - 5:
                    break
                for v1 in range(u1 + 1, n):
                    if not ((adj[u1] >> v1) & 1):
                        continue
                    for u2 in range(n):
                        if u2 == u1:
                            continue
                        for v2 in range(u2 + 1, n):
                            if (adj[u2] >> v2) & 1:
                                continue
                            # Skip if the swap would just re-add the same edge
                            if (u1 == u2 and v1 == v2) or (u1 == v2 and v1 == u2):
                                continue
                            remove_edge(u1, v1)
                            if can_add(u2, v2):
                                add_edge(u2, v2)
                                if edge_count() > best_count:
                                    best_count = edge_count()
                                    best_adj = adj[:]
                                    improved = True
                            else:
                                add_edge(u1, v1)
                            # Restore if we didn't improve
                            if not improved and not ((adj[u1] >> v1) & 1):
                                add_edge(u1, v1)
        
        # If we made improvements, reset adj to best and continue
        if improved:
            adj = best_adj[:]
        else:
            break

    # Convert best_adj back to list of edges
    edges = []
    for u in range(n):
        for v in range(u + 1, n):
            if (best_adj[u] >> v) & 1:
                edges.append([u, v])
                
    return edges
