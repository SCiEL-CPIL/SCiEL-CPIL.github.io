import random
import time
import math

def construct():
    n = 20
    adj = [0] * n
    
    def popcount(x):
        return bin(x).count('1')
        
    def is_valid_addition(u, v):
        # Check if adding (u, v) creates a K3,3
        # For any w != u, v:
        # 1. If v in N(w) and v in N(z) for some z? No.
        # Condition:
        # For all x, y != u, v:
        #   If v in N(x) and v in N(y):
        #       popcount(N(u) & N(x) & N(y)) must be <= 1
        #   If u in N(x) and u in N(y):
        #       popcount(N(v) & N(x) & N(y)) must be <= 1
        
        # Optimization:
        # Iterate x in range(n):
        #   if x == u or x == v: continue
        #   Iterate y in range(x+1, n):
        #       if y == u or y == v: continue
        #       if (adj[x] >> v) & 1 and (adj[y] >> v) & 1:
        #           if popcount(adj[u] & adj[x] & adj[y]) >= 2: return False
        #       if (adj[x] >> u) & 1 and (adj[y] >> u) & 1:
        #           if popcount(adj[v] & adj[x] & adj[y]) >= 2: return False
        #       return True
        
        for x in range(n):
            if x == u or x == v:
                continue
            for y in range(x + 1, n):
                if y == u or y == v:
                    continue
                # Check if v is common neighbor of x and y
                if ((adj[x] >> v) & 1) and ((adj[y] >> v) & 1):
                    if popcount(adj[u] & adj[x] & adj[y]) >= 2:
                        return False
                # Check if u is common neighbor of x and y
                if ((adj[x] >> u) & 1) and ((adj[y] >> u) & 1):
                    if popcount(adj[v] & adj[x] & adj[y]) >= 2:
                        return False
        return True

    def add_edge(u, v):
        adj[u] |= (1 << v)
        adj[v] |= (1 << u)
        
    def remove_edge(u, v):
        adj[u] &= ~(1 << v)
        adj[v] &= ~(1 << u)
        
    def edge_count():
        count = 0
        for u in range(n):
            count += popcount(adj[u])
        return count // 2
        
    def find_k33():
        # Returns (u, v, w, x, y, z) or None
        # u,v,w are the "left" part, x,y,z are the "right" part (common neighbors)
        for u in range(n):
            for v in range(u + 1, n):
                for w in range(v + 1, n):
                    common = adj[u] & adj[v] & adj[w]
                    if popcount(common) >= 3:
                        # Extract 3 bits
                        bits = []
                        temp = common
                        while len(bits) < 3 and temp:
                            if temp & 1:
                                bits.append(0)
                            temp >>= 1
                            # This logic is wrong for extracting indices
                        # Better:
                        indices = [i for i in range(n) if (common >> i) & 1]
                        if len(indices) >= 3:
                            x, y, z = indices[0], indices[1], indices[2]
                            return (u, v, w, x, y, z)
        return None

    def repair_graph():
        # Remove edges until no K3,3
        # To make it stochastic, we can pick random edges from the K3,3 to remove
        while True:
            k33 = find_k33()
            if not k33:
                break
            u, v, w, x, y, z = k33
            # The K3,3 edges are between {u,v,w} and {x,y,z}
            # There are 9 edges. Pick one at random.
            left = [u, v, w]
            right = [x, y, z]
            lu = random.choice(left)
            rv = random.choice(right)
            # Ensure edge exists (it should)
            if (adj[lu] >> rv) & 1:
                remove_edge(lu, rv)
            else:
                # Fallback: remove any edge
                remove_edge(u, x)

    def generate_initial():
        nonlocal adj
        adj = [0] * n
        # Start with a random dense graph
        p = 0.5
        for u in range(n):
            for v in range(u + 1, n):
                if random.random() < p:
                    add_edge(u, v)
        # Repair
        repair_graph()

    def simulate_annealing(start_time, deadline):
        nonlocal adj
        best_adj = adj[:]
        best_count = edge_count()
        
        current_count = best_count
        T = 10.0
        cooling_rate = 0.999
        min_T = 0.1
        
        all_edges = [(u, v) for u in range(n) for v in range(u + 1, n)]
        
        while time.time() < deadline:
            if T < min_T:
                # Restart
                generate_initial()
                current_count = edge_count()
                if current_count > best_count:
                    best_count = current_count
                    best_adj = adj[:]
                T = 10.0
                continue
                
            move_type = random.random()
            
            if move_type < 0.8:
                # Add edge
                # Pick random missing edge
                # Optimization: Precompute missing? Or just pick u,v and check if missing
                u = random.randint(0, n-1)
                v = random.randint(0, n-1)
                if u == v:
                    continue
                if u > v:
                    u, v = v, u
                if (adj[u] >> v) & 1:
                    continue # Already present
                
                if is_valid_addition(u, v):
                    add_edge(u, v)
                    current_count += 1
                    if current_count > best_count:
                        best_count = current_count
                        best_adj = adj[:]
                else:
                    # Don't accept
                    pass
            else:
                # Remove edge
                u = random.randint(0, n-1)
                v = random.randint(0, n-1)
                if u == v:
                    continue
                if u > v:
                    u, v = v, u
                if not ((adj[u] >> v) & 1):
                    continue # Not present
                
                remove_edge(u, v)
                current_count -= 1
                
                # Accept with prob exp(-delta/T) where delta = -1
                prob = math.exp(1.0 / T)
                if random.random() < prob:
                    if current_count > best_count:
                        best_count = current_count
                        best_adj = adj[:]
                else:
                    add_edge(u, v)
                    current_count += 1
                    
            T *= cooling_rate
            
        return best_adj, best_count

    start_time = time.time()
    deadline = start_time + 115.0
    
    # Run multiple restarts implicitly via the SA loop
    best_adj, best_count = simulate_annealing(start_time, deadline)
    
    best_edges = []
    for u in range(n):
        for v in range(u + 1, n):
            if (best_adj[u] >> v) & 1:
                best_edges.append([u, v])
                
    if not best_edges:
        return [[a, b] for a in range(2) for b in range(2, 20)]
        
    return best_edges
