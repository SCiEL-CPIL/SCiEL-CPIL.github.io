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
        # Condition:
        # For all x, y != u, v:
        #   If v in N(x) and v in N(y):
        #       popcount(N(u) & N(x) & N(y)) must be <= 1
        #   If u in N(x) and u in N(y):
        #       popcount(N(v) & N(x) & N(y)) must be <= 1
        
        # Optimization:
        # Iterate x in N(v) (neighbors of v, excluding u)
        # For each x, iterate y in N(v) (neighbors of v, excluding u, x)
        # Check popcount(N(u) & N(x) & N(y)) <= 1
        
        # Also iterate x in N(u) (neighbors of u, excluding v)
        # For each x, iterate y in N(u) (neighbors of u, excluding v, x)
        # Check popcount(N(v) & N(x) & N(y)) <= 1
        
        # Note: N(v) does not contain u yet, so u is not in the list.
        # But we must ensure x, y are not u or v.
        # Since u not in N(v) and v not in N(u), this is naturally handled if we just iterate neighbors.
        # However, x and y must be distinct.
        
        mask_v = adj[v]
        mask_u = adj[u]
        
        # Check pairs in N(v)
        # We need to check if any pair x,y in N(v) has >= 2 common neighbors with u.
        # i.e., popcount(mask_u & adj[x] & adj[y]) >= 2
        
        # Efficient way:
        # For each x in N(v):
        #    For each y in N(v), y > x:
        #       if popcount(mask_u & adj[x] & adj[y]) >= 2: return False
        
        # This is O(deg(v)^2).
        
        # To make it faster, we can precompute? No, adj changes.
        
        # Let's just iterate. n=20 is small.
        
        # Get list of neighbors of v
        # Since n=20, we can just iterate 0..19 and check bit.
        # Or iterate bits.
        
        # Iterate x in 0..19
        for x in range(n):
            if x == u or x == v: continue
            if not (mask_v & (1 << x)): continue # x not in N(v)
            
            # Iterate y in x+1..19
            for y in range(x + 1, n):
                if y == u or y == v: continue
                if not (mask_v & (1 << y)): continue # y not in N(v)
                
                # Check common neighbors of u, x, y
                # We want to know if count >= 2
                common = mask_u & adj[x] & adj[y]
                # Quick check: if common has >= 2 bits
                # (common & (common - 1)) != 0 implies >= 2 bits
                if common & (common - 1):
                    return False
        
        # Check pairs in N(u)
        # For each x in N(u):
        #    For each y in N(u), y > x:
        #       if popcount(mask_v & adj[x] & adj[y]) >= 2: return False
        
        for x in range(n):
            if x == u or x == v: continue
            if not (mask_u & (1 << x)): continue # x not in N(u)
            
            for y in range(x + 1, n):
                if y == u or y == v: continue
                if not (mask_u & (1 << y)): continue # y not in N(u)
                
                common = mask_v & adj[x] & adj[y]
                if common & (common - 1):
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
        for u in range(n):
            for v in range(u + 1, n):
                for w in range(v + 1, n):
                    common = adj[u] & adj[v] & adj[w]
                    # Check if >= 3 bits
                    if common & (common - 1) & (common - 2): # This logic is tricky
                        # Just use popcount for simplicity in repair
                        pass
                    if popcount(common) >= 3:
                        indices = [i for i in range(n) if (common >> i) & 1]
                        if len(indices) >= 3:
                            x, y, z = indices[0], indices[1], indices[2]
                            return (u, v, w, x, y, z)
        return None

    def repair_graph():
        # Remove edges until no K3,3
        while True:
            k33 = find_k33()
            if not k33:
                break
            u, v, w, x, y, z = k33
            left = [u, v, w]
            right = [x, y, z]
            lu = random.choice(left)
            rv = random.choice(right)
            if (adj[lu] >> rv) & 1:
                remove_edge(lu, rv)
            else:
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
        repair_graph()

    def simulate_annealing(start_time, deadline):
        nonlocal adj
        best_adj = adj[:]
        best_count = edge_count()
        
        current_count = best_count
        T = 10.0
        cooling_rate = 0.999
        min_T = 0.1
        
        # Precompute all possible edges
        all_edges = [(u, v) for u in range(n) for v in range(u + 1, n)]
        
        while time.time() < deadline:
            if T < min_T:
                generate_initial()
                current_count = edge_count()
                if current_count > best_count:
                    best_count = current_count
                    best_adj = adj[:]
                T = 10.0
                continue
                
            move_type = random.random()
            
            if move_type < 0.6:
                # Add edge
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
                    pass
            elif move_type < 0.8:
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
                
                prob = math.exp(1.0 / T)
                if random.random() < prob:
                    if current_count > best_count:
                        best_count = current_count
                        best_adj = adj[:]
                else:
                    add_edge(u, v)
                    current_count += 1
            else:
                # Swap: Remove (u,v), Add (x,y)
                # Pick random existing edge
                u = random.randint(0, n-1)
                v = random.randint(0, n-1)
                if u == v:
                    continue
                if u > v:
                    u, v = v, u
                if not ((adj[u] >> v) & 1):
                    continue # Not present
                    
                # Pick random non-existing edge
                x = random.randint(0, n-1)
                y = random.randint(0, n-1)
                if x == y:
                    continue
                if x > y:
                    x, y = y, x
                if (adj[x] >> y) & 1:
                    continue # Already present
                
                # Perform swap
                remove_edge(u, v)
                # Check if adding (x,y) is valid
                if is_valid_addition(x, y):
                    add_edge(x, y)
                    # Edge count remains same
                    if current_count > best_count:
                        best_count = current_count
                        best_adj = adj[:]
                else:
                    # Revert
                    add_edge(u, v)
                    # Don't accept swap, revert
                    # Note: we removed (u,v) and didn't add (x,y).
                    # We must add (u,v) back to restore state.
                    # The state is now restored.
                    
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
