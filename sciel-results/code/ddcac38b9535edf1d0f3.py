import random
import time
import math

def construct():
    n = 20
    # adj[u] is a bitmask of neighbors of u
    adj = [0] * n
    
    def popcount(x):
        return bin(x).count('1')
        
    # Precompute bit masks for vertices
    # Not strictly necessary, 1 << i is fast enough
    
    def is_valid_addition(u, v):
        # Check if adding edge (u, v) violates the constraint
        # Constraint: For all triples {a,b,c}, |N(a) & N(b) & N(c)| <= 2
        # Adding (u,v) updates N(u) and N(v).
        # Only triples containing u or v are affected.
        
        # Triples {u, w, z} where w, z != u, v
        # New common neighbors for {u, w, z} = (N(u) | (1<<v)) & N(w) & N(z)
        # Old common neighbors = N(u) & N(w) & N(z)
        # The new common neighbors are Old UNION ( {v} & N(w) & N(z) )
        # So the count increases by 1 if v is in N(w) and N(z).
        # We need to ensure that if v is in N(w) & N(z), then the old count was < 2.
        # i.e., if v in N(w) & N(z), then popcount(N(u) & N(w) & N(z)) must be <= 1.
        
        # Similarly for triples {v, w, z}
        # If u in N(w) & N(z), then popcount(N(v) & N(w) & N(z)) must be <= 1.
        
        # Iterate over all pairs w, z
        # Optimization: Only check pairs (w, z) where v is in N(w) & N(z) for u's check
        # and u is in N(w) & N(z) for v's check.
        
        for w in range(n):
            if w == u or w == v:
                continue
            for z in range(w + 1, n):
                if z == u or z == v:
                    continue
                
                # Check triple {u, w, z}
                if (adj[w] >> v) & 1 and (adj[z] >> v) & 1:
                    # v is a neighbor of both w and z
                    # Check if u already has 2 common neighbors with w and z
                    common = adj[u] & adj[w] & adj[z]
                    if popcount(common) >= 2:
                        return False
                
                # Check triple {v, w, z}
                if (adj[w] >> u) & 1 and (adj[z] >> u) & 1:
                    # u is a neighbor of both w and z
                    common = adj[v] & adj[w] & adj[z]
                    if popcount(common) >= 2:
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
        
    def generate_greedy_initial():
        nonlocal adj
        adj = [0] * n
        all_edges = [(u, v) for u in range(n) for v in range(u + 1, n)]
        random.shuffle(all_edges)
        for u, v in all_edges:
            if is_valid_addition(u, v):
                add_edge(u, v)
                
    # Simulated Annealing
    def simulate_annealing(start_time, deadline):
        nonlocal adj
        best_adj = adj[:]
        best_count = edge_count()
        
        # Initialize with greedy
        generate_greedy_initial()
        current_count = edge_count()
        if current_count > best_count:
            best_count = current_count
            best_adj = adj[:]
            
        # Parameters
        T = 50.0
        cooling_rate = 0.9995
        min_T = 0.01
        
        all_edges = [(u, v) for u in range(n) for v in range(u + 1, n)]
        
        while time.time() < deadline:
            if T < min_T:
                # Restart with a new greedy or random perturbation
                generate_greedy_initial()
                current_count = edge_count()
                if current_count > best_count:
                    best_count = current_count
                    best_adj = adj[:]
                T = 50.0
                continue
                
            # Pick a random move
            move_type = random.random()
            
            if move_type < 0.7:
                # Try to add a random missing edge
                # Find a missing edge
                missing = [e for e in all_edges if not ((adj[e[0]] >> e[1]) & 1)]
                if not missing:
                    continue
                u, v = random.choice(missing)
                if is_valid_addition(u, v):
                    add_edge(u, v)
                    current_count += 1
                    if current_count > best_count:
                        best_count = current_count
                        best_adj = adj[:]
                else:
                    # Don't accept invalid add
                    pass
            elif move_type < 0.9:
                # Try to remove a random edge
                # Find an existing edge
                existing = []
                for u in range(n):
                    for v in range(u + 1, n):
                        if (adj[u] >> v) & 1:
                            existing.append((u, v))
                if not existing:
                    continue
                u, v = random.choice(existing)
                remove_edge(u, v)
                current_count -= 1
                # Accept removal with probability based on temperature?
                # Since we want to maximize edges, we only accept removal if it helps later.
                # In SA, we accept worse moves with prob exp(-delta/T).
                # Delta = -1 (loss of 1 edge).
                prob = math.exp(1.0 / T)
                if random.random() < prob:
                    if current_count > best_count:
                        best_count = current_count
                        best_adj = adj[:]
                else:
                    # Revert
                    add_edge(u, v)
                    current_count += 1
            else:
                # Swap: Remove one edge, add another
                # Find existing and missing
                existing = []
                for u in range(n):
                    for v in range(u + 1, n):
                        if (adj[u] >> v) & 1:
                            existing.append((u, v))
                missing = [e for e in all_edges if not ((adj[e[0]] >> e[1]) & 1)]
                
                if existing and missing:
                    u1, v1 = random.choice(existing)
                    u2, v2 = random.choice(missing)
                    
                    # Try swap
                    remove_edge(u1, v1)
                    
                    if is_valid_addition(u2, v2):
                        add_edge(u2, v2)
                        # Count stays same
                        # Accept swap? Usually yes, as it explores the space of same size.
                        # But we need to ensure we don't loop.
                        if current_count > best_count:
                            best_count = current_count
                            best_adj = adj[:]
                    else:
                        # Revert
                        add_edge(u1, v1)
                        
            T *= cooling_rate
            
        return best_adj, best_count

    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin
    
    # Run SA
    best_adj, best_count = simulate_annealing(start_time, deadline)
    
    # Reconstruct edge list
    best_edges = []
    for u in range(n):
        for v in range(u + 1, n):
            if (best_adj[u] >> v) & 1:
                best_edges.append([u, v])
                
    if not best_edges:
        return [[a, b] for a in range(2) for b in range(2, 20)]
        
    return best_edges
