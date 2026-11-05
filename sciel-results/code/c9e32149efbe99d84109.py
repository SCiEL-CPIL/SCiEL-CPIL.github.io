import time
import random
import math

def construct():
    start_time = time.time()
    deadline = start_time + 145.0
    n = 20
    all_edges = [(u, v) for u in range(n) for v in range(u + 1, n)]
    E = len(all_edges)
    
    # Precompute edge index for quick lookup? Not strictly needed if we use bitmasks.
    
    def get_edges(adj):
        edges = []
        for u in range(n):
            mask = adj[u]
            while mask:
                lsb = mask & -mask
                v = lsb.bit_length() - 1
                if u < v:
                    edges.append([u, v])
                mask -= lsb
        return edges

    def count_edges(adj):
        c = 0
        for u in range(n):
            c += adj[u].bit_count()
        return c // 2

    def is_valid(adj):
        for u in range(n):
            for v in range(u + 1, n):
                if (adj[u] & adj[v]).bit_count() >= 3:
                    return False
        return True

    def repair(adj):
        # Remove edges to fix violations
        # Find a violating pair
        while True:
            found_violation = False
            for a in range(n):
                for b in range(a + 1, n):
                    common = adj[a] & adj[b]
                    if common.bit_count() >= 3:
                        found_violation = True
                        # Pick a common neighbor w
                        # Remove edge (a, w) or (b, w)
                        # Randomly pick one to remove
                        w_list = []
                        mask = common
                        while mask:
                            lsb = mask & -mask
                            w = lsb.bit_length() - 1
                            w_list.append(w)
                            mask -= lsb
                        w = random.choice(w_list)
                        # Decide to remove (a,w) or (b,w)
                        if random.random() < 0.5:
                            adj[a] &= ~(1 << w)
                            adj[w] &= ~(1 << a)
                        else:
                            adj[b] &= ~(1 << w)
                            adj[w] &= ~(1 << b)
                        break
                if found_violation:
                    break
            if not found_violation:
                break

    def can_add(adj, u, v):
        # Check if adding (u,v) is valid
        # 1. For each w in N(v), if w != u, check common(u,w) < 2
        mask_v = adj[v]
        # Exclude u from mask_v? u is not in N(v) yet.
        # We need to check w in N(v).
        while mask_v:
            lsb = mask_v & -mask_v
            w = lsb.bit_length() - 1
            if w != u:
                if (adj[u] & adj[w]).bit_count() >= 2:
                    return False
            mask_v -= lsb
            
        # 2. For each w in N(u), if w != v, check common(v,w) < 2
        mask_u = adj[u]
        while mask_u:
            lsb = mask_u & -mask_u
            w = lsb.bit_length() - 1
            if w != v:
                if (adj[v] & adj[w]).bit_count() >= 2:
                    return False
            mask_u -= lsb
            
        return True

    best_adj = [0] * n
    best_count = 0
    
    def update_best(adj):
        nonlocal best_adj, best_count
        c = count_edges(adj)
        if c > best_count:
            best_count = c
            best_adj = adj[:]

    # Simulated Annealing
    num_restarts = 100
    
    for restart in range(num_restarts):
        if time.time() > deadline - 5.0:
            break
            
        # Initialize
        adj = [0] * n
        p = random.uniform(0.3, 0.5)
        for u, v in all_edges:
            if random.random() < p:
                adj[u] |= (1 << v)
                adj[v] |= (1 << u)
                
        repair(adj)
        
        T = 10.0
        cooling = 0.999
        min_T = 0.01
        steps = 0
        max_steps = 20000
        
        edge_count = count_edges(adj)
        
        while T > min_T and steps < max_steps:
            if time.time() > deadline - 2.0:
                break
                
            # Random move
            if random.random() < 0.7:
                # Try to add
                # Pick random missing edge
                while True:
                    u = random.randint(0, n-1)
                    v = random.randint(0, n-1)
                    if u == v: continue
                    if u > v: u, v = v, u
                    if (adj[u] >> v) & 1:
                        continue
                    # Found a missing edge
                    break
                
                if can_add(adj, u, v):
                    # Accept
                    adj[u] |= (1 << v)
                    adj[v] |= (1 << u)
                    edge_count += 1
                    if edge_count > best_count:
                        update_best(adj)
                else:
                    # Reject add
                    pass
            else:
                # Try to remove
                # Pick random existing edge
                if edge_count == 0:
                    continue
                while True:
                    u = random.randint(0, n-1)
                    v = random.randint(0, n-1)
                    if u == v: continue
                    if u > v: u, v = v, u
                    if not ((adj[u] >> v) & 1):
                        continue
                    break
                
                # Calculate probability
                # Delta E = -1 (since we maximize edges, removing decreases score)
                # Accept if random() < exp(1/T) ? No, exp(-delta/T) where delta is increase in energy.
                # Energy = -edge_count.
                # New Energy = -(edge_count - 1) = -edge_count + 1.
                # Delta E = 1.
                # Prob = exp(-1/T)
                if random.random() < math.exp(-1.0 / T):
                    adj[u] &= ~(1 << v)
                    adj[v] &= ~(1 << u)
                    edge_count -= 1
            T *= cooling
            steps += 1
            
        # Final greedy fill
        improved = True
        while improved:
            improved = False
            for u, v in all_edges:
                if not ((adj[u] >> v) & 1):
                    if can_add(adj, u, v):
                        adj[u] |= (1 << v)
                        adj[v] |= (1 << u)
                        edge_count += 1
                        improved = True
        update_best(adj)

    # Return best
    return get_edges(best_adj)

def _construct_greedy():
    pass

def _greedy_construct(n, all_edges, seed):
    pass

def _can_add_edge_fast(adj, u, v):
    pass

def _local_search(n, edges, all_edges, deadline):
    pass
