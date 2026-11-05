import random
import time
import math

def construct():
    n = 20
    adj = [0] * n
    all_edges = [(u, v) for u in range(n) for v in range(u + 1, n)]
    
    def popcount(x):
        return bin(x).count('1')

    def is_valid_addition(u, v):
        # Check if adding edge (u, v) keeps the graph K3,3-free
        # Constraint: For all triples {a,b,c}, |N(a) & N(b) & N(c)| <= 2
        # Adding (u,v) affects triples containing u or v.
        
        # Triples {u, w, z}
        # New common neighbors = (N(u) | {v}) & N(w) & N(z)
        # Increase by 1 if v in N(w) & N(z).
        # If v in N(w) & N(z), we need old count < 2.
        # Old count = popcount(N(u) & N(w) & N(z)).
        
        # Triples {v, w, z}
        # If u in N(w) & N(z), we need popcount(N(v) & N(w) & N(z)) < 2.
        
        mask_u = adj[u]
        mask_v = adj[v]
        
        for w in range(n):
            if w == u or w == v:
                continue
            mask_w = adj[w]
            
            # Check if v is connected to w
            v_in_w = (mask_w >> v) & 1
            
            # Check if u is connected to w
            u_in_w = (mask_w >> u) & 1
            
            for z in range(w + 1, n):
                if z == u or z == v:
                    continue
                mask_z = adj[z]
                
                # Check triple {u, w, z}
                # Condition: v in N(w) AND v in N(z)
                if v_in_w and ((mask_z >> v) & 1):
                    common = mask_u & mask_w & mask_z
                    if popcount(common) >= 2:
                        return False
                
                # Check triple {v, w, z}
                # Condition: u in N(w) AND u in N(z)
                if u_in_w and ((mask_z >> u) & 1):
                    common = mask_v & mask_w & mask_z
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

    def get_edges():
        edges = []
        for u in range(n):
            for v in range(u + 1, n):
                if (adj[u] >> v) & 1:
                    edges.append((u, v))
        return edges

    def greedy_construct():
        nonlocal adj
        adj = [0] * n
        edges_copy = all_edges[:]
        random.shuffle(edges_copy)
        for u, v in edges_copy:
            if is_valid_addition(u, v):
                add_edge(u, v)
        
        # Polish
        improved = True
        while improved:
            improved = False
            for u, v in all_edges:
                if not ((adj[u] >> v) & 1):
                    if is_valid_addition(u, v):
                        add_edge(u, v)
                        improved = True
        return edge_count()

    def simulated_annealing(start_adj, time_limit):
        nonlocal adj
        adj = start_adj[:]
        current_edges = edge_count()
        best_adj = adj[:]
        best_edges = current_edges
        
        # List of current edges and missing edges for faster selection
        # Re-calculating these every step is O(E), which is fine for n=20
        
        T = 1.0
        cooling_rate = 0.999
        min_T = 0.01
        start_time = time.time()
        
        while time.time() - start_time < time_limit:
            if T < min_T:
                T = min_T
            
            # Choose move
            r = random.random()
            
            if r < 0.6:
                # Try to add a random missing edge
                # Pick random u, v
                u = random.randint(0, n - 1)
                v = random.randint(0, n - 1)
                if u == v:
                    continue
                if u > v:
                    u, v = v, u
                
                if (adj[u] >> v) & 1:
                    continue # Already an edge
                
                if is_valid_addition(u, v):
                    add_edge(u, v)
                    current_edges += 1
                    if current_edges > best_edges:
                        best_edges = current_edges
                        best_adj = adj[:]
                # If invalid, do nothing (reject)
                
            elif r < 0.8:
                # Try to remove a random existing edge
                # Get list of edges
                edges = get_edges()
                if not edges:
                    continue
                u, v = random.choice(edges)
                remove_edge(u, v)
                current_edges -= 1
                # Accept always? Or with probability?
                # Since we want to maximize, we should accept removals only if it helps escape a local min.
                # In SA, we accept worse moves with prob exp(-delta/T).
                # Here delta = -1 (loss of 1 edge).
                # Prob = exp(1/T). Since T is small, this is large? No, exp(-(-1)/T) = exp(1/T) > 1.
                # Wait, standard SA: accept if new_energy < old_energy OR with prob exp((new-old)/T).
                # Energy = -edge_count.
                # new_energy = -(E-1) = -E + 1.
                # old_energy = -E.
                # delta = 1.
                # Prob = exp(-1/T).
                # If T is high, prob is high. If T is low, prob is low.
                # So we should REMOVE with prob exp(-1/T).
                
                # But I already removed it. Let's re-add if we reject.
                if random.random() > math.exp(-1.0 / T):
                    # Reject removal
                    add_edge(u, v)
                    current_edges += 1
                else:
                    # Accept removal
                    pass
                    
            else:
                # Try a swap: remove one, add another
                edges = get_edges()
                if not edges:
                    continue
                u, v = random.choice(edges)
                
                # Tentatively remove
                remove_edge(u, v)
                
                # Try to add a random missing edge
                added = False
                for _ in range(10): # Try a few candidates
                    x = random.randint(0, n - 1)
                    y = random.randint(0, n - 1)
                    if x == y:
                        continue
                    if x > y:
                        x, y = y, x
                    if (adj[x] >> y) & 1:
                        continue
                    if is_valid_addition(x, y):
                        add_edge(x, y)
                        added = True
                        break
                
                if added:
                    # Net change in edges is 0.
                    # But structure changed.
                    # We accept this move always? 
                    # It's a neutral move in terms of edge count.
                    # In SA, we might accept neutral moves to explore.
                    # Let's accept with prob 0.5 or always?
                    # Let's accept always to allow mixing.
                    if random.random() < 0.5:
                        # Reject swap, revert
                        remove_edge(x, y)
                        add_edge(u, v)
                else:
                    # Revert removal
                    add_edge(u, v)
                
            T *= cooling_rate
            # Occasionally reheat?
            if random.random() < 0.01:
                T = 0.5
                
        return best_adj, best_edges

    start_time = time.time()
    deadline = start_time + 115.0
    
    best_adj_overall = [0] * n
    best_count_overall = 0
    
    # Initial greedy
    count = greedy_construct()
    if count > best_count_overall:
        best_count_overall = count
        best_adj_overall = adj[:]
        
    # Run SA
    # We can run multiple SA instances with different starting points
    # Or one long SA.
    # Given 115s, we can do several restarts.
    
    while time.time() < deadline:
        # Get a fresh start or perturb the best
        # Let's perturb the current best
        start_adj = best_adj_overall[:]
        
        # Time limit for this SA run
        time_left = deadline - time.time()
        if time_left < 5:
            break
            
        # Use a portion of remaining time
        sa_time = min(10.0, time_left * 0.8)
        
        best_adj_sa, best_count_sa = simulated_annealing(start_adj, sa_time)
        
        if best_count_sa > best_count_overall:
            best_count_overall = best_count_sa
            best_adj_overall = best_adj_sa[:]
            
    # Final polish on the best found
    adj = best_adj_overall[:]
    improved = True
    while improved:
        improved = False
        for u, v in all_edges:
            if not ((adj[u] >> v) & 1):
                if is_valid_addition(u, v):
                    add_edge(u, v)
                    improved = True
    
    final_count = edge_count()
    if final_count > best_count_overall:
        best_count_overall = final_count
        best_adj_overall = adj[:]
        
    # Reconstruct edge list
    best_edges = []
    for u in range(n):
        for v in range(u + 1, n):
            if (best_adj_overall[u] >> v) & 1:
                best_edges.append([u, v])
                
    if not best_edges:
        return [[a, b] for a in range(2) for b in range(2, 20)]
        
    return best_edges
