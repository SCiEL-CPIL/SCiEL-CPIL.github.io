import time
import random
import numpy as np

def construct():
    """
    Constructs a C6-free graph on 40 vertices maximizing edges.
    Uses multiple restarts of a greedy local search with adaptive kicks.
    """
    N = 40
    start_time = time.time()
    deadline = start_time + 115  # Safety margin before 120s limit
    
    # Precompute bitmasks for efficiency
    # adj_bits[i] is a bitmask where bit j is 1 if edge (i,j) exists
    
    def can_add(adj_bits, u, v):
        """
        Check if adding edge (u,v) creates a C6.
        A C6 is u-a-b-c-d-v-u.
        We need a path u-a-b-c-d-v in G.
        """
        nu = adj_bits[u] & ~(1 << v)
        nv = adj_bits[v] & ~(1 << u)
        
        if nu == 0 or nv == 0:
            return True
        
        # Iterate over a in N(u)
        temp_nu = nu
        while temp_nu:
            lsb_a = temp_nu & (-temp_nu)
            a = lsb_a.bit_length() - 1
            temp_nu ^= lsb_a
            
            # b must be in N(a) \ {u, v, a}
            # Note: b cannot be v (handled by nu/nv exclusions? No, v is in nv, but b is neighbor of a)
            # If b=v, then u-a-v... but v is not connected to a in the path?
            # Path: u-a-b-c-d-v.
            # b is neighbor of a.
            # If b=v, then u-a-v. Then we need c,d such that v-c-d-v? No, v is the end.
            # So b cannot be v.
            ba_mask = adj_bits[a] & ~(1 << u) & ~(1 << v) & ~(1 << a)
            
            if ba_mask == 0:
                continue
            
            # Iterate over d in N(v)
            # d cannot be a (else u-a-d-v is C4, not C6, but we check C6)
            # If d=a, path u-a-a... invalid.
            temp_nv = nv & ~(1 << a)
            while temp_nv:
                lsb_d = temp_nv & (-temp_nv)
                d = lsb_d.bit_length() - 1
                temp_nv ^= lsb_d
                
                # b must be in N(a) \ {u, v, a, d}
                # c must be in N(d) \ {u, v, a, d}
                # We need an edge (b,c) with b != c.
                
                # Optimize: 
                # Let B_set = N(a) \ {u, v, a, d}
                # Let C_set = N(d) \ {u, v, a, d}
                # We need to check if there is an edge between B_set and C_set.
                
                B_set = ba_mask & ~(1 << d)
                if B_set == 0:
                    continue
                
                C_set = adj_bits[d] & ~(1 << u) & ~(1 << v) & ~(1 << a) & ~(1 << d)
                if C_set == 0:
                    continue
                
                # Check for edge (b,c)
                # Iterate over b in B_set
                temp_b = B_set
                found = False
                while temp_b:
                    lsb_b = temp_b & (-temp_b)
                    b = lsb_b.bit_length() - 1
                    temp_b ^= lsb_b
                    
                    # c must be in N(b) \ {u, v, a, b, d}
                    # And c must be in C_set
                    # N(b) & C_set
                    # If c == b, we have a loop, but simple graph, so N(b) doesn't have b.
                    # However, we must ensure c != a, c != d, c != u, c != v.
                    # C_set already excludes u, v, a, d.
                    # So we just need N(b) & C_set != 0.
                    # But wait, if c == b, it's not in C_set? 
                    # b is in B_set. C_set excludes d, a, u, v.
                    # If b is in C_set, then b is a neighbor of d.
                    # If b is in N(b)? No.
                    # So if N(b) & C_set is not empty, there is a c.
                    # Is it possible that c == b?
                    # c is in C_set. b is in B_set.
                    # If b is in C_set, then b is a neighbor of d.
                    # But c is a neighbor of b.
                    # If c == b, then b is neighbor of b, impossible.
                    # So any c in N(b) & C_set is valid and distinct from b.
                    
                    if adj_bits[b] & C_set:
                        return False
                        
        return True

    def edge_count(adj_bits):
        return sum(bin(b).count('1') for b in adj_bits) // 2

    def run_local_search(initial_adj, time_limit):
        adj_bits = initial_adj[:]
        current_edges = edge_count(adj_bits)
        best_edges = current_edges
        best_adj = adj_bits[:]
        
        # Build list of missing edges
        all_pairs = [(i, j) for i in range(N) for j in range(i + 1, N)]
        
        # Greedy phase
        while time.time() < time_limit:
            # Shuffle to avoid bias
            random.shuffle(all_pairs)
            
            added_any = False
            # We iterate over ALL pairs, not just missing, to handle re-adds after kicks
            # But checking existing edges is fast (just a bit check).
            for u, v in all_pairs:
                if time.time() > time_limit:
                    break
                
                if adj_bits[u] & (1 << v):
                    continue
                
                if can_add(adj_bits, u, v):
                    adj_bits[u] |= (1 << v)
                    adj_bits[v] |= (1 << u)
                    current_edges += 1
                    added_any = True
            
            if current_edges > best_edges:
                best_edges = current_edges
                best_adj = adj_bits[:]
            
            if not added_any:
                # Stuck. Perform a kick.
                if time.time() > time_limit - 2:
                    break
                
                # Collect existing edges
                existing_edges = []
                for u in range(N):
                    mask = adj_bits[u]
                    while mask:
                        lsb = mask & (-mask)
                        v = lsb.bit_length() - 1
                        mask ^= lsb
                        if u < v:
                            existing_edges.append((u, v))
                
                if not existing_edges:
                    break
                
                # Adaptive kick size: start small, increase if stuck repeatedly?
                # For now, fixed kick of 2-3 edges, biased towards high-degree nodes to break dense clusters
                kick_size = random.randint(2, 4)
                
                # Sort by degree sum to target "tight" connections
                # Degree sum is a proxy for how "constrained" the edge is
                def edge_weight(e):
                    u, v = e
                    return bin(adj_bits[u]).count('1') + bin(adj_bits[v]).count('1')
                
                existing_edges.sort(key=edge_weight, reverse=True)
                
                # Pick kick_size edges, mix of top and random
                kick_set = set()
                # 50% from top, 50% random
                for _ in range(kick_size):
                    if random.random() < 0.5 and existing_edges:
                        u, v = existing_edges[0]
                    else:
                        u, v = random.choice(existing_edges)
                    kick_set.add((u, v))
                
                for u, v in kick_set:
                    if adj_bits[u] & (1 << v):
                        adj_bits[u] &= ~(1 << v)
                        adj_bits[v] &= ~(1 << u)
                        current_edges -= 1
                
                # Immediately try to re-add
                random.shuffle(all_pairs)
                for u, v in all_pairs:
                    if time.time() > time_limit:
                        break
                    if adj_bits[u] & (1 << v):
                        continue
                    if can_add(adj_bits, u, v):
                        adj_bits[u] |= (1 << v)
                        adj_bits[v] |= (1 << u)
                        current_edges += 1
                
                if current_edges > best_edges:
                    best_edges = current_edges
                    best_adj = adj_bits[:]
        
        return best_adj, best_edges

    # Initial state: 8 disjoint K5s
    init_adj = [0] * N
    for k in range(8):
        base = 5 * k
        for a in range(5):
            for b in range(a + 1, 5):
                u, v = base + a, base + b
                init_adj[u] |= (1 << v)
                init_adj[v] |= (1 << u)
    
    best_overall_adj = init_adj[:]
    best_overall_edges = edge_count(init_adj)
    
    # Multi-restart strategy
    # We have ~115 seconds.
    # Let's do 4 restarts, each with ~25s budget, leaving 15s for final polish/safety.
    num_restarts = 4
    time_per_restart = (deadline - time.time()) / num_restarts - 1
    
    for r in range(num_restarts):
        if time.time() > deadline - 5:
            break
        
        if r == 0:
            start_adj = init_adj[:]
        else:
            # Perturb the best solution found so far
            start_adj = best_overall_adj[:]
            # Remove 3 random edges to escape local optimum
            existing_edges = []
            for u in range(N):
                mask = start_adj[u]
                while mask:
                    lsb = mask & (-mask)
                    v = lsb.bit_length() - 1
                    mask ^= lsb
                    if u < v:
                        existing_edges.append((u, v))
            
            if existing_edges:
                edges_to_remove = random.sample(existing_edges, min(3, len(existing_edges)))
                for u, v in edges_to_remove:
                    start_adj[u] &= ~(1 << v)
                    start_adj[v] &= ~(1 << u)
        
        t_end = time.time() + time_per_restart
        adj, count = run_local_search(start_adj, t_end)
        
        if count > best_overall_edges:
            best_overall_edges = count
            best_overall_adj = adj
    
    # Final polish: One more greedy pass on the best solution
    if time.time() < deadline - 2:
        adj, count = run_local_search(best_overall_adj, deadline - 2)
        if count > best_overall_edges:
            best_overall_edges = count
            best_overall_adj = adj
    
    # Build final edge list
    final_edges = []
    for u in range(N):
        for v in range(u + 1, N):
            if best_overall_adj[u] & (1 << v):
                final_edges.append([u, v])
    
    return final_edges

if __name__ == "__main__":
    e = construct()
    print(f"Edges: {len(e)}")
