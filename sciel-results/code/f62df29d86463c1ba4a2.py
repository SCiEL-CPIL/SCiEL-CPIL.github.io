import time
import random
import math

def construct():
    start_time = time.time()
    deadline = start_time + 115  # Safety margin for 120s limit
    
    n = 20
    all_pairs = [(i, j) for i in range(n) for j in range(i + 1, n)]
    num_edges = len(all_pairs)
    
    # Pre-compute triples for validity check
    triples = [(i, j, k) for i in range(n) for j in range(i+1, n) for k in range(j+1, n)]
    
    def is_valid_fast(adj_bits):
        for i, j, k in triples:
            if (adj_bits[i] & adj_bits[j] & adj_bits[k]).bit_count() >= 3:
                return False
        return True

    def adj_to_edges(adj_bits):
        edges = []
        for u in range(n):
            v = adj_bits[u]
            while v:
                low = v & (-v)
                vv = low.bit_length() - 1
                if vv > u:
                    edges.append([u, vv])
                v ^= low
        return edges

    def count_edges(adj_bits):
        return sum(adj_bits[i].bit_count() for i in range(n)) // 2

    def generate_random(p):
        adj_bits = [0] * n
        for i, j in all_pairs:
            if random.random() < p:
                adj_bits[i] |= (1 << j)
                adj_bits[j] |= (1 << i)
        return adj_bits

    def repair(adj_bits):
        # Greedily remove edges that cause violations
        # We need to identify which edges are part of violations.
        # Simple approach: while invalid, find a violating triple, remove a random edge from it.
        # To find a violating triple:
        def find_violation(adj):
            for i, j, k in triples:
                common = adj[i] & adj[j] & adj[k]
                if common.bit_count() >= 3:
                    # Find 3 neighbors in common
                    neighbors = []
                    mask = common
                    while len(neighbors) < 3 and mask:
                        low = mask & (-mask)
                        vv = low.bit_length() - 1
                        neighbors.append(vv)
                        mask ^= low
                    return (i, j, k, neighbors)
            return None
        
        # To speed up repair, we can try removing edges that are in many potential violations?
        # Or just random removal.
        # Since n is small, random removal is fine.
        changed = True
        while changed:
            changed = False
            viol = find_violation(adj_bits)
            if viol:
                i, j, k, neighbors = viol
                # The violation is formed by {i,j,k} and 3 neighbors.
                # Edges involved: (i,n1), (i,n2), (i,n3), (j,n1)... (k,n3)
                # Remove a random edge from this set.
                # Actually, removing any edge that breaks the K3,3 works.
                # The K3,3 is between {i,j,k} and {n1,n2,n3}.
                # Wait, the definition is "no three vertices have three common neighbours".
                # If i,j,k have common neighbors n1,n2,n3, then edges (i,n1)..(k,n3) exist.
                # This forms a K3,3 with parts {i,j,k} and {n1,n2,n3}.
                # Removing any one of these 9 edges breaks this specific K3,3.
                # But there might be other K3,3s.
                
                # Pick a random edge from the 9 edges connecting {i,j,k} to {n1,n2,n3}
                # Actually, we can just pick a random edge incident to i, j, or k that goes to a common neighbor.
                # Let's just pick a random edge from the list of all edges in the graph?
                # No, that's inefficient.
                
                # Let's pick a random edge from the 9 edges.
                edges_to_remove = []
                for u in [i, j, k]:
                    for v in neighbors:
                        if u != v: # distinct vertices
                            edges_to_remove.append((u, v))
                
                # Ensure we only pick edges that actually exist (they should, by definition of common neighbors)
                # But wait, if (u,v) is an edge, it's in adj.
                # We pick one at random.
                u, v = random.choice(edges_to_remove)
                
                # Remove edge (u,v)
                adj_bits[u] &= ~(1 << v)
                adj_bits[v] &= ~(1 << u)
                changed = True
        return adj_bits

    best_adj = None
    best_count = 0
    best_edges = []
    
    def update_best(adj_bits):
        nonlocal best_adj, best_count, best_edges
        if not is_valid_fast(adj_bits):
            return
        c = count_edges(adj_bits)
        if c > best_count:
            best_count = c
            best_adj = adj_bits[:]
            best_edges = adj_to_edges(adj_bits)

    random.seed(42)
    
    # Multi-start Local Search
    num_restarts = 20
    
    for s in range(num_restarts):
        if time.time() > deadline - 5:
            break
            
        # Generate initial graph
        # Density around 0.5-0.7 is good for K3,3 free?
        # Max degree in K3,3 free graph?
        # Let's try random density.
        p = random.uniform(0.4, 0.7)
        adj = generate_random(p)
        
        # Repair
        adj = repair(adj)
        
        # Local Search
        # 1. Try to add edges
        # 2. If stuck, try swaps
        
        improved = True
        while improved and time.time() < deadline - 5:
            improved = False
            
            # Phase 1: Add edges
            # Shuffle edges to avoid bias
            edges_to_try = all_pairs[:]
            random.shuffle(edges_to_try)
            
            for u, v in edges_to_try:
                if adj[u] & (1 << v):
                    continue
                
                # Try adding
                adj[u] |= (1 << v)
                adj[v] |= (1 << u)
                
                if is_valid_fast(adj):
                    improved = True
                    break
                else:
                    # Revert
                    adj[u] &= ~(1 << v)
                    adj[v] &= ~(1 << u)
            
            if improved:
                continue
            
            # Phase 2: Swaps
            # Try removing one edge and adding another
            # This is O(E^2) which is 190^2 = 36100 checks.
            # Each check is ~1ms? 36s. Too slow for a single step.
            # We need to be smarter or limit the number of swaps tried.
            
            # Strategy: Pick a random edge to remove, then try to add a random edge.
            # If valid, accept.
            # Repeat many times.
            
            # Let's try a limited number of random swaps
            swaps_tried = 0
            max_swaps = 500 # Limit per local search step to avoid timeout
            
            while swaps_tried < max_swaps and time.time() < deadline - 5:
                # Pick random edge to remove
                existing_edges = []
                for i in range(n):
                    v = adj[i]
                    while v:
                        low = v & (-v)
                        vv = low.bit_length() - 1
                        if vv > i:
                            existing_edges.append((i, vv))
                        v ^= low
                
                if not existing_edges:
                    break
                
                u_rem, v_rem = random.choice(existing_edges)
                
                # Remove it
                adj[u_rem] &= ~(1 << v_rem)
                adj[v_rem] &= ~(1 << u_rem)
                
                # Try to add a random missing edge
                # Pick random pair
                u_add, v_add = random.choice(all_pairs)
                
                # If it's the same edge, skip
                if (u_add == u_rem and v_add == v_rem) or (u_add == v_rem and v_add == u_rem):
                    # Revert and try next
                    adj[u_rem] |= (1 << v_rem)
                    adj[v_rem] |= (1 << u_rem)
                    swaps_tried += 1
                    continue
                
                # Check if already exists
                if adj[u_add] & (1 << v_add):
                    # Revert
                    adj[u_rem] |= (1 << v_rem)
                    adj[v_rem] |= (1 << u_rem)
                    swaps_tried += 1
                    continue
                
                # Try adding
                adj[u_add] |= (1 << v_add)
                adj[v_add] |= (1 << u_add)
                
                if is_valid_fast(adj):
                    improved = True
                    # Keep the swap
                    break
                else:
                    # Revert add
                    adj[u_add] &= ~(1 << v_add)
                    adj[v_add] &= ~(1 << u_add)
                    # Revert remove
                    adj[u_rem] |= (1 << v_rem)
                    adj[v_rem] |= (1 << u_rem)
                    swaps_tried += 1
            
            if improved:
                continue
            
            # If no improvement, break local search
            break
        
        update_best(adj)
        
        # If we have a very good result, we can stop early?
        # Theoretical max is unknown, but high.
        # Let's just run all restarts if time permits.

    if not best_edges:
        # Fallback
        adj = [0] * n
        for i in range(n):
            for j in range(i + 1, n):
                if (j - i) % 2 == 1:
                    adj[i] |= (1 << j)
                    adj[j] |= (1 << i)
        if is_valid_fast(adj):
            return adj_to_edges(adj)
        return [[i, i + 1] for i in range(n - 1)]

    return best_edges
