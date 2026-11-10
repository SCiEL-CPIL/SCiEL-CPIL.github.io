import time
import random
import numpy as np

def construct():
    deadline = time.time() + 110
    n = 40
    
    # 1. Initialize with 8 disjoint K5s (104 edges)
    # This is a strong starting point known to be C6-free.
    edges = []
    adj_mask = [0] * n
    for k in range(8):
        for a in range(5):
            for b in range(a + 1, 5):
                u = 5 * k + a
                v = 5 * k + b
                edges.append((u, v))
                adj_mask[u] |= (1 << v)
                adj_mask[v] |= (1 << u)
    
    edge_set = set(edges)
    
    # 2. Fast C6 Detection using Bitsets
    # To check if adding (u, v) creates a C6, we need to check if there is a path of length 4
    # between u and v in the current graph.
    # Path: u - w - x - y - z - v
    # w in N(u), x in N(w), y in N(x), z in N(y)
    # And z must be in N(v).
    # Also all vertices u,v,w,x,y,z must be distinct.
    
    def creates_c6(u, v):
        # Optimization: Only check if u and v are "close" enough?
        # In a sparse graph like 104 edges on 40 nodes, average degree is ~5.2.
        # Most pairs are far apart. But we must be exact.
        
        # Neighbors of u
        mask_u = adj_mask[u]
        # Neighbors of v
        mask_v = adj_mask[v]
        
        # We need w in N(u) such that there is a path w -> x -> y -> z -> v
        # z must be in N(v).
        # Let's iterate w in N(u).
        # For each w, we need a path of length 3 from w to v.
        # Path w -> x -> y -> z -> v.
        
        # Precompute N(N(v)) ? No, that's just length 2.
        
        # Efficient check:
        # For each w in N(u):
        #   For each x in N(w) \ {u, v, w}:
        #     For each y in N(x) \ {u, v, w, x}:
        #       If N(y) intersects (N(v) \ {u, w, x, y}):
        #          Return True (Found z)
        
        # To speed this up, we can precompute "reachable in 2 steps" masks?
        # Let R2[v] be the set of nodes reachable from v in exactly 2 steps.
        # Then we need w in N(u) such that there exists x in N(w)
        # such that N(x) intersects R2[v] ?
        # Wait, the path is w-x-y-z-v.
        # z is in N(v). y is in N(z).
        # So y is in R2[v] (reachable from v in 2 steps: v-z-y).
        # So we need w in N(u), x in N(w), y in N(x)
        # such that y is in R2[v].
        # And we must ensure distinctness.
        
        # Let's precompute R2 for all nodes? 
        # R2[i] = union of N(N(i))
        # This is O(n * E) to build, which is fine.
        # But distinctness is the tricky part.
        # If we just use R2, we might find a path that reuses vertices.
        # However, in sparse graphs, short cycles are rare.
        # Given the time limit, let's try a hybrid:
        # 1. Quick bitset check for existence of length 4 path (ignoring distinctness initially? No, C6 requires distinct).
        # 2. If bitset check fails, definitely no C6.
        # 3. If bitset check passes, do a careful DFS to confirm distinctness.
        
        # Actually, the DFS in the previous solution was the bottleneck.
        # Let's optimize the DFS with bitsets.
        
        # Iterate w
        w_mask = mask_u & ~(1 << v)
        while w_mask:
            w_bit = w_mask & -w_mask
            w = w_bit.bit_length() - 1
            w_mask ^= w_bit
            
            # Iterate x in N(w)
            # x cannot be u, v, w
            x_mask = adj_mask[w] & ~(1 << u) & ~(1 << v) & ~(1 << w)
            while x_mask:
                x_bit = x_mask & -x_mask
                x = x_bit.bit_length() - 1
                x_mask ^= x_bit
                
                # Iterate y in N(x)
                # y cannot be u, v, w, x
                y_mask = adj_mask[x] & ~(1 << u) & ~(1 << v) & ~(1 << w) & ~(1 << x)
                while y_mask:
                    y_bit = y_mask & -y_mask
                    y = y_bit.bit_length() - 1
                    y_mask ^= y_bit
                    
                    # Check if any neighbor z of y is in N(v)
                    # z cannot be u, v, w, x, y
                    # z in N(y) AND z in N(v)
                    # z_mask = N(y) & N(v) & ~(1<<u) & ~(1<<v) & ~(1<<w) & ~(1<<x) & ~(1<<y)
                    # If z_mask is not 0, we found a C6.
                    
                    # Optimization: Precompute common neighbors?
                    # common = adj_mask[y] & adj_mask[v]
                    # If common has a bit not in {u,v,w,x,y}, return True.
                    
                    z_mask = adj_mask[y] & mask_v
                    # Exclude forbidden nodes
                    forbidden = (1 << u) | (1 << v) | (1 << w) | (1 << x) | (1 << y)
                    if z_mask & ~forbidden:
                        return True
        
        return False

    # 3. Greedy Construction Phase
    # We want to add inter-cluster edges.
    # Generate all possible edges not in the initial set.
    # Sort them by a heuristic? 
    # Randomized Greedy with Backtracking?
    
    # Let's use a "Greedy with Randomized Restart" approach.
    # 1. Generate list of candidate edges (inter-cluster).
    # 2. Shuffle.
    # 3. Try to add each. If it creates C6, skip.
    # 4. If we get stuck, try to remove a few edges to unblock?
    
    # Better: Local Search with fast moves.
    # Since we have a fast bitset check, we can do more iterations.
    
    current_edges = set(edge_set)
    best_edges = set(edge_set)
    best_score = len(best_edges)
    
    # Candidate edges: all pairs (u, v) not in current_edges
    # To make this fast, we can just iterate through all pairs in random order repeatedly.
    
    # Pre-compute list of all possible edges
    all_pairs = [(i, j) for i in range(n) for j in range(i+1, n)]
    
    # Filter to only inter-cluster edges? No, intra-cluster are already full (K5).
    # So only inter-cluster edges are candidates.
    # Cluster of u is u // 5.
    
    candidates = []
    for i, j in all_pairs:
        if i // 5 != j // 5:
            candidates.append((i, j))
    
    random.shuffle(candidates)
    
    # Main Loop: Greedy Add / Random Remove
    T = 1.0
    T_min = 0.01
    
    while time.time() < deadline:
        # Try to add a random candidate edge
        # To be efficient, we can pick a random edge from the full list of non-edges.
        # But maintaining a list of non-edges is expensive.
        # Instead, pick random u, v.
        
        # Strategy: 
        # 70% chance: Try to add a random edge.
        # 30% chance: Try to remove a random edge (to escape local optima).
        
        if random.random() < 0.7:
            u = random.randint(0, n - 1)
            v = random.randint(0, n - 1)
            if u == v:
                continue
            if u > v:
                u, v = v, u
            
            if (u, v) in current_edges:
                continue
                
            # Check C6
            if creates_c6(u, v):
                continue
            
            # Add edge
            adj_mask[u] |= (1 << v)
            adj_mask[v] |= (1 << u)
            current_edges.add((u, v))
            
            if len(current_edges) > best_score:
                best_score = len(current_edges)
                best_edges = set(current_edges)
                
        else:
            # Remove a random edge
            if not current_edges:
                continue
            e = random.choice(list(current_edges))
            u, v = e
            
            # Check if removal helps? 
            # In SA, we accept removal with prob exp(-1/T).
            # Here, removal always decreases score, so we only do it to escape.
            # We accept if random < exp(-1/T).
            
            if random.random() < np.exp(-1.0 / T):
                adj_mask[u] &= ~(1 << v)
                adj_mask[v] &= ~(1 << u)
                current_edges.remove(e)
            
        T *= 0.999
        if T < T_min:
            T = 0.5 # Restart temperature occasionally

    return [[a, b] for (a, b) in sorted(best_edges)]
