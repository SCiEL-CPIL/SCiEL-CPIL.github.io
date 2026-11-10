import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 110  # 110s safety margin for 120s limit
    
    n = 18
    # Initialize with K6,6,6
    # Parts: 0-5, 6-11, 12-17
    adj = [0] * n
    for i in range(6):
        for j in range(6, 12):
            adj[i] |= 1 << j
            adj[j] |= 1 << i
        for j in range(12, 18):
            adj[i] |= 1 << j
            adj[j] |= 1 << i
    for i in range(6, 12):
        for j in range(12, 18):
            adj[i] |= 1 << j
            adj[j] |= 1 << i
            
    # Precompute triangles
    triangles = []
    def update_triangles():
        nonlocal triangles
        triangles = []
        for u in range(n):
            for v in range(u + 1, n):
                if adj[u] & (1 << v):
                    # Check common neighbors
                    common = adj[u] & adj[v]
                    # Iterate through common neighbors w > v to avoid duplicates?
                    # Or just add all and dedupe later?
                    # Better: for w in common, if w > v, then (u,v,w) is a unique triangle if we enforce u < v < w?
                    # No, u < v is fixed. w can be anything.
                    # To ensure uniqueness, we can just store the bitmask.
                    c = common
                    while c:
                        lsb = c & -c
                        w = lsb.bit_length() - 1
                        if w > v: # Enforce u < v < w
                            t = (1 << u) | (1 << v) | (1 << w)
                            triangles.append(t)
                        c -= lsb
                            
    update_triangles()
    
    def is_k4_free():
        for u in range(n):
            for v in range(u + 1, n):
                if adj[u] & (1 << v):
                    common = adj[u] & adj[v]
                    # If common neighbors contain an edge, K4
                    c = common
                    while c:
                        lsb = c & -c
                        w = lsb.bit_length() - 1
                        if adj[w] & (1 << v): # Check if (w, v) is edge? No, (w, u) and (w, v) are edges by definition of common.
                             # We need to check if there is an edge between any two vertices in common?
                             # No, K4 is u, v, w, x.
                             # If u, v are connected, and w, x are common neighbors, and (w, x) is an edge, then u, v, w, x form K4?
                             # No, u is connected to w, x. v is connected to w, x.
                             # If (w, x) is an edge, then {u, v, w, x} is a K4.
                             if adj[w] & (1 << v): pass # This is always true if w is common neighbor of u, v? No.
                             # w is common neighbor of u, v means (u, w) and (v, w) are edges.
                             # We need to check if there exists another common neighbor x such that (w, x) is an edge.
                             # So for each w in common, check if common \ {w} contains a neighbor of w.
                             # This is O(|common|^2).
                             pass
                        c -= lsb
        # Simplified K4 check:
        for u in range(n):
            for v in range(u + 1, n):
                if adj[u] & (1 << v):
                    common = adj[u] & adj[v]
                    # Check if subgraph induced by common has any edge
                    c = common
                    while c:
                        lsb = c & -c
                        w = lsb.bit_length() - 1
                        if adj[w] & common:
                            return False
                        c -= lsb
        return True

    def max_tf_size():
        # If no triangles, return 18
        if not triangles:
            return 18
            
        # Sort triangles?
        # Iterate subsets in decreasing size
        # To speed up, we can just check all subsets and keep max
        # But we want max.
        # We can iterate k from 18 down to 1
        # And check if any subset of size k is TF.
        
        # Precompute list of all subsets? No.
        
        # Optimization:
        # If the number of triangles is high, the max size is likely low.
        
        # Let's just iterate all subsets 0..2^18-1
        # And check if they are TF.
        # This is 262k iterations.
        # For each, check against triangles.
        
        best = 0
        # We can skip subsets that are subsets of known non-TF sets?
        
        # Simple loop
        for S in range(1 << n):
            # Check if S is TF
            is_tf = True
            for t in triangles:
                if (S & t) == t:
                    is_tf = False
                    break
            if is_tf:
                sz = bin(S).count('1')
                if sz > best:
                    best = sz
        return best

    # Initial score
    current_score = max_tf_size()
    best_score = current_score
    best_adj = adj[:]
    
    # SA Parameters
    T = 1.0
    cooling = 0.999
    min_T = 0.01
    
    # Greedy step first?
    # Just SA
    
    while time.time() < deadline and T > min_T:
        # Pick random edge to flip
        u = random.randint(0, n - 1)
        v = random.randint(0, n - 1)
        if u == v:
            continue
            
        # Flip
        mask = 1 << v
        if adj[u] & mask:
            adj[u] &= ~mask
            adj[v] &= ~(1 << u)
        else:
            adj[u] |= mask
            adj[v] |= 1 << u
            
        # Check K4
        if not is_k4_free():
            # Revert
            if adj[u] & mask:
                adj[u] &= ~mask
                adj[v] &= ~(1 << u)
            else:
                adj[u] |= mask
                adj[v] |= 1 << u
            continue
            
        # Update triangles
        update_triangles()
        
        # Compute score
        new_score = max_tf_size()
        
        if new_score < best_score:
            best_score = new_score
            best_adj = adj[:]
            
        # Accept/Reject
        delta = new_score - current_score
        if delta < 0 or random.random() < pow(2.718, -delta / T):
            current_score = new_score
        else:
            # Revert
            if adj[u] & mask:
                adj[u] &= ~mask
                adj[v] &= ~(1 << u)
            else:
                adj[u] |= mask
                adj[v] |= 1 << u
            update_triangles()
            current_score = max_tf_size() # Revert score? No, we can store current_score before flip
            
        T *= cooling
        
    # Return edges of best_adj
    edges = []
    for u in range(n):
        for v in range(u + 1, n):
            if best_adj[u] & (1 << v):
                edges.append([u, v])
                
    return edges
