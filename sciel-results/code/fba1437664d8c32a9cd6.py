import time
import random

def construct():
    start_time = time.time()
    timeout = 140.0
    
    n = 18
    # Initial graph: K6,6,6
    parts = [list(range(0, 6)), list(range(6, 12)), list(range(12, 18))]
    adj = [0] * n
    for i in range(3):
        for j in range(i + 1, 3):
            for a in parts[i]:
                for b in parts[j]:
                    adj[a] |= (1 << b)
                    adj[b] |= (1 << a)
    
    # Precompute is_tf for all masks
    # is_tf[mask] = 1 if mask induces a triangle-free subgraph
    is_tf = [0] * (1 << n)
    is_tf[0] = 1
    
    # To compute is_tf efficiently, we can iterate masks and add one vertex at a time
    # But $2^{18}$ is small enough to just check each mask?
    # Checking each mask takes $O(k^2)$. Total $O(n^2 2^n) \approx 18^2 \cdot 262144 \approx 8.5 \cdot 10^7$.
    # This might take a few seconds in Python.
    
    # Faster:
    # A mask is TF iff it contains no triangle.
    # We can precompute for each pair (u,v) the set of common neighbors.
    # Then for a mask M, it is TF iff for all u,v in M, (common_neighbors[u][v] & M) == 0.
    
    # Let's just do the brute force check for initialization, it's a one-time cost.
    for mask in range(1, 1 << n):
        # Get lowest set bit
        lsb = mask & -mask
        v = lsb.bit_length() - 1
        prev_mask = mask ^ lsb
        
        # If prev_mask is not TF, mask is not TF
        if not is_tf[prev_mask]:
            is_tf[mask] = 0
            continue
            
        # Check if v forms a triangle with any pair in prev_mask
        # v forms a triangle with u, w in prev_mask if (u,v) and (v,w) and (u,w) are edges.
        # Equivalently, if N(v) & N(u) contains w.
        # Or simpler: if there is u in prev_mask such that (adj[v] & adj[u] & prev_mask) != 0?
        # No, that checks if u and v have a common neighbor in prev_mask.
        # If they do, say w, then u,v,w is a triangle.
        # So we need to check if for any u in prev_mask, (adj[v] & adj[u] & prev_mask) is non-zero?
        # Wait, if (adj[v] & adj[u] & prev_mask) is non-zero, it means there is w in prev_mask connected to both u and v.
        # So u, v, w is a triangle.
        # So mask is NOT TF.
        
        valid = True
        pm = prev_mask
        while pm:
            lsb_u = pm & -pm
            u = lsb_u.bit_length() - 1
            pm ^= lsb_u
            
            if (adj[v] & adj[u] & prev_mask):
                valid = False
                break
        
        is_tf[mask] = 1 if valid else 0

    # Compute initial max size
    max_size = 0
    for mask in range(1, 1 << n):
        if is_tf[mask]:
            cnt = bin(mask).count('1')
            if cnt > max_size:
                max_size = cnt

    best_max_size = max_size
    best_adj = adj[:]
    
    # Local Search
    while time.time() - start_time < timeout:
        improved = False
        # Generate list of non-edges
        non_edges = []
        for u in range(n):
            for v in range(u+1, n):
                if not ((adj[u] >> v) & 1):
                    non_edges.append((u, v))
        
        random.shuffle(non_edges)
        
        for u, v in non_edges:
            # Check K4 constraint
            common = adj[u] & adj[v]
            has_edge = False
            c = common
            while c:
                x = (c & -c).bit_length() - 1
                c &= c - 1
                if (adj[x] & common) & ~(1 << x):
                    has_edge = True
                    break
            if has_edge:
                continue
            
            # Try adding edge (u,v)
            # Update is_tf for masks containing u and v
            # Only masks that were TF can become non-TF
            # Condition: If M contains u,v and M & common is not empty, then M is not TF.
            
            # We need to iterate over all masks containing u and v
            # Let M = (1<<u) | (1<<v) | rest, where rest is any subset of remaining n-2 vertices
            base_mask = (1 << u) | (1 << v)
            rest_mask = ((1 << n) - 1) ^ base_mask
            
            # Iterate over subsets of rest_mask
            # We can use submask iteration
            sub = rest_mask
            while True:
                mask = base_mask | sub
                if is_tf[mask]:
                    # Check if this mask becomes non-TF
                    # It becomes non-TF if it contains a common neighbor of u and v
                    if (mask & common):
                        is_tf[mask] = 0
                if sub == 0:
                    break
                sub = (sub - 1) & rest_mask
            
            # Update max_size
            # Since we only set 1 to 0, max_size can only decrease
            # We need to check if current max_size is still achievable
            # If not, find new max
            # This might be slow if we do it every time.
            # But we can just check if any mask of size max_size is still TF.
            # If not, decrement max_size.
            
            # Optimization: Keep a count of TF masks by size
            # counts[size] = number of TF masks of that size
            # Initially compute counts.
            
            # For now, just recompute if needed.
            # But we can be smarter.
            
            # Let's just check if max_size is still valid
            valid_max = False
            # Iterate masks of size max_size?
            # This is expensive.
            
            # Alternative: Just accept the move if the *potential* max size doesn't increase.
            # Since we are only removing TF sets, the max size cannot increase.
            # So we can always accept the move?
            # No, we want to *minimize* the max size.
            # So if we remove a large TF set, we improve.
            # If we remove a small TF set, we don't improve, but we might enable future improvements.
            # So we should accept moves that reduce the max size.
            # How to know if max size reduced?
            # We can check if the specific large sets we broke were the maximum.
            
            # Let's just compute the new max size by scanning is_tf array.
            # 2^18 scan is fast.
            new_max = 0
            for m in range(1, 1 << n):
                if is_tf[m]:
                    cnt = bin(m).count('1')
                    if cnt > new_max:
                        new_max = cnt
                        if new_max >= max_size:
                            break # Can't get better than current max
            if new_max < max_size:
                max_size = new_max
                improved = True
                # Update adj
                adj[u] |= (1 << v)
                adj[v] |= (1 << u)
                if max_size < best_max_size:
                    best_max_size = max_size
                    best_adj = adj[:]
                break # Restart search
            
        if not improved:
            # Perturbation
            edge_list = []
            for u in range(n):
                for v in range(u+1, n):
                    if (adj[u] >> v) & 1:
                        edge_list.append((u, v))
            if edge_list:
                u, v = random.choice(edge_list)
                adj[u] &= ~(1 << v)
                adj[v] &= ~(1 << u)
                # Recompute is_tf? This is expensive.
                # Maybe just restart with a new random graph if stuck?
                pass

    # Reconstruct edges from best_adj
    best_edges = []
    for u in range(n):
        for v in range(u+1, n):
            if (best_adj[u] >> v) & 1:
                best_edges.append([u, v])
                
    return best_edges
