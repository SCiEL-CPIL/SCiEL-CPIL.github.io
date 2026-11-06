import random
import time

def _check_valid_and_get_bad_mask(adj_mask, n):
    """
    Returns (True, 0) if valid.
    Returns (False, mask) if an invalid regular induced subgraph is found.
    """
    # Iterate subsets of size >= 6
    # Use Gosper's hack for each k.
    for k in range(6, n + 1):
        # Generate all masks of size k
        mask = (1 << k) - 1
        limit = 1 << n
        while mask < limit:
            # Check if mask is regular
            m = mask
            first_v = -1
            d0 = -1
            is_reg = True
            while m:
                lsb = m & -m
                v = lsb.bit_length() - 1
                if first_v == -1:
                    first_v = v
                    d0 = (adj_mask[v] & mask).bit_count()
                else:
                    dv = (adj_mask[v] & mask).bit_count()
                    if dv != d0:
                        is_reg = False
                        break
                m ^= lsb
            
            if is_reg:
                return False, mask
            
            # Next mask of size k
            c = mask & -mask
            r = mask + c
            mask = (((r ^ mask) >> 2) // c) | r
            
    return True, 0

def _generate_expander_like_graph(n, target_deg, seed):
    """
    Generate a random graph with degree distribution centered around target_deg.
    This mimics the properties of an expander, which tends to have fewer regular subgraphs.
    """
    rng = random.Random(seed)
    adj_mask = [0] * n
    
    # First, build a random regular graph of degree target_deg (if possible)
    # Simple method: random permutation of multiset of edges?
    # Or just random graph with p = target_deg / (n-1)
    p = target_deg / (n - 1)
    
    for i in range(n):
        for j in range(i + 1, n):
            if rng.random() < p:
                adj_mask[i] |= (1 << j)
                adj_mask[j] |= (1 << i)
                
    return adj_mask

def construct():
    start_time = time.time()
    deadline = start_time + 140  # 10s margin
    
    # We want to maximize n.
    # Try n=22, then 21, then 20.
    
    for n in [22, 21, 20]:
        if time.time() > deadline - 10:
            break
            
        # Strategy 1: Random Graphs (G(n, 0.5))
        # Strategy 2: Expander-like (G(n, p) with p tuned to degree ~5 or 6)
        
        # For n=21, target degree 5 implies p = 5/20 = 0.25
        # For n=22, target degree 5 implies p = 5/21 ~ 0.24
        # Let's try a few probabilities: 0.5, 0.25, 0.75
        
        probabilities = [0.5, 0.25, 0.75, 0.4, 0.6]
        
        found = False
        total_attempts = 100
        
        # Interleave attempts with different probabilities
        for attempt in range(total_attempts):
            if time.time() > deadline - 5:
                break
                
            p = probabilities[attempt % len(probabilities)]
            
            # Generate random graph
            adj_mask = [0] * n
            for i in range(n):
                for j in range(i + 1, n):
                    if random.random() < p:
                        adj_mask[i] |= (1 << j)
                        adj_mask[j] |= (1 << i)
                        
            # Local Search / Repair
            # We iterate, finding a bad mask, fixing it, and re-checking.
            # To avoid infinite loops, we limit repair steps per start.
            # Repair steps should be limited because full check is expensive.
            # C(21, 6) is 54k checks. Each check is ~21 ops. ~1M ops per full check.
            # Python can do ~10M-50M ops/sec. So ~0.02-0.1s per full check.
            # We can afford maybe 10-20 full checks in 1 second.
            
            max_repair_steps = 20
            
            for repair_step in range(max_repair_steps):
                if time.time() > deadline - 2:
                    break
                    
                is_valid, bad_mask = _check_valid_and_get_bad_mask(adj_mask, n)
                
                if is_valid:
                    # Convert to edge list
                    edges = []
                    for i in range(n):
                        for j in range(i + 1, n):
                            if adj_mask[i] & (1 << j):
                                edges.append([i, j])
                    return {'n': n, 'edges': edges}
                
                # Not valid. Repair.
                # bad_mask is a regular induced subgraph.
                # Pick two vertices u, v in bad_mask.
                # If there is an edge between them, remove it.
                # If not, add it.
                # This changes degrees of u and v in the subgraph, breaking regularity.
                
                vertices_in_mask = []
                m = bad_mask
                while m:
                    lsb = m & -m
                    v = lsb.bit_length() - 1
                    vertices_in_mask.append(v)
                    m ^= lsb
                    
                if len(vertices_in_mask) < 2:
                    # Should not happen for k>=6
                    break
                    
                u, v = random.sample(vertices_in_mask, 2)
                
                # Flip edge (u, v)
                if adj_mask[u] & (1 << v):
                    adj_mask[u] ^= (1 << v)
                    adj_mask[v] ^= (1 << u)
                else:
                    adj_mask[u] |= (1 << v)
                    adj_mask[v] |= (1 << u)
            
            # If we exhaust repair steps, try a new random start
    
    # Fallback: n=5
    return {'n': 5, 'edges': []}
