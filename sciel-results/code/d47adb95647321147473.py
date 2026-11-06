import random
import time
from itertools import combinations

def _check_valid_and_get_bad_mask(adj_mask, n):
    """
    Returns (True, 0) if valid.
    Returns (False, mask) if an invalid regular induced subgraph is found.
    """
    # Iterate subsets of size >= 6
    # To be fast, we check smaller sizes first? 
    # Size 6 is most numerous, so likely to find a violation there.
    # But checking size 6 is C(n,6). For n=21, C(21,6)=54264.
    # Checking size 7 is C(21,7)=116280.
    # It's better to just iterate all masks and check popcount >= 6.
    # 2^21 = 2M masks.
    
    # Optimization: 
    # A regular subgraph of size k must have average degree d.
    # We can iterate masks of size k for k=6..n.
    
    # Let's use Gosper's hack for each k.
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

def construct():
    start_time = time.time()
    deadline = start_time + 140  # 10s margin
    
    # We want to maximize n.
    # Try n=22, then 21, then 20.
    
    for n in [22, 21, 20]:
        if time.time() > deadline - 10:
            break
            
        # Try multiple random starts
        max_starts = 50 if n <= 20 else 20
        found = False
        
        for _ in range(max_starts):
            if time.time() > deadline - 5:
                break
                
            # Generate random graph
            adj_mask = [0] * n
            for i in range(n):
                for j in range(i + 1, n):
                    if random.random() < 0.5:
                        adj_mask[i] |= (1 << j)
                        adj_mask[j] |= (1 << i)
                        
            # Local Search / Repair
            # We iterate, finding a bad mask, fixing it, and re-checking.
            # To avoid infinite loops, we limit repair steps per start.
            for repair_step in range(200):
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
