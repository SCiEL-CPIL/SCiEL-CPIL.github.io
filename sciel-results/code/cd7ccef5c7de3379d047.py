import time
import random
from itertools import combinations

def _is_regular_subgraph(G, subset_mask, k):
    # G is list of ints
    # subset_mask is int with k bits set
    # returns True if regular
    # Extract vertices
    verts = []
    m = subset_mask
    while m:
        lsb = m & -m
        idx = lsb.bit_length() - 1
        verts.append(idx)
        m ^= lsb
    
    # Calculate degrees
    degs = []
    for v in verts:
        # neighbors in subset
        # G[v] has bits for neighbors
        # mask for subset without v
        sub_mask = subset_mask & ~(1 << v)
        d = (G[v] & sub_mask).bit_count()
        degs.append(d)
        
    return len(set(degs)) == 1

def _get_subsets_containing(n, u, v, k):
    # Yield masks of size k containing u and v
    # Remaining k-2 vertices from n-2 others
    others = [i for i in range(n) if i != u and i != v]
    base_mask = (1 << u) | (1 << v)
    for combo in combinations(others, k-2):
        mask = base_mask
        for x in combo:
            mask |= (1 << x)
        yield mask

def construct():
    start_time = time.time()
    deadline = start_time + 110 # 10s margin
    
    best_n = 5
    best_G = [0]*5
    
    for n in range(20, 4, -1):
        if time.time() > deadline:
            break
            
        found = False
        # Try multiple random restarts
        for seed in range(50):
            if time.time() > deadline - 5:
                break
                
            random.seed(seed)
            # Init random graph
            G = [0]*n
            for i in range(n):
                for j in range(i+1, n):
                    if random.random() < 0.5:
                        G[i] |= (1 << j)
                        G[j] |= (1 << i)
                        
            # Local search to eliminate size 6 regular induced subgraphs
            # Maintain set of bad masks
            bad_masks = set()
            
            # Initial scan for size 6
            for mask in _generate_masks(n, 6):
                if _is_regular_subgraph(G, mask, 6):
                    bad_masks.add(mask)
                    
            # Iterate
            max_iters = 10000
            for _ in range(max_iters):
                if time.time() > deadline - 5:
                    break
                
                if not bad_masks:
                    # Check larger sizes
                    is_valid = True
                    for k in range(7, n+1):
                        # Sample or full check? Full check is slow if many subsets.
                        # But if we reach here, size 6 is clean.
                        # Likely size 7 is also clean.
                        # Let's do a full check for safety, but optimized.
                        # Actually, if n=20, subsets of size 7 is 77k.
                        # Checking 77k subsets takes ~0.5s.
                        # We can afford this check occasionally.
                        for mask in _generate_masks(n, k):
                             if _is_regular_subgraph(G, mask, k):
                                 # Found a larger bad subset
                                 # Add to bad_masks? 
                                 # Our logic above only handles size 6 updates efficiently.
                                 # If we find a size 7 bad subset, we should probably restart 
                                 # or handle it.
                                 # For now, mark invalid and break.
                                 is_valid = False
                                 break
                        if not is_valid:
                            break
                    
                    if is_valid:
                        best_n = n
                        best_G = G
                        found = True
                        break
                    
                    # If not valid, we need to break the larger subset.
                    # Pick a random edge in the bad subset and flip?
                    # This is complex. Let's just restart with a new seed.
                    # Or, add the larger bad subset to a list and try to break it.
                    # Given time constraints, restarting is safer.
                    break 

                # Pick a random bad mask
                bad_mask = random.choice(list(bad_masks))
                # Get vertices in bad_mask
                verts = []
                m = bad_mask
                while m:
                    lsb = m & -m
                    idx = lsb.bit_length() - 1
                    verts.append(idx)
                    m ^= lsb
                
                # Pick a random edge (u,v) in this subset
                u, v = random.sample(verts, 2)
                
                # Flip edge (u,v)
                if (G[u] >> v) & 1:
                    G[u] &= ~(1 << v)
                    G[v] &= ~(1 << u)
                    is_present = False
                else:
                    G[u] |= (1 << v)
                    G[v] |= (1 << u)
                    is_present = True
                
                # Update bad_masks
                # Only subsets of size 6 containing u and v are affected
                # 1. Check existing bad masks
                to_remove = []
                for bm in bad_masks:
                    if (bm >> u) & 1 and (bm >> v) & 1:
                        # This mask contains u and v
                        # Re-check regularity
                        if not _is_regular_subgraph(G, bm, 6):
                            to_remove.append(bm)
                
                for rm in to_remove:
                    bad_masks.remove(rm)
                    
                # 2. Check new candidates
                # Subsets of size 6 containing u and v that were NOT bad
                # We can iterate all such subsets and check if they are now bad
                # But we don't know which ones were not bad.
                # We can just iterate ALL subsets containing u,v and check if regular.
                # If regular, add to bad_masks (if not already there).
                for mask in _get_subsets_containing(n, u, v, 6):
                    if _is_regular_subgraph(G, mask, 6):
                        bad_masks.add(mask)
                
            if found:
                break
                
        if found:
            break
            
    # Convert G to edges
    edges = []
    for i in range(best_n):
        for j in range(i+1, best_n):
            if (G[i] >> j) & 1:
                edges.append([i, j])
                
    return {'n': best_n, 'edges': edges}

def _generate_masks(n, k):
    # Generator for all masks of size k
    # Simple recursion or itertools
    for combo in combinations(range(n), k):
        mask = 0
        for v in combo:
            mask |= (1 << v)
        yield mask
