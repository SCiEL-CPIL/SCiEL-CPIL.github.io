import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # 10s margin
    
    def is_regular_subset(adj, mask, n):
        # Check if induced subgraph on mask is regular
        # Get degrees
        degs = []
        m = mask
        while m:
            lsb = m & -m
            v = lsb.bit_length() - 1
            deg = (adj[v] & mask).bit_count()
            degs.append(deg)
            m -= lsb
        
        if not degs:
            return False
        first = degs[0]
        for d in degs:
            if d != first:
                return False
        return True

    def find_bad_subset(adj, n, k, tries=100):
        # Try to find a bad subset of size k
        for _ in range(tries):
            # Pick random k vertices
            subset = random.sample(range(n), k)
            mask = 0
            for v in subset:
                mask |= (1 << v)
            if is_regular_subset(adj, mask, n):
                return mask
        return -1

    for n in range(20, 5, -1):
        for restart in range(500):
            if time.time() > deadline:
                break
            
            # Init random graph
            adj = [0] * n
            for i in range(n):
                for j in range(i + 1, n):
                    if random.random() < 0.5:
                        adj[i] |= (1 << j)
                        adj[j] |= (1 << i)
            
            # Local search
            success = False
            for step in range(5000):
                if time.time() > deadline:
                    break
                
                # Check for bad subsets
                bad_found = False
                for k in range(6, n + 1):
                    bad_mask = find_bad_subset(adj, n, k, tries=50)
                    if bad_mask != -1:
                        bad_found = True
                        # Flip a random edge in bad_mask
                        # Get vertices in mask
                        verts = []
                        m = bad_mask
                        while m:
                            lsb = m & -m
                            v = lsb.bit_length() - 1
                            verts.append(v)
                            m -= lsb
                        
                        # Pick random pair
                        i, j = random.sample(verts, 2)
                        # Flip edge
                        if adj[i] & (1 << j):
                            adj[i] ^= (1 << j)
                            adj[j] ^= (1 << i)
                        else:
                            adj[i] |= (1 << j)
                            adj[j] |= (1 << i)
                        break # Restart size check
                
                if not bad_found:
                    # Assume valid
                    edges = []
                    for i in range(n):
                        for j in range(i + 1, n):
                            if adj[i] & (1 << j):
                                edges.append([i, j])
                    return {'n': n, 'edges': edges}
            
        # If we fail for n, try n-1

    return {'n': 5, 'edges': []}
