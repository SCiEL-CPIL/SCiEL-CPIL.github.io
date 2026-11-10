import itertools
import random
import time

def construct():
    """
    Constructs a graph with no induced regular subgraph on 6 or more vertices.
    Returns a dictionary with 'n' and 'edges'.
    """
    
    def is_regular_mask(mask, adj):
        """Check if induced subgraph on vertices in 'mask' is regular."""
        m = mask
        first_deg = -1
        while m:
            lsb = m & -m
            v = lsb.bit_length() - 1
            deg = (adj[v] & mask).bit_count()
            if first_deg == -1:
                first_deg = deg
            elif deg != first_deg:
                return False
            m ^= lsb
        return True

    target_n = 20
    best_adj = None
    start_time = time.time()
    time_limit = 100
    
    # Precompute 6-masks for speed
    six_masks = []
    for subset in itertools.combinations(range(target_n), 6):
        mask = 0
        for v in subset:
            mask |= (1 << v)
        six_masks.append(mask)
            
    for trial in range(50):
        if time.time() - start_time > time_limit:
            break
            
        n = target_n
        adj = [0] * n
        # Random graph
        for i in range(n):
            for j in range(i + 1, n):
                if random.random() < 0.5:
                    adj[i] |= (1 << j)
                    adj[j] |= (1 << i)
                    
        # Greedy fix for 6-sets
        for _ in range(500):
            # Find a bad 6-set
            bad_mask = None
            # Random sample of 6-sets
            for _ in range(20):
                m = random.choice(six_masks)
                if is_regular_mask(m, adj):
                    bad_mask = m
                    break
            
            if bad_mask is None:
                break
                
            # Break it
            vertices = [v for v in range(n) if (bad_mask >> v) & 1]
            u = random.choice(vertices)
            v = random.randint(0, n - 1)
            if u == v:
                continue
            if (adj[u] >> v) & 1:
                adj[u] &= ~(1 << v)
                adj[v] &= ~(1 << u)
            else:
                adj[u] |= (1 << v)
                adj[v] |= (1 << u)
                
        # Verify 6-sets fully
        valid_6 = True
        for m in six_masks:
            if is_regular_mask(m, adj):
                valid_6 = False
                break
                
        if valid_6:
            # Check 7-sets
            valid_7 = True
            for subset in itertools.combinations(range(n), 7):
                mask = 0
                for v in subset:
                    mask |= (1 << v)
                if is_regular_mask(mask, adj):
                    valid_7 = False
                    break
            
            if valid_7:
                # Check 8-10 sets
                valid_large = True
                for size in range(8, 11):
                    for subset in itertools.combinations(range(n), size):
                        mask = 0
                        for v in subset:
                            mask |= (1 << v)
                        if is_regular_mask(mask, adj):
                            valid_large = False
                            break
                    if not valid_large:
                        break
                
                if valid_large:
                    best_adj = adj
                    break

    if best_adj:
        # Final full verification
        n = target_n
        adj = best_adj
        valid = True
        for size in range(6, n + 1):
            for subset in itertools.combinations(range(n), size):
                mask = 0
                for v in subset:
                    mask |= (1 << v)
                if is_regular_mask(mask, adj):
                    valid = False
                    break
            if not valid:
                break
        
        if valid:
            edges = []
            for i in range(n):
                for j in range(i + 1, n):
                    if (adj[i] >> j) & 1:
                        edges.append([i, j])
            return {'n': n, 'edges': edges}

    # Fallback: n=10
    n = 10
    for _ in range(1000):
        adj = [0] * n
        for i in range(n):
            for j in range(i + 1, n):
                if random.random() < 0.5:
                    adj[i] |= (1 << j)
                    adj[j] |= (1 << i)
        valid = True
        for size in range(6, n + 1):
            for subset in itertools.combinations(range(n), size):
                mask = 0
                for v in subset:
                    mask |= (1 << v)
                if is_regular_mask(mask, adj):
                    valid = False
                    break
            if not valid:
                break
        if valid:
            edges = []
            for i in range(n):
                for j in range(i + 1, n):
                    if (adj[i] >> j) & 1:
                        edges.append([i, j])
            return {'n': n, 'edges': edges}
            
    return {'n': 5, 'edges': []}
