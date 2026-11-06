import random
import time
import itertools
import sys

def construct():
    start_time = time.time()
    deadline = start_time + 115  # Safety margin

    def pc(x):
        return x.bit_count()

    def check_regular(adj, mask, n):
        """Check if induced subgraph on mask is regular."""
        degs = []
        m = mask
        while m:
            lsb = m & -m
            v = lsb.bit_length() - 1
            m ^= lsb
            deg = pc(adj[v] & mask)
            degs.append(deg)
            if len(degs) > 1 and degs[0] != deg:
                return False
        return len(set(degs)) == 1

    def count_violations(adj, n, sizes):
        """Count regular induced subgraphs of given sizes."""
        total = 0
        for k in sizes:
            if k > n: continue
            for combo in itertools.combinations(range(n), k):
                mask = 0
                for v in combo:
                    mask |= (1 << v)
                if check_regular(adj, mask, n):
                    total += 1
        return total

    def full_check(adj, n):
        """Check all subsets of size >= 6."""
        for k in range(6, n + 1):
            for combo in itertools.combinations(range(n), k):
                mask = 0
                for v in combo:
                    mask |= (1 << v)
                if check_regular(adj, mask, n):
                    return False
        return True

    def get_edges(adj, n):
        edges = []
        for i in range(n):
            for j in range(i + 1, n):
                if adj[i] & (1 << j):
                    edges.append([i, j])
        return edges

    # Try n=21 first
    for n in [21, 20]:
        if time.time() > deadline - 10:
            break
            
        # Pre-compute masks for size 6, 7, 8 to speed up count_violations
        # Actually, itertools.combinations is fast enough for 54k items if called repeatedly?
        # No, generating combinations every time is slow.
        # Let's pre-generate the masks for sizes 6, 7, 8.
        
        masks_6 = []
        for combo in itertools.combinations(range(n), 6):
            mask = 0
            for v in combo:
                mask |= (1 << v)
            masks_6.append(mask)
            
        masks_7 = []
        if n >= 7:
            for combo in itertools.combinations(range(n), 7):
                mask = 0
                for v in combo:
                    mask |= (1 << v)
                masks_7.append(mask)
                
        masks_8 = []
        if n >= 8:
            for combo in itertools.combinations(range(n), 8):
                mask = 0
                for v in combo:
                    mask |= (1 << v)
                masks_8.append(mask)

        def count_violations_fast(adj):
            total = 0
            for mask in masks_6:
                if check_regular(adj, mask, n):
                    total += 1
            if total > 0:
                return total
            for mask in masks_7:
                if check_regular(adj, mask, n):
                    total += 1
            if total > 0:
                return total
            for mask in masks_8:
                if check_regular(adj, mask, n):
                    total += 1
            return total

        best_adj = None
        best_violations = float('inf')
        
        # Random restarts
        for restart in range(50):
            if time.time() > deadline - 5:
                break
                
            adj = [0] * n
            for i in range(n):
                for j in range(i + 1, n):
                    if random.random() < 0.5:
                        adj[i] |= (1 << j)
                        adj[j] |= (1 << i)
            
            curr_violations = count_violations_fast(adj)
            
            if curr_violations < best_violations:
                best_violations = curr_violations
                best_adj = adj[:]
                
            if curr_violations == 0:
                if full_check(adj, n):
                    return {'n': n, 'edges': get_edges(adj, n)}

            # Simulated Annealing
            T = 5.0
            T_min = 0.01
            cooling = 0.95
            
            for step in range(2000):
                if time.time() > deadline - 5:
                    break
                    
                u = random.randrange(n)
                v = random.randrange(n)
                if u == v: continue
                
                # Flip edge
                if adj[u] & (1 << v):
                    adj[u] &= ~(1 << v)
                    adj[v] &= ~(1 << u)
                    is_present = False
                else:
                    adj[u] |= (1 << v)
                    adj[v] |= (1 << u)
                    is_present = True
                
                new_violations = count_violations_fast(adj)
                delta = new_violations - curr_violations
                
                if delta < 0 or random.random() < pow(2.718, -delta / T):
                    curr_violations = new_violations
                    if curr_violations < best_violations:
                        best_violations = curr_violations
                        best_adj = adj[:]
                        
                    if curr_violations == 0:
                        if full_check(adj, n):
                            return {'n': n, 'edges': get_edges(adj, n)}
                else:
                    # Revert
                    if is_present:
                        adj[u] &= ~(1 << v)
                        adj[v] &= ~(1 << u)
                    else:
                        adj[u] |= (1 << v)
                        adj[v] |= (1 << u)
                
                T *= cooling
                if T < T_min:
                    T = T_min

    # Fallback to best found
    if best_adj and best_violations == 0:
        if full_check(best_adj, len(best_adj)):
            return {'n': len(best_adj), 'edges': get_edges(best_adj, len(best_adj))}
            
    # Fallback to n=10 random
    n = 10
    random.seed(42)
    adj = [0] * n
    for i in range(n):
        for j in range(i + 1, n):
            if random.random() < 0.5:
                adj[i] |= (1 << j)
                adj[j] |= (1 << i)
    if full_check(adj, n):
        return {'n': n, 'edges': get_edges(adj, n)}
        
    return {'n': 5, 'edges': []}
