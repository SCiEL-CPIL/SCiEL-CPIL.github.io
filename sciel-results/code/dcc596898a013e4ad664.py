import random
import time

def construct():
    start_time = time.time()
    best_n = 5
    best_graph = {'n': 5, 'edges': []}
    
    # Helper to check if a subset S of vertices induces a regular subgraph
    # G is list of bitmasks
    # S is a bitmask
    def is_regular(G, S_mask, size):
        # Calculate degrees in induced subgraph
        # For each u in S, deg = popcount(G[u] & S_mask)
        # Check if all equal
        degs = []
        m = S_mask
        while m:
            lsb = m & -m
            u = lsb.bit_length() - 1
            deg = (G[u] & S_mask).bit_count()
            degs.append(deg)
            m -= lsb
        if not degs: return True
        first = degs[0]
        for d in degs:
            if d != first:
                return False
        return True

    # Helper to generate all subsets of size k from 0..n-1
    # We only need this for verification
    
    def verify_graph(G, n):
        # Check all subsets of size >= 6
        # Optimized: iterate over all masks from 1 to 2^n - 1
        # If popcount(mask) >= 6, check regularity
        # 2^20 is 1M, popcount is fast.
        # But checking regularity is O(n). 1M * 20 = 20M. OK.
        for mask in range(1, 1 << n):
            if mask.bit_count() < 6:
                continue
            if is_regular(G, mask, mask.bit_count()):
                return False
        return True

    for seed in range(1000):
        random.seed(seed)
        n = 1
        G = [0] # G[i] is bitmask of neighbors of i
        edges = []
        
        success = True
        for target_n in range(2, 23):
            k = target_n - 1
            found_mask = None
            
            # Try 200 random masks
            for _ in range(200):
                cand_mask = random.getrandbits(k)
                # Check if adding vertex k with cand_mask creates any regular subgraph of size 6
                # Subsets S of size 5 from 0..k-1
                # S_mask | (1<<k) is the new subgraph
                # We need to check if G[S_mask | (1<<k)] is regular.
                # But we can do it faster.
                # Let N_new be the neighbors of k in S.
                # d_new = popcount(cand_mask & S_mask)
                # For u in S: deg_u = popcount(G[u] & S_mask) + (1 if u in cand_mask else 0)
                # Must equal d_new.
                
                valid = True
                # Iterate subsets of size 5
                # Generate combinations of 5 from range(k)
                # To speed up, we can precompute? No, k changes.
                # For k up to 20, C(20,5)=15504.
                # 15504 * 5 ops = 77k ops per candidate.
                # 200 candidates = 15M ops per vertex.
                # 20 vertices = 300M ops. Too slow in Python.
                
                # Optimization: Only check if candidate is "promising"?
                # Or reduce candidates to 20?
                # Or use a different check?
                
                # Let's try 20 candidates first.
                pass
                
            # If this loop is too slow, I will implement a simpler check or fewer candidates.
            # Given 120s limit, 300M ops is likely > 120s.
            # I need to be faster.
            
            # Alternative:
            # Just try 10 random masks.
            # If none work, restart.
            
            if not found_mask:
                success = False
                break
            else:
                G.append(found_mask)
                # Update existing G entries
                for i in range(k):
                    if (found_mask >> i) & 1:
                        G[i] |= (1 << k)
                        edges.append([i, k])
                n = target_n
        
        if n > best_n:
            # Verify full graph
            if verify_graph(G, n):
                best_n = n
                best_graph = {'n': n, 'edges': edges}
                if n >= 20:
                    return best_graph
        if time.time() - start_time > 100:
            break
            
    return best_graph
