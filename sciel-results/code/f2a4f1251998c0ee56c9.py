import random
import time
from itertools import combinations

def construct():
    # We want to maximize n.
    # Known: n=20 is possible. n=21 is the target.
    # Strategy: Try to find a valid graph on n=21.
    # If not found quickly, fallback to n=20.
    
    def is_valid(n, adj):
        # adj is a list of int bitmasks
        # Check all subsets of size >= 6
        # Optimization: Iterate masks
        for mask in range(1, 1 << n):
            # Count bits
            k = mask.bit_count()
            if k < 6:
                continue
            
            # Check if induced subgraph on 'mask' is regular
            # Get degrees of vertices in mask
            degs = []
            # We can break early if we find two different degrees
            first_deg = -1
            is_reg = True
            m = mask
            while m:
                # Get lowest set bit
                lsb = m & -m
                v = lsb.bit_length() - 1
                m -= lsb
                # Degree of v within mask
                d = (adj[v] & mask).bit_count()
                if first_deg == -1:
                    first_deg = d
                elif d != first_deg:
                    is_reg = False
                    break
            
            if is_reg:
                return False
        
        return True

    # Try to find a valid graph on n=21
    # We will try random regular graphs of degree 5 and 6
    # and random graphs with p=0.5
    
    def generate_random_regular(n, d, seed):
        # Simple random regular graph generator
        # This can be tricky to implement correctly from scratch quickly.
        # Alternative: Generate random graph and check degree sequence?
        # Or just use random graphs with p ~ d/n.
        # For n=21, d=5, p ~ 0.24. d=6, p ~ 0.28.
        # Let's just use random graphs with varying p.
        pass

    def try_random_graph(n, p, seed):
        random.seed(seed)
        adj = [0] * n
        for i in range(n):
            for j in range(i+1, n):
                if random.random() < p:
                    adj[i] |= (1 << j)
                    adj[j] |= (1 << i)
        return adj

    # Try n=21
    found_21 = False
    start_time = time.time()
    max_time = 60.0  # Leave time for n=20 if needed
    
    for seed in range(1000):
        if time.time() - start_time > max_time:
            break
            
        # Try different densities
        for p in [0.4, 0.5, 0.6]:
            adj = try_random_graph(21, p, seed*10 + int(p*10))
            if is_valid(21, adj):
                edges = []
                for i in range(21):
                    for j in range(i+1, 21):
                        if adj[i] & (1 << j):
                            edges.append([i, j])
                return {'n': 21, 'edges': edges}
    
    # Fallback to n=20
    # n=20 is known to be solvable.
    # We will try to find a valid graph on n=20 quickly.
    for seed in range(1000):
        if time.time() - start_time > 110.0:
            break
        for p in [0.4, 0.5, 0.6]:
            adj = try_random_graph(20, p, seed*10 + int(p*10))
            if is_valid(20, adj):
                edges = []
                for i in range(20):
                    for j in range(i+1, 20):
                        if adj[i] & (1 << j):
                            edges.append([i, j])
                return {'n': 20, 'edges': edges}
                
    # If all else fails, return a trivially valid small graph
    # But n=20 should be found.
    # Just in case, return n=5
    return {'n': 5, 'edges': []}
