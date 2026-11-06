import random
import time
from itertools import combinations

def construct():
    # We want to maximize n.
    # Known: n=20 is possible. n=21 is the target.
    # Strategy: Use local search (repair) to find a valid graph on n=21.
    # If not found, fallback to n=20.
    
    def find_regular_subgraph(n, adj):
        # Returns a mask of a regular induced subgraph of size >= 6, or None.
        # Optimized to check smaller sizes first.
        for k in range(6, n + 1):
            # Generate combinations
            for comb in combinations(range(n), k):
                mask = 0
                for v in comb:
                    mask |= (1 << v)
                
                # Check if induced subgraph on 'mask' is regular
                first_deg = -1
                is_reg = True
                m = mask
                while m:
                    lsb = m & -m
                    v = lsb.bit_length() - 1
                    m -= lsb
                    d = (adj[v] & mask).bit_count()
                    if first_deg == -1:
                        first_deg = d
                    elif d != first_deg:
                        is_reg = False
                        break
                
                if is_reg:
                    return mask
        return None

    def generate_random_graph(n, p, seed):
        random.seed(seed)
        adj = [0] * n
        for i in range(n):
            for j in range(i+1, n):
                if random.random() < p:
                    adj[i] |= (1 << j)
                    adj[j] |= (1 << i)
        return adj

    def flip_edge(adj, n, u, v):
        # Flip edge between u and v
        if adj[u] & (1 << v):
            adj[u] &= ~(1 << v)
            adj[v] &= ~(1 << u)
        else:
            adj[u] |= (1 << v)
            adj[v] |= (1 << u)

    def local_search(n, p, seed, max_iters=200):
        random.seed(seed)
        adj = generate_random_graph(n, p, seed)
        
        # Check initial state
        # We want to find a graph with NO regular subgraphs.
        # So if find_regular_subgraph returns None, we are done.
        
        for i in range(max_iters):
            mask = find_regular_subgraph(n, adj)
            if mask is None:
                return adj
            
            # Found a regular subgraph. Try to break it.
            # Pick a random vertex in the subgraph and a random neighbor (in or out) to flip.
            # Or just pick a random edge in the subgraph and flip it.
            vertices_in_subgraph = []
            m = mask
            while m:
                lsb = m & -m
                v = lsb.bit_length() - 1
                m -= lsb
                vertices_in_subgraph.append(v)
            
            # Pick a random vertex from the subgraph
            v = random.choice(vertices_in_subgraph)
            # Pick a random other vertex to flip edge with
            # Bias towards vertices in the subgraph to break the regularity locally
            if random.random() < 0.7:
                u = random.choice(vertices_in_subgraph)
            else:
                u = random.randint(0, n - 1)
            
            if u != v:
                flip_edge(adj, n, u, v)
                
        return None # Failed to find valid graph

    # Try n=21
    start_time = time.time()
    max_time_21 = 90.0
    
    for seed in range(50):
        if time.time() - start_time > max_time_21:
            break
        # Try different densities
        for p in [0.4, 0.5, 0.6]:
            adj = local_search(21, p, seed * 10 + int(p * 10), max_iters=50)
            if adj is not None:
                # Verify one last time just in case
                if find_regular_subgraph(21, adj) is None:
                    edges = []
                    for i in range(21):
                        for j in range(i+1, 21):
                            if adj[i] & (1 << j):
                                edges.append([i, j])
                    return {'n': 21, 'edges': edges}

    # Fallback to n=20
    for seed in range(50):
        if time.time() - start_time > 115.0:
            break
        for p in [0.4, 0.5, 0.6]:
            adj = local_search(20, p, seed * 10 + int(p * 10), max_iters=50)
            if adj is not None:
                if find_regular_subgraph(20, adj) is None:
                    edges = []
                    for i in range(20):
                        for j in range(i+1, 20):
                            if adj[i] & (1 << j):
                                edges.append([i, j])
                    return {'n': 20, 'edges': edges}
                
    # If all else fails, return a trivially valid small graph
    return {'n': 5, 'edges': []}
