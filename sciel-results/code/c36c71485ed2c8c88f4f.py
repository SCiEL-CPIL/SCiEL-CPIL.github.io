import random
import time
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin for 150s limit

    def check_valid(adj, n, min_deg):
        """Check if graph is C4-free and has min degree >= min_deg using bitmasks."""
        # Check min degree
        for i in range(n):
            if bin(adj[i]).count('1') < min_deg:
                return False
        # Check C4: any two vertices share at most 1 common neighbor
        for i in range(n):
            for j in range(i + 1, n):
                common = adj[i] & adj[j]
                if bin(common).count('1') > 1:
                    return False
        return True

    def get_edges(adj, n):
        """Extract edge list from bitmask adjacency."""
        edges = []
        for i in range(n):
            mask = adj[i]
            j = 0
            while mask:
                if mask & 1:
                    if i < j:
                        edges.append([i, j])
                mask >>= 1
                j += 1
        return edges

    def polarity_graph(q):
        """
        Construct polarity graph of projective plane of order q.
        Returns (n, adj) where adj is list of bitmasks.
        Only works for prime powers q.
        """
        # Check if q is a prime power
        def is_prime_power(q):
            if q < 2:
                return False
            for p in range(2, int(q**0.5) + 1):
                power = p
                while power <= q:
                    if power == q:
                        return True
                    power *= p
            return q == 2  # 2 is prime power

        if not is_prime_power(q):
            return None, None
        
        n = q * q + q + 1
        adj = [0] * n
        
        # For polarity graph, we need to define which pairs are adjacent
        # In a projective plane, a polarity makes points and lines correspond
        # Two points are adjacent if one lies on the polar line of the other
        
        # For simplicity, use a standard construction for small q
        # This is complex, so let's use a simpler approach:
        # For q=2 (Fano plane), 7 vertices, each degree 3
        # For q=3, 13 vertices, each degree 4
        # etc.
        
        # Actually, constructing polarity graphs correctly is non-trivial.
        # Let's use a different deterministic approach:
        # Start with empty graph and greedily add edges maintaining C4-freeness
        
        return None, None

    def greedy_construct(n, min_deg):
        """
        Greedily construct a C4-free graph with min degree >= min_deg.
        Returns adj list of bitmasks or None if failed.
        """
        adj = [0] * n
        deg = [0] * n
        
        # Try to add edges in a way that maintains C4-freeness
        # Process vertices in order, try to connect to existing vertices
        for u in range(n):
            # Try to connect u to vertices 0..u-1
            candidates = []
            for v in range(u):
                if not (adj[u] & (1 << v)):
                    # Check if adding (u,v) creates C4
                    # Need common_neighbors(u,v) == 0
                    common = adj[u] & adj[v]
                    if common == 0:
                        candidates.append(v)
            
            # Add edges to candidates until degree is sufficient
            random.shuffle(candidates)
            for v in candidates:
                if deg[u] >= min_deg:
                    break
                # Add edge (u, v)
                adj[u] |= (1 << v)
                adj[v] |= (1 << u)
                deg[u] += 1
                deg[v] += 1
        
        # Check if we achieved min degree
        if min(deg) >= min_deg:
            # Verify C4-freeness
            for i in range(n):
                for j in range(i + 1, n):
                    if bin(adj[i] & adj[j]).count('1') > 1:
                        return None
            return adj
        return None

    def local_search(adj, n, min_deg, max_iters=10000):
        """
        Local search to maximize minimum degree while maintaining C4-freeness.
        Returns improved adj or original if no improvement.
        """
        best_adj = adj[:]
        best_min_deg = min(bin(adj[i]).count('1') for i in range(n))
        
        for it in range(max_iters):
            if time.time() > deadline:
                break
            
            # Find vertex with minimum degree
            min_deg_vertex = min(range(n), key=lambda i: bin(adj[i]).count('1'))
            current_deg = bin(adj[min_deg_vertex]).count('1')
            
            if current_deg >= min_deg:
                # Check if we can improve further
                # Try to add edges to low-degree vertices
                low_verts = [i for i in range(n) if bin(adj[i]).count('1') < min_deg + 2]
                if not low_verts:
                    break
                u = random.choice(low_verts)
            else:
                u = min_deg_vertex
            
            # Find non-neighbors of u that can be connected
            candidates = []
            for v in range(n):
                if v == u:
                    continue
                if not (adj[u] & (1 << v)):
                    # Check if adding (u,v) creates C4
                    common = adj[u] & adj[v]
                    if common == 0:
                        candidates.append(v)
            
            if not candidates:
                continue
            
            # Try adding an edge to a random candidate
            v = random.choice(candidates)
            
            # Tentatively add edge
            adj[u] |= (1 << v)
            adj[v] |= (1 << u)
            
            # Check if C4-free
            c4_free = True
            for i in range(n):
                for j in range(i + 1, n):
                    if bin(adj[i] & adj[j]).count('1') > 1:
                        c4_free = False
                        break
                if not c4_free:
                    break
            
            if c4_free:
                new_min_deg = min(bin(adj[i]).count('1') for i in range(n))
                if new_min_deg > best_min_deg:
                    best_adj = adj[:]
                    best_min_deg = new_min_deg
                # Keep the edge
            else:
                # Revert
                adj[u] &= ~(1 << v)
                adj[v] &= ~(1 << u)
        
        return best_adj, best_min_deg

    def solve_for_n(n, min_deg):
        """
        Try to find a C4-free graph on n vertices with min degree >= min_deg.
        Returns adj list of bitmasks or None.
        """
        best_adj = None
        best_score = -1
        
        # Strategy 1: Random initialization with repair
        for attempt in range(300):
            if time.time() > deadline:
                break
            
            random.seed(attempt * 7919 + 13)
            
            # Initialize with random graph
            p = min(min_deg / n * 1.3, 0.7)
            adj = [0] * n
            
            for i in range(n):
                for j in range(i + 1, n):
                    if random.random() < p:
                        adj[i] |= (1 << j)
                        adj[j] |= (1 << i)
            
            # Repair phase
            improved = True
            max_iter = 300
            for _ in range(max_iter):
                if time.time() > deadline:
                    break
                
                # Check for C4s
                violating_pairs = []
                for i in range(n):
                    for j in range(i + 1, n):
                        common = adj[i] & adj[j]
                        if bin(common).count('1') > 1:
                            violating_pairs.append((i, j))
                
                if violating_pairs:
                    # Pick a violating pair and fix it
                    i, j = random.choice(violating_pairs)
                    common = adj[i] & adj[j]
                    common_list = []
                    mask = common
                    k = 0
                    while mask:
                        if mask & 1:
                            common_list.append(k)
                        mask >>= 1
                        k += 1
                    
                    # Remove edges to reduce common neighbors to 1
                    deg_i = bin(adj[i]).count('1')
                    deg_j = bin(adj[j]).count('1')
                    
                    for w in common_list[1:]:
                        if deg_i > deg_j:
                            adj[i] &= ~(1 << w)
                            adj[w] &= ~(1 << i)
                            deg_i -= 1
                        elif deg_j > deg_i:
                            adj[j] &= ~(1 << w)
                            adj[w] &= ~(1 << j)
                            deg_j -= 1
                        else:
                            if random.random() < 0.5:
                                adj[i] &= ~(1 << w)
                                adj[w] &= ~(1 << i)
                                deg_i -= 1
                            else:
                                adj[j] &= ~(1 << w)
                                adj[w] &= ~(1 << j)
                                deg_j -= 1
                else:
                    # No C4s, check min degree
                    min_deg_actual = min(bin(adj[i]).count('1') for i in range(n))
                    if min_deg_actual >= min_deg:
                        # Success!
                        if check_valid(adj, n, min_deg):
                            return adj
                        break
                    else:
                        # Need to add edges
                        low_verts = [i for i in range(n) if bin(adj[i]).count('1') < min_deg]
                        if not low_verts:
                            break
                        u = random.choice(low_verts)
                        candidates = []
                        for v in range(n):
                            if v == u:
                                continue
                            if not (adj[u] & (1 << v)):
                                common = adj[u] & adj[v]
                                if common == 0:
                                    candidates.append(v)
                        
                        if candidates:
                            v = random.choice(candidates)
                            adj[u] |= (1 << v)
                            adj[v] |= (1 << u)
                        else:
                            pass
        
        return None

    def solve_with_local_search(n, min_deg):
        """
        Use greedy construction followed by local search.
        """
        # Try greedy construction
        adj = greedy_construct(n, min_deg)
        
        if adj is not None:
            # Apply local search to improve
            adj, min_deg_achieved = local_search(adj, n, min_deg, max_iters=5000)
            if min_deg_achieved >= min_deg and check_valid(adj, n, min_deg):
                return adj
        
        return None

    # Try n from 47 down to 43
    for n in range(47, 42, -1):
        min_deg = n - 39
        if min_deg < 1:
            continue
        
        # Try random repair first
        result_adj = solve_for_n(n, min_deg)
        
        if result_adj is not None:
            if check_valid(result_adj, n, min_deg):
                edges = get_edges(result_adj, n)
                return {'n': n, 'edges': edges}
        
        # Try greedy + local search
        result_adj = solve_with_local_search(n, min_deg)
        
        if result_adj is not None:
            if check_valid(result_adj, n, min_deg):
                edges = get_edges(result_adj, n)
                return {'n': n, 'edges': edges}
        
        # If we fail for n, try n-1

    # Fallback: Return a valid small graph
    # For n=40, min_deg=1. Easy to construct.
    n = 40
    min_deg = 1
    adj = [0] * n
    # Create a simple cycle
    for i in range(n):
        adj[i] |= (1 << ((i + 1) % n))
        adj[(i + 1) % n] |= (1 << i)
    
    # Verify
    if check_valid(adj, n, min_deg):
        edges = get_edges(adj, n)
        return {'n': n, 'edges': edges}
    
    # Absolute fallback: empty graph on 39 vertices
    return {'n': 39, 'edges': []}
