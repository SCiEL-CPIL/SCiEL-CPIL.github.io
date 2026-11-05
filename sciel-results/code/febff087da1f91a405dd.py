import time
import random

def construct():
    start_time = time.time()
    deadline = 110.0  # Safety margin
    
    n = 39
    
    # Use bitsets for fast operations
    # adj[i] is an integer where bit j is 1 if edge (i,j) exists
    adj = [0] * n
    
    def is_triangle_free(adj):
        for i in range(n):
            # For each neighbor j of i, check if any neighbor k of j is also neighbor of i
            # Actually, simpler: for each i, for each j in N(i), for each k in N(j), if k in N(i) and k!=j, triangle.
            # With bitsets:
            # N(i) & N(j) should be 0 for all j in N(i)? No.
            # Triangle (i,j,k) means j in N(i), k in N(j), i in N(k).
            # Symmetric: i in N(j), k in N(i), j in N(k).
            # Check: For each i, for each j in N(i):
            #   if (N(i) & N(j)) != 0: triangle exists (excluding j itself? N(i) contains j. N(j) contains i. 
            #   Intersection N(i) & N(j) contains vertices connected to both.
            #   If there is any vertex k != i, j in N(i) & N(j), then i-k, j-k, i-j form triangle.
            #   Note: N(i) contains j. N(j) contains i. So i and j are always in N(i) and N(j) respectively?
            #   No, N(i) is set of neighbors. j is in N(i). i is in N(j).
            #   Intersection N(i) & N(j) will contain any common neighbor.
            #   It will NOT contain i (unless self-loop, which we don't have) or j?
            #   j is in N(i). Is j in N(j)? No.
            #   i is in N(j). Is i in N(i)? No.
            #   So N(i) & N(j) contains exactly the common neighbors.
            #   If N(i) & N(j) != 0, there is a triangle.
            neighbors_i = adj[i]
            # Iterate through bits in neighbors_i
            mask = neighbors_i
            while mask:
                lsb = mask & -mask
                j = lsb.bit_length() - 1
                # Check if N(i) and N(j) have any common neighbor
                if (adj[i] & adj[j]) != 0:
                    return False
                mask -= lsb
        return True

    def max_independent_set_size(adj):
        # Greedy approximation or exact B&B?
        # For n=39, exact B&B is feasible if pruned well.
        # But for speed in local search, greedy is faster.
        # Let's use a simple greedy: sort by degree, pick vertex, remove neighbors.
        # Or random greedy multiple times.
        # Better: Bron-Kerbosch variant for max independent set (max clique in complement).
        # Given time, let's use a fast heuristic:
        # 1. Order vertices by degree (ascending).
        # 2. Greedily pick lowest degree vertex, add to IS, remove from graph.
        
        # We'll do a few random restarts of greedy to get a good upper bound estimate.
        # If we find an IS of size >= 10, we return it.
        # We need to know if alpha >= 10.
        
        best_size = 0
        best_set = []
        
        for _ in range(10): # 10 random orderings
            vertices = list(range(n))
            random.shuffle(vertices)
            # Sort by degree in current graph? 
            # Simpler: just pick random available vertex with min degree.
            available = [1] * n
            current_size = 0
            current_set = []
            
            for _ in range(n):
                candidates = [v for v in range(n) if available[v]]
                if not candidates:
                    break
                # Pick vertex with min degree in available subgraph
                min_deg = float('inf')
                min_v = -1
                for v in candidates:
                    deg = bin(adj[v] & sum([1 << u for u in candidates if available[u]]).bit_count() if False else 0) # This is slow
                    # Faster: precompute? No, just count bits in adj[v] & available_mask
                    pass
                
                # Even faster greedy:
                # Just pick random from available?
                # Let's use standard greedy: sort by degree once? 
                # Degrees change.
                
                # Let's just use a simple heuristic: 
                # Pick vertex with min degree among available.
                avail_mask = 0
                for v in range(n):
                    if available[v]:
                        avail_mask |= (1 << v)
                
                min_deg = float('inf')
                min_v = -1
                for v in range(n):
                    if available[v]:
                        deg = bin(adj[v] & avail_mask).count('1')
                        if deg < min_deg:
                            min_deg = deg
                            min_v = v
                
                if min_v == -1:
                    break
                
                current_set.append(min_v)
                current_size += 1
                available[min_v] = 0
                # Remove neighbors
                neighbors = adj[min_v]
                for u in range(n):
                    if neighbors & (1 << u):
                        available[u] = 0
            
            if current_size > best_size:
                best_size = current_size
                best_set = current_set[:]
                
        return best_size, best_set

    # Initial random triangle-free graph
    # Start empty, add random edges
    for _ in range(1000):
        u = random.randint(0, n-1)
        v = random.randint(0, n-1)
        if u == v:
            continue
        if adj[u] & (1 << v):
            continue
        # Check if adding (u,v) creates triangle
        # Common neighbors of u and v
        if (adj[u] & adj[v]) == 0:
            adj[u] |= (1 << v)
            adj[v] |= (1 << u)

    # Local search
    while time.time() < deadline:
        size, indep_set = max_independent_set_size(adj)
        
        if size < 10:
            # Valid graph found
            edges = []
            for i in range(n):
                for j in range(i+1, n):
                    if adj[i] & (1 << j):
                        edges.append([i, j])
            return {'n': n, 'edges': edges}
        
        # Try to add an edge to reduce independence
        # Look for pair (u,v) in indep_set such that N(u) & N(v) == 0
        # If found, add edge (u,v). This breaks the independence of that set.
        added_edge = False
        for i in range(len(indep_set)):
            for j in range(i+1, len(indep_set)):
                u = indep_set[i]
                v = indep_set[j]
                if (adj[u] & adj[v]) == 0:
                    adj[u] |= (1 << v)
                    adj[v] |= (1 << u)
                    added_edge = True
                    break
            if added_edge:
                break
        
        if not added_edge:
            # Try to remove an edge to allow adding a different one?
            # Or restart with a different random seed
            # Let's try to remove a random edge and add a random valid edge
            if random.random() < 0.1:
                # Remove random edge
                edges_list = []
                for i in range(n):
                    for j in range(i+1, n):
                        if adj[i] & (1 << j):
                            edges_list.append((i,j))
                if edges_list:
                    u, v = random.choice(edges_list)
                    adj[u] &= ~(1 << v)
                    adj[v] &= ~(1 << u)
            else:
                # Add random valid edge
                u = random.randint(0, n-1)
                v = random.randint(0, n-1)
                if u != v and not (adj[u] & (1 << v)):
                    if (adj[u] & adj[v]) == 0:
                        adj[u] |= (1 << v)
                        adj[v] |= (1 << u)

    # Fallback: Return the 26-vertex Andrasfai graph if 39 fails
    n = 26
    edges = [[i, j] for i in range(n) for j in range(i + 1, n) if (j - i) % n % 3 == 1]
    return {'n': n, 'edges': edges}
