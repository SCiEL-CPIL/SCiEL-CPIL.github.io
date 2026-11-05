import random
import itertools
import time

def _find_violation(n, adj_mask):
    """Find a regular induced subgraph of size >= 6.
    Returns the subset as a tuple if found, else None."""
    vertices = list(range(n))
    # Check sizes from 6 to n
    for k in range(6, n + 1):
        for subset in itertools.combinations(vertices, k):
            mask_S = 0
            for v in subset:
                mask_S |= (1 << v)
            
            degs = []
            for v in subset:
                common = adj_mask[v] & mask_S
                d = bin(common).count('1')
                degs.append(d)
            
            if len(set(degs)) == 1:
                return subset
    return None

def _check_valid(n, adj_mask):
    return _find_violation(n, adj_mask) is None

def _random_adj_mask(n, p=0.5):
    adj_mask = [0] * n
    for i in range(n):
        for j in range(i + 1, n):
            if random.random() < p:
                adj_mask[i] |= (1 << j)
                adj_mask[j] |= (1 << i)
    return adj_mask

def _flip_edge(adj_mask, u, v):
    if adj_mask[u] & (1 << v):
        adj_mask[u] &= ~(1 << v)
        adj_mask[v] &= ~(1 << u)
    else:
        adj_mask[u] |= (1 << v)
        adj_mask[v] |= (1 << u)

def _local_search(n, time_limit=10):
    """Try to find a valid graph on n vertices using local search."""
    start_time = time.time()
    best_adj = None
    
    for attempt in range(1000):
        if time.time() - start_time > time_limit:
            break
            
        adj_mask = _random_adj_mask(n, 0.5)
        
        # Local search phase
        for step in range(500):
            if time.time() - start_time > time_limit:
                break
                
            viol = _find_violation(n, adj_mask)
            if viol is None:
                return adj_mask
            
            # Pick a random edge in the violating subset to flip
            u, v = random.sample(list(viol), 2)
            _flip_edge(adj_mask, u, v)
            
    return None

def _try_extend(base_adj, new_vertex_idx, time_limit=20):
    """
    Try to extend a valid graph on 'n' vertices to 'n+1' vertices.
    base_adj: adjacency masks for vertices 0..n-1
    new_vertex_idx: index of the new vertex (n)
    Returns adj_mask for n+1 vertices if successful, else None.
    """
    n = len(base_adj)
    start_time = time.time()
    
    # Pre-compute data for base graph subsets of size 5 to n-1
    # We need to check subsets S of V_base such that S U {new} is regular.
    # |S| >= 5.
    
    # Optimization: 
    # For a subset S, let degs_S[u] be degree of u in G[S].
    # In G[S U {new}], deg(u) = degs_S[u] + (1 if u connected to new else 0).
    # deg(new) = number of neighbors of new in S.
    # We need all deg(u) for u in S to be equal to deg(new).
    
    # Let's use Simulated Annealing on the connections of the new vertex.
    # State: mask_new (n bits).
    
    def count_violations(mask_new):
        """Count regular induced subgraphs involving the new vertex."""
        count = 0
        vertices = list(range(n))
        # Subsets S of V_base with size k-1, where k >= 6 => |S| >= 5.
        for k in range(5, n): # |S| = k, total size k+1
            for S in itertools.combinations(vertices, k):
                mask_S = 0
                for v in S:
                    mask_S |= (1 << v)
                
                # Degree of new vertex in S U {new}
                deg_new = bin(mask_new & mask_S).count('1')
                
                # Degrees of vertices in S
                is_regular = True
                for v in S:
                    # Degree in S
                    d_S = bin(base_adj[v] & mask_S).count('1')
                    # Degree in S U {new}
                    conn_to_new = 1 if (mask_new & (1 << v)) else 0
                    d_total = d_S + conn_to_new
                    
                    if d_total != deg_new:
                        is_regular = False
                        break
                
                if is_regular:
                    count += 1
        return count

    # SA Initialization
    mask_new = 0
    for i in range(n):
        if random.random() < 0.5:
            mask_new |= (1 << i)
            
    current_energy = count_violations(mask_new)
    best_mask = mask_new
    best_energy = current_energy
    
    T = 10.0
    T_min = 0.01
    cooling = 0.99
    
    while T > T_min and (time.time() - start_time) < time_limit:
        # Move: flip a random bit
        i = random.randint(0, n - 1)
        new_mask = mask_new ^ (1 << i)
        new_energy = count_violations(new_mask)
        
        delta = new_energy - current_energy
        
        if delta < 0 or random.random() < pow(2.718, -delta / T):
            mask_new = new_mask
            current_energy = new_energy
            
            if current_energy < best_energy:
                best_energy = current_energy
                best_mask = mask_new
                
                if best_energy == 0:
                    break
        T *= cooling
        
    if best_energy == 0:
        # Construct full adj mask
        new_adj = base_adj + [0]
        for i in range(n):
            if best_mask & (1 << i):
                new_adj[i] |= (1 << n)
                new_adj[n] |= (1 << i)
        return new_adj
    return None

def construct():
    start_time = time.time()
    deadline = start_time + 115.0
    
    best_n = 0
    best_edges = []
    
    # Step 1: Find a valid graph for n=20
    # Use a faster local search or multiple attempts
    adj_20 = None
    for attempt in range(5):
        if time.time() > deadline - 30:
            break
        adj_20 = _local_search(20, time_limit=15)
        if adj_20 is not None:
            break
            
    if adj_20 is not None and _check_valid(20, adj_20):
        best_n = 20
        edges = []
        for i in range(20):
            for j in range(i + 1, 20):
                if adj_20[i] & (1 << j):
                    edges.append([i, j])
        best_edges = edges
        
        # Step 2: Try to extend to n=21
        if time.time() < deadline - 20:
            adj_21 = _try_extend(adj_20, 20, time_limit=60)
            if adj_21 is not None:
                if _check_valid(21, adj_21):
                    best_n = 21
                    edges = []
                    for i in range(21):
                        for j in range(i + 1, 21):
                            if adj_21[i] & (1 << j):
                                edges.append([i, j])
                    best_edges = edges
                    # If 21 found, try 22?
                    if time.time() < deadline - 10:
                        adj_22 = _try_extend(adj_21, 21, time_limit=30)
                        if adj_22 is not None and _check_valid(22, adj_22):
                            best_n = 22
                            edges = []
                            for i in range(22):
                                for j in range(i + 1, 22):
                                    if adj_22[i] & (1 << j):
                                        edges.append([i, j])
                            best_edges = edges

    # Fallback if 20 failed
    if best_n == 0:
        for target_n in [18, 19, 20]:
            if time.time() > deadline:
                break
            adj = _local_search(target_n, time_limit=10)
            if adj is not None and _check_valid(target_n, adj):
                best_n = target_n
                edges = []
                for i in range(target_n):
                    for j in range(i + 1, target_n):
                        if adj[i] & (1 << j):
                            edges.append([i, j])
                best_edges = edges
                break

    return {'n': best_n, 'edges': best_edges}
