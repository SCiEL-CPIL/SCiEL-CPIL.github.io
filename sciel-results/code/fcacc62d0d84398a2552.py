"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random

def _is_triangle_free_circulant(n, D):
    """Check if circulant graph C(n, D) is triangle-free.
    Triangle-free iff no three elements d1,d2,d3 in D (with repetitions allowed)
    sum to 0 mod n, i.e., no d1+d2+d3 ≡ 0 (mod n) for d1,d2,d3 in D.
    """
    Dset = set(D)
    for d1 in D:
        for d2 in D:
            for d3 in D:
                if (d1 + d2 + d3) % n == 0:
                    return False
    return True

def _max_independent_set_size(n, adj_mask):
    """Find max independent set size using branch and bound with bitmasks."""
    full_mask = (1 << n) - 1
    best = [0]
    
    # Precompute popcount for fast counting
    # For n <= 40, we can use bin(x).count('1')
    
    def popcount(x):
        return bin(x).count('1')
    
    def dfs(candidates, current_size):
        if current_size + popcount(candidates) <= best[0]:
            return
        if candidates == 0:
            best[0] = max(best[0], current_size)
            return
        if current_size >= best[0]:
            # Update best
            if current_size > best[0]:
                best[0] = current_size
            
        # Pick vertex with highest degree in candidate subgraph for better pruning
        # Try a few heuristics: pick vertex with max degree in induced subgraph
        v_list = []
        c = candidates
        while c:
            lb = c & -c
            v = lb.bit_length() - 1
            v_list.append(v)
            c ^= lb
        
        # Sort by degree in candidate subgraph (descending)
        # Degree of v in candidates = popcount(adj_mask[v] & candidates)
        v_list.sort(key=lambda v: popcount(adj_mask[v] & candidates), reverse=True)
        
        for v in v_list:
            if not (candidates & (1 << v)):
                continue
            # Branch 1: Include v
            new_candidates = candidates & ~adj_mask[v] & ~(1 << v)
            dfs(new_candidates, current_size + 1)
            
            # Branch 2: Exclude v
            new_candidates = candidates & ~(1 << v)
            dfs(new_candidates, current_size)
            
            break  # Only branch on first vertex to avoid exponential blowup
    
    dfs(full_mask, 0)
    return best[0]

def _check_no_independent_10(n, adj_mask):
    """Check if alpha < 10 using efficient branch and bound."""
    full_mask = (1 << n) - 1
    
    # Quick upper bound: if n < 10, alpha < 10 trivially
    if n < 10:
        return True
    
    # Use branch and bound to find if independent set of size 10 exists
    found = [False]
    
    def popcount(x):
        return bin(x).count('1')
    
    def dfs(candidates, current_size):
        if found[0]:
            return
        if current_size >= 10:
            found[0] = True
            return
        if current_size + popcount(candidates) < 10:
            return
        if candidates == 0:
            return
        
        # Pick vertex with highest degree in candidate subgraph
        v_list = []
        c = candidates
        while c:
            lb = c & -c
            v = lb.bit_length() - 1
            v_list.append(v)
            c ^= lb
        
        v_list.sort(key=lambda v: popcount(adj_mask[v] & candidates), reverse=True)
        v = v_list[0]
        
        # Branch 1: Include v
        new_candidates = candidates & ~adj_mask[v] & ~(1 << v)
        dfs(new_candidates, current_size + 1)
        
        # Branch 2: Exclude v
        if not found[0]:
            new_candidates = candidates & ~(1 << v)
            dfs(new_candidates, current_size)
    
    dfs(full_mask, 0)
    return not found[0]

def _build_edges_circulant(n, D):
    """Build edge list for circulant graph C(n, D)."""
    edges = []
    Dset = set(D)
    for i in range(n):
        for d in D:
            j = (i + d) % n
            if i < j:
                edges.append([i, j])
    return edges

def _try_circulant(n, D, time_limit):
    """Try a circulant graph with difference set D.
    Returns (n, edges) if valid, else (0, []).
    """
    start = time.time()
    
    # Check triangle-free
    if not _is_triangle_free_circulant(n, D):
        return 0, []
    
    # Build adjacency masks
    adj_mask = [0] * n
    Dset = set(D)
    for i in range(n):
        mask = 0
        for d in D:
            j = (i + d) % n
            mask |= (1 << j)
            j2 = (i - d) % n
            mask |= (1 << j2)
        adj_mask[i] = mask
    
    # Check no independent set of size 10
    if time.time() > time_limit:
        return 0, []
    
    if _check_no_independent_10(n, adj_mask):
        edges = _build_edges_circulant(n, D)
        return n, edges
    
    return 0, []

def _construct_maximal_triangle_free(n, seed):
    """Construct a maximal triangle-free graph on n vertices using random edge ordering."""
    rng = random.Random(seed)
    adj = [set() for _ in range(n)]
    
    # Generate all possible edges
    edges = []
    for i in range(n):
        for j in range(i + 1, n):
            edges.append((i, j))
    
    rng.shuffle(edges)
    
    for u, v in edges:
        if adj[u].isdisjoint(adj[v]):
            adj[u].add(v)
            adj[v].add(u)
    
    edge_list = []
    for i in range(n):
        for j in adj[i]:
            if i < j:
                edge_list.append([i, j])
    return edge_list

def _check_alpha_less_than(n, adj, k=10):
    """Check if graph has no independent set of size k using bitmask BFS."""
    adj_mask = [0] * n
    for i in range(n):
        mask = 0
        for j in adj[i]:
            mask |= (1 << j)
        adj_mask[i] = mask
    
    return _check_no_independent_10(n, adj_mask)

def construct():
    start_time = time.time()
    deadline = start_time + 130  # Safety margin for 150s limit
    
    best_n = 0
    best_edges = []
    
    # Strategy 1: Try algebraic circulant constructions for large n
    # These are structured and can be verified quickly
    
    # Try circulant graphs for n from 40 down to 20
    # A circulant C(n, D) is triangle-free iff no 3 elements of D sum to 0 mod n
    # We want alpha < 10
    
    # Known good constructions:
    # For n=39: try various difference sets
    # The key insight: for circulant graphs, we can enumerate small D sets
    
    # First, try some known promising circulant structures
    # A(3, n) type constructions
    
    candidate_sets = []
    
    # For each n from 40 down, try to find a valid D
    for n in range(40, 15, -1):
        if time.time() > deadline:
            break
        
        # Try difference sets of various sizes
        # The graph should be relatively dense to keep alpha small
        # But sparse enough to be triangle-free
        
        # Try D as sets of consecutive integers or symmetric sets
        # For triangle-free: no d1+d2+d3 ≡ 0 mod n
        
        # Try small symmetric sets
        tried_D = set()
        
        # Generate candidate D sets
        # For triangle-free circulant, D should not contain elements that form triangles
        # Simple: D = {d, -d} for various d, but this gives a cycle/disjoint cycles
        # More complex: D with 3+ elements
        
        # Try D of size 2: {a, b} where a+b != 0 and a+a+a != 0 etc.
        # Actually for |D|=2, D={a,b}, triangle-free means no x+y+z=0
        # With only a,b: a+a+a=3a!=0, a+a+b=2a+b!=0, a+b+b=a+2b!=0, b+b+b=3b!=0
        
        # For larger n, we need denser graphs to reduce alpha
        # Try D with 3-5 elements
        
        max_D_size = min(6, n // 3)
        
        for d_size in range(2, max_D_size + 1):
            if time.time() > deadline:
                break
            
            # Try random D sets
            for attempt in range(20):
                if time.time() > deadline:
                    break
                
                # Generate random D
                D = set()
                while len(D) < d_size:
                    d = random.randint(1, n - 1)
                    D.add(d)
                
                D = sorted(D)
                D_tuple = tuple(D)
                if D_tuple in tried_D:
                    continue
                tried_D.add(D_tuple)
                
                # Quick triangle-free check
                if not _is_triangle_free_circulant(n, D):
                    continue
                
                # Check alpha
                adj_mask = [0] * n
                for i in range(n):
                    mask = 0
                    for d in D:
                        j = (i + d) % n
                        mask |= (1 << j)
                        j2 = (i - d) % n
                        mask |= (1 << j2)
                    adj_mask[i] = mask
                
                if _check_no_independent_10(n, adj_mask):
                    edges = _build_edges_circulant(n, D)
                    best_n = n
                    best_edges = edges
                    return {'n': best_n, 'edges': best_edges}
    
    # Strategy 2: Random maximal triangle-free graphs
    for n in range(40, 5, -1):
        if time.time() > deadline:
            break
        
        for seed in range(30):
            if time.time() > deadline:
                break
            
            edges = _construct_maximal_triangle_free(n, seed)
            
            adj = [set() for _ in range(n)]
            for u, v in edges:
                adj[u].add(v)
                adj[v].add(u)
            
            if _check_alpha_less_than(n, adj, 10):
                best_n = n
                best_edges = edges
                break
        
        if best_n == n:
            break
    
    # Fallback
    if best_n == 0:
        best_n = 5
        best_edges = [[0,1], [1,2], [2,3], [3,4], [4,0]]
    
    return {'n': best_n, 'edges': best_edges}
# EVOLVE-BLOCK-END
