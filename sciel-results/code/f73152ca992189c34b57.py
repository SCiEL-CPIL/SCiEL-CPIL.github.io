import time
import random
import itertools

def construct():
    start_time = time.time()
    deadline = start_time + 115.0
    n = 18
    
    # Use bitmask for fast triangle-free checks
    # adj_mask[v] is a bitmask of neighbors of v
    
    def build_adj_mask(edge_set):
        adj = [0] * n
        for u, v in edge_set:
            adj[u] |= (1 << v)
            adj[v] |= (1 << u)
        return adj
    
    def is_tf_mask(mask, adj):
        """Check if vertex set represented by bitmask is triangle-free."""
        # For each pair of adjacent vertices in the set, check if they share a common neighbor in the set
        m = mask
        while m:
            u_bit = m & (-m)
            u = u_bit.bit_length() - 1
            m ^= u_bit
            # Neighbors of u within the set
            common = adj[u] & mask
            while common:
                v_bit = common & (-common)
                v = v_bit.bit_length() - 1
                common ^= v_bit
                if v < u:
                    continue
                # Check if u and v share a common neighbor w in the set
                # w must be > v to avoid double counting, but we just need any
                w_common = adj[u] & adj[v] & mask
                if w_common:
                    return False
        return True
    
    def max_tf_size(adj):
        """Find the maximum size of a triangle-free induced subgraph."""
        # Try sizes from 18 down
        # Use a branch-and-bound approach for efficiency
        # For n=18, checking all C(18,k) for k=12 is 185640 which is feasible with fast check
        
        # First try to find if any set of size 12 is TF
        # If yes, score >= 12
        # If no, try 11, etc.
        
        # Precompute all combinations for each size
        # Actually, let's do it smartly with pruning
        
        def has_tf_of_size(target_size, adj):
            """Check if there exists a triangle-free set of the given size."""
            all_verts = list(range(n))
            # Use recursive search with pruning
            # A set is TF if no three vertices form a triangle
            # Equivalently, the induced subgraph has no K3
            
            # Branch and bound: build set incrementally
            # At each step, check if partial set + remaining can form a TF set of target_size
            
            results = []
            
            def search(current_set, mask, start_idx, remaining):
                if len(current_set) == target_size:
                    if is_tf_mask(mask, adj):
                        results.append(True)
                        return
                    return
                if start_idx >= n:
                    return
                # Pruning: not enough vertices left
                if n - start_idx < remaining:
                    return
                
                for i in range(start_idx, n):
                    if results:
                        return
                    new_set = current_set + [i]
                    new_mask = mask | (1 << i)
                    search(new_set, new_mask, i + 1, remaining - 1)
                    if results:
                        return
            
            search([], 0, 0, target_size)
            return len(results) > 0
        
        # Check from high to low
        for size in range(18, 0, -1):
            if has_tf_of_size(size, adj):
                return size
        return 0
    
    def has_tf_of_size_fast(target_size, adj):
        """Faster check using itertools for small target sizes."""
        for subset in itertools.combinations(range(n), target_size):
            mask = 0
            for v in subset:
                mask |= (1 << v)
            if is_tf_mask(mask, adj):
                return True
        return False
    
    def max_tf_size_fast(adj):
        """Find max TF size, checking from 12 down (most likely range)."""
        for size in range(12, 0, -1):
            if has_tf_of_size_fast(size, adj):
                return size
        return 0
    
    def check_k4_free(adj, u, v):
        """Check if adding edge (u,v) would create a K4."""
        # Need w, z such that all 6 edges among {u,v,w,z} exist
        common_uv = adj[u] & adj[v] & ~(1 << u) & ~(1 << v)
        m = common_uv
        while m:
            w_bit = m & (-m)
            w = w_bit.bit_length() - 1
            m ^= w_bit
            # Need z adjacent to u, v, w
            common_uvw = adj[u] & adj[v] & adj[w] & ~(1 << u) & ~(1 << v) & ~(1 << w)
            if common_uvw:
                return False
        return True
    
    # Initial construction: start with Turan graph T_18,3 and modify
    # T_18,3 = K6,6,6: edges between different parts, none within
    parts = [list(range(0, 6)), list(range(6, 12)), list(range(12, 18))]
    edges = set()
    for i in range(3):
        for j in range(i + 1, 3):
            for a in parts[i]:
                for b in parts[j]:
                    edges.add((min(a, b), max(a, b)))
    
    adj = build_adj_mask(edges)
    current_score = max_tf_size_fast(adj)
    
    best_edges = set(edges)
    best_score = current_score
    best_adj = adj[:]
    
    # Local search with multiple restarts
    rng = random.Random(42)
    
    for restart in range(20):
        if time.time() > deadline - 5:
            break
        
        if restart == 0:
            # Start from current
            pass
        else:
            # Perturb: randomly add/remove some edges
            edges = set(best_edges)
            for _ in range(rng.randint(3, 8)):
                if edges and rng.random() < 0.5:
                    e = rng.choice(list(edges))
                    edges.discard(e)
                else:
                    u = rng.randint(0, n - 1)
                    v = rng.randint(0, n - 1)
                    if u == v:
                        continue
                    u, v = min(u, v), max(u, v)
                    if (u, v) not in edges:
                        adj_temp = build_adj_mask(edges)
                        if check_k4_free(adj_temp, u, v):
                            edges.add((u, v))
        
        adj = build_adj_mask(edges)
        current_score = max_tf_size_fast(adj)
        
        if current_score < best_score:
            best_score = current_score
            best_edges = set(edges)
            best_adj = adj[:]
        
        # Hill climbing
        no_improve = 0
        max_no_improve = 50
        
        while time.time() < deadline - 5:
            if no_improve >= max_no_improve:
                break
            
            # Try a random move: add an edge
            u = rng.randint(0, n - 1)
            v = rng.randint(0, n - 1)
            if u == v:
                no_improve += 1
                continue
            u, v = min(u, v), max(u, v)
            
            if (u, v) in edges:
                no_improve += 1
                continue
            
            # Check K4-free
            if not check_k4_free(adj, u, v):
                no_improve += 1
                continue
            
            # Try the move
            edges.add((u, v))
            adj[u] |= (1 << v)
            adj[v] |= (1 << u)
            
            new_score = max_tf_size_fast(adj)
            
            if new_score < current_score:
                current_score = new_score
                no_improve = 0
                if new_score < best_score:
                    best_score = new_score
                    best_edges = set(edges)
                    best_adj = adj[:]
            else:
                # Revert
                edges.discard((u, v))
                adj[u] &= ~(1 << v)
                adj[v] &= ~(1 << u)
                no_improve += 1
    
    return [[u, v] for u, v in sorted(best_edges)]
