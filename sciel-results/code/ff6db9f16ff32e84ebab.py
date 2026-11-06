"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import random
import time

# EVOLVE-BLOCK-START
def construct():
    """
    Constructs a triangle-free graph with no independent set of size 10.
    Maximizes n, with upper bound 40.
    """
    
    def has_independent_set(adj, n, limit=10):
        """
        Checks if there is an independent set of size 'limit' in graph with adjacency bitmasks.
        """
        # Precompute neighbor masks for speed in recursion
        # adj is already list of bitmasks
        
        def dfs(available_mask, current_size):
            if current_size >= limit:
                return True
            if available_mask == 0:
                return False
            # Pruning: if remaining vertices + current size < limit
            if current_size + bin(available_mask).count('1') < limit:
                return False
            
            # Heuristic: pick vertex with highest degree in subgraph
            best_v = -1
            best_deg = -1
            m = available_mask
            while m:
                lsb = m & (-m)
                v = lsb.bit_length() - 1
                deg = bin(adj[v] & available_mask).count('1')
                if deg > best_deg:
                    best_deg = deg
                    best_v = v
                m ^= lsb
            
            if best_v == -1:
                return False
                
            # Branch 1: Include best_v
            # Remove best_v and its neighbors
            new_mask = available_mask & ~(adj[best_v] | (1 << best_v))
            if dfs(new_mask, current_size + 1):
                return True
                
            # Branch 2: Exclude best_v
            new_mask = available_mask & ~(1 << best_v)
            if dfs(new_mask, current_size):
                return True
                
            return False

        return dfs((1 << n) - 1, 0, limit)

    def is_triangle_free(adj, n):
        """
        Checks if graph is triangle-free.
        """
        for i in range(n):
            # For each neighbor j of i with j > i
            m = adj[i] & ~((1 << (i + 1)) - 1) # neighbors > i
            while m:
                lsb = m & (-m)
                j = lsb.bit_length() - 1
                # Check if i and j have a common neighbor
                if adj[i] & adj[j]:
                    return False
                m ^= lsb
        return True

    def generate_random_regular(n, d):
        """
        Generates a random d-regular graph.
        Returns None if fails.
        """
        if n * d % 2 != 0:
            return None
            
        # Use configuration model with retries for self-loops/multi-edges
        for attempt in range(100):
            stubs = []
            for i in range(n):
                for _ in range(d):
                    stubs.append(i)
            random.shuffle(stubs)
            
            adj = [0] * n
            valid = True
            for i in range(0, n * d, 2):
                u = stubs[i]
                v = stubs[i+1]
                if u == v:
                    valid = False
                    break
                # Check for multi-edge
                if (adj[u] >> v) & 1:
                    valid = False
                    break
                adj[u] |= (1 << v)
                adj[v] |= (1 << u)
            
            if valid:
                # Verify degree
                for i in range(n):
                    if bin(adj[i]).count('1') != d:
                        valid = False
                        break
                if valid:
                    return adj
        return None

    def eliminate_triangles(adj, n, max_iters=5000):
        """
        Local search to remove triangles.
        Returns True if triangle-free.
        """
        for _ in range(max_iters):
            # Find a triangle
            tri_found = False
            # Randomize order of checking to avoid getting stuck?
            # Just iterate
            for i in range(n):
                m = adj[i] & ~((1 << (i + 1)) - 1)
                while m:
                    lsb_j = m & (-m)
                    j = lsb_j.bit_length() - 1
                    common = adj[i] & adj[j]
                    if common:
                        lsb_k = common & (-common)
                        k = lsb_k.bit_length() - 1
                        # Remove random edge from triangle (i,j,k)
                        u, v = random.choice([(i, j), (i, k), (j, k)])
                        adj[u] &= ~(1 << v)
                        adj[v] &= ~(1 << u)
                        tri_found = True
                        break
                    m ^= lsb_j
                if tri_found:
                    break
            if not tri_found:
                return True
        return False

    def try_n(n, num_seeds=50):
        """
        Try to find a valid graph on n vertices.
        """
        for seed in range(num_seeds):
            random.seed(seed * 1000 + n)
            # Try degrees 4 to 7
            for d in [4, 5, 6, 7]:
                adj = generate_random_regular(n, d)
                if adj is None:
                    continue
                
                if eliminate_triangles(adj, n):
                    if not has_independent_set(adj, n, 10):
                        edges = []
                        for i in range(n):
                            for j in range(i + 1, n):
                                if (adj[i] >> j) & 1:
                                    edges.append([i, j])
                        return {'n': n, 'edges': edges}
        return None

    # Try n from 40 down to 26
    # Limit time for 40 to avoid wasting too much time
    start_time = time.time()
    
    for n in range(40, 25, -1):
        # Allow less time for larger n if we are running out of time
        time_elapsed = time.time() - start_time
        if time_elapsed > 100:
            break
            
        result = try_n(n, num_seeds=30)
        if result:
            return result
            
    # Fallback: Andrasfai graph on 26 vertices
    n = 26
    adj = [0] * n
    for i in range(n):
        for j in range(i + 1, n):
            if (j - i) % n % 3 == 1:
                adj[i] |= (1 << j)
                adj[j] |= (1 << i)
                
    edges = []
    for i in range(n):
        for j in range(i + 1, n):
            if (adj[i] >> j) & 1:
                edges.append([i, j])
    return {'n': n, 'edges': edges}

# EVOLVE-BLOCK-END
