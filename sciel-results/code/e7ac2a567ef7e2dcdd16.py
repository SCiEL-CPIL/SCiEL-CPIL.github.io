import random
import time

def construct():
    # We aim for the maximum n possible (43)
    # If we fail, we step down.
    
    def check_c4_free(adj_bits, n):
        # adj_bits[i] is an integer bitmask of neighbors of i
        for u in range(n):
            for v in range(u + 1, n):
                common = adj_bits[u] & adj_bits[v]
                # Check if common has more than 1 bit set
                if common & (common - 1):
                    return False
        return True

    def check_alpha_le(adj_bits, n, max_alpha):
        # Returns True if alpha <= max_alpha
        # i.e., No independent set of size max_alpha + 1
        # We search for an independent set of size max_alpha + 1
        target = max_alpha + 1
        
        # Order vertices by degree (ascending) for better pruning?
        # Or just use index order.
        # For speed, we use a simple recursive DFS with pruning.
        
        # Pre-compute neighbor masks
        # adj_bits is already neighbor masks.
        
        def dfs(start, current_set, current_mask, remaining_needed):
            if remaining_needed == 0:
                return True
            if start >= n:
                return False
            
            # Pruning: If remaining vertices < remaining_needed, fail
            if n - start < remaining_needed:
                return False
            
            # Try to include 'start'
            # Check if 'start' is independent from current_set
            # current_mask has bits for vertices in current_set
            if not (adj_bits[start] & current_mask):
                new_mask = current_mask | (1 << start)
                # Optimization: If we include start, we can't include its neighbors
                # But for simplicity, we just recurse start+1.
                # A stronger pruning: 
                # Available vertices are those > start that are not neighbors of start and not in current_set.
                # If count of available < remaining_needed - 1, prune.
                
                if dfs(start + 1, current_set + [start], new_mask, remaining_needed - 1):
                    return True
            
            # Try to exclude 'start'
            if dfs(start + 1, current_set, current_mask, remaining_needed):
                return True
                
            return False

        return not dfs(0, [], 0, target)

    def build_graph(n, seed=0):
        random.seed(seed)
        adj_bits = [0] * n
        edges = []
        
        # Shuffle vertex order for greedy addition
        order = list(range(n))
        random.shuffle(order)
        
        # Greedy Add Edges
        for u in order:
            # Try to connect u to all v < u (in original index) or just all others?
            # To ensure simple graph, we define edges (u,v) with u<v
            for v in range(u):
                # Check if adding (u,v) creates C4
                # Common neighbors of u and v
                common = adj_bits[u] & adj_bits[v]
                if not (common & (common - 1)):
                    # Safe to add
                    adj_bits[u] |= (1 << v)
                    adj_bits[v] |= (1 << u)
                    edges.append([u, v])
                    
        return adj_bits, edges

    def verify_and_format(adj_bits, edges, n):
        if not check_c4_free(adj_bits, n):
            return None
        if not check_alpha_le(adj_bits, n, 10):
            return None
        return {'n': n, 'edges': edges}

    start_time = time.time()
    max_time = 115
    
    # Try n=43 down to 30
    for n in range(43, 29, -1):
        # Multiple restarts for each n
        for attempt in range(200):
            if time.time() - start_time > max_time:
                break
            
            adj_bits, edges = build_graph(n, seed=random.randint(0, 1000000))
            result = verify_and_format(adj_bits, edges, n)
            if result:
                return result
                
    # Fallback: Return a known small valid graph
    # n=30 seed graph from prompt
    edges = []
    for t in range(10):
        a, b, c = 3 * t, 3 * t + 1, 3 * t + 2
        edges += [[a, b], [a, c], [b, c]]
    return {'n': 30, 'edges': edges}
