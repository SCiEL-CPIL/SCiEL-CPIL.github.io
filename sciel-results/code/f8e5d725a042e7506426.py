import time
import random

def construct():
    start_time = time.time()
    # The evaluator kills after 150s. We use a 135s deadline to be safe.
    deadline = start_time + 130.0

    # Fallback: Andrásfai graph And(9) on 26 vertices
    # It is triangle-free and has independence number 9.
    n_seed = 26
    edges_seed = [[i, j] for i in range(n_seed) for j in range(i + 1, n_seed) if (j - i) % n_seed % 3 == 1]
    best_result = {'n': n_seed, 'edges': edges_seed}

    def edges_from_adj(adj, n):
        edges = []
        for i in range(n):
            mask = adj[i]
            while mask:
                lsb = mask & -mask
                j = lsb.bit_length() - 1
                if i < j:
                    edges.append([i, j])
                mask ^= lsb
        return edges

    def has_independent_set_10(adj, n):
        """
        Check if there exists an independent set of size 10.
        Uses branch and bound with memoization on (mask, size) for speedup.
        """
        all_mask = (1 << n) - 1
        non_neighbors = [0] * n
        for i in range(n):
            non_neighbors[i] = (all_mask ^ (1 << i)) & ~adj[i]
        
        # Precompute bit counts for masks up to some size for speed
        # Use a dictionary for memoization
        memo = {}
        
        def dfs(mask, size):
            if size == 10:
                return True
            if mask == 0:
                return False
            remaining = 10 - size
            if mask.bit_count() < remaining:
                return False
            
            key = (mask, size)
            if key in memo:
                return memo[key]
            
            # Find the vertex in mask with the fewest non-neighbors in mask
            min_count = n + 1
            best_v = -1
            temp_mask = mask
            while temp_mask:
                lsb = temp_mask & -temp_mask
                v = lsb.bit_length() - 1
                cnt = (non_neighbors[v] & mask).bit_count()
                if cnt < min_count:
                    min_count = cnt
                    best_v = v
                temp_mask ^= lsb
                
            if best_v == -1:
                memo[key] = False
                return False
                
            v_bit = 1 << best_v
            new_mask_include = non_neighbors[best_v] & mask
            if dfs(new_mask_include, size + 1):
                memo[key] = True
                return True
            new_mask_exclude = mask & ~v_bit
            if dfs(new_mask_exclude, size):
                memo[key] = True
                return True
            memo[key] = False
            return False
            
        return dfs(all_mask, 0)

    def try_cayley(n, max_attempts):
        """
        Attempts to construct a valid triangle-free Cayley graph on Z_n.
        Uses a more efficient sum-free check.
        """
        for _ in range(max_attempts):
            if time.time() > deadline:
                return None
            
            S = set()
            candidates = list(range(1, n // 2 + 1))
            random.shuffle(candidates)
            
            for x in candidates:
                if x == n - x:
                    pair = {x}
                else:
                    pair = {x, n - x}
                
                valid = True
                # Check internal consistency of pair
                for p in pair:
                    for q in pair:
                        if (p + q) % n in pair:
                            valid = False
                            break
                    if not valid:
                        break
                
                if valid:
                    # Check interaction with existing S
                    for s in S:
                        for p in pair:
                            sum_val = (s + p) % n
                            if sum_val in S or sum_val in pair:
                                valid = False
                                break
                        if not valid:
                            break
                
                if valid:
                    S.update(pair)
            
            if not S:
                continue
                
            # Build adjacency masks
            adj = [0] * n
            for u in range(n):
                for v in S:
                    neighbor = (u + v) % n
                    adj[u] |= (1 << neighbor)
            
            if not has_independent_set_10(adj, n):
                return adj
        return None

    def try_random_greedy(n, max_attempts):
        """
        Random greedy triangle-free graph construction.
        Improved: use a priority queue-like approach where we process vertices
        in a random order and connect to non-conflicting vertices.
        """
        for _ in range(max_attempts):
            if time.time() > deadline:
                return None
            
            adj = [0] * n
            # Random permutation of vertices
            order = list(range(n))
            random.shuffle(order)
            
            # For each vertex in order, connect to all previous vertices it can connect to
            for i, u in enumerate(order):
                for j in range(i):
                    v = order[j]
                    common = adj[u] & adj[v]
                    if common == 0:
                        adj[u] |= (1 << v)
                        adj[v] |= (1 << u)
            
            if not has_independent_set_10(adj, n):
                return adj
        return None

    def try_refine(adj, n):
        """
        Try to refine an existing graph by removing edges that are not needed
        or by restructuring. For now, just try to find if we can add more
        structure. This is a placeholder for future refinement strategies.
        """
        pass

    # Main search loop
    # We try to find a graph with n vertices, starting from the maximum possible (40)
    # down to 27.
    for n in range(40, 26, -1):
        if time.time() > deadline:
            break
        
        # Strategy 1: Cayley Graphs
        # Allocate time proportional to remaining budget
        remaining = deadline - time.time()
        cayley_budget = max(20, int(remaining * 0.3))
        adj = try_cayley(n, cayley_budget)
        if adj is not None:
            edges = edges_from_adj(adj, n)
            return {'n': n, 'edges': edges}
            
        # Strategy 2: Random Greedy
        remaining = deadline - time.time()
        greedy_budget = max(20, int(remaining * 0.5))
        adj = try_random_greedy(n, greedy_budget)
        if adj is not None:
            edges = edges_from_adj(adj, n)
            return {'n': n, 'edges': edges}
            
    return best_result
