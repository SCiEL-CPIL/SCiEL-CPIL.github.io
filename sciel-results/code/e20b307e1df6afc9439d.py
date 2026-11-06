import time
import random
import math
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin for 120s limit

    def get_edges(adj, n):
        edges = []
        for u in range(n):
            for v in range(u + 1, n):
                if (adj[u] >> v) & 1:
                    edges.append([u, v])
        return edges

    def has_k4(n, adj):
        """Check if graph contains a K4. Returns True if found."""
        for u in range(n):
            for v in range(u + 1, n):
                if (adj[u] >> v) & 1:
                    common = adj[u] & adj[v]
                    common &= ~(1 << u)
                    common &= ~(1 << v)
                    m = common
                    while m:
                        lsb = m & -m
                        a = lsb.bit_length() - 1
                        m ^= lsb
                        rest = common & ~(1 << a)
                        if adj[a] & rest:
                            return True
        return False

    def has_indep_6(n, adj):
        """Check if graph contains an independent set of size 6. Returns True if found."""
        all_mask = (1 << n) - 1
        
        def find_indep(remaining, size):
            if size == 0:
                return True
            if remaining.bit_count() < size:
                return False
            # Heuristic: pick vertex with fewest neighbors in remaining (max non-neighbors)
            # to prune the search space for independent sets.
            # Actually, for independent set, we want vertices that are NOT connected.
            # So we want to pick a vertex and remove its neighbors.
            # To find an independent set, we pick a vertex v, and then must pick from non-neighbors of v.
            # To prune, we should pick the vertex that leaves the largest set of non-neighbors?
            # No, if we pick v, the next candidates are non-neighbors of v in 'remaining'.
            # If we pick a vertex with many neighbors, the non-neighbor set is small, which might prune faster if we fail?
            # Or pick vertex with few neighbors, so non-neighbor set is large?
            # Standard branch and bound: pick vertex, branch on include/exclude.
            # Here we just need existence.
            
            # Simple recursion:
            m = remaining
            while m:
                lsb = m & -m
                u = lsb.bit_length() - 1
                m ^= lsb
                # Option 1: Include u. Then next candidates are non-neighbors of u in 'remaining' (excluding u).
                non_n = remaining & ~(adj[u] | (1 << u))
                if find_indep(non_n, size - 1):
                    return True
                # Option 2: Exclude u. (Implicitly handled by moving to next lsb in the loop? 
                # No, the loop iterates over vertices. If we don't pick u, we just try the next vertex in 'remaining'.
                # But 'remaining' is fixed for this level.
                # The loop tries each vertex u in 'remaining' as the first element of the independent set.
                # If including u doesn't lead to a solution, we try the next u.
            return False

        return find_indep(all_mask, 6)

    def exact_check(n, adj):
        if has_k4(n, adj):
            return False
        if has_indep_6(n, adj):
            return False
        return True

    def get_paley_17():
        """Fallback: Paley graph of order 17. Known to be K4-free and alpha=5."""
        n = 17
        adj = [0] * n
        squares = {x * x % 17 for x in range(1, 17)}
        for i in range(n):
            for j in range(i + 1, n):
                if (j - i) % 17 in squares:
                    adj[i] |= (1 << j)
                    adj[j] |= (1 << i)
        return adj

    def local_search(n, time_budget):
        """
        Local search to find a valid graph on n vertices.
        Starts with random graph, flips edges to reduce violations.
        """
        deadline_local = start_time + time_budget
        rng = random.Random(42)
        
        # Initialize random graph
        adj = [0] * n
        for u in range(n):
            for v in range(u + 1, n):
                if rng.random() < 0.3:
                    adj[u] |= (1 << v)
                    adj[v] |= (1 << u)
        
        # Check if already valid
        if exact_check(n, adj):
            return adj

        best_adj = adj[:]
        best_score = float('inf')
        
        # Simple score: number of K4s + number of I6s (approximate via sampling or just existence? 
        # For local search, we need a gradient. Counting is expensive.
        # Let's just do random flips and check validity. 
        # This is a "blind" local search.
        
        # To make it smarter, we can count K4s and I6s but limit the counting.
        # Given time constraints, blind search with validity check is safer.
        
        for _ in range(50000):
            if time.time() > deadline_local:
                break
            
            # Flip a random edge
            u = rng.randint(0, n - 1)
            v = rng.randint(0, n - 1)
            if u == v:
                continue
            if u > v:
                u, v = v, u
            
            adj[u] ^= (1 << v)
            adj[v] ^= (1 << u)
            
            if exact_check(n, adj):
                return adj
        
        return None

    # Try to find a valid graph on larger n first
    # We prioritize finding a valid graph over maximizing n if it risks timeout.
    # However, the task says "Maximize n".
    # We will try 35, then 30, then 25, then fallback to 17.
    
    targets = [35, 30, 25, 20, 17]
    
    for target_n in targets:
        if time.time() > deadline - 5:
            break
        
        # Allocate remaining time to this target
        time_budget = deadline - time.time() - 2
        if time_budget < 5:
            continue
            
        result = local_search(target_n, time_budget)
        
        if result is not None:
            return {'n': target_n, 'edges': get_edges(result, target_n)}
        
        # If local search failed, try a few random restarts if time permits?
        # The local_search function already does 50k iterations.
        # If it fails, it's likely difficult. Move to smaller n.

    # Fallback to Paley 17
    adj = get_paley_17()
    return {'n': 17, 'edges': get_edges(adj, 17)}
