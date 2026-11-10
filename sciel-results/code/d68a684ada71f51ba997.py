import time
import random
import numpy as np
from itertools import combinations

# EVOLVE-BLOCK-START

def construct():
    """
    Constructs a graph with no regular induced subgraph on 6 or more vertices.
    Strategy:
    1. Target n=20 (known to exist).
    2. Use local search with efficient bit-mask based violation detection.
    3. Primary focus: Eliminate regular induced subgraphs of size 6.
    4. Secondary focus: Verify and repair larger violations (size > 6) only when size-6 is clear.
    5. Deadline guard ensures we return the best valid graph found.
    """
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin for 150s limit
    
    def is_regular_subset(adj, subset, n):
        """
        Check if the induced subgraph on subset is regular.
        adj: list of int bitmasks.
        subset: list of vertex indices.
        """
        mask = 0
        for v in subset:
            mask |= (1 << v)
        
        degrees = []
        for v in subset:
            deg = (adj[v] & mask).bit_count()
            degrees.append(deg)
        
        return len(set(degrees)) == 1

    def find_violation_fast(adj, n):
        """
        Find a violating subset of size 6 using random sampling.
        Returns list of vertices or None.
        """
        for _ in range(50):
            subset = random.sample(range(n), 6)
            if is_regular_subset(adj, subset, n):
                return subset
        return None

    def find_violation_full(adj, n):
        """
        Find any violating subset of size >= 6.
        Checks all size 6, then all size 7, etc.
        Returns list of vertices or None.
        """
        # Check all subsets of size 6
        for subset in combinations(range(n), 6):
            if is_regular_subset(adj, list(subset), n):
                return list(subset)
        
        # Check all subsets of size 7 to n
        for k in range(7, n + 1):
            for subset in combinations(range(n), k):
                if is_regular_subset(adj, list(subset), n):
                    return list(subset)
        return None

    def local_search(adj, n, max_iterations=20000):
        """
        Local search to eliminate violations.
        Returns a valid adj if found, else None.
        """
        current_adj = adj[:]
        
        for i in range(max_iterations):
            if time.time() > deadline:
                break
            
            # 1. Try to find a size-6 violation quickly
            violation = find_violation_fast(current_adj, n)
            
            if violation is None:
                # No quick violation found. Do a full check.
                # This is expensive, so we do it only when we think we are close.
                # We can throttle this: do full check every 100 iterations if no quick violation.
                if i % 100 == 0:
                    violation = find_violation_full(current_adj, n)
                    if violation is None:
                        # Valid graph found
                        return current_adj
                    # Found a violation (size > 6)
            
            if violation is not None:
                # Repair: Flip a random edge within the violation
                u, v = random.sample(violation, 2)
                if current_adj[u] & (1 << v):
                    current_adj[u] ^= (1 << v)
                    current_adj[v] ^= (1 << u)
                else:
                    current_adj[u] |= (1 << v)
                    current_adj[v] |= (1 << u)
            else:
                # No violation found in quick check, and not time for full check
                # Random move to escape local minima
                u, v = random.sample(range(n), 2)
                if current_adj[u] & (1 << v):
                    current_adj[u] ^= (1 << v)
                    current_adj[v] ^= (1 << u)
                else:
                    current_adj[u] |= (1 << v)
                    current_adj[v] |= (1 << u)
        
        return None

    def solve_for_n(n):
        """
        Attempt to construct a valid graph on n vertices.
        """
        # Try multiple random restarts
        for restart in range(100):
            if time.time() > deadline:
                break
            
            # Initialize random graph
            adj = [0] * n
            for i in range(n):
                for j in range(i + 1, n):
                    if random.random() < 0.5:
                        adj[i] |= (1 << j)
                        adj[j] |= (1 << i)
            
            # Run local search
            result = local_search(adj, n, max_iterations=5000)
            
            if result is not None:
                # Final verification (just in case)
                if find_violation_full(result, n) is None:
                    return result
        
        return None

    # Try n=20 first, then fall back
    for n in range(20, 5, -1):
        if time.time() > deadline - 5:
            break
        
        result = solve_for_n(n)
        if result is not None:
            edges = []
            for i in range(n):
                for j in range(i + 1, n):
                    if result[i] & (1 << j):
                        edges.append([i, j])
            return {'n': n, 'edges': edges}
    
    # Fallback: n=5 is always valid (no subset of size >= 6)
    return {'n': 5, 'edges': []}

# EVOLVE-BLOCK-END
