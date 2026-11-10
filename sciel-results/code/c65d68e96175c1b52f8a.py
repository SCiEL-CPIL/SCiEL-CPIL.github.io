import time
import random
import numpy as np
from itertools import combinations

# EVOLVE-BLOCK-START

def construct():
    """
    Constructs a graph with no regular induced subgraph on 6 or more vertices.
    Strategy: Simulated Annealing with fast incremental verification for n=20.
    1. Start with a random graph on n=20.
    2. Use a cost function based on the number of violating subsets (size >= 6).
    3. Use simulated annealing to minimize the cost.
    4. Periodically verify full validity.
    5. If n=20 fails, fall back to n=19, etc.
    """
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin for 120s limit
    
    def is_regular_subset(n, adj, subset):
        """Check if the induced subgraph on subset is regular."""
        mask = 0
        for v in subset:
            mask |= (1 << v)
        
        degrees = []
        for v in subset:
            deg = (adj[v] & mask).bit_count()
            degrees.append(deg)
        
        return len(set(degrees)) == 1

    def count_violations(n, adj, sample_size=500):
        """
        Count violations using random sampling.
        Returns the number of violating subsets found in the sample.
        """
        count = 0
        for _ in range(sample_size):
            # Pick a random subset size >= 6
            k = random.randint(6, n)
            subset = tuple(random.sample(range(n), k))
            if is_regular_subset(n, adj, subset):
                count += 1
        return count

    def full_verify(n, adj):
        """
        Full verification: check all subsets of size >= 6.
        Returns True if no regular induced subgraph on 6+ vertices exists.
        """
        for k in range(6, n + 1):
            for subset in combinations(range(n), k):
                if is_regular_subset(n, adj, subset):
                    return False
        return True

    def solve_for_n(n):
        """
        Use simulated annealing to find a valid graph on n vertices.
        """
        # Initialize random graph
        adj = [0] * n
        for i in range(n):
            for j in range(i + 1, n):
                if random.random() < 0.5:
                    adj[i] |= (1 << j)
                    adj[j] |= (1 << i)
        
        # Current cost (sampled)
        current_cost = count_violations(n, adj)
        
        # Simulated annealing parameters
        T = 1.0  # Initial temperature
        alpha = 0.995  # Cooling rate
        max_steps = 100000
        
        for step in range(max_steps):
            if time.time() > deadline:
                break
            
            # Pick a random edge to flip
            u = random.randint(0, n - 1)
            v = random.randint(0, n - 1)
            if u == v:
                continue
            
            # Flip the edge
            if adj[u] & (1 << v):
                adj[u] ^= (1 << v)
                adj[v] ^= (1 << u)
            else:
                adj[u] |= (1 << v)
                adj[v] |= (1 << u)
            
            # New cost
            new_cost = count_violations(n, adj)
            delta = new_cost - current_cost
            
            # Accept with probability exp(-delta / T) if delta > 0
            if delta <= 0 or random.random() < np.exp(-delta / T):
                current_cost = new_cost
            else:
                # Revert
                if adj[u] & (1 << v):
                    adj[u] ^= (1 << v)
                    adj[v] ^= (1 << u)
                else:
                    adj[u] |= (1 << v)
                    adj[v] |= (1 << u)
            
            # Cool down
            T *= alpha
            
            # Periodic full check if cost is low
            if current_cost == 0 and step % 200 == 0:
                if full_verify(n, adj):
                    # Convert to edge list
                    edges = []
                    for i in range(n):
                        for j in range(i + 1, n):
                            if adj[i] & (1 << j):
                                edges.append([i, j])
                    return {'n': n, 'edges': edges}
        
        # Final check
        if current_cost == 0:
            if full_verify(n, adj):
                edges = []
                for i in range(n):
                    for j in range(i + 1, n):
                        if adj[i] & (1 << j):
                            edges.append([i, j])
                return {'n': n, 'edges': edges}
        
        return None

    # Try n=20, then 19, etc.
    for n in range(20, 5, -1):
        result = solve_for_n(n)
        if result is not None:
            return result
    
    # Fallback: n=5 is always valid
    return {'n': 5, 'edges': []}

# EVOLVE-BLOCK-END
