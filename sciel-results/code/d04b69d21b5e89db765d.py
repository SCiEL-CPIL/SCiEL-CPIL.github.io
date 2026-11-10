import time
import random
import numpy as np
from itertools import combinations

# EVOLVE-BLOCK-START

def construct():
    """
    Constructs a graph with no regular induced subgraph on 6 or more vertices.
    Strategy: Optimized Local Search for n=20.
    1. Start with a random graph on n=20.
    2. Verify validity by checking all subsets of size >= 6 for regularity.
    3. If a violation is found, identify the offending subset and flip a random 
       edge within that subset to break the regularity.
    4. Repeat until valid or time limit reached.
    5. If n=20 fails, fall back to n=19, etc.
    
    Verification is optimized using bitsets (integers) and iterating only over 
    combinations of size >= 6, rather than all 2^n masks.
    """
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin for 120s limit

    def verify_and_find_violation(n, adj):
        """
        Checks if the graph has no regular induced subgraph on k >= 6 vertices.
        Returns (is_valid, violation_subset) where violation_subset is a tuple of vertices 
        forming a regular induced subgraph, or None if valid.
        """
        # We check sizes from 6 up to n
        for k in range(6, n + 1):
            # Iterate over all subsets of size k
            for subset in combinations(range(n), k):
                mask = 0
                for v in subset:
                    mask |= (1 << v)
                
                # Check if induced subgraph on this subset is regular
                # Degree of v in subset is popcount(adj[v] & mask)
                degrees = []
                for v in subset:
                    deg = (adj[v] & mask).bit_count()
                    degrees.append(deg)
                
                # Check if all degrees are equal
                if len(set(degrees)) == 1:
                    return False, subset
        
        return True, None

    def local_search(n, max_steps=5000):
        """
        Attempts to find a valid graph on n vertices using local search.
        """
        # Initialize random graph
        adj = [0] * n
        for i in range(n):
            for j in range(i + 1, n):
                if random.random() < 0.5:
                    adj[i] |= (1 << j)
                    adj[j] |= (1 << i)
        
        for step in range(max_steps):
            if time.time() > deadline:
                return None
            
            is_valid, violation = verify_and_find_violation(n, adj)
            if is_valid:
                # Convert adj to edge list
                edges = []
                for i in range(n):
                    for j in range(i + 1, n):
                        if adj[i] & (1 << j):
                            edges.append([i, j])
                return edges
            
            # If violation found, try to fix it
            # Flip a random edge within the violation subset
            if violation:
                # Pick a random pair in the subset
                u, v = random.sample(violation, 2)
                # Flip the edge
                if adj[u] & (1 << v):
                    adj[u] ^= (1 << v)
                    adj[v] ^= (1 << u)
                else:
                    adj[u] |= (1 << v)
                    adj[v] |= (1 << u)
            else:
                # Should not happen if is_valid is False
                break
                
        return None

    # Target n=20 first
    for n in range(20, 4, -1):
        # Try local search
        # We can run multiple independent local searches if time permits
        while time.time() < deadline:
            edges = local_search(n, max_steps=100)
            if edges is not None:
                return {'n': n, 'edges': edges}
            
            # If we've done enough local searches for this n, try smaller n
            # Limit total time spent on this n
            if time.time() > deadline - 10:
                break
            # Add a small delay or just loop? 
            # To avoid infinite loop if local search is stuck, we break after some attempts
            # The inner loop in local_search has max_steps. 
            # Let's add a counter for restarts.
            # For simplicity, we just break the while loop after a few restarts if not found
            # Actually, local_search does 100 steps. Let's do a few restarts.
            # We'll structure it as: for restart in range(10): ...
            
        # If we exited the while loop without finding, try smaller n
        # But we should try a few restarts for each n.
        
    # Fallback: Return a small valid graph if all else fails
    # n=5 is always valid (no subset of size 6)
    return {'n': 5, 'edges': []}

# EVOLVE-BLOCK-END
