"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import numpy as np
from itertools import combinations

# EVOLVE-BLOCK-START

def construct():
    """
    Construct a graph on as many vertices as possible with no K5 and no independent 5-set.
    Strategy: 
    1. Attempt to find a valid graph on n=42 vertices using local search (minimizing K5 + Ind5 count).
    2. If unsuccessful, try n=41, n=40, etc.
    3. Fallback to the Paley graph on 37 vertices.
    """
    start_time = time.time()
    deadline = start_time + 290.0  # 300s limit, keep 10s margin

    def get_paley(q):
        """Generate Paley graph on q vertices (q must be prime 1 mod 4)."""
        residues = {(x * x) % q for x in range(1, q)}
        edges = []
        for i in range(q):
            for j in range(i + 1, q):
                if (j - i) % q in residues:
                    edges.append((i, j))
        return q, edges

    def edges_to_adj(n, edges):
        adj = np.zeros((n, n), dtype=np.uint8)
        for u, v in edges:
            adj[u, v] = 1
            adj[v, u] = 1
        return adj

    def adj_to_edges(n, adj):
        edges = []
        for i in range(n):
            for j in range(i + 1, n):
                if adj[i, j]:
                    edges.append((i, j))
        return edges

    def count_bad_sets(n, adj):
        """
        Count the number of K5s and independent 5-sets in the graph.
        Returns (k5_count, ind5_count).
        """
        k5_count = 0
        ind5_count = 0
        
        # Precompute complement for Ind5
        # ind_adj[i][j] = 1 if NOT adj[i][j] (and i!=j)
        # We can just use (1 - adj) but need to handle diagonal.
        
        # Optimization: 
        # A K5 is a set of 5 vertices where all pairs are edges.
        # An Ind5 is a set of 5 vertices where all pairs are non-edges.
        
        # Naive O(n^5) is too slow if we do it many times?
        # n=42. C(42,5) = 850,668.
        # Inside loop: 10 checks.
        # 8.5M operations. In Python, this is ~0.5-1.0 seconds.
        # If we do 100 iterations, that's 50-100 seconds. This is tight but possible.
        # We need to be efficient.
        
        # Better: Iterate over 5-combinations.
        for combo in combinations(range(n), 5):
            i, j, k, l, m = combo
            # Check K5
            if (adj[i, j] and adj[i, k] and adj[i, l] and adj[i, m] and 
                adj[j, k] and adj[j, l] and adj[j, m] and 
                adj[k, l] and adj[k, m] and 
                adj[l, m]):
                k5_count += 1
            # Check Ind5
            # Non-edge if adj is 0
            if (not adj[i, j] and not adj[i, k] and not adj[i, l] and not adj[i, m] and 
                not adj[j, k] and not adj[j, l] and not adj[j, m] and 
                not adj[k, l] and not adj[k, m] and 
                not adj[l, m]):
                ind5_count += 1
                
        return k5_count, ind5_count

    def local_search(n, time_limit, initial_adj=None):
        """
        Perform local search to minimize bad sets.
        Returns edges if a valid graph is found, else None.
        """
        if initial_adj is None:
            # Random initialization
            adj = np.zeros((n, n), dtype=np.uint8)
            for i in range(n):
                for j in range(i + 1, n):
                    if random.random() < 0.5:
                        adj[i, j] = 1
                        adj[j, i] = 1
        else:
            adj = initial_adj.copy()
            
        k5_c, ind5_c = count_bad_sets(n, adj)
        current_cost = k5_c + ind5_c
        
        if current_cost == 0:
            return adj_to_edges(n, adj)
            
        best_cost = current_cost
        best_adj = adj.copy()
        
        # Simple hill climbing with random restarts if stuck
        # To avoid getting stuck in local minima, we can use a "plateau" strategy
        # or just random restarts.
        
        stuck_counter = 0
        max_stuck = 500 # If no improvement for 500 steps, restart?
        
        while time.time() < time_limit:
            # Pick random edge
            u = random.randint(0, n - 1)
            v = random.randint(0, n - 1)
            if u == v:
                continue
            if u > v:
                u, v = v, u
                
            # Flip edge
            old_val = adj[u, v]
            adj[u, v] = 1 - old_val
            adj[v, u] = adj[u, v]
            
            # Recalculate cost
            new_k5, new_ind5 = count_bad_sets(n, adj)
            new_cost = new_k5 + new_ind5
            
            if new_cost <= current_cost:
                current_cost = new_cost
                stuck_counter = 0
                if new_cost == 0:
                    return adj_to_edges(n, adj)
                if new_cost < best_cost:
                    best_cost = new_cost
                    best_adj = adj.copy()
            else:
                # Revert
                adj[u, v] = old_val
                adj[v, u] = old_val
                stuck_counter += 1
                
                # If stuck too long, random restart
                if stuck_counter > max_stuck:
                    # Reinitialize with random graph
                    adj = np.zeros((n, n), dtype=np.uint8)
                    for i in range(n):
                        for j in range(i + 1, n):
                            if random.random() < 0.5:
                                adj[i, j] = 1
                                adj[j, i] = 1
                    k5_c, ind5_c = count_bad_sets(n, adj)
                    current_cost = k5_c + ind5_c
                    stuck_counter = 0
                    if current_cost == 0:
                        return adj_to_edges(n, adj)
                    if current_cost < best_cost:
                        best_cost = current_cost
                        best_adj = adj.copy()

        # If we exited due to time, return the best found if it's valid, else None
        # Note: best_adj might not be valid if best_cost > 0
        if best_cost == 0:
            return adj_to_edges(n, best_adj)
        return None

    # Try n=42 first
    # We allocate 150s to n=42, 100s to n=41, 40s to n=40, rest to 37
    # Actually, if we find a valid 42, we return immediately.
    
    # Strategy: 
    # 1. Try n=42 for 180s.
    # 2. If fail, try n=41 for 90s.
    # 3. If fail, try n=40 for 15s.
    # 4. Fallback to Paley 37.
    
    # To improve success rate for n=42, we can try a few different initializations
    # e.g., Random, or Paley-based extension.
    
    # Let's try Random first.
    # We can also try a "biased" random where we start with a known good structure?
    # No, just random is more robust for search.
    
    # Optimization: If we don't find 42, we might find 41.
    
    # We will run local_search for n=42.
    # To make it more effective, we can run it in parallel? No.
    # We can run multiple short runs?
    
    # Let's do one long run for n=42.
    time_limit_42 = start_time + 180
    edges_42 = local_search(42, time_limit_42)
    if edges_42 is not None:
        return 42, edges_42
        
    # If 42 failed, try 41
    if time.time() < deadline - 10:
        time_limit_41 = min(deadline - 10, time.time() + 90)
        edges_41 = local_search(41, time_limit_41)
        if edges_41 is not None:
            return 41, edges_41
            
    # If 41 failed, try 40
    if time.time() < deadline - 5:
        time_limit_40 = min(deadline - 5, time.time() + 15)
        edges_40 = local_search(40, time_limit_40)
        if edges_40 is not None:
            return 40, edges_40

    # Fallback to Paley 37
    n, edges = get_paley(37)
    return n, edges

# EVOLVE-BLOCK-END
