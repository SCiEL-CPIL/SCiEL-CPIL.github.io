import time
import random
import itertools
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 320  # Safety margin (evaluator kills at 330s)

    # 1. Initialize with Paley graph on 37 vertices
    q = 37
    residues = {(x * x) % q for x in range(1, q)}
    n = q
    # adj[i] is a bitmask of neighbors of i (Python int)
    adj = [0] * n
    for i in range(n):
        for j in range(i + 1, n):
            if (j - i) % q in residues:
                adj[i] |= (1 << j)
                adj[j] |= (1 << i)

    best_n = n
    best_adj = adj[:]

    # Precompute all 4-subsets of range(n) as (a,b,c,d) tuples
    # For n up to ~45, C(45,4) ≈ 148995, manageable
    def get_4subsets(n):
        return list(itertools.combinations(range(n), 4))

    # Fast validity check using numpy
    def check_validity(new_mask, adj, n):
        """Check if adding a vertex with neighbor mask new_mask keeps graph (5,5)-free.
        Returns True if valid."""
        # S: neighbors of new vertex (need no K4)
        # T: non-neighbors of new vertex (need no independent set of size 4)
        
        # Build S and T as lists
        S = []
        T = []
        for i in range(n):
            if (new_mask >> i) & 1:
                S.append(i)
            else:
                T.append(i)
        
        # Check S for K4
        if len(S) >= 4:
            for combo in itertools.combinations(S, 4):
                a, b, c, d = combo
                # Check all 6 edges
                if ((adj[a] >> b) & 1) and ((adj[a] >> c) & 1) and ((adj[a] >> d) & 1) and \
                   ((adj[b] >> c) & 1) and ((adj[b] >> d) & 1) and ((adj[c] >> d) & 1):
                    return False
        
        # Check T for independent set of size 4
        if len(T) >= 4:
            for combo in itertools.combinations(T, 4):
                a, b, c, d = combo
                # Check NO edges exist
                if not ((adj[a] >> b) & 1) and not ((adj[a] >> c) & 1) and not ((adj[a] >> d) & 1) and \
                   not ((adj[b] >> c) & 1) and not ((adj[b] >> d) & 1) and not ((adj[c] >> d) & 1):
                    return False
        
        return True

    def add_vertex(adj, new_mask):
        n = len(adj)
        new_adj = adj[:]
        new_adj.append(0)
        for i in range(n):
            if (new_mask >> i) & 1:
                new_adj[i] |= (1 << n)
                new_adj[n] |= (1 << i)
        return new_adj

    # Random restart search
    best_overall_n = best_n
    best_overall_adj = best_adj[:]
    
    num_restarts = 20
    
    for restart in range(num_restarts):
        if time.time() > deadline:
            break
        
        # Reset to best known
        current_adj = best_overall_adj[:]
        current_n = best_overall_n
        
        # If we're starting fresh from the Paley graph, use a different random seed
        random.seed(42 + restart * 7)
        
        max_add_attempts = 500
        add_success_count = 0
        
        while current_n < 46 and add_success_count < max_add_attempts:
            if time.time() > deadline:
                break
            
            # Try to find a valid connection pattern for a new vertex
            found = False
            
            # Strategy 1: Biased random sampling
            # Bias towards ~50% density, but also try some structured patterns
            max_attempts = 2000
            attempts = 0
            
            while attempts < max_attempts and not found:
                if time.time() > deadline:
                    break
                
                # Generate random mask with ~50% density
                mask = 0
                for i in range(current_n):
                    if random.random() < 0.5:
                        mask |= (1 << i)
                
                if check_validity(mask, current_adj, current_n):
                    current_adj = add_vertex(current_adj, mask)
                    current_n += 1
                    found = True
                    if current_n > best_overall_n:
                        best_overall_n = current_n
                        best_overall_adj = current_adj[:]
                    add_success_count += 1
                    break
                
                attempts += 1
            
            # Strategy 2: Systematic search over masks (only for small n, say n <= 20)
            # This is exponential, so only use when n is small
            if not found and current_n <= 20:
                # Try all masks (2^20 = 1M, might be slow but worth trying for small n)
                # Actually, let's be more selective: try masks with specific densities
                for density in [0.3, 0.4, 0.5, 0.6, 0.7]:
                    if time.time() > deadline:
                        break
                    # Generate masks with approximately this density
                    for trial in range(500):
                        if time.time() > deadline:
                            break
                        mask = 0
                        for i in range(current_n):
                            if random.random() < density:
                                mask |= (1 << i)
                        
                        if check_validity(mask, current_adj, current_n):
                            current_adj = add_vertex(current_adj, mask)
                            current_n += 1
                            found = True
                            if current_n > best_overall_n:
                                best_overall_n = current_n
                                best_overall_adj = current_adj[:]
                            add_success_count += 1
                            break
                    if found:
                        break
            
            if not found:
                # Backtrack: remove the last added vertex and try a different connection
                if current_n > best_overall_n:
                    # We've gone beyond our best, so backtracking is risky
                    # Just break and try a new restart
                    break
                elif len(current_adj) > q:
                    # Remove the last vertex
                    current_adj = current_adj[:-1]
                    current_n -= 1
                else:
                    # Can't go below the Paley graph, break
                    break

    # Reconstruct edges from best graph
    best_edges = []
    for i in range(best_overall_n):
        for j in range(i + 1, best_overall_n):
            if (best_overall_adj[i] >> j) & 1:
                best_edges.append((i, j))
    
    return best_overall_n, best_edges
