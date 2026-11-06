import random
import time
import numpy as np
from itertools import combinations

def construct():
    start_time = time.time()
    time_limit = 110  # Leave 10s buffer
    
    def is_regular_subset(subset, adj):
        k = len(subset)
        if k < 6:
            return False
        # Extract subgraph
        sub_adj = adj[np.ix_(subset, subset)]
        # Calculate degrees
        degrees = np.sum(sub_adj, axis=1)
        # Check if all degrees are the same
        return np.all(degrees == degrees[0])

    def find_violations(adj, n, k):
        """Find all subsets of size k that induce a regular subgraph."""
        violations = []
        for subset in combinations(range(n), k):
            if is_regular_subset(subset, adj):
                violations.append(subset)
        return violations

    def is_valid_graph(adj, n):
        for k in range(6, n + 1):
            if find_violations(adj, n, k):
                return False
        return True

    # Try to find a valid graph for n=20, then 19, etc.
    # Using a repair-based local search approach.
    for n in [20, 19, 18, 17, 16, 15, 14, 13, 12, 11, 10, 9, 8, 7, 6]:
        if time.time() - start_time > time_limit:
            break
            
        found = False
        best_edges = None
        
        # Try multiple random seeds
        for seed in range(50):
            if time.time() - start_time > time_limit:
                break
            
            random.seed(seed)
            # Generate random graph
            adj = np.zeros((n, n), dtype=bool)
            for i in range(n):
                for j in range(i + 1, n):
                    if random.random() < 0.5:
                        adj[i][j] = True
                        adj[j][i] = True
            
            # Repair loop
            # We focus on eliminating regular induced subgraphs of size 6 first.
            # If we can eliminate all size 6, we then check larger sizes.
            # Flipping an edge in a violating size-6 subset breaks its regularity.
            # It might create new violations, but we iterate.
            
            max_iterations = 1000
            for _ in range(max_iterations):
                if time.time() - start_time > time_limit:
                    break
                
                # Check for violations of size 6
                viol_6 = find_violations(adj, n, 6)
                
                if not viol_6:
                    # Check for larger violations
                    valid = True
                    for k in range(7, n + 1):
                        viol_k = find_violations(adj, n, k)
                        if viol_k:
                            valid = False
                            # If we find a violation of size k>6, we can try to fix it
                            # by flipping an edge in that subset.
                            # But usually, if size 6 are gone, larger ones are rarer?
                            # Not necessarily.
                            # Let's pick a violation from the smallest k found.
                            viol_subset = viol_k[0]
                            break
                    if valid:
                        # Graph is valid
                        found = True
                        edges = []
                        for i in range(n):
                            for j in range(i + 1, n):
                                if adj[i][j]:
                                    edges.append([i, j])
                        best_edges = edges
                        break
                    else:
                        # Flip an edge in viol_subset
                        u, v = random.sample(list(viol_subset), 2)
                        adj[u][v] = not adj[u][v]
                        adj[v][u] = adj[u][v]
                else:
                    # Pick a violating subset of size 6
                    viol_subset = random.choice(viol_6)
                    # Flip an edge in it
                    u, v = random.sample(list(viol_subset), 2)
                    adj[u][v] = not adj[u][v]
                    adj[v][u] = adj[u][v]
                
                if found:
                    break
            
            if found:
                break
        
        if found:
            return {'n': n, 'edges': best_edges}
            
    # Fallback: n=6, K6 minus an edge
    n = 6
    edges = []
    for i in range(n):
        for j in range(i + 1, n):
            if not (i == 0 and j == 1):
                edges.append([i, j])
    return {'n': n, 'edges': edges}
