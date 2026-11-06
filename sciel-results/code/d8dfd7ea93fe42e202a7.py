import time
import random
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 115.0
    
    # We aim for n=20. If that fails, try 19, etc.
    for n in [20, 19, 18]:
        if time.time() > deadline:
            break
            
        # Try to find a valid graph on n vertices
        # Using local search / repair
        max_attempts = 100
        for attempt in range(max_attempts):
            if time.time() > deadline - 10: # Reserve time for final verification
                break
                
            # Initialize random graph
            adj = [0] * n
            for i in range(n):
                for j in range(i + 1, n):
                    if random.random() < 0.5:
                        adj[i] |= 1 << j
                        adj[j] |= 1 << i
            
            # Local search to remove regular induced subgraphs
            # We iterate and try to fix violations
            iterations = 0
            max_iterations = 500
            found_valid = False
            
            while iterations < max_iterations:
                if time.time() > deadline - 5:
                    break
                iterations += 1
                
                # Find a regular induced subgraph
                # Strategy: Check random subsets of size 6, 7, 8 first for speed.
                # If none found, do a full scan of size 6, 7.
                # If still none, assume valid (high probability for random graphs if no small ones).
                # But we must be rigorous for the final answer.
                
                violation = None
                found_violation = False
                
                # 1. Random sampling of subsets
                # Sample 1000 random subsets of sizes 6, 7, 8
                for _ in range(1000):
                    size = random.choice([6, 7, 8])
                    if size > n:
                        continue
                    subset = random.sample(range(n), size)
                    if is_regular_subset(subset, adj):
                        violation = subset
                        found_violation = True
                        break
                
                # 2. If not found in random sampling, do a more exhaustive check for size 6 and 7
                if not found_violation:
                    # Check all size 6
                    for comb in combinations(range(n), 6):
                        if is_regular_subset(comb, adj):
                            violation = list(comb)
                            found_violation = True
                            break
                    if found_violation:
                        break
                    
                    # Check all size 7
                    for comb in combinations(range(n), 7):
                        if is_regular_subset(comb, adj):
                            violation = list(comb)
                            found_violation = True
                            break
                    if found_violation:
                        break
                    
                    # If no size 6 or 7 regular subgraphs, check size 8, 9...
                    # For n=20, if no 6 or 7, it's likely valid.
                    # But to be safe, check up to n?
                    # Full check is ~1M subsets. Might be slow if done every iteration.
                    # Let's do full check only if we suspect validity.
                    
                    # Full check for sizes 8 to n
                    for size in range(8, n + 1):
                        if size > n:
                            break
                        for comb in combinations(range(n), size):
                            if is_regular_subset(comb, adj):
                                violation = list(comb)
                                found_violation = True
                                break
                        if found_violation:
                            break
                    
                    if not found_violation:
                        # Graph is valid!
                        found_valid = True
                        break
                
                if found_valid:
                    break
                
                if found_violation:
                    # Flip an edge in the violation to break it
                    # Pick random pair in violation
                    u, v = random.sample(violation, 2)
                    # Check if edge exists
                    if (adj[u] >> v) & 1:
                        adj[u] &= ~(1 << v)
                        adj[v] &= ~(1 << u)
                    else:
                        adj[u] |= 1 << v
                        adj[v] |= 1 << u
                else:
                    # No violation found in sampling or partial scan?
                    # Should have triggered found_valid in the full scan part.
                    # If we didn't do full scan, we might have missed one.
                    # Let's force a full scan check before declaring valid.
                    pass
            
            if found_valid:
                # Convert adj to edge list
                edges = []
                for i in range(n):
                    for j in range(i + 1, n):
                        if (adj[i] >> j) & 1:
                            edges.append([i, j])
                
                # Final verification just in case
                if verify_graph(adj, n):
                    return {'n': n, 'edges': edges}
    
    # Fallback
    return {'n': 5, 'edges': []}

def is_regular_subset(vertices, adj):
    """Check if induced subgraph on vertices is regular."""
    if len(vertices) < 6:
        return False
    v_set = set(vertices)
    mask = 0
    for v in vertices:
        mask |= 1 << v
    
    degrees = []
    for v in vertices:
        deg = (adj[v] & mask).bit_count()
        degrees.append(deg)
    return all(d == degrees[0] for d in degrees)

def verify_graph(adj, n):
    """Full verification of graph."""
    for size in range(6, n + 1):
        for comb in combinations(range(n), size):
            if is_regular_subset(comb, adj):
                return False
    return True
