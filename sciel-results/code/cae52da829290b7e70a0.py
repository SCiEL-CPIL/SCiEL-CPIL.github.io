import time
import random
import itertools
import sys

def construct():
    """
    Constructs a graph with no regular induced subgraph on 6+ vertices.
    Maximizes n.
    """
    start_time = time.time()
    deadline = start_time + 110  # Safety margin for 120s limit
    
    # We aim for n=21 first, then n=20
    for target_n in [21, 20]:
        if time.time() > deadline:
            break
        
        # Try multiple restarts
        found = False
        for restart in range(100):
            if time.time() > deadline:
                break
            
            # Initialize random graph
            n = target_n
            # adj[i] is a bitmask of neighbors of i
            adj = [0] * n
            for i in range(n):
                for j in range(i + 1, n):
                    if random.random() < 0.5:
                        adj[i] |= (1 << j)
                        adj[j] |= (1 << i)
            
            # Local Search
            # We want to minimize the number of regular induced subgraphs of size >= 6
            # Ideally 0.
            
            # Pre-compute all subsets of size 6 to 10? 
            # For n=21, subsets of size 6: 54264.
            # Subsets of size 7: 116280.
            # It's better to generate on the fly or pre-compute masks.
            
            # Let's pre-compute masks for size 6, as they are the most frequent violations.
            # Actually, checking all sizes is safer.
            
            # Helper: Check if graph is valid (no regular subgraph size >= 6)
            def is_valid(adj):
                # Check size 6
                for subset in itertools.combinations(range(n), 6):
                    mask = 0
                    for v in subset:
                        mask |= (1 << v)
                    # Check if regular
                    degs = []
                    for v in subset:
                        deg = bin(adj[v] & mask).count('1')
                        degs.append(deg)
                    if len(set(degs)) == 1:
                        return False, subset
                # Check size 7
                for subset in itertools.combinations(range(n), 7):
                    mask = 0
                    for v in subset:
                        mask |= (1 << v)
                    degs = []
                    for v in subset:
                        deg = bin(adj[v] & mask).count('1')
                        degs.append(deg)
                    if len(set(degs)) == 1:
                        return False, subset
                # Check size 8+ (less likely, but must check)
                for size in range(8, n + 1):
                    for subset in itertools.combinations(range(n), size):
                        mask = 0
                        for v in subset:
                            mask |= (1 << v)
                        degs = []
                        for v in subset:
                            deg = bin(adj[v] & mask).count('1')
                            degs.append(deg)
                        if len(set(degs)) == 1:
                            return False, subset
                return True, None

            # Initial check
            valid, viol_subset = is_valid(adj)
            if valid:
                # Extract edges
                edges = []
                for i in range(n):
                    for j in range(i + 1, n):
                        if adj[i] & (1 << j):
                            edges.append([i, j])
                return {'n': n, 'edges': edges}
            
            # Local Search Loop
            best_adj = adj[:]
            best_violations = 1 # Just a marker
            
            for step in range(2000):
                if time.time() > deadline:
                    break
                
                # Find a violation
                # To speed up, we can just check size 6 subsets first.
                # If we find one, we flip an edge.
                # If no size 6 violation, check size 7, etc.
                
                viol_subset = None
                # Check size 6
                for subset in itertools.combinations(range(n), 6):
                    mask = 0
                    for v in subset:
                        mask |= (1 << v)
                    degs = [bin(adj[v] & mask).count('1') for v in subset]
                    if len(set(degs)) == 1:
                        viol_subset = subset
                        break
                
                if viol_subset is None:
                    # Check size 7
                    for subset in itertools.combinations(range(n), 7):
                        mask = 0
                        for v in subset:
                            mask |= (1 << v)
                        degs = [bin(adj[v] & mask).count('1') for v in subset]
                        if len(set(degs)) == 1:
                            viol_subset = subset
                            break
                
                if viol_subset is None:
                    # Check larger sizes if necessary, but this is rare
                    # If we reach here, it might be valid or have a larger violation
                    # For efficiency, assume if no 6 or 7, it's likely valid or close.
                    # Let's do a full check occasionally.
                    if step % 10 == 0:
                        valid, _ = is_valid(adj)
                        if valid:
                            edges = []
                            for i in range(n):
                                for j in range(i + 1, n):
                                    if adj[i] & (1 << j):
                                        edges.append([i, j])
                            return {'n': n, 'edges': edges}
                    
                    # If no violation found in quick check, try flipping a random edge to escape local minima
                    # Or break and restart
                    break
                
                # Flip an edge related to the violation
                # Pick a random vertex from the violation and a random other vertex
                v1 = random.choice(viol_subset)
                v2 = random.choice([v for v in range(n) if v != v1])
                
                # Flip edge (v1, v2)
                if adj[v1] & (1 << v2):
                    adj[v1] &= ~(1 << v2)
                    adj[v2] &= ~(1 << v1)
                else:
                    adj[v1] |= (1 << v2)
                    adj[v2] |= (1 << v1)

    # Fallback: Return a known valid small graph or empty
    # A graph with n=5 is trivially valid (no subset of size 6)
    return {'n': 5, 'edges': []}
