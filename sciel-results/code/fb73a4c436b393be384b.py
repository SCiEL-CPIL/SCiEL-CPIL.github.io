import time
import random
import itertools
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 280.0
    
    # Helper to check if a graph is (5,5)-free
    # Graph is represented as an NxN numpy array of 0s and 1s
    def is_valid(adj):
        n = adj.shape[0]
        # Check for K5
        for combo in itertools.combinations(range(n), 5):
            # If all edges exist, it's a K5
            # adj[i][j] == 1 for all pairs
            if all(adj[i][j] == 1 for i, j in itertools.combinations(combo, 2)):
                return False
        # Check for Independent Set 5
        for combo in itertools.combinations(range(n), 5):
            # If no edges exist, it's an independent set
            # adj[i][j] == 0 for all pairs
            if all(adj[i][j] == 0 for i, j in itertools.combinations(combo, 2)):
                return False
        return True

    # Convert adj matrix to edge list
    def to_edges(adj):
        n = adj.shape[0]
        edges = []
        for i in range(n):
            for j in range(i+1, n):
                if adj[i][j] == 1:
                    edges.append((i, j))
        return edges

    # Fallback: Paley 37
    def paley_37():
        q = 37
        residues = {(x * x) % q for x in range(1, q)}
        return q, [(i, j) for i in range(q) for j in range(i + 1, q) if (j - i) % q in residues]

    # Try to find a graph of size N using local search
    def find_graph(N, time_limit):
        # Random initialization
        # Use a random graph with p=0.5
        best_adj = None
        best_violations = float('inf')
        
        # Number of restarts depends on time
        while time.time() < start_time + time_limit:
            adj = np.random.randint(0, 2, (N, N))
            np.fill_diagonal(adj, 0)
            # Symmetrize
            adj = np.maximum(adj, adj.T)
            
            # Local search: Simulated Annealing or simple hill climbing
            # Objective: Number of K5s + Number of I5s
            # This is expensive to calculate fully. 
            # Instead, we can just try to repair violations.
            
            # Let's use a simpler heuristic: 
            # 1. Count violations. If 0, return.
            # 2. If > 0, pick a random violating 5-set, flip a random edge in it.
            
            # To speed up violation detection, we can precompute degrees etc, 
            # but for N<=42, checking all combinations is C(42,5) ~ 850k.
            # Checking all pairs in a 5-set is 10 checks. 8.5M ops per full scan.
            # In Python, this is slow. 
            # We need a faster violation check or a smarter search.
            
            # Alternative: Use the fact that we only need *one* valid graph.
            # We can use a SAT-like approach or just random restarts with greedy repair.
            
            # Let's try a greedy repair on a random graph.
            adj = np.random.randint(0, 2, (N, N))
            np.fill_diagonal(adj, 0)
            adj = np.maximum(adj, adj.T)
            
            # Repair loop
            for _ in range(10000): # Max repair steps
                if time.time() > start_time + time_limit:
                    break
                
                # Find a violation
                # We can sample random 5-sets to find a violation faster than exhaustive
                # But if we miss it, we might think it's valid. 
                # For correctness, we MUST verify at the end.
                # So, we can do a quick check on a subset, and a full check at the end.
                
                # Let's do a full check every K steps to ensure we are converging.
                # But full check is expensive.
                
                # Let's try to find a K5 or I5 by sampling.
                # If we find one, repair it.
                # If we don't find one in 100 samples, do a full check.
                
                found_violation = False
                for _ in range(50):
                    combo = random.sample(range(N), 5)
                    # Check K5
                    is_k5 = True
                    is_i5 = True
                    for i, j in itertools.combinations(combo, 2):
                        if adj[i][j] == 1:
                            is_i5 = False
                        else:
                            is_k5 = False
                        if not is_k5 and not is_i5:
                            break
                    if is_k5:
                        # Found K5, flip a random edge in it
                        u, v = random.sample(combo, 2)
                        adj[u][v] = 0
                        adj[v][u] = 0
                        found_violation = True
                        break
                    elif is_i5:
                        # Found I5, flip a random edge in it
                        u, v = random.sample(combo, 2)
                        adj[u][v] = 1
                        adj[v][u] = 1
                        found_violation = True
                        break
                
                if not found_violation:
                    # No violation found in sampling, do a full check
                    if is_valid(adj):
                        return adj
                    else:
                        # There is a violation we missed, do a full search to find one
                        # This is a fallback, should be rare if sampling is good
                        # But for N=42, C(42,5) is large. 
                        # We might just restart.
                        break

        return None

    # Try sizes from 42 down to 38
    for N in range(42, 37, -1):
        if time.time() > deadline:
            break
        # Allocate more time for larger N? No, just try each for a bit.
        # Let's give 50s to N=42, 20s to N=41, etc.
        if N == 42:
            t_limit = 150.0
        elif N == 41:
            t_limit = 60.0
        else:
            t_limit = 20.0
            
        adj = find_graph(N, t_limit)
        if adj is not None:
            # Final verification
            if is_valid(adj):
                return N, to_edges(adj)
            
    # Fallback to Paley 37
    return paley_37()
