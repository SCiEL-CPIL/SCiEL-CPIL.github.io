import numpy as np
import time
import random
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 140  # Safety margin (evaluator kills at 150s)
    
    # The task is to find a graph on n vertices with NO regular induced subgraph on 6 vertices.
    # "Regular induced subgraph" means all degrees in the induced subgraph are equal.
    # For a subset S of size 6, the induced subgraph is regular if:
    # deg(v) in S is constant for all v in S.
    # Possible degrees: 0, 1, 2, 3, 4, 5.
    
    # Strategy:
    # 1. Start with a random graph on n=20.
    # 2. Define a "violation" score: count of subsets of size 6 that induce a regular subgraph.
    # 3. Use local search (Simulated Annealing) to minimize this score to 0.
    #    - Move: Flip a random edge.
    #    - Cost: Change in violation count.
    #    - Accept: If cost < 0, or with probability exp(-cost/T) if cost > 0.
    
    # Optimization for Violation Counting:
    # Checking all C(20,6)=38760 subsets every time is ~38k * 6*6 ops ~ 1.4M ops.
    # In Python/NumPy, this might be too slow for thousands of iterations.
    # We need a faster way to estimate or calculate violations.
    
    # Alternative: Precompute all subsets of size 6.
    # For n=20, there are 38,760 subsets.
    # We can represent the graph as an adjacency matrix A (20x20 bool).
    # For a subset S, the induced subgraph is A[S, S].
    # Degrees are sum(A[S, S], axis=1).
    # It is regular if min(deg) == max(deg).
    
    # To speed up, we can use a "delta" approach or just optimize the check.
    # Let's try to vectorize the check for ALL subsets at once?
    # That would require a tensor of shape (38760, 6, 6), which is 38760*36 ~ 1.4M booleans.
    # This is feasible in memory (1.4MB).
    # We can extract all induced subgraphs at once using advanced indexing.
    
    n_target = 20
    
    # Generate all subsets of size 6 from 0..19
    # This takes a second or two, but is done once.
    subsets = list(combinations(range(n_target), 6))
    subsets_arr = np.array(subsets) # Shape (38760, 6)
    
    # Initial random graph
    # It is often better to start with a graph that is "close" to valid.
    # Random p=0.5 is a good start, but maybe p=0.4 or 0.6 helps avoid dense/sparse regulars?
    # Let's stick to 0.5 for symmetry.
    
    def count_violations(adj):
        """
        Count the number of subsets of size 6 that induce a regular subgraph.
        adj: (n, n) boolean array.
        """
        # Extract all induced subgraphs
        # rows = subsets_arr (38760, 6)
        # We need adj[i, j] for all i, j in subset.
        # We can do: sub_graphs = adj[subsets_arr[:, :, None], subsets_arr[:, None, :]]
        # This creates an array of shape (38760, 6, 6)
        
        # This operation is memory intensive but fast in C (NumPy).
        # 38760 * 6 * 6 = 1,395,360 elements.
        
        try:
             # Advanced indexing
             # subsets_arr is (M, 6).
             # We want adj[u, v] for u, v in each subset.
             # Let's create indices for rows and columns.
             row_idx = subsets_arr[:, :, None] # (M, 6, 1)
             col_idx = subsets_arr[:, None, :] # (M, 1, 6)
             
             sub_graphs = adj[row_idx, col_idx] # (M, 6, 6)
             
             # Calculate degrees
             degrees = sub_graphs.sum(axis=2) # (M, 6)
             
             # Check if regular: min == max
             min_deg = degrees.min(axis=1)
             max_deg = degrees.max(axis=1)
             
             violations = np.sum(min_deg == max_deg)
             return int(violations)
        except Exception:
             # Fallback if memory fails (unlikely for 20 vertices)
             return 100000

    def flip_edge(adj, u, v):
        """Flip edge (u, v) in adj. Returns new adj (or modifies in place? In place is faster)."""
        adj[u, v] = not adj[u, v]
        adj[v, u] = not adj[v, u]

    best_n = 0
    best_edges = []
    
    # Try to find a valid graph for n=20 first.
    # If we fail, we can try n=19, etc., but n=20 is the target.
    
    # Simulated Annealing Parameters
    T_start = 10.0
    T_end = 0.1
    cooling_rate = 0.999
    max_iterations = 50000
    current_T = T_start
    
    # Initial random graph
    adj = np.random.rand(n_target, n_target) < 0.5
    np.fill_diagonal(adj, False)
    adj = (adj + adj.T).astype(bool)
    
    current_violations = count_violations(adj)
    best_violations = current_violations
    best_adj = adj.copy()
    
    # Precompute all possible edges to pick randomly
    edges_to_flip = []
    for i in range(n_target):
        for j in range(i+1, n_target):
            edges_to_flip.append((i, j))
            
    iter_count = 0
    while current_T > T_end and time.time() < deadline - 10:
        iter_count += 1
        
        # Pick a random edge to flip
        u, v = edges_to_flip[random.randint(0, len(edges_to_flip)-1)]
        
        # Flip
        flip_edge(adj, u, v)
        
        # Calculate new violations
        new_violations = count_violations(adj)
        delta = new_violations - current_violations
        
        if delta < 0 or random.random() < np.exp(-delta / current_T):
            # Accept
            current_violations = new_violations
            if current_violations < best_violations:
                best_violations = current_violations
                best_adj = adj.copy()
                
                # If we found a valid graph (0 violations), we can stop early!
                if best_violations == 0:
                    break
        else:
            # Reject: flip back
            flip_edge(adj, u, v)
            
        current_T *= cooling_rate
        
        # Time check inside loop
        if time.time() > deadline - 5:
            break

    # If we found a valid graph (0 violations)
    if best_violations == 0:
        e_list = []
        for i in range(n_target):
            for j in range(i+1, n_target):
                if best_adj[i, j]:
                    e_list.append([i, j])
        return {'n': n_target, 'edges': e_list}
    
    # If we didn't reach 0 violations for n=20, try n=19?
    # The task says "Graphs with 20 vertices are known".
    # If we can't find one in time, returning a valid 19-vertex graph is better than 5.
    # But 20 is the target. Let's try a few more restarts if we have time.
    
    # Restart logic
    remaining_time = deadline - time.time()
    if remaining_time > 10:
        for restart in range(5):
            if time.time() > deadline - 5:
                break
            
            # New random start
            adj = np.random.rand(n_target, n_target) < 0.5
            np.fill_diagonal(adj, False)
            adj = (adj + adj.T).astype(bool)
            
            current_violations = count_violations(adj)
            if current_violations < best_violations:
                best_violations = current_violations
                best_adj = adj.copy()
                
            if best_violations == 0:
                e_list = []
                for i in range(n_target):
                    for j in range(i+1, n_target):
                        if best_adj[i, j]:
                            e_list.append([i, j])
                return {'n': n_target, 'edges': e_list}
            
            # Short SA run
            current_T = 5.0
            local_iter = 0
            while current_T > 0.5 and time.time() < deadline - 5 and local_iter < 5000:
                u, v = edges_to_flip[random.randint(0, len(edges_to_flip)-1)]
                flip_edge(adj, u, v)
                new_violations = count_violations(adj)
                delta = new_violations - current_violations
                
                if delta < 0 or random.random() < np.exp(-delta / current_T):
                    current_violations = new_violations
                    if current_violations < best_violations:
                        best_violations = current_violations
                        best_adj = adj.copy()
                        if best_violations == 0:
                            break
                else:
                    flip_edge(adj, u, v)
                
                current_T *= 0.99
                local_iter += 1
                
                if best_violations == 0:
                    break
            
            if best_violations == 0:
                e_list = []
                for i in range(n_target):
                    for j in range(i+1, n_target):
                        if best_adj[i, j]:
                            e_list.append([i, j])
                return {'n': n_target, 'edges': e_list}

    # If we still have violations, we might try to reduce n?
    # No, the score is n. A valid graph on 20 is score 4 (20/5).
    # An invalid graph scores 0.
    # So we MUST return a valid graph.
    # If we can't find a valid 20-vertex graph, we should try to find a valid 19-vertex graph.
    # Or even 18, 17, 16...
    # The seed returns 5 (empty graph is valid? No, empty graph on 5 vertices has no 6-vertex subset, so it's vacuously valid? 
    # Wait. "No set of 6 or more vertices may induce a regular subgraph".
    # If n < 6, there are no sets of 6 vertices. So it is valid.
    # Score = n/5.
    # Seed returns 5. Score 1.0.
    
    # Let's try to find a valid graph on n=19 if 20 fails.
    # And n=18, etc.
    
    for target_n in [19, 18, 17, 16]:
        if time.time() > deadline - 5:
            break
            
        # Quick random search for smaller n (easier to satisfy)
        # For n=19, C(19,6) = 27132.
        subsets_19 = list(combinations(range(target_n), 6))
        subsets_arr_19 = np.array(subsets_19)
        
        def count_violations_n(adj, subsets_arr):
             row_idx = subsets_arr[:, :, None]
             col_idx = subsets_arr[:, None, :]
             sub_graphs = adj[row_idx, col_idx]
             degrees = sub_graphs.sum(axis=2)
             min_deg = degrees.min(axis=1)
             max_deg = degrees.max(axis=1)
             return int(np.sum(min_deg == max_deg))

        # Try a few random graphs
        for attempt in range(20):
            if time.time() > deadline - 2:
                break
            adj_t = np.random.rand(target_n, target_n) < 0.5
            np.fill_diagonal(adj_t, False)
            adj_t = (adj_t + adj_t.T).astype(bool)
            
            v = count_violations_n(adj_t, subsets_arr_19)
            if v == 0:
                e_list = []
                for i in range(target_n):
                    for j in range(i+1, target_n):
                        if adj_t[i, j]:
                            e_list.append([i, j])
                return {'n': target_n, 'edges': e_list}
                
            # If not 0, do a tiny local search?
            # Might be too slow. Let's just try more randoms.
            
    # Fallback: Return the best found (which might be invalid, scoring 0) or a known valid small graph.
    # We know n=5 is valid (vacuously).
    # Can we find a valid n=10, 11, 12 quickly?
    # Let's just return the seed result if we fail, to ensure score 1.0.
    # Actually, returning an invalid graph scores 0.
    # The seed program returns n=5, edges=[] which is valid.
    
    return {'n': 5, 'edges': []}
