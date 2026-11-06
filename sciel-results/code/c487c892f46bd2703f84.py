import time
import random
import numpy as np

def count_k5_and_ind5(adj):
    """
    Count K5 cliques and independent sets of size 5.
    adj is a numpy boolean matrix (n x n).
    """
    n = adj.shape[0]
    if n < 5:
        return 0, 0
    
    k5_count = 0
    ind5_count = 0
    
    # Precompute combinations of 5 vertices? 
    # For n=40, C(40,5) = 658008. 
    # Iterating through all is ~0.1s in C, but Python loop is slow.
    # We can use a smarter way or just accept the cost if we limit iterations.
    
    # Actually, let's use a simpler metric: 
    # Sum of degrees in induced subgraphs? No.
    # Let's just iterate over 5-sets but efficiently.
    
    # To make it faster, we can check if the 5-vertices form a clique or independent set.
    # A 5-set is a clique if all 10 pairs are edges.
    # A 5-set is independent if all 10 pairs are non-edges.
    
    # We can precompute the number of edges in each 5-set?
    # Let's just do a quick check.
    
    # For performance, we will sample or use a heuristic.
    # However, for correctness, we must count exactly.
    
    # Let's use itertools.combinations
    from itertools import combinations
    for combo in combinations(range(n), 5):
        # Check if clique
        is_clique = True
        is_ind = True
        for i in range(5):
            for j in range(i+1, 5):
                u, v = combo[i], combo[j]
                if adj[u, v]:
                    is_ind = False
                else:
                    is_clique = False
                if not is_clique and not is_ind:
                    break
            if not is_clique and not is_ind:
                break
        
        if is_clique:
            k5_count += 1
        if is_ind:
            ind5_count += 1
            
    return k5_count, ind5_count

def construct():
    start_time = time.time()
    deadline = start_time + 290.0  # Leave 10s margin
    
    # Seed: Paley 37
    q = 37
    residues = {(x * x) % q for x in range(1, q)}
    seed_edges = [(i, j) for i in range(q) for j in range(i + 1, q) if (j - i) % q in residues]
    
    best_n = q
    best_edges = seed_edges
    
    # Try to find larger graphs
    # We will try n = 38, 39, 40, 41, 42, 43
    for target_n in range(q + 1, 44):
        if time.time() > deadline - 30:
            break
            
        # Initialize adjacency matrix
        # Start with a random graph or extend Paley
        # Extending Paley: keep 37 vertices as is, add (target_n - 37) new vertices connected randomly
        adj = np.zeros((target_n, target_n), dtype=bool)
        
        # Copy Paley edges
        for i, j in seed_edges:
            adj[i, j] = True
            adj[j, i] = True
            
        # Add random edges for new vertices
        for i in range(q, target_n):
            for j in range(i + 1, target_n):
                if random.random() < 0.5:
                    adj[i, j] = True
                    adj[j, i] = True
                    
        # Also randomize some existing edges? No, keep structure.
        # Actually, pure random restarts might be better for larger n.
        # Let's try a few random restarts for each n.
        
        found_valid = False
        restarts = 10
        for r in range(restarts):
            if time.time() > deadline - 5:
                break
                
            # Reset adj for this restart
            adj = np.zeros((target_n, target_n), dtype=bool)
            # Random graph
            np.fill_diagonal(adj, False)
            mask = (np.random.rand(target_n, target_n) < 0.5)
            adj = (mask | mask.T) & ~np.eye(target_n, dtype=bool)
            
            # Simulated Annealing / Hill Climbing
            # Objective: Minimize (K5 + Ind5)
            current_k5, current_ind5 = count_k5_and_ind5(adj)
            current_score = current_k5 + current_ind5
            
            if current_score == 0:
                found_valid = True
                break
            
            # Local search steps
            # We will try flipping edges that are part of violations.
            # To save time, we will do a limited number of steps.
            steps = 500
            for s in range(steps):
                if time.time() > deadline - 5:
                    break
                    
                if current_score == 0:
                    found_valid = True
                    break
                    
                # Pick a random edge to flip
                i, j = np.random.randint(0, target_n, 2)
                if i == j:
                    continue
                
                # Flip
                old_val = adj[i, j]
                adj[i, j] = not old_val
                adj[j, i] = not old_val
                
                new_k5, new_ind5 = count_k5_and_ind5(adj)
                new_score = new_k5 + new_ind5
                
                # Accept if better, or with some probability (SA)
                # For simplicity, accept if better (Hill Climbing)
                # To escape local minima, we might need SA, but Hill Climbing is faster.
                if new_score <= current_score:
                    current_score = new_score
                    current_k5, current_ind5 = new_k5, new_ind5
                else:
                    # Revert
                    adj[i, j] = old_val
                    adj[j, i] = old_val
                    # With small probability, accept worse (SA)
                    if random.random() < 0.05:
                        adj[i, j] = not old_val
                        adj[j, i] = not old_val
                        current_score = new_score
                        current_k5, current_ind5 = new_k5, new_ind5
            
            if found_valid:
                break
        
        if found_valid:
            best_n = target_n
            # Extract edges
            edges = []
            for i in range(target_n):
                for j in range(i + 1, target_n):
                    if adj[i, j]:
                        edges.append((i, j))
            best_edges = edges
            # If we found a valid graph, we can stop or try for larger?
            # The task says "maximized n". So we should try to go higher.
            # But if we found 38, we should try 39.
            # The loop continues.
            
    return best_n, best_edges
