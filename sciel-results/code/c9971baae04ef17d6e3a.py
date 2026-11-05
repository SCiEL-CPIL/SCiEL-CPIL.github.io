"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import random

def construct():
    n = 40
    # Precompute all possible edges
    all_edges = [(u, v) for u in range(n) for v in range(u + 1, n)]
    edge_set = set()
    
    def is_c6_free(A, u, v):
        """Check if adding edge (u, v) to graph with adjacency matrix A creates a C6."""
        # We need to check if there is a path of length 5 between u and v in A.
        # If A[u,v] is 0, we are checking if A^5[u,v] > 0.
        # Note: A does NOT include the edge (u,v) yet.
        
        # Calculate A^5
        # A^2 = A @ A
        # A^4 = A^2 @ A^2
        # A^5 = A^4 @ A
        
        A2 = A @ A
        A4 = A2 @ A2
        A5 = A4 @ A
        
        return A5[u, v] == 0

    # 1. Initialize a random valid graph
    # Start with empty graph, add edges greedily in random order
    random.seed(42)
    edges = []
    A = np.zeros((n, n), dtype=np.int32)
    
    # Greedy initialization
    candidates = list(all_edges)
    random.shuffle(candidates)
    for u, v in candidates:
        if A[u, v] == 0:
            if is_c6_free(A, u, v):
                A[u, v] = 1
                A[v, u] = 1
                edges.append((u, v))
                edge_set.add((u, v))
    
    best_edges = edges.copy()
    best_count = len(edges)
    
    # 2. Local Search: Edge Swapping
    # We try to swap an existing edge with a non-edge to increase density or just move around.
    # Since we want to maximize edges, we are already at a local optimum for "adding" edges 
    # (greedy might not be global, but we can't add more without removing).
    # So we must remove one and add another? No, that keeps count same.
    # To increase count, we need to find a set of edges that allows adding more.
    # Actually, the greedy phase might get stuck. 
    # A better strategy: 
    # 1. Greedy construction.
    # 2. Try to add more edges? No, greedy adds all possible.
    # 3. So we must perform swaps to reach a different configuration that allows MORE edges to be added later?
    # Or, we can use a "randomized greedy" with restarts.
    
    # Let's combine: Randomized Greedy with multiple restarts + Local Search for improvement.
    # Actually, the current seed does 50 restarts of greedy. 
    # The new approach: 
    # 1. Randomized Greedy to get a starting point.
    # 2. Local Search: Try to swap edges. If we can swap such that we can subsequently add an edge, we do it.
    #    This is complex.
    
    # Alternative New Approach: **Simulated Annealing on Edge Subsets** is hard.
    # Let's stick to **Multiple Randomized Greedy Restarts** but with a **Smarter Check** or **Better Ordering**.
    # The seed uses random order. 
    # What if we use **Degree-based Ordering**? 
    # Or **Confidence-based**?
    
    # Let's try a different family: **Genetic Algorithm**? Too slow.
    
    # Let's refine the **Local Search** idea.
    # Current state: C6-free graph.
    # Move: Remove edge e1, Add edge e2.
    # If the new graph is C6-free, accept.
    # This explores the space of C6-free graphs with the same number of edges.
    # It doesn't directly increase edges.
    # HOWEVER, if we get stuck in a local optimum where no single edge can be added, 
    # a swap might move us to a region where an edge CAN be added.
    # So the strategy is:
    # 1. Greedy to max.
    # 2. While no edge can be added:
    #    a. Try random swaps.
    #    b. If a swap is valid, perform it.
    #    c. Check if any edge can now be added.
    #    d. If yes, add it (increase count), repeat.
    #    e. If no, continue swapping.
    
    # This is a valid new approach: **Escaping Local Optima via Swaps**.
    
    for restart in range(20):
        random.seed(restart * 100 + 42)
        np.random.seed(restart * 100 + 42)
        
        A = np.zeros((n, n), dtype=np.int32)
        edges = []
        edge_set = set()
        
        # Phase 1: Greedy Construction
        candidates = list(all_edges)
        random.shuffle(candidates)
        for u, v in candidates:
            if A[u, v] == 0:
                if is_c6_free(A, u, v):
                    A[u, v] = 1
                    A[v, u] = 1
                    edges.append((u, v))
                    edge_set.add((u, v))
        
        current_count = len(edges)
        
        # Phase 2: Escape Local Optima
        # Try to add more edges. If none can be added, do swaps.
        # Limit time/iterations per restart to ensure we finish in 120s.
        # 20 restarts * (Greedy + Swaps).
        
        swaps_per_iter = 100
        max_iters = 500 # 20 * 500 * 100 = 1M swap checks. 
        # Each check is ~3 matrix mults. 1M * 3 * 40^3 ops? 
        # 40^3 = 64000. 3 * 64000 = 192000 ops.
        # 1M * 192000 = 1.9e11 ops. Too slow for Python/NumPy if not optimized?
        # NumPy is C-optimized. 40x40 mult is very fast, likely <10us.
        # 1M * 10us = 10s. This is feasible.
        
        for _ in range(max_iters):
            # Check if we can add an edge
            added = False
            # Randomly sample non-edges to try adding
            # To be efficient, maintain a list of non-edges? 
            # Or just sample random pairs.
            
            # Try to add a random edge
            for _ in range(50): # Try 50 random additions
                u = random.randint(0, n-1)
                v = random.randint(0, n-1)
                if u == v: continue
                if A[u, v] == 1: continue
                if is_c6_free(A, u, v):
                    A[u, v] = 1
                    A[v, u] = 1
                    edges.append((u, v))
                    edge_set.add((u, v))
                    current_count += 1
                    added = True
                    break
            
            if added:
                if current_count > best_count:
                    best_count = current_count
                    best_edges = edges.copy()
                # Reset loop to try more additions
                continue
            
            # If no edge could be added, perform a swap
            # Remove a random edge
            if not edges:
                break
            idx = random.randint(0, len(edges)-1)
            e_remove = edges[idx]
            u_r, v_r = e_remove
            A[u_r, v_r] = 0
            A[v_r, u_r] = 0
            edges.pop(idx)
            edge_set.remove(e_remove)
            current_count -= 1
            
            # Try to add a random edge (different from removed)
            swap_success = False
            for _ in range(20):
                u = random.randint(0, n-1)
                v = random.randint(0, n-1)
                if u == v: continue
                if A[u, v] == 1: continue
                if (u, v) == e_remove or (v, u) == e_remove: continue
                
                if is_c6_free(A, u, v):
                    A[u, v] = 1
                    A[v, u] = 1
                    edges.append((u, v))
                    edge_set.add((u, v))
                    current_count += 1
                    swap_success = True
                    break
            
            if not swap_success:
                # Revert swap
                A[u_r, v_r] = 1
                A[v_r, u_r] = 1
                edges.append(e_remove)
                edge_set.add(e_remove)
                current_count += 1
    
    return best_edges
# EVOLVE-BLOCK-END
