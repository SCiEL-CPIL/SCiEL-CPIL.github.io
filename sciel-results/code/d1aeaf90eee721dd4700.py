"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def partition(adjacency):
    n = len(adjacency)
    if n == 0:
        return np.array([], dtype=np.int8)
    
    # Use a deadline guard: aim to finish well under 120s.
    start_time = time.time()
    deadline = start_time + 115.0  # 5s safety margin
    
    # Precompute adjacency as a boolean numpy array for fast operations
    A = np.asarray(adjacency, dtype=bool)
    degrees = A.sum(axis=1)
    
    # Helper to compute cut size efficiently
    def cut_size(labels):
        # Cut edges are those where labels differ
        # diff = labels[:, None] != labels[None, :]
        # return (A & diff).sum() // 2
        # Faster:
        # Sum of edges within groups is (A[0,0] + A[1,1]).sum() // 2 ? No.
        # Let's stick to the reliable one for N=100, it's fast enough.
        diff = labels[:, None] != labels[None, :]
        return (A & diff).sum() // 2
    
    # Helper to compute gain of flipping vertex v
    # Gain = (edges to same) - (edges to diff)
    # If gain > 0, flipping increases cut.
    def get_gain(labels, v):
        # edges_to_same = A[v, labels == labels[v]].sum()
        # edges_to_diff = A[v, labels != labels[v]].sum()
        # gain = edges_to_same - edges_to_diff
        # Note: sum(A[v]) = degree(v) = edges_to_same + edges_to_diff
        # So gain = 2 * edges_to_same - degree(v)
        edges_to_same = A[v, labels == labels[v]].sum()
        return 2 * edges_to_same - degrees[v]

    best_labels = None
    best_cut = -1
    
    # Simulated Annealing + ILS strategy
    # We run multiple iterations. Each iteration:
    # 1. Initialize (random or perturbed best)
    # 2. Run SA to convergence
    # 3. Update best
    # 4. Perturb best for next iteration
    
    num_iterations = 200 # Upper bound, limited by time
    
    # Initial random start
    current_labels = np.random.randint(0, 2, n)
    
    for it in range(num_iterations):
        if time.time() > deadline:
            break
            
        # --- Simulated Annealing Phase ---
        T = 1.0 # Initial temperature
        T_min = 0.01
        cooling_rate = 0.95
        
        labels = current_labels.copy()
        cur_cut = cut_size(labels)
        
        # Local search / SA loop
        # We will iterate until no improvements for a while or T < T_min
        stagnant_rounds = 0
        max_stagnant = 10
        
        while T > T_min:
            if time.time() > deadline:
                break
            
            improved = False
            # Random order for fairness
            order = np.random.permutation(n)
            
            for v in order:
                if time.time() > deadline:
                    break
                    
                gain = get_gain(labels, v)
                
                if gain > 0:
                    # Always accept
                    labels[v] = 1 - labels[v]
                    cur_cut += gain // 2 # Wait, gain is diff in sum, cut is sum/2?
                    # Let's verify cut update.
                    # Cut = 1/2 * sum_{u,v} A[u,v] (1 - x_u x_v) ? No, simpler:
                    # Cut = sum_{u in 0, v in 1} A[u,v].
                    # If we flip v, edges (u,v) change status.
                    # If u in same group as v, edge was internal, becomes crossing. Cut increases by 1.
                    # If u in diff group as v, edge was crossing, becomes internal. Cut decreases by 1.
                    # So change in cut = (edges_to_same) - (edges_to_diff) = gain.
                    # Wait, my get_gain returns 2*same - deg.
                    # same - diff = same - (deg - same) = 2*same - deg.
                    # So yes, change in cut is exactly `gain`.
                    cur_cut += gain
                    improved = True
                else:
                    # Accept with probability exp(gain / T)
                    # gain is negative or zero
                    if np.random.random() < np.exp(gain / T):
                        labels[v] = 1 - labels[v]
                        cur_cut += gain
                        improved = True
            
            if improved:
                stagnant_rounds = 0
            else:
                stagnant_rounds += 1
                if stagnant_rounds > max_stagnant:
                    break
            
            T *= cooling_rate
        
        # Update best
        if cur_cut > best_cut:
            best_cut = cur_cut
            best_labels = labels.copy()
        
        # --- ILS Perturbation Phase ---
        # Perturb the best solution to start the next SA run
        # Flip k random vertices
        # k can be adaptive, e.g., 2 to 5
        k = np.random.randint(2, 6)
        indices = np.random.choice(n, k, replace=False)
        current_labels = best_labels.copy()
        current_labels[indices] = 1 - current_labels[indices]
        
        # Optionally, do a quick local search on the perturbed solution to clean it up before next SA?
        # Or just let SA handle it. SA is slower but more robust.
        # Let's do a quick hill climb to ensure we start from a local optimum.
        # This is "Iterated Local Search" in the true sense: LocalSearch(Perturb(LocalSearch(Perturb(...))))
        
        # Quick Hill Climb
        improved = True
        while improved:
            if time.time() > deadline:
                break
            improved = False
            order = np.random.permutation(n)
            for v in order:
                gain = get_gain(current_labels, v)
                if gain > 0:
                    current_labels[v] = 1 - current_labels[v]
                    improved = True
        
        # Update best if this quick climb found something better (unlikely but possible if perturbation was lucky)
        cur_cut = cut_size(current_labels)
        if cur_cut > best_cut:
            best_cut = cur_cut
            best_labels = current_labels.copy()
            
    if best_labels is None:
        # Fallback
        best_labels = np.arange(n) % 2
    
    return best_labels.astype(np.int8)
# EVOLVE-BLOCK-END
