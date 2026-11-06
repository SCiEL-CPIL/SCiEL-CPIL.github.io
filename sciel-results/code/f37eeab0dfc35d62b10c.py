"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def partition(adjacency):
    """
    Maximum Cut via Multi-Start Simulated Annealing.
    
    1. Multi-Start Strategy:
       Run multiple independent simulated annealing processes from different
       random initial states to avoid getting stuck in local optima.
       
    2. Simulated Annealing:
       - Start with a random partition.
       - Repeatedly flip a random vertex.
       - Accept the move if it improves the cut, or with a probability
         that decreases over time (cooling), allowing escape from local minima.
       - Keep the best solution found across all runs.
    """
    n = len(adjacency)
    if n == 0:
        return np.array([], dtype=int)
    
    # Precompute adjacency as a numpy array
    adj = np.asarray(adjacency, dtype=np.float64)
    
    # Ensure symmetry (just in case)
    adj = (adj + adj.T) / 2.0
    np.fill_diagonal(adj, 0)
    
    # Helper to compute cut size efficiently
    # Cut size = 0.5 * sum_{i,j} A_ij (1 - s_i s_j)
    # Let s be in {-1, 1}.
    # sum_{i,j} A_ij s_i s_j = s^T A s
    # Cut = 0.5 * (2 * E - s^T A s) where E is total edge weight?
    # Actually, let's stick to the direct formula:
    # C(s) = 0.5 * sum_{i != j} A_ij (1 - s_i s_j)
    # Note: sum_{i,j} A_ij = 2E.
    # C(s) = 0.5 * (2E - sum_{i,j} A_ij s_i s_j) = E - 0.5 * (s^T A s)
    
    total_edges = 0.5 * np.sum(adj)
    
    def calc_cut(s):
        return total_edges - 0.5 * (s @ adj @ s)

    # Parameters for Simulated Annealing
    # We will run multiple independent chains.
    num_runs = 10
    initial_temp = 10.0
    cooling_factor = 0.95
    min_temp = 0.01
    max_steps_per_run = 2000
    
    best_global_cut = -np.inf
    best_global_labels = None
    
    # Use a fixed seed for reproducibility in the function if needed, 
    # but usually random is fine. Let's use a local RNG for thread safety if applicable, 
    # though standard np.random is fine here.
    rng = np.random.default_rng()
    
    for run in range(num_runs):
        # Initialize random partition
        # s: 1 for group 1, -1 for group 0
        s = rng.choice([-1, 1], size=n)
        labels = ((s + 1) // 2).astype(np.int8)
        
        current_cut = calc_cut(s)
        best_local_cut = current_cut
        best_local_s = s.copy()
        
        T = initial_temp
        # Optional: slightly vary initial temperature or steps per run? 
        # Keeping them same is simpler.
        
        for step in range(max_steps_per_run):
            # Pick a random vertex to flip
            v = rng.integers(n)
            
            # Calculate change in cut if we flip v
            # Flipping v changes s[v] to -s[v].
            # Delta in s^T A s:
            # Old term: 2 * s[v] * (A s)_v (since A is symmetric, diag 0)
            # New term: 2 * (-s[v]) * (A s)_{v}' where (A s)_{v}' = (A s)_v - 2 * A_{v,v} * s[v] ... wait.
            # Let's use the delta formula derived earlier:
            # Delta_Cut = 2 * s[v] * (A @ s)[v]
            
            As_v = (adj @ s)[v]
            delta = 2 * s[v] * As_v
            
            if delta >= 0:
                # Accept improvement
                s[v] = -s[v]
                current_cut += delta
            else:
                # Accept worsening with probability exp(delta / T)
                if T > 0:
                    prob = np.exp(delta / T)
                    if rng.random() < prob:
                        s[v] = -s[v]
                        current_cut += delta
            
            if current_cut > best_local_cut:
                best_local_cut = current_cut
                best_local_s = s.copy()
                
            # Cool down
            T *= cooling_factor
            if T < min_temp:
                break
        
        # Update global best
        if best_local_cut > best_global_cut:
            best_global_cut = best_local_cut
            best_global_labels = best_local_s
            
    # Convert best_global_s (-1, 1) to labels (0, 1)
    if best_global_labels is not None:
        final_labels = ((best_global_labels + 1) // 2).astype(np.int8)
        return final_labels
    else:
        # Fallback (should not happen if n>0)
        return np.zeros(n, dtype=np.int8)
# EVOLVE-BLOCK-END
