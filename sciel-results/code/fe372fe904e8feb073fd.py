"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def partition(adjacency):
    n = len(adjacency)
    if n == 0:
        return np.array([], dtype=int)
    
    # Convert to numpy array for fast operations
    A = np.array(adjacency, dtype=np.int8)
    
    # Deadline guard: The evaluator kills after 150s. 
    # We aim to finish well before that to be safe, say 145s.
    start_time = time.time()
    deadline = 145.0
    
    best_labels = None
    best_cut = -1
    
    # Helper to calculate cut size
    def calc_cut_size(labels):
        # labels are 0/1. 
        # Cut size = sum_{i,j} A[i][j] * (labels[i] != labels[j]) / 2 
        # Since A is symmetric and unweighted, and we count edges once.
        mask = labels[:, None] != labels[None, :]
        return 0.5 * np.sum(A * mask)
    
    # Helper for local search (polish)
    def local_search(labels):
        # Steepest ascent local search
        x = labels.copy()
        x[x == 0] = -1
        x[x == 1] = 1
        
        improved = True
        # Limit iterations to prevent infinite loops in case of bugs, though it should converge fast
        for _ in range(1000):
            if time.time() > deadline - 5:
                break
            
            # Calculate gain for each vertex
            # Ax = A @ x
            Ax = A @ x
            # Gain for flipping v is proportional to x_v * Ax[v]
            # If x_v * Ax[v] > 0, flipping reduces x^T A x (increases cut)
            gains = x * Ax
            
            # Find the vertex with the maximum positive gain
            max_gain_idx = np.argmax(gains)
            max_gain = gains[max_gain_idx]
            
            if max_gain > 0:
                # Flip the vertex
                x[max_gain_idx] = -x[max_gain_idx]
                improved = True
            else:
                improved = False
                break
        
        # Convert back to 0/1
        labels_out = (x > 0).astype(np.int8)
        return labels_out

    # Simulated Annealing
    def simulated_annealing(initial_labels, t_start, t_end, num_steps):
        x = initial_labels.copy()
        x[x == 0] = -1
        x[x == 1] = 1
        
        # Precompute row sums for potential speedups if needed, but A@x is fast for n=100
        current_cut = calc_cut_size((x > 0).astype(np.int8))
        best_x = x.copy()
        best_cut_val = current_cut
        
        # Cooling schedule
        # Linear or exponential? Exponential is standard.
        # We will do this in chunks to check time frequently.
        
        # To make SA faster, we can vectorize the acceptance step?
        # Standard SA is sequential. For n=100, num_steps can be large.
        # Let's try to do many steps in a loop, checking time every 1000 steps.
        
        for i in range(num_steps):
            if time.time() > deadline - 5:
                break
            
            # Pick a random vertex
            v = np.random.randint(n)
            
            # Calculate delta in cut if we flip v
            # In spin formulation:
            # Current contribution of v to x^T A x is x_v * sum_{j != v} A[v,j] x_j
            # Let S = sum_{j != v} A[v,j] x_j
            # New contribution is -x_v * S
            # Change in x^T A x = New - Old = -x_v S - x_v S = -2 x_v S
            # Wait, x^T A x = sum_i x_i (Ax)_i.
            # (Ax)_v = sum_j A[v,j] x_j.
            # Change in energy (to be minimized) when flipping v:
            # E_new - E_old = -2 x_v (Ax)_v
            # We want to MINIMIZE x^T A x, which MAXIMIZES Cut.
            # So if Change < 0, Cut increases.
            # Change = -2 x_v (Ax)_v.
            
            Ax_v = np.dot(A[v], x)
            delta_energy = -2 * x[v] * Ax_v
            
            # delta_cut = -delta_energy / 4 ?
            # Cut = 0.25 (n^2 - x^T A x) approx? 
            # Let's just calculate cut directly for simplicity in acceptance? No, too slow.
            # Let's stick to energy minimization.
            # If delta_energy < 0, always accept.
            # Else, accept with prob exp(-delta_energy / T).
            
            if delta_energy < 0:
                # Always accept
                x[v] = -x[v]
                current_cut += -delta_energy / 4.0 # Rough update? 
                # Actually, let's just recalculate cut every N steps or use exact delta.
                # Delta Cut = -Delta Energy / 4 is exact for this formulation?
                # Cut = (1/4) sum_{i,j} A_{ij} (1 - x_i x_j) = (1/4) (sum A_{ij} - sum A_{ij} x_i x_j)
                # So Delta Cut = -(1/4) Delta (x^T A x).
                # Yes.
            else:
                # Metropolis criterion
                # T is current temperature
                # We need T. 
                # Let's define T based on i.
                T = t_start * (t_end / t_start) ** (i / num_steps)
                if T > 1e-9:
                    prob = np.exp(-delta_energy / T)
                    if np.random.random() < prob:
                        x[v] = -x[v]
                        current_cut += -delta_energy / 4.0
            
            # Update best
            # Checking cut every step is expensive if we calc_cut_size. 
            # But we updated current_cut incrementally. 
            # Is current_cut accurate? 
            # Delta Cut = -1/4 Delta Energy. 
            # This is exact. 
            if current_cut > best_cut_val:
                best_cut_val = current_cut
                best_x = x.copy()
                
        # Final polish
        best_labels_sa = (best_x > 0).astype(np.int8)
        return best_labels_sa, best_cut_val

    # Multi-Start with SA
    # We will run several SA instances with different temperatures and durations.
    
    # Initial best from a quick local search on random
    init_labels = np.random.randint(0, 2, size=n).astype(np.int8)
    init_labels = local_search(init_labels)
    init_cut = calc_cut_size(init_labels)
    if init_cut > best_cut:
        best_cut = init_cut
        best_labels = init_labels.copy()
        
    # SA Parameters
    # n=100. 
    # We have ~140s.
    # Each SA run takes time. 
    # Let's estimate time per step.
    # np.dot(A[v], x) is O(n). 
    # If n=100, that's very fast.
    # Python loop overhead is the bottleneck.
    # Maybe 10,000 steps per second?
    # If so, 140s * 10,000 = 1.4 million steps.
    # We can split this into multiple SA runs.
    
    num_sa_runs = 5
    steps_per_run = 200000 # 5 * 200k = 1M steps. Should fit in time.
    
    for run in range(num_sa_runs):
        if time.time() > deadline - 10:
            break
            
        # Random start or perturbed best?
        # Random is better for diversity in SA.
        init = np.random.randint(0, 2, size=n).astype(np.int8)
        
        # SA Parameters
        # High T start to explore, low T to converge.
        T_start = 5.0 # Average degree is ~4-10. Energy change per flip is ~ O(degree).
        T_end = 0.01
        
        labels, cut_val = simulated_annealing(init, T_start, T_end, steps_per_run)
        
        # Polish the result
        labels = local_search(labels)
        cut_val = calc_cut_size(labels)
        
        if cut_val > best_cut:
            best_cut = cut_val
            best_labels = labels.copy()
            
    return best_labels
# EVOLVE-BLOCK-END
