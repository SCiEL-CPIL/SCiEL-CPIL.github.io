"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def _cut_size(adjacency, labels):
    """Compute the number of edges crossing the partition."""
    # diff[i,j] is True if labels[i] != labels[j]
    # We want sum of adjacency[i,j] where diff[i,j] is True
    # Since adjacency is symmetric, this counts each edge twice.
    diff = labels[None, :] != labels[:, None]
    return int((adjacency & diff).sum() // 2)

def _simulated_annealing(adjacency, labels, deadline):
    """
    Quantum-Inspired Simulated Annealing.
    Uses a cooling schedule to escape local optima.
    """
    n = len(labels)
    # Precompute degrees for gain calculation? 
    # Gain of flipping v: 
    # Let S_v be set of neighbors.
    # Current cut contribution of edges incident to v:
    # c_v = sum_{u in S_v} (labels[u] != labels[v])
    # New cut contribution if we flip v:
    # c'_v = sum_{u in S_v} (labels[u] == labels[v]) = deg(v) - c_v
    # Delta = c'_v - c_v = deg(v) - 2*c_v
    
    # To speed up, we can maintain the cut size incrementally.
    # But for simplicity and robustness within 150s for n=100, 
    # we can recompute delta efficiently using vector ops or precomputed sums.
    
    # Let's use the standard SA loop.
    # Initial temperature
    T = 10.0
    T_min = 1e-4
    cooling_rate = 0.995
    
    current_labels = labels.copy()
    current_cut = _cut_size(adjacency, current_labels)
    
    best_labels = current_labels.copy()
    best_cut = current_cut
    
    # Precompute adjacency row sums (degrees)
    degrees = adjacency.sum(axis=1)
    
    # For faster delta calculation, we can maintain 
    # same_count[v] = number of neighbors of v with same label as v
    # other_count[v] = number of neighbors of v with different label as v
    # delta = other_count[v] - same_count[v]
    
    # Initialize counts
    # same_count[v] = sum( (labels[u] == labels[v]) for u in neighbors(v) )
    # We can compute this via matrix multiplication or loops.
    # labels is 0/1. 
    # same[v] = sum( (labels[u] == labels[v]) * adj[v,u] )
    # If labels[v]=0, same[v] = sum( (1-labels[u]) * adj[v,u] )
    # If labels[v]=1, same[v] = sum( labels[u] * adj[v,u] )
    
    # Let's maintain 'gain' array: gain[v] = other_count[v] - same_count[v]
    # If we flip v, the cut changes by gain[v].
    # After flipping v, how do gains of neighbors change?
    # Let u be a neighbor of v.
    # Before flip: 
    #   if labels[u] == labels[v], u contributed to same_count[u] from v.
    #   if labels[u] != labels[v], u contributed to other_count[u] from v.
    # After flip:
    #   labels[v] becomes 1 - labels[v].
    #   Now labels[u] and new_labels[v] are different if they were same before, and same if they were different.
    #   So for neighbor u:
    #     If labels[u] == old_labels[v]:
    #        Before: u had v in same. After: u has v in other.
    #        same_count[u] decreases by 1. other_count[u] increases by 1.
    #        gain[u] changes by +2.
    #     If labels[u] != old_labels[v]:
    #        Before: u had v in other. After: u has v in same.
    #        other_count[u] decreases by 1. same_count[u] increases by 1.
    #        gain[u] changes by -2.
    
    # Let's compute initial gains.
    # gain[v] = other_count[v] - same_count[v]
    # other_count[v] = sum( (labels[u] != labels[v]) * adj[v,u] )
    # same_count[v] = sum( (labels[u] == labels[v]) * adj[v,u] )
    
    # Vectorized init:
    # Create matrix M where M[i,j] = 1 if labels[i] != labels[j] else 0
    diff_mat = current_labels[None, :] != current_labels[:, None]
    # other_counts = (adjacency & diff_mat).sum(axis=1)
    # same_counts = (adjacency & ~diff_mat).sum(axis=1)
    # But ~diff_mat includes diagonal? adjacency is 0 on diagonal usually.
    # Let's just compute:
    other_counts = (adjacency & diff_mat).sum(axis=1)
    same_counts = (adjacency & ~diff_mat).sum(axis=1)
    # Note: if adjacency has 0s on diagonal, ~diff_mat has 1s on diagonal where labels[i]==labels[i].
    # But adjacency[i,i] is 0, so it doesn't matter.
    gains = other_counts - same_counts
    
    # SA Loop
    start_time = time.time()
    
    # Number of iterations estimate
    # We want to finish before deadline.
    # Let's run until deadline or T < T_min.
    
    while time.time() < deadline and T > T_min:
        # Pick random vertex
        v = np.random.randint(n)
        
        # Calculate delta
        delta = gains[v]
        
        # Metropolis criterion
        if delta > 0 or np.random.random() < np.exp(delta / T):
            # Accept move
            # Update cut
            current_cut += delta
            
            # Flip label
            old_label = current_labels[v]
            current_labels[v] = 1 - old_label
            
            # Update gains for v and its neighbors
            # v's own gain changes sign?
            # gain[v] = other - same.
            # After flip, other becomes same, same becomes other.
            # So gain[v] becomes -gain[v].
            gains[v] = -gains[v]
            
            # Update neighbors
            neighbors = np.where(adjacency[v])[0]
            for u in neighbors:
                # If u had same label as v (before flip), u's gain increases by 2
                # If u had different label as v (before flip), u's gain decreases by 2
                if current_labels[u] == 1 - old_label: # current label of u is same as NEW label of v?
                    # Wait.
                    # Before flip: labels[u] vs old_label.
                    # If labels[u] == old_label, then u and v were same.
                    # After flip, u and v are different.
                    # u's 'other' count increases by 1, 'same' decreases by 1.
                    # gain[u] increases by 2.
                    # Condition: labels[u] == old_label
                    gains[u] += 2
                else:
                    # labels[u] != old_label
                    # u and v were different.
                    # After flip, u and v are same.
                    # u's 'other' count decreases by 1, 'same' increases by 1.
                    # gain[u] decreases by 2.
                    gains[u] -= 2
            
            # Track best
            if current_cut > best_cut:
                best_cut = current_cut
                best_labels = current_labels.copy()
        
        # Cool down
        T *= cooling_rate
        
        # Optional: Restart if stuck? 
        # For now, simple cooling.
        
    return best_labels

def partition(adjacency):
    """
    Main partition function using Simulated Annealing.
    """
    n = len(adjacency)
    deadline = time.time() + 140.0 # Safety margin
    
    # Initial random label
    labels = np.random.randint(0, 2, size=n)
    
    # Run SA
    best_labels = _simulated_annealing(adjacency, labels, deadline)
    
    return best_labels
# EVOLVE-BLOCK-END
