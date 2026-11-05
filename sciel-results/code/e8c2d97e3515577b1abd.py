"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def partition(adjacency):
    """
    Solve Maximum Cut using Simulated Annealing.
    
    Args:
        adjacency: bool matrix of shape (n, n)
        
    Returns:
        labels: array of 0/1 labels per vertex
    """
    n = len(adjacency)
    if n == 0:
        return np.array([], dtype=int)
    
    # Convert to numpy array for efficient operations
    adj = np.asarray(adjacency, dtype=np.int8)
    
    # Precompute degree for each vertex
    degrees = adj.sum(axis=1)
    
    # Best solution tracking
    best_labels = None
    best_cut = -1
    
    # Number of restarts to escape local optima
    n_restarts = 10
    rng = np.random.default_rng(42)
    
    for restart in range(n_restarts):
        # Random initialization
        if restart == 0:
            # Start with alternating pattern for one run
            labels = np.arange(n) % 2
        else:
            # Random initialization for other runs
            labels = rng.integers(0, 2, size=n)
        
        # Compute initial cut size
        # Cut size = sum of edges where labels differ
        # For each edge (i,j), it's in the cut if labels[i] != labels[j]
        # cut = 0.5 * sum_{i,j} adj[i,j] * (1 - 2*abs(labels[i]-labels[j])/2) ... 
        # Simpler: cut = sum of adj[i,j] where labels[i] != labels[j], divided by 2 (since undirected)
        # Actually for a bool adjacency matrix representing undirected graph,
        # we count each edge once. Let's compute directly.
        
        def compute_cut(labels, adj, n):
            """Compute the number of edges crossing the cut."""
            # An edge (i,j) crosses if labels[i] != labels[j]
            # Using the fact that for binary labels, labels[i] != labels[j] iff 1 - labels[i]*labels[j] when in {0,1}
            # But careful: if labels are 0/1, then labels[i] != labels[j] means one is 0 and other is 1
            # So the indicator is labels[i] * (1 - labels[j]) + (1 - labels[i]) * labels[j]
            # This equals labels[i] - labels[i]*labels[j] + labels[j] - labels[i]*labels[j]
            # = labels[i] + labels[j] - 2*labels[i]*labels[j]
            # Sum over all i,j and divide by 2 since each edge counted twice
            
            # More efficient: 
            # cut = sum_{i:label=0, j:label=1} adj[i,j]
            mask0 = labels == 0
            mask1 = labels == 1
            # Edges from 0-side to 1-side
            cut = adj[np.ix_(mask0, mask1)].sum()
            return cut
        
        current_cut = compute_cut(labels, adj, n)
        
        # Simulated Annealing parameters
        # Temperature schedule: geometric cooling
        T_start = 10.0
        T_end = 0.01
        n_iterations = 5000
        cooling_rate = (T_end / T_start) ** (1.0 / n_iterations)
        
        current_T = T_start
        best_local_cut = current_cut
        best_local_labels = labels.copy()
        
        for it in range(n_iterations):
            if it % 100 == 0 and it > 0:
                # Update temperature
                current_T *= cooling_rate
            
            # Pick a random vertex to flip
            v = rng.integers(0, n)
            
            # Compute the change in cut size if we flip vertex v
            # Currently, edges from v to same side contribute 0 to cut
            # Edges from v to other side contribute 1 to cut
            # After flipping, edges from v to same side (now other) contribute 1
            # Edges from v to other side (now same) contribute 0
            
            # Let's compute:
            # same_side = neighbors of v with same label
            # other_side = neighbors of v with different label
            # Current contribution from v's edges to cut = len(other_side)
            # After flip, contribution = len(same_side)
            # Delta = len(same_side) - len(other_side)
            
            same_label = labels == labels[v]
            other_label = labels != labels[v]
            
            same_count = adj[v, same_label].sum()
            other_count = adj[v, other_label].sum()
            
            delta = same_count - other_count
            
            # Acceptance: always accept if delta > 0 (improvement)
            # If delta < 0 (decrease in cut), accept with probability exp(delta / T)
            # Note: delta is negative here, so exp(negative/T) < 1
            
            if delta >= 0:
                # Improvement or no change, always accept
                labels[v] = 1 - labels[v]
                current_cut += delta
            else:
                # Deterioration, accept with probability exp(delta / T)
                # delta is negative
                prob = np.exp(delta / current_T)
                if rng.random() < prob:
                    labels[v] = 1 - labels[v]
                    current_cut += delta
            
            # Track best
            if current_cut > best_local_cut:
                best_local_cut = current_cut
                best_local_labels = labels.copy()
        
        # After SA, do local improvement (greedy flips)
        improved = True
        while improved:
            improved = False
            for v in range(n):
                same_label = best_local_labels == best_local_labels[v]
                other_label = best_local_labels != best_local_labels[v]
                
                same_count = adj[v, same_label].sum()
                other_count = adj[v, other_label].sum()
                
                if same_count > other_count:
                    best_local_labels[v] = 1 - best_local_labels[v]
                    best_local_cut += (same_count - other_count)
                    improved = True
        
        # Update global best
        if best_local_cut > best_cut:
            best_cut = best_local_cut
            best_labels = best_local_labels.copy()
    
    return best_labels
# EVOLVE-BLOCK-END
