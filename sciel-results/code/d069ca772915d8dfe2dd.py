"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def partition(adjacency):
    """
    Solve Maximum Cut using Simulated Annealing with spin glass formulation.
    
    Args:
        adjacency: bool matrix of shape (n, n)
        
    Returns:
        labels: array of 0/1 labels per vertex
    """
    n = len(adjacency)
    if n == 0:
        return np.array([], dtype=int)
    
    adj = np.asarray(adjacency, dtype=np.int8)
    
    # Precompute neighbor indices for fast delta computation
    neighbors = [np.where(adj[i] == 1)[0] for i in range(n)]
    
    best_labels = None
    best_cut = -1
    
    n_restarts = 12
    rng = np.random.default_rng(123)
    
    for restart in range(n_restarts):
        # Diverse initialization strategies
        if restart == 0:
            # Greedy initialization: place each vertex on the side that maximizes immediate cut
            spins = np.zeros(n, dtype=np.int8)  # 0 or 1
            for i in range(n):
                # Count neighbors already placed on each side
                placed = spins[:i]
                if i > 0:
                    neighbor_mask = adj[i, :i]
                    if neighbor_mask.sum() == 0:
                        spins[i] = rng.integers(0, 2)
                    else:
                        # Put on side opposite to majority of placed neighbors
                        same_side = (placed == 0) & neighbor_mask
                        diff_side = (placed == 1) & neighbor_mask
                        if same_side.sum() > diff_side.sum():
                            spins[i] = 1
                        elif diff_side.sum() > same_side.sum():
                            spins[i] = 0
                        else:
                            spins[i] = rng.integers(0, 2)
                else:
                    spins[i] = 0
        elif restart == 1:
            # Alternating pattern
            spins = np.arange(n) % 2
        elif restart == 2:
            # Spectral-like: use random projection
            r = rng.standard_normal(n)
            spins = (r > 0).astype(np.int8)
        else:
            # Random with varying density
            p = 0.3 + 0.4 * (restart % 5) / 5.0
            spins = (rng.random(n) < p).astype(np.int8)
        
        # Compute initial cut
        mask0 = spins == 0
        mask1 = spins == 1
        current_cut = adj[np.ix_(mask0, mask1)].sum()
        
        # Simulated Annealing
        T_start = max(5.0, n * 0.1)
        T_end = 0.005
        n_iterations = 8000
        cooling_rate = (T_end / T_start) ** (1.0 / n_iterations)
        
        current_T = T_start
        best_local_cut = current_cut
        best_local_spins = spins.copy()
        
        for it in range(n_iterations):
            # Update temperature every 50 steps
            if it % 50 == 0 and it > 0:
                current_T *= cooling_rate
            
            # Pick random vertex
            v = rng.integers(0, n)
            
            # Compute delta for flipping v
            # Current: edges from v to opposite side count toward cut
            # After flip: edges from v to same side (now opposite) count
            # delta = (# neighbors on same side) - (# neighbors on opposite side)
            
            v_spin = spins[v]
            same_side_mask = (spins == v_spin)
            diff_side_mask = (spins != v_spin)
            
            # Only count actual neighbors
            same_count = adj[v, same_side_mask].sum()
            diff_count = adj[v, diff_side_mask].sum()
            
            delta = same_count - diff_count
            
            if delta >= 0:
                spins[v] = 1 - spins[v]
                current_cut += delta
            else:
                prob = np.exp(delta / current_T)
                if rng.random() < prob:
                    spins[v] = 1 - spins[v]
                    current_cut += delta
            
            if current_cut > best_local_cut:
                best_local_cut = current_cut
                best_local_spins = spins.copy()
        
        # Local improvement: greedy passes
        improved = True
        while improved:
            improved = False
            # Random order to break symmetry
            order = rng.permutation(n)
            for v in order:
                v_spin = best_local_spins[v]
                same_side_mask = (best_local_spins == v_spin)
                diff_side_mask = (best_local_spins != v_spin)
                
                same_count = adj[v, same_side_mask].sum()
                diff_count = adj[v, diff_side_mask].sum()
                
                if same_count > diff_count:
                    best_local_spins[v] = 1 - best_local_spins[v]
                    best_local_cut += (same_count - diff_count)
                    improved = True
        
        if best_local_cut > best_cut:
            best_cut = best_local_cut
            best_labels = best_local_spins.copy()
    
    return best_labels
# EVOLVE-BLOCK-END
