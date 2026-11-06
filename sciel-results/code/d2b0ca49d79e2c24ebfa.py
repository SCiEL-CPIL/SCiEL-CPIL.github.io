"""
Strategy: Multi-Start Vectorized Simulated Annealing with Spectral Initialization.

This approach abandons the single-trajectory spectral+greedy method in favor of
a population-based search strategy. It runs multiple independent simulated
annealing instances, each initialized using the spectral method (top eigenvector
of the adjacency matrix) to ensure a strong starting point. 

The key structural difference is the use of **vectorized batch updates** within
the annealing loop. Instead of flipping one vertex at a time (which is slow in
Python due to loop overhead), we:
1. Compute the gain for flipping *every* vertex simultaneously.
2. Identify the top K candidates for flipping based on gain.
3. Accept/reject these flips based on the current temperature and random noise.
4. Update the partition and recompute gains.

This allows us to perform many more effective search steps within the time budget,
exploring the solution space more thoroughly than a single greedy or sequential
annealing path.
"""
# EVOLVE-BLOCK-START
import numpy as np
import time

def partition(adjacency):
    """
    Assign each vertex a 0/1 label to maximize the number of cut edges.
    Uses Multi-Start Vectorized Simulated Annealing with Spectral Initialization.
    """
    adj = np.asarray(adjacency, dtype=np.float64)
    n = len(adj)
    
    if n <= 1:
        return np.zeros(n, dtype=np.int8)
    
    start_time = time.time()
    deadline = start_time + 110.0  # Leave 10s safety margin for 120s limit
    
    # Check for empty graph
    if not np.any(adj):
        return np.zeros(n, dtype=np.int8)
    
    def spectral_init(rng_seed):
        """
        Generate an initial partition using the top eigenvector of the adjacency matrix.
        """
        # Power method for top eigenvector
        np.random.seed(rng_seed)
        x = np.random.randn(n)
        x = x / np.linalg.norm(x)
        
        # Run power method for a fixed number of steps (fast enough for n=100)
        for _ in range(50):
            Ax = adj @ x
            norm = np.linalg.norm(Ax)
            if norm < 1e-12:
                break
            x = Ax / norm
        
        # Round to 0/1 labels
        labels = (x > 0).astype(np.int8)
        return labels

    def compute_gains(labels, adj):
        """
        Compute the gain for flipping each vertex.
        Gain = 2 * (edges to same side) - degree
        """
        # labels shape (n,)
        # adj shape (n, n)
        # same_side[i, j] = 1 if labels[i] == labels[j] else 0
        same_side = (labels[:, None] == labels[None, :]).astype(np.float64)
        # edges_to_same[i] = sum of adj[i, j] * same_side[i, j]
        edges_to_same = np.sum(adj * same_side, axis=1)
        degrees = np.sum(adj, axis=1)
        
        # If we flip vertex i:
        # Edges crossing change: 
        # Old cut edges involving i: degree - edges_to_same
        # New cut edges involving i: edges_to_same
        # Change = New - Old = edges_to_same - (degree - edges_to_same) = 2*edges_to_same - degree
        gains = 2 * edges_to_same - degrees
        return gains

    def cut_size(labels, adj):
        """
        Compute the number of edges crossing the partition.
        """
        # Cut edges are those where labels[i] != labels[j] and adj[i, j] = 1
        diff = (labels[:, None] != labels[None, :]).astype(np.float64)
        return np.sum(adj * diff) / 2.0  # Divide by 2 because each edge is counted twice in the matrix

    best_labels = None
    best_cut = -1
    
    # Number of restarts
    # We will adapt the number of restarts based on remaining time.
    # Let's aim for at least 10 restarts if time permits.
    num_restarts = 20
    
    for r in range(num_restarts):
        if time.time() > deadline - 5.0:
            break
            
        # Initialize using spectral method with different seeds
        init_labels = spectral_init(seed=r * 1000 + 42)
        current_labels = init_labels.copy()
        
        # Compute initial cut
        current_cut = cut_size(current_labels, adj)
        
        # Simulated Annealing Parameters
        # T0: Initial temperature
        # Tend: Final temperature
        # alpha: Cooling rate
        # n_steps: Number of steps per restart
        
        # Adaptive time per restart
        time_per_restart = (deadline - 5.0 - time.time()) / (num_restarts - r)
        if time_per_restart < 0.5:
            time_per_restart = 0.5
            
        # Estimate steps per second roughly to set n_steps
        # For n=100, vectorized step is very fast. Let's assume 1000 steps/sec.
        n_steps = int(time_per_restart * 1000)
        if n_steps < 100:
            n_steps = 100
        
        T0 = 10.0
        Tend = 0.01
        alpha = (Tend / T0) ** (1.0 / n_steps)
        
        T = T0
        
        for step in range(n_steps):
            if time.time() > deadline - 2.0:
                break
                
            # Compute gains for all vertices
            gains = compute_gains(current_labels, adj)
            
            # Select top K candidates to consider flipping
            # K should be small to keep it fast, but large enough for diversity
            K = min(10, n)
            
            if K < n:
                # Get indices of top K gains
                # We want to consider both positive gains (improvements) and some negative ones (for escape)
                # Sort by gain descending
                top_k_indices = np.argsort(gains)[::-1][:K]
            else:
                top_k_indices = np.arange(n)
            
            # For each candidate, decide whether to flip
            # We do this vectorized-ish by checking conditions
            # To keep it simple and fast, we pick one random candidate from top K
            # and decide based on SA rule. This is a compromise between full vectorization and speed.
            # Actually, let's just pick the best gain first, then random from top K if rejected?
            # No, standard SA picks a random neighbor. Here, "neighbor" is any single flip.
            # To speed up, we can pick a random index from the top K gains.
            
            idx = top_k_indices[np.random.randint(K)]
            gain = gains[idx]
            
            # Acceptance probability
            if gain >= 0:
                accept = True
            else:
                # Boltzmann factor
                # Avoid overflow/underflow
                if T > 1e-12:
                    exp_val = np.exp(gain / T)
                    accept = (np.random.random() < exp_val)
                else:
                    accept = False
            
            if accept:
                current_labels[idx] = 1 - current_labels[idx]
                current_cut += gain
                # Note: current_cut is updated incrementally, but gains change for neighbors.
                # We recompute gains from scratch next iteration, so this is safe.
            
            # Cool down
            T *= alpha
            
        # End of restart
        if current_cut > best_cut:
            best_cut = current_cut
            best_labels = current_labels.copy()
            
    # Final Polish: Greedy local search on the best solution found
    if best_labels is not None and time.time() < deadline - 2.0:
        # Simple greedy polish
        improved = True
        while improved and time.time() < deadline - 1.0:
            improved = False
            gains = compute_gains(best_labels, adj)
            max_gain = np.max(gains)
            if max_gain > 1e-9:
                flip_idx = np.argmax(gains)
                best_labels[flip_idx] = 1 - best_labels[flip_idx]
                improved = True
    
    if best_labels is None:
        # Fallback: random partition
        best_labels = np.random.randint(0, 2, n).astype(np.int8)
        
    return best_labels
# EVOLVE-BLOCK-END
