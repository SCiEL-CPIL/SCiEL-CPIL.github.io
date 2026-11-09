"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.linalg import eigh


def partition(adjacency):
    """
    Solve Maximum Cut using:
    1. Spectral initialization (Fiedler vector)
    2. Fast local search with incremental gain updates
    3. Simulated Annealing with restarts and optimized delta updates
    4. Deadline guard to ensure completion within time limit
    """
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin before 150s kill
    
    n = len(adjacency)
    if n == 0:
        return np.array([], dtype=np.int8)
    
    adj = np.asarray(adjacency, dtype=np.int8)
    degrees = adj.sum(axis=1)
    
    # Get neighbors for fast iteration
    neighbors = [np.where(adj[i] == 1)[0] for i in range(n)]
    
    def compute_initial_labels():
        """Generate initial labels using Fiedler vector or random."""
        try:
            L = np.diag(degrees.astype(float)) - adj.astype(float)
            eigenvalues, eigenvectors = eigh(L)
            # Handle potential degeneracy or disconnected components
            fiedler = eigenvectors[:, 1]
            return (fiedler > 0).astype(np.int8)
        except Exception:
            rng = np.random.default_rng(42)
            return rng.integers(0, 2, size=n)
    
    def initialize_state(labels):
        """Initialize same_count array and current cut."""
        same_count = np.zeros(n, dtype=np.int32)
        for i in range(n):
            same_count[i] = np.sum(adj[i] * (labels == labels[i]))
        
        # Cut size = 0.5 * sum(degrees) - 0.5 * sum(same_count)
        # Or just sum(degrees - same_count) / 2
        current_cut = int(np.sum(degrees - same_count) // 2)
        return same_count, current_cut
    
    def run_sa(labels, num_iterations, rng):
        """Run Simulated Annealing for a specific number of iterations."""
        same_count, current_cut = initialize_state(labels)
        best_labels = labels.copy()
        best_cut = current_cut
        
        # Cooling schedule
        T_start = 10.0
        T_end = 0.1
        # Geometric cooling: T_k = T_start * alpha^k
        # We want T_K = T_end where K is total iterations.
        alpha = (T_end / T_start) ** (1.0 / max(num_iterations, 1))
        T = T_start
        
        for _ in range(num_iterations):
            if time.time() > deadline:
                break
            
            v = rng.integers(0, n)
            
            # Delta calculation
            # delta = same_count[v] - (degrees[v] - same_count[v])
            same_v = same_count[v]
            opp_v = degrees[v] - same_v
            delta = same_v - opp_v
            
            # Acceptance
            if delta >= 0 or rng.random() < np.exp(delta / T):
                # Flip v
                old_label = labels[v]
                labels[v] = 1 - old_label
                
                # Update same_count for v and its neighbors
                # v's new same_count is opp_v
                same_count[v] = opp_v
                
                for u in neighbors[v]:
                    if labels[u] == old_label:
                        # u was same as v, now u is opposite to v
                        # u's same_count decreases by 1
                        same_count[u] -= 1
                    else:
                        # u was opposite to v, now u is same as v
                        # u's same_count increases by 1
                        same_count[u] += 1
                
                current_cut += delta
                
                if current_cut > best_cut:
                    best_cut = current_cut
                    best_labels = labels.copy()
            
            T *= alpha
        
        return best_labels, best_cut
    
    def polish(labels):
        """Greedy hill climbing to ensure local optimality."""
        same_count, current_cut = initialize_state(labels)
        improved = True
        while improved:
            improved = False
            for v in range(n):
                same_v = same_count[v]
                opp_v = degrees[v] - same_v
                delta = same_v - opp_v
                if delta > 0:
                    # Flip v
                    old_label = labels[v]
                    labels[v] = 1 - old_label
                    same_count[v] = opp_v
                    for u in neighbors[v]:
                        if labels[u] == old_label:
                            same_count[u] -= 1
                        else:
                            same_count[u] += 1
                    current_cut += delta
                    improved = True
        return labels, current_cut
    
    best_global_labels = None
    best_global_cut = 0
    
    rng_master = np.random.default_rng(12345)
    
    # Run multiple restarts
    # Estimate time per restart.
    # Let's allow up to 20 restarts or until deadline.
    max_restarts = 20
    
    for i in range(max_restarts):
        if time.time() > deadline - 5.0:  # Leave 5s for final polish
            break
        
        # Initialize
        if i == 0:
            labels = compute_initial_labels()
        else:
            # Random initialization for diversity
            labels = rng_master.integers(0, 2, size=n)
        
        rng_run = np.random.default_rng(12345 + i)
        
        # Run SA
        sa_labels, sa_cut = run_sa(labels, 5000, rng_run)
        
        # Polish
        final_labels, final_cut = polish(sa_labels)
        
        if final_cut > best_global_cut:
            best_global_cut = final_cut
            best_global_labels = final_labels
        
        # Check time
        if time.time() > deadline - 2.0:
            break
    
    if best_global_labels is None:
        # Fallback if something went wrong
        best_global_labels = compute_initial_labels()
        best_global_labels, _ = polish(best_global_labels)
        
    return best_global_labels
# EVOLVE-BLOCK-END
