"""Block Flip Simulated Annealing for Maximum Cut."""
# EVOLVE-BLOCK-START
import time
import numpy as np


def partition(adjacency):
    """
    Assign each vertex a 0/1 label to maximize the number of cut edges.
    Uses Block Flip Simulated Annealing with multiple restarts.
    """
    adj = np.asarray(adjacency, dtype=np.int8)
    n = len(adj)
    
    if n <= 1:
        return np.zeros(n, dtype=np.int8)
    
    # Precompute total edges for quick cut calculation
    total_edges = adj.sum() / 2.0
    
    deadline = time.time() + 115.0  # Safety margin before 150s kill
    
    best_labels = None
    best_cut = -1
    
    def compute_cut(labels):
        """
        Compute cut size using matrix multiplication.
        s = 2*labels - 1 (values -1, 1)
        Q = s @ adj @ s
        cut = (total_edges - Q) / 2
        """
        s = 2.0 * labels - 1.0
        Q = s @ adj @ s
        return (total_edges - Q) / 2.0

    def local_search_sa(labels, time_limit):
        """
        Simulated Annealing with Block Flips.
        """
        current_labels = labels.copy()
        current_cut = compute_cut(current_labels)
        
        # Parameters
        T = 5.0  # Higher initial temperature to allow larger block flips
        cooling_rate = 0.9995
        min_T = 0.05
        start_time = time.time()
        
        # Precompute s vector for current state to avoid recomputation in loop if possible,
        # but since we flip blocks, s changes. We will recompute s from labels when needed.
        
        while time.time() < time_limit and T > min_T:
            # Determine block size
            # Smaller blocks as T decreases to refine solution
            # At high T, allow larger blocks (up to n/2)
            # At low T, restrict to small blocks (1-3)
            max_block_size = max(1, int(n * (0.1 + 0.4 * (T / 5.0))))
            block_size = np.random.randint(1, max_block_size + 1)
            
            # Select random vertices for the block
            if block_size == n:
                # Flip all? Rarely useful but valid. Actually, flipping all doesn't change cut.
                # So max block size should be n-1 or we skip if block_size == n.
                # Let's cap max_block_size to n-1 effectively by randint logic or just handle it.
                # If we flip all, gain is 0.
                pass 
            
            # Ensure block_size < n to avoid trivial 0-gain moves
            if block_size >= n:
                block_size = n - 1
            
            block_idx = np.random.choice(n, size=block_size, replace=False)
            
            # Calculate gain of flipping this block
            # Let B be the set of vertices to flip.
            # Let s be the current spin vector (-1/1).
            # Let s_new be the spin vector after flipping B.
            # s_new[i] = -s[i] if i in B else s[i].
            
            # Gain in Cut = -0.5 * (Delta Q)
            # Q = s^T A s
            # Q_new = s_new^T A s_new
            # Delta Q = Q_new - Q
            
            # Let's compute Delta Q directly.
            # s_new = s - 2 * e_B (where e_B is 1 for i in B, 0 otherwise)
            # Let u be the vector with u_i = 1 if i in B, 0 otherwise.
            # s_new = s - 2u
            
            # Q_new = (s - 2u)^T A (s - 2u)
            # = s^T A s - 2 s^T A u - 2 u^T A s + 4 u^T A u
            # Since A is symmetric, s^T A u = u^T A s.
            # Delta Q = -4 u^T A s + 4 u^T A u
            # Delta Cut = -0.5 * Delta Q = 2 u^T A s - 2 u^T A u
            
            s = 2.0 * current_labels - 1.0
            
            # u is a vector of 0s and 1s
            u = np.zeros(n, dtype=np.float64)
            u[block_idx] = 1.0
            
            # u^T A s is the sum of A[i,j] s_j for i in B
            # This is equivalent to summing the (A @ s) values at indices in B
            A_s = adj @ s
            term1 = np.sum(A_s[block_idx])
            
            # u^T A u is the number of edges within the block B
            # Note: A is 0/1. u^T A u = sum_{i,j in B} A[i,j]
            # Since A is symmetric and 0 diagonal, this counts each edge within B twice.
            # Wait, u^T A u = sum_i sum_j u_i A_ij u_j.
            # If i,j in B, u_i=1, u_j=1. So it sums A_ij for all i,j in B.
            # For an undirected edge (i,j) in B, both A_ij and A_ji are 1.
            # So u^T A u = 2 * (number of edges within B).
            
            # Let's verify with a small example.
            # B={0,1}, edge (0,1).
            # u=[1,1,0...]
            # A_01=1, A_10=1.
            # u^T A u = 1*1*1 + 1*1*1 = 2.
            # Edges within B = 1.
            # So term2 = 2 * edges_within_B.
            
            # Delta Cut = 2 * term1 - 2 * term2
            # term2 = u @ adj @ u
            
            term2 = u @ adj @ u
            
            gain = 2.0 * term1 - 2.0 * term2
            
            # Acceptance criterion
            if gain >= 0:
                # Always accept
                current_labels[block_idx] = 1 - current_labels[block_idx]
                current_cut += gain
            else:
                # Accept with probability exp(gain / T)
                # gain is negative
                if np.random.random() < np.exp(gain / T):
                    current_labels[block_idx] = 1 - current_labels[block_idx]
                    current_cut += gain
            
            T *= cooling_rate
            
        return current_labels, current_cut

    # Multiple restarts
    # Budget: 115s
    # Block flips are more expensive per step than single flips?
    # Single flip: O(n) for gain (dot product of row).
    # Block flip: O(n^2) for u @ adj @ u if done naively, or O(n * block_size) if optimized?
    # u @ adj is O(n^2) if u is dense, but u is sparse.
    # However, in numpy, u @ adj will likely be O(n^2) unless we use sparse.
    # But n=100, so n^2 = 10,000 operations. Very fast.
    # Let's estimate steps.
    # Single flip SA might do ~10,000 steps/sec.
    # Block flip SA might do ~5,000 steps/sec due to matrix mult overhead.
    # But each step explores a larger part of the space.
    
    n_restarts = 30
    time_per_restart = 110.0 / n_restarts
    
    for i in range(n_restarts):
        if time.time() > deadline:
            break
            
        # Random initialization
        labels = np.random.randint(0, 2, n).astype(np.int8)
        
        # Run SA
        labels, cut = local_search_sa(labels, time.time() + time_per_restart)
        
        if cut > best_cut:
            best_cut = cut
            best_labels = labels.copy()
            
    if best_labels is None:
        # Fallback: alternating
        best_labels = np.arange(n) % 2
        
    return best_labels
# EVOLVE-BLOCK-END
