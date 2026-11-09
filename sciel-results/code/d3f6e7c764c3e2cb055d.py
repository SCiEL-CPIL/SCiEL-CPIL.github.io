"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def _cut_size(adj, labels):
    """
    Compute the number of edges crossing the partition.
    adj: (n, n) boolean or int matrix
    labels: (n,) array of 0s and 1s
    """
    # An edge (i, j) is in the cut if labels[i] != labels[j].
    # We can compute this as:
    # sum(adj * (labels[:, None] != labels[None, :])) / 2
    # But a more efficient way for symmetric binary matrices:
    # Let S0 be set of nodes with label 0, S1 be set of nodes with label 1.
    # Cut size = sum of adj[i, j] for i in S0, j in S1.
    
    # Vectorized approach:
    # diff = (labels[:, None] != labels[None, :])
    # return np.sum(adj * diff) // 2
    
    # Even faster using matrix multiplication logic?
    # Let x be vector where x_i = 1 if label 1, -1 if label 0.
    # Then x_i x_j = -1 if different, 1 if same.
    # We want sum where x_i x_j = -1.
    # Total edges = E.
    # Sum_{i,j} adj[i,j] x_i x_j = Sum_{same} 1 - Sum_{diff} 1
    # Let Q = x^T A x.
    # Q = 2 * (E - cut) - 2 * cut ? No.
    # Let's stick to the simple diff method, it's O(N^2) but vectorized.
    
    diff = (labels[:, None] != labels[None, :])
    return np.sum(adj * diff) // 2

def _spectral_init(adj, n, rng):
    """
    Initialize labels using the Fiedler vector (second eigenvector of Laplacian).
    This provides a strong starting point for max-cut.
    """
    # Laplacian L = D - A
    # For sparse graphs, scipy is better, but for N=100, dense is fine.
    # However, we can use numpy.linalg.eigh.
    
    # Compute Laplacian
    # D is diagonal matrix of degrees
    degrees = np.sum(adj, axis=1)
    L = np.diag(degrees) - adj
    
    # Get eigenvalues and eigenvectors
    # We need the second smallest eigenvalue's eigenvector (Fiedler vector).
    # eigh returns eigenvalues in ascending order.
    try:
        vals, vecs = np.linalg.eigh(L)
        fiedler = vecs[:, 1]
    except:
        # Fallback to random if eigh fails (unlikely for 100x100)
        return rng.integers(0, 2, size=n).astype(np.int8)
    
    # Threshold the Fiedler vector at 0 (or median) to get labels
    # Usually 0 is a good threshold for Fiedler vector
    labels = (fiedler > 0).astype(np.int8)
    
    # If the partition is trivial (all 0s or all 1s), flip some randomly
    if np.all(labels == 0) or np.all(labels == 1):
         labels = rng.integers(0, 2, size=n).astype(np.int8)
         
    return labels

def _local_improve(adj, labels, n):
    """
    Greedy local improvement pass.
    """
    improved = True
    while improved:
        improved = False
        for v in range(n):
            current_label = labels[v]
            # Count neighbors with same and different labels
            # same_count = sum(adj[v, labels == current_label])
            # diff_count = sum(adj[v, labels != current_label])
            
            same_mask = (labels == current_label)
            diff_mask = (labels != current_label)
            
            same_count = np.sum(adj[v, same_mask])
            diff_count = np.sum(adj[v, diff_mask])
            
            if same_count > diff_count:
                labels[v] = 1 - current_label
                improved = True
    return labels

def partition(adjacency):
    n = len(adjacency)
    adj = np.asarray(adjacency, dtype=np.int8)
    
    # Deadline guard: 150s wall-clock limit, return best with safety margin (10s)
    start_time = time.time()
    deadline = start_time + 140.0
    
    best_labels = None
    best_cut = -1
    
    # Multi-restart Simulated Annealing
    # Number of restarts depends on available time, but we'll run a fixed number 
    # of SA runs, each with a time limit, or just loop until deadline.
    
    # We will perform several restarts.
    # Each restart: Spectral Init -> SA -> Local Improve
    
    rng = np.random.default_rng(42) # Fixed seed for reproducibility in evaluation
    
    while time.time() < deadline:
        # 1. Initialization
        labels = _spectral_init(adj, n, rng)
        
        # 2. Simulated Annealing
        # Cooling schedule: T_start, T_end, cooling_rate
        # For N=100, a few hundred iterations per restart is reasonable.
        # Let's aim for ~0.5s per restart to allow ~280 restarts in 140s.
        # Or fewer restarts with longer SA.
        
        T_start = 1.0
        T_end = 0.01
        # Number of iterations per SA run
        # Let's try 200 iterations per restart.
        # 200 iterations * 100 nodes = 20,000 flips per restart.
        # Each flip calculation is O(N) if not vectorized, or O(1) if precomputed?
        # We can precompute the "gain" for each node.
        
        # Let's use a vectorized SA step for speed.
        # Current cut: C
        # If we flip node v, the change in cut size is:
        # delta = 2 * (sum(adj[v, labels != labels[v]]) - sum(adj[v, labels == labels[v]]))
        # Actually, if we flip v:
        # Edges previously in cut (v, u where u != v) become NOT in cut.
        # Edges previously NOT in cut (v, u where u == v) become IN cut.
        # New Cut = Old Cut - (edges to other) + (edges to same)
        # Delta = (edges to same) - (edges to other)
        
        # We can maintain an array `gain` where gain[v] = delta if we flip v.
        # When we flip v, gain for all neighbors u changes.
        
        # For N=100, O(N^2) per step is 10,000 ops.
        # 200 steps = 2,000,000 ops. Very fast in Python/NumPy.
        
        current_labels = labels.copy()
        current_cut = _cut_size(adj, current_labels)
        
        # Precompute initial gains?
        # gain[v] = sum(adj[v, same]) - sum(adj[v, diff])
        # Note: if gain[v] > 0, flipping increases cut.
        
        # Let's do a simple SA loop.
        # To speed up, we'll use a vectorized approach for the flip decision.
        
        # However, updating gains after a flip is O(degree).
        # With N=100, even naive O(N^2) per step is fine.
        
        # Let's try 500 iterations of SA per restart.
        iterations = 500
        
        # Cooling
        alpha = (T_end / T_start) ** (1.0 / iterations)
        
        for i in range(iterations):
            if time.time() > deadline - 5.0: # Safety margin
                break
                
            T = T_start * (alpha ** i)
            
            # Compute gains for all nodes
            # same_counts = adj @ (current_labels == 0) ? No.
            # Let's just compute the delta for each node vectorized.
            
            # labels: 0/1
            # same_mask: (N, N) bool
            # This is O(N^2) memory and compute per step.
            # 100x100 = 10k bools. 500 steps = 5M ops. Fine.
            
            # Actually, we can be smarter.
            # Let's just pick a random node to flip to keep it simple and robust.
            # Single-node flip SA.
            
            v = rng.integers(0, n)
            
            # Calculate delta for flipping v
            same_count = np.sum(adj[v, current_labels == current_labels[v]])
            diff_count = np.sum(adj[v, current_labels != current_labels[v]])
            delta = same_count - diff_count
            
            # Acceptance probability
            if delta > 0:
                accept = True
            else:
                # delta is negative
                # P = exp(-delta / T)  (since delta is change in cost, we want to minimize cost? 
                # Wait, Max Cut. We want to MAXIMIZE cut.
                # So delta is the change in Cut Size.
                # If delta < 0, we are decreasing the cut.
                # We accept with prob exp(delta / T) because delta is negative, so exp(negative/T) < 1.
                # Yes.
                accept = (rng.random() < np.exp(delta / T))
            
            if accept:
                current_labels[v] = 1 - current_labels[v]
                current_cut += delta
                
                if current_cut > best_cut:
                    best_cut = current_cut
                    best_labels = current_labels.copy()
        
        # 3. Local Improvement
        # Run greedy local improvement to polish the solution
        current_labels = _local_improve(adj, current_labels, n)
        current_cut = _cut_size(adj, current_labels)
        
        if current_cut > best_cut:
            best_cut = current_cut
            best_labels = current_labels.copy()
            
        # Check time
        if time.time() > deadline - 2.0:
            break

    # If no best found (shouldn't happen), return random
    if best_labels is None:
        best_labels = np.arange(n) % 2
        
    return best_labels
# EVOLVE-BLOCK-END
