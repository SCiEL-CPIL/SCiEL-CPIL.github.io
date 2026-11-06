"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import scipy.sparse as sp
import scipy.sparse.linalg as spla

def partition(adjacency):
    """
    Maximum Cut via Spectral Initialization + Multi-Start Simulated Annealing.
    """
    n = len(adjacency)
    if n == 0:
        return np.array([], dtype=int)
    
    # Convert to numpy array for fast operations
    adj = np.asarray(adjacency, dtype=np.float64)
    
    # Compute initial cut size
    def get_cut_size(labels):
        # labels are 0/1
        # Edge (i,j) crosses if labels[i] != labels[j]
        # Sum over all i<j where adj[i,j]=1 and labels[i]!=labels[j]
        # Efficient way: 
        # Let s = 2*labels - 1 (so labels are -1, 1)
        # Cut size = 0.5 * sum_{i,j} adj[i,j] * (1 - s[i]*s[j]) / 2 ? No.
        # Cut size = 0.5 * sum_{i,j} adj[i,j] * (1 - s[i]*s[j]) is wrong because of double counting.
        # Let's use: cut = sum_{i<j} adj[i,j] * (labels[i] != labels[j])
        # Vectorized:
        # Create matrix of differences?
        # Or: cut = 0.5 * (sum(adj * (labels[:, None] != labels[None, :])) - np.trace(adj * (labels[:, None] != labels[None, :])))
        # Simpler:
        # Convert labels to -1/1
        s = 2 * labels - 1
        # The number of crossing edges is 0.5 * sum_{i,j} adj[i,j] * (1 - s[i]*s[j])
        # Since adj is symmetric and 0 on diagonal.
        # sum_{i,j} adj[i,j] s[i] s[j] = s^T A s
        # Total edges E = 0.5 * sum_{i,j} adj[i,j]
        # Cut = E - 0.5 * s^T A s
        E = 0.5 * np.sum(adj)
        sAs = s @ (adj @ s)
        return E - 0.5 * sAs

    # 1. Spectral Initialization
    # Construct Laplacian L = D - A
    # Since we only need the second eigenvector, we can use sparse solver.
    # Convert adj to sparse for speed in eigensolver
    A_sp = sp.csr_matrix(adj)
    D = np.sum(adj, axis=1)
    L = sp.diags(D) - A_sp
    
    try:
        # Get the smallest eigenvalues. The first is 0 (constant vector).
        # We want the second smallest (Fiedler vector).
        # k=2 to be safe, or k=1 if we shift.
        # spla.eigsh(L, k=2, which='SM')
        # However, for disconnected graphs, there might be multiple 0 eigenvalues.
        # The Fiedler vector is the eigenvector corresponding to the smallest non-zero eigenvalue.
        # Let's just get the first two.
        vals, vecs = spla.eigsh(L, k=2, which='SM')
        # Sort by eigenvalue
        idx = np.argsort(vals)
        fiedler_vec = vecs[:, idx[1]]
        
        # Initialize labels based on sign of Fiedler vector
        # If vector is very close to 0, it might be unstable.
        # Threshold at 0.
        labels = (fiedler_vec >= 0).astype(int)
        
        # Ensure it's not a trivial solution (all 0 or all 1)
        if np.all(labels == 0) or np.all(labels == 1):
            labels = np.arange(n) % 2
            
    except Exception:
        # Fallback to random or alternating if spectral fails
        labels = np.arange(n) % 2

    best_labels = labels.copy()
    best_cut = get_cut_size(labels)
    
    # 2. Multi-Start Simulated Annealing
    start_time = time.time()
    deadline = start_time + 100.0  # Safety margin for 120s limit
    
    # Parameters for SA
    T_start = 10.0
    T_end = 0.01
    alpha = 0.95  # Cooling rate
    
    # We will run several restarts.
    # The spectral init is our best bet. Let's try to improve it first.
    # Then do a few random restarts if time permits.
    
    num_restarts = 5
    
    for r in range(num_restarts):
        if time.time() > deadline:
            break
            
        if r == 0:
            # Use spectral init
            curr_labels = best_labels.copy()
        else:
            # Random restart
            curr_labels = np.random.randint(0, 2, n)
            
        curr_cut = get_cut_size(curr_labels)
        T = T_start
        
        # Simulated Annealing Loop
        while T > T_end:
            if time.time() > deadline:
                break
                
            # Pick a random vertex to flip
            v = np.random.randint(0, n)
            
            # Calculate delta in cut size if we flip v
            # Flipping v changes edges (v, u) for all u.
            # If u was in same set, edge becomes crossing (+1).
            # If u was in other set, edge becomes non-crossing (-1).
            # neighbors = adj[v]
            # same_set_count = sum(adj[v, labels == labels[v]])
            # other_set_count = sum(adj[v, labels != labels[v]])
            # delta = other_set_count - same_set_count
            
            # Vectorized delta calculation
            # Let s = 2*labels - 1
            # s_v = 2*labels[v] - 1
            # s_u = 2*labels[u] - 1
            # Edge (v,u) contributes to cut if s_v * s_u = -1.
            # After flip, s_v becomes -s_v.
            # New product: -s_v * s_u.
            # Change in "non-cut" contribution (s_v s_u):
            # Old: s_v s_u. New: -s_v s_u.
            # Cut size = E - 0.5 * sum(s_i s_j A_ij)
            # Delta Cut = -0.5 * (New Sum - Old Sum) = -0.5 * (-2 * sum_{u} A_vu s_v s_u) = sum_{u} A_vu s_v s_u
            # Wait, let's derive carefully.
            # C = E - 0.5 * s^T A s
            # s^T A s = sum_i sum_j s_i A_ij s_j
            # Flip v: s_v -> -s_v
            # New term for v: sum_j (-s_v) A_vj s_j = -s_v sum_j A_vj s_j
            # Old term for v: s_v sum_j A_vj s_j
            # Change in sum = -2 * s_v * (A s)_v
            # Delta C = -0.5 * (-2 * s_v * (A s)_v) = s_v * (A s)_v
            
            s = 2 * curr_labels - 1
            As = adj @ s
            delta = s[v] * As[v]
            
            if delta > 0 or (T > 0 and np.random.rand() < np.exp(-delta / T)):
                curr_labels[v] = 1 - curr_labels[v]
                curr_cut += delta
                if curr_cut > best_cut:
                    best_cut = curr_cut
                    best_labels = curr_labels.copy()
            
            T *= alpha
            
        # After SA, do a quick local search to polish
        # Simple greedy local search
        improved = True
        while improved and time.time() < deadline:
            improved = False
            for v in range(n):
                s = 2 * curr_labels - 1
                As = adj @ s
                delta = s[v] * As[v]
                if delta > 0:
                    curr_labels[v] = 1 - curr_labels[v]
                    curr_cut += delta
                    improved = True
                    if curr_cut > best_cut:
                        best_cut = curr_cut
                        best_labels = curr_labels.copy()
                
                if time.time() > deadline:
                    break
    
    return best_labels
# EVOLVE-BLOCK-END
