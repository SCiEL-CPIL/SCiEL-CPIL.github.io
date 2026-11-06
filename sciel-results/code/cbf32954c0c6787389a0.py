"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import random
import scipy.sparse as sp
import scipy.sparse.linalg as spla


def partition(adjacency):
    n = len(adjacency)
    if n == 0:
        return np.array([], dtype=int)
    
    # Convert to numpy array for efficient operations
    adj = np.asarray(adjacency, dtype=np.int8)
    
    # Deadline: 120 seconds total, leave 10s safety margin
    start_time = time.time()
    deadline = start_time + 110.0
    
    def cut_size(labels):
        """Compute the number of edges crossing the cut."""
        s = 2 * labels - 1  # ±1 vector
        val = s @ (adj @ s)
        total_edges = adj.sum() // 2
        return total_edges - val // 2
    
    def spectral_init(adj):
        """Generate initial labeling using Fiedler vector of graph Laplacian."""
        # Build Laplacian L = D - A
        degree = adj.sum(axis=1)
        D = sp.diags(degree)
        A_sparse = sp.csr_matrix(adj)
        L = D - A_sparse
        
        # Compute smallest eigenvalues of L
        # We want the second smallest (Fiedler vector)
        try:
            # Use eigsh to find smallest eigenvalues
            # k=2 to get the smallest and second smallest
            eigenvalues, eigenvectors = spla.eigsh(L, k=2, which='SM')
            # Sort by eigenvalue
            idx = np.argsort(eigenvalues)
            fiedler = eigenvectors[:, idx[1]]
            
            # Convert to 0/1 labels based on sign
            labels = (fiedler > 0).astype(int)
            
            # If all labels are the same, try random
            if np.all(labels == labels[0]):
                labels = np.random.randint(0, 2, size=n)
            return labels
        except Exception:
            # Fallback to random if eigsh fails
            return np.random.randint(0, 2, size=n)
    
    def flip_gain(v, labels, adj):
        """Gain in cut size if we flip vertex v."""
        same = adj[v, labels == labels[v]].sum()
        deg = adj[v].sum()
        return 2 * same - deg
    
    def local_search_2flip(labels, adj, max_iters=50000, time_limit=None):
        """Local search with 1-flip and 2-flip moves, vectorized."""
        labels = labels.copy()
        s = 2 * labels - 1  # ±1 vector
        deg = adj.sum(axis=1)
        
        # Vectorized gain computation for all single flips
        def compute_single_gains(s):
            # gain_v = 2*same_v - deg_v where same_v = number of neighbors with same sign
            # same_v = (s @ adj @ s) component... no
            # For vertex v: same_v = sum_{w: adj[v,w]=1, s[w]=s[v]} 1
            # = (s[v] * (adj @ s)[v] + deg[v]) / 2
            adj_s = adj @ s
            same = (s * adj_s + deg) // 2
            gains = 2 * same - deg
            return gains
        
        # Compute 2-flip gain for pair (u,v):
        # Flipping both u and v: delta = gain_u + gain_v - 2*adj[u,v]*(1 if s[u]==s[v] else -1)
        # Actually: when s[u]==s[v], flipping both: edge (u,v) stays non-crossing, gain contribution 0
        # When s[u]!=s[v], flipping both: edge (u,v) goes from crossing to non-crossing, gain -1
        # So: pair_gain = gain_u + gain_v - adj[u,v]*(0 if s[u]==s[v] else 1)
        # Wait: if s[u]!=s[v], the edge was crossing. After flipping both, s'[u]=-s[u], s'[v]=-s[v], 
        # s'[u] != s'[v] still (since -s[u] != -s[v] iff s[u] != s[v]). So edge stays crossing!
        # So actually the edge (u,v) contribution doesn't change regardless.
        # But the individual gains already account for the edge (u,v) in same/diff counts.
        # gain_u includes the effect of edge (u,v) in its same/diff count.
        # When we flip both, the combined effect is not simply gain_u + gain_v because
        # flipping v changes the "same/diff" status for u's other neighbors too... no, it doesn't.
        # Actually flipping v only affects the label of v. For u's gain calculation, 
        # the neighbor v's label changes, so the same/diff count for u changes.
        # This makes the simple sum incorrect.
        
        # Let me just compute it directly:
        # cut(s) = (sum(A) - s^T A s) / 2
        # cut(s') where s' differs from s at u and v:
        # s'^T A s' = s^T A s - 2*(A @ s)[u]*s[u] - 2*(A @ s)[v]*s[v] + 2*adj[u,v]*s[u]*s[v]
        # Wait: s' = s - 2*s[u]*e_u - 2*s[v]*e_v
        # s'^T A s' = (s - 2*s[u]*e_u - 2*s[v]*e_v)^T A (s - 2*s[u]*e_u - 2*s[v]*e_v)
        # = s^T A s - 2*s[u]*(e_u^T A s) - 2*s[v]*(e_v^T A s) + 4*s[u]*s[v]*(e_u^T A e_v)
        #   + 4*s[u]^2*(e_u^T A e_u) + 4*s[v]^2*(e_v^T A e_v) + ... 
        # Since A is symmetric and A[u,u]=0:
        # = s^T A s - 2*s[u]*(A s)[u] - 2*s[v]*(A s)[v] + 4*s[u]*s[v]*adj[u,v]
        
        # delta_cut = (s^T A s - s'^T A s') / 2
        # = (2*s[u]*(A s)[u] + 2*s[v]*(A s)[v] - 4*s[u]*s[v]*adj[u,v]) / 2
        # = s[u]*(A s)[u] + s[v]*(A s)[v] - 2*s[u]*s[v]*adj[u,v]
        
        # This is the gain! Let me verify with single flip: if v only, 
        # gain_v = s[v]*(A s)[v] which should equal 2*same_v - deg_v
        # (A s)[v] = sum_w adj[v,w]*s[w] = same_v - diff_v = 2*same_v - deg_v
        # Yes! So single flip gain = s[v] * (A s)[v].
        
        # So 2-flip gain for (u,v) = s[u]*(A s)[u] + s[v]*(A s)[v] - 2*s[u]*s[v]*adj[u,v]
        
        def compute_2flip_gains(s, gains_single):
            """Compute 2-flip gains for all pairs. Returns array of shape (n,n) or top-k."""
            # For efficiency, compute in batches
            As = adj @ s
            # pair_gain[u,v] = s[u]*As[u] + s[v]*As[v] - 2*s[u]*s[v]*adj[u,v]
            # = gains_single[u] + gains_single[v] - 2*s[u]*s[v]*adj[u,v]
            # We want pairs where this is > max(gains_single[u], gains_single[v]) typically
            # or just > 0
            # For efficiency, only consider pairs where at least one has positive single gain
            # or where the -2*s[u]*s[v]*adj[u,v] term is positive (i.e., s[u]*s[v] < 0 and adj[u,v]>0)
            
            # Vectorized: compute full matrix
            # pair_gains = gains_single[:,None] + gains_single[None,:] - 2*(s[:,None]*s[None,:])*adj
            # This is n x n = 100x100 = 10000 entries, fine!
            pair_gains = gains_single[:, None] + gains_single[None, :] - 2 * (s[:, None] * s[None, :]) * adj
            # Zero out diagonal and upper triangle (we only need lower triangle to avoid duplicates)
            np.fill_diagonal(pair_gains, 0)
            # Only keep lower triangle
            idx = np.tril_indices(n, k=-1)
            return pair_gains[idx]
        
        best_labels = labels.copy()
        best_cut = cut_size(labels)
        
        for it in range(max_iters):
            if time_limit is not None and time.time() > time_limit:
                break
            
            s = 2 * labels - 1
            As = adj @ s
            gains_single = s * As  # = 2*same - deg for each vertex
            
            # Try best single flip
            max_single = np.max(gains_single)
            if max_single > 0:
                v = np.argmax(gains_single)
                labels[v] = 1 - labels[v]
                continue
            
            # Try best 2-flip
            pair_gains = compute_2flip_gains(s, gains_single)
            if len(pair_gains) == 0:
                break
            max_pair = np.max(pair_gains)
            if max_pair <= 0:
                break
            
            # Find the pair
            flat_idx = np.argmax(pair_gains)
            # Convert flat index in lower triangle to (u,v)
            # For n x n lower triangle (excluding diagonal), element (i,j) with i>j
            # Flat index in tril: for row i, there are i elements before it (columns 0..i-1)
            # So we need to find which row
            # Use np.unravel_index on the full matrix
            full_pair_gains = np.zeros((n, n))
            full_pair_gains[np.tril_indices(n, k=-1)] = pair_gains
            u, v = np.unravel_index(np.argmax(full_pair_gains), (n, n))
            
            # Flip both u and v
            labels[u] = 1 - labels[u]
            labels[v] = 1 - labels[v]
        
        # Final cut
        final_cut = cut_size(labels)
        if final_cut > best_cut:
            best_labels = labels.copy()
            best_cut = final_cut
        
        return best_labels
    
    def simulated_annealing(labels, adj, iters=50000, t_start=1.0, t_end=0.01, time_limit=None):
        """Simulated annealing for max cut."""
        labels = labels.copy()
        current_cut = cut_size(labels)
        best_labels = labels.copy()
        best_cut = current_cut
        
        t = t_start
        cooling = (t_end / t_start) ** (1.0 / iters)
        
        for it in range(iters):
            if time_limit is not None and time.time() > time_limit:
                break
            t *= cooling
            # Pick random vertex
            v = random.randint(0, n - 1)
            gain = flip_gain(v, labels, adj)
            if gain > 0 or (t > 0.001 and random.random() < np.exp(gain / t)):
                labels[v] = 1 - labels[v]
                current_cut += gain
                if current_cut > best_cut:
                    best_cut = current_cut
                    best_labels = labels.copy()
        
        return best_labels, best_cut
    
    # Multi-start with different initializations
    best_labels = None
    best_cut = -1
    
    # Start 1: spectral initialization
    labels = spectral_init(adj)
    labels = local_search_2flip(labels, adj, max_iters=20000, time_limit=deadline)
    c = cut_size(labels)
    if c > best_cut:
        best_cut = c
        best_labels = labels.copy()
    
    # Start 2: alternating
    labels = np.arange(n) % 2
    labels = local_search_2flip(labels, adj, max_iters=20000, time_limit=deadline)
    c = cut_size(labels)
    if c > best_cut:
        best_cut = c
        best_labels = labels.copy()
    
    # Start 3: random
    for start_iter in range(5):
        if time.time() > deadline:
            break
        labels = np.random.randint(0, 2, size=n)
        labels = local_search_2flip(labels, adj, max_iters=20000, time_limit=deadline)
        c = cut_size(labels)
        if c > best_cut:
            best_cut = c
            best_labels = labels.copy()
    
    # Simulated annealing from best solution found so far
    if time.time() < deadline - 5:
        sa_labels, sa_cut = simulated_annealing(
            best_labels, adj, iters=100000, t_start=1.0, t_end=0.01,
            time_limit=deadline - 5
        )
        if sa_cut > best_cut:
            best_cut = sa_cut
            best_labels = sa_labels.copy()
    
    # Final high-precision polish pass with more iterations
    if time.time() < deadline:
        best_labels = local_search_2flip(best_labels, adj, max_iters=50000, time_limit=deadline)
    
    return best_labels
# EVOLVE-BLOCK-END
# EVOLVE-BLOCK-END
