# EVOLVE-BLOCK-START
import numpy as np
import time


def partition(adjacency):
    """
    Partition an undirected graph into two sets to maximize the number of crossing edges.
    Uses spectral initialization + multi-start local search with single-vertex flips
    and 2-swap moves.
    """
    if not isinstance(adjacency, np.ndarray):
        adjacency = np.array(adjacency)
    n = adjacency.shape[0]

    start_time = time.time()
    deadline = start_time + 110.0  # 10s safety margin before 120s limit

    # Work with int8 adjacency for speed
    adj = adjacency.astype(np.int8)
    total_edges = int(np.sum(np.triu(adj, k=1)))

    if n == 0:
        return np.array([], dtype=int)

    def cut_size(labels):
        """Compute the number of edges crossing the partition."""
        # labels is 0/1 array of length n
        # An edge (i,j) crosses if labels[i] != labels[j]
        # Sum over all i<j of adj[i,j] * (labels[i] != labels[j])
        # Efficient: (labels * labels.T) gives 1 where same, 0 where diff
        # diff = 1 - labels*labels.T (for binary labels)
        # But we only want upper triangle to avoid double counting
        diff = (labels[:, None] != labels[None, :]).astype(np.int8)
        return int(np.sum(adj * diff)) // 2  # divide by 2 since symmetric

    def spectral_init():
        """
        Use the Fiedler vector (second eigenvector of the graph Laplacian) to initialize.
        For maximum cut, we want to split based on the sign of the Fiedler vector.
        """
        # Graph Laplacian L = D - A
        degree = adj.sum(axis=1)
        L = np.diag(degree) - adj.astype(np.float64)
        
        try:
            # Use scipy if available for faster eigendecomposition
            from scipy.sparse import diags, csr_matrix
            from scipy.sparse.linalg import eigs
            L_sparse = csr_matrix(L)
            # Find the smallest eigenvalue (which is 0) and the next one
            # eigs finds eigenvalues near a value; we want the smallest non-zero
            vals, vecs = eigs(L_sparse, k=2, which='SM')
            # Sort by eigenvalue
            idx = np.argsort(vals)
            fiedler = np.real(vecs[:, idx[1]])
        except Exception:
            # Fall back to numpy
            try:
                vals, vecs = np.linalg.eigh(L)
                fiedler = vecs[:, 1]
            except Exception:
                return np.arange(n) % 2  # fallback
        
        # Split based on median or zero
        median_val = np.median(fiedler)
        labels = (fiedler > median_val).astype(int)
        return labels

    def improve_single(labels):
        """
        Local improvement via single-vertex flips.
        For vertex v with label l, flipping it changes the cut by:
        gain = (# neighbors on opposite side) - (# neighbors on same side)
        """
        improved = True
        while improved and (time.time() < deadline):
            improved = False
            # Compute for all vertices at once
            # For vertex v: same_side = sum(adj[v] & (labels == labels[v]))
            #               opp_side = sum(adj[v] & (labels != labels[v]))
            #               gain = opp_side - same_side
            # If gain > 0, flipping v increases the cut
            
            labels_copy = labels.copy()
            for v in range(n):
                if time.time() >= deadline:
                    break
                l = labels[v]
                same_side = np.sum(adj[v] & (labels == l))
                opp_side = np.sum(adj[v] & (labels != l))
                gain = int(opp_side) - int(same_side)
                if gain > 0:
                    labels[v] = 1 - l
                    improved = True
        return labels

    def improve_2swap(labels):
        """
        Local improvement via 2-swap moves.
        A 2-swap move exchanges the labels of two vertices.
        This is useful when single flips are stuck but swapping two vertices helps.
        
        For vertices i and j with labels li and lj:
        If li == lj, swapping doesn't change anything.
        If li != lj, swapping is equivalent to flipping both.
        
        Actually, a more useful 2-move is: flip two vertices on the SAME side.
        Let's implement: try flipping pairs of vertices that are on the same side.
        """
        improved = True
        while improved and (time.time() < deadline - 2.0):
            improved = False
            # For each pair of vertices on the same side, check if flipping both improves
            # To keep it efficient, we only try a subset or use a heuristic
            # For n=100, O(n^2) per iteration is 10000 pairs, which is fine
            
            for i in range(n):
                if time.time() >= deadline - 2.0:
                    break
                li = labels[i]
                for j in range(i + 1, n):
                    lj = labels[j]
                    if li != lj:
                        continue  # Only consider pairs on the same side
                    
                    # Compute gain from flipping both i and j
                    # Before: both have label li
                    # After: both have label 1-li
                    # 
                    # For vertex i:
                    #   Before: same_i = # neighbors with label li, opp_i = # neighbors with label 1-li
                    #   After:  same_i' = # neighbors with label 1-li = opp_i (but j is now on same side)
                    #   Actually, let's compute the change in cut size directly.
                    #
                    # The cut size change from flipping vertex i alone is:
                    #   delta_i = opp_i - same_i
                    # Flipping vertex j alone:
                    #   delta_j = opp_j - same_j
                    # But when flipping both, there's an interaction term for the edge (i,j).
                    #
                    # Let's compute more carefully.
                    # Let S0 = set of vertices with label 0, S1 = set with label 1.
                    # Suppose li = lj = 0. So i, j in S0.
                    # After flip, i, j in S1.
                    #
                    # Edges affected:
                    # 1. Edge (i,j): was internal to S0, now internal to S1. No change in cut.
                    # 2. Edges from i to S0 (excluding j): were internal, now crossing. Gain: +1 each.
                    # 3. Edges from i to S1: were crossing, now internal. Loss: -1 each.
                    # 4. Edges from j to S0 (excluding i): were internal, now crossing. Gain: +1 each.
                    # 5. Edges from j to S1: were crossing, now internal. Loss: -1 each.
                    #
                    # Let:
                    #   a = # neighbors of i in S0 (excluding j)
                    #   b = # neighbors of i in S1
                    #   c = # neighbors of j in S0 (excluding i)
                    #   d = # neighbors of j in S1
                    #
                    # Gain = a + c - b - d
                    #
                    # Note: same_i (before) = a + 1 (if j is neighbor) + ... wait
                    # same_i = # neighbors of i with label 0 = a + (1 if adj[i,j] else 0)
                    # opp_i = # neighbors of i with label 1 = b
                    # Similarly for j.
                    
                    same_i = np.sum(adj[i] & (labels == li))
                    opp_i = np.sum(adj[i] & (labels != li))
                    same_j = np.sum(adj[j] & (labels == li))
                    opp_j = np.sum(adj[j] & (labels != li))
                    
                    # same_i includes j if adj[i,j] is 1
                    # Let's compute a, b, c, d
                    if adj[i, j]:
                        a = same_i - 1  # neighbors of i in S0 excluding j
                        c = same_j - 1  # neighbors of j in S0 excluding i
                    else:
                        a = same_i
                        c = same_j
                    b = opp_i
                    d = opp_j
                    
                    gain = a + c - b - d
                    if gain > 0:
                        labels[i] = 1 - li
                        labels[j] = 1 - li
                        improved = True
                        break  # Restart the outer loop since labels changed
        return labels

    def improve(labels):
        """Combined improvement: alternate single flips and 2-swaps."""
        labels = improve_single(labels)
        labels = improve_2swap(labels)
        labels = improve_single(labels)
        return labels

    # Multi-start local search
    best_labels = None
    best_cut = -1

    # Strategy 1: Spectral initialization
    labels_spec = spectral_init()
    labels_spec = improve(labels_spec.copy())
    c = cut_size(labels_spec)
    if c > best_cut:
        best_cut = c
        best_labels = labels_spec.copy()

    if best_cut >= total_edges:
        return best_labels

    # Strategy 2: Multiple random restarts
    num_restarts = 15
    for restart in range(num_restarts):
        if time.time() >= deadline - 10.0:
            break
        
        # Different random initializations
        if restart == 0:
            labels = np.arange(n) % 2
        elif restart == 1:
            labels = np.ones(n, dtype=int)
        elif restart == 2:
            labels = np.zeros(n, dtype=int)
        else:
            # Random with varying bias
            p = 0.3 + 0.4 * (restart / num_restarts)
            labels = (np.random.random(n) < p).astype(int)
        
        labels = labels.copy()
        labels = improve(labels)
        
        c = cut_size(labels)
        if c > best_cut:
            best_cut = c
            best_labels = labels.copy()
        
        if best_cut >= total_edges:
            break

    # Final safety check
    if time.time() >= deadline:
        return best_labels if best_labels is not None else np.arange(n) % 2

    return best_labels if best_labels is not None else np.arange(n) % 2
# EVOLVE-BLOCK-END
