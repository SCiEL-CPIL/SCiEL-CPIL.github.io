# EVOLVE-BLOCK-START
import numpy as np
import time
import scipy.sparse as sp
import scipy.sparse.linalg as spla


def partition(adjacency):
    """
    Maximum Cut via Spectral Initialization + Multi-Start Local Search.
    """
    n = len(adjacency)
    if n == 0:
        return np.array([], dtype=int)

    # Convert to numpy array for fast operations
    adj = np.asarray(adjacency, dtype=np.float64)

    # Deadline guard: 140 seconds (10s safety margin from 150s limit)
    deadline = time.time() + 140.0

    def get_cut_size(labels):
        """Compute the number of edges crossing the partition."""
        # For each edge (i,j) where i < j, it crosses if labels[i] != labels[j]
        # Efficient computation:
        # cut = sum over all edges where labels differ
        # Using the fact that for a given vertex v, the number of neighbors in the other partition
        # is the degree of v minus the number of neighbors in the same partition
        # Total cut = (1/2) * sum over all v of (neighbors of v in other partition)
        
        # More efficient: for each vertex, count neighbors with different label
        # cut_size = 0
        # for v in range(n):
        #     cut_size += adj[v].sum() - adj[v, labels == labels[v]].sum()
        # But this is O(n^2) per call. Let's use a vectorized approach.
        
        # cut = 0.5 * sum_v (deg(v) - same_count(v))
        # where same_count(v) = number of neighbors of v with same label as v
        
        # Vectorized:
        # For each vertex v, same_count = adj[v] @ (labels == labels[v])
        # This is still O(n^2) but can be done with matrix operations
        
        # Alternative: 
        # Let S = labels (0/1 array)
        # cut = 0.5 * sum over all edges (i,j) of |S[i] - S[j]|
        # = 0.5 * sum over all i,j of adj[i,j] * |S[i] - S[j]|
        # Since adj is symmetric, this equals sum over i<j of adj[i,j] * |S[i] - S[j]|
        
        # Efficient computation using matrix multiplication:
        # |S[i] - S[j]| for binary S is 1 if S[i] != S[j], 0 otherwise
        # = 1 - (1 - 2*S[i])*(1 - 2*S[j]) ... no, let's think differently
        # For binary labels, |S[i]-S[j]| = S[i] + S[j] - 2*S[i]*S[j] when S[i]!=S[j]
        # Actually, simpler: define Z = 2*S - 1, so Z[i] in {-1, 1}
        # Then |S[i]-S[j]| = (1 - Z[i]*Z[j])/2
        # So cut = sum_{i<j} adj[i,j] * (1 - Z[i]*Z[j])/2
        #       = (1/4) * sum_{i,j} adj[i,j] * (1 - Z[i]*Z[j])   [since adj is symmetric and adj[i,i]=0]
        #       = (1/4) * (sum_{i,j} adj[i,j] - sum_{i,j} adj[i,j]*Z[i]*Z[j])
        #       = (1/4) * (2*E - Z^T * adj * Z)
        # where E is the number of edges
        
        Z = 2.0 * labels - 1.0
        # Z^T * adj * Z
        temp = adj @ Z
        zadjz = Z @ temp
        # Number of edges
        E = 0.5 * adj.sum()
        cut = 0.25 * (2.0 * E - zadjz)
        return cut

    def local_search(labels, max_iterations=100):
        """Greedy local search: flip a vertex if it improves the cut."""
        labels = labels.copy()
        current_cut = get_cut_size(labels)
        
        for iteration in range(max_iterations):
            if time.time() > deadline - 2.0:
                break
            improved = False
            for v in range(n):
                # Compute the gain from flipping vertex v
                # Currently, v has label labels[v]
                # Neighbors with same label: contribute to "wasted" edges
                # Neighbors with different label: contribute to cut
                # If we flip v, the same/different status reverses for all neighbors
                
                # Number of neighbors of v with same label as v
                same_label = (labels == labels[v])
                # Number of neighbors of v with different label
                diff_label = (labels != labels[v])
                
                # Current contribution of edges incident to v to the cut:
                # For each neighbor u with different label, edge (v,u) is in the cut
                # For each neighbor u with same label, edge (v,u) is not in the cut
                # After flipping v:
                # For each neighbor u with same label (before flip), edge (v,u) becomes cut
                # For each neighbor u with different label (before flip), edge (v,u) becomes non-cut
                
                # Gain = (new cut contribution from v's edges) - (old cut contribution from v's edges)
                # Old: adj[v] @ diff_label (number of neighbors with different label)
                # New: adj[v] @ same_label (number of neighbors with same label, which become different after flip)
                # But we need to be careful: we're counting each edge once for v, but the edge (v,u) is counted
                # when we process v and when we process u. However, since we're doing a single pass and
                # evaluating the change for flipping v, we should consider the full effect.
                
                # Actually, let's compute the change in total cut:
                # When we flip v, for each neighbor u:
                #   If labels[u] == labels[v] (before flip): edge (v,u) was NOT in cut, now IS in cut -> +1
                #   If labels[u] != labels[v] (before flip): edge (v,u) WAS in cut, now IS NOT in cut -> -1
                # So the change in cut = (number of neighbors with same label) - (number of neighbors with different label)
                
                same_count = adj[v] @ same_label
                diff_count = adj[v] @ diff_label
                
                gain = same_count - diff_count
                if gain > 0:
                    labels[v] = 1 - labels[v]
                    current_cut += gain
                    improved = True
            
            if not improved:
                break
        
        return labels, current_cut

    # Spectral initialization using the Fiedler vector (second eigenvector of Laplacian)
    def spectral_init():
        """Create initial partition using the Fiedler vector."""
        # Build the Laplacian: L = D - A
        # D is the degree matrix
        degrees = adj.sum(axis=1)
        # Create sparse Laplacian
        D = sp.diags(degrees, format='csr')
        A = sp.csr_matrix(adj)
        L = D - A
        
        # Find the second smallest eigenvalue and its eigenvector
        # The Fiedler vector corresponds to the second smallest eigenvalue
        try:
            # Use eigsh to find the two smallest eigenvalues/eigenvectors
            # k=2, sigma=0 to find eigenvalues near 0
            evals, evecs = spla.eigsh(L, k=2, sigma=0, which='LM', tol=1e-8)
            # Sort by eigenvalue
            idx = np.argsort(evals)
            fiedler = evecs[:, idx[1]]
        except Exception:
            # Fallback to random initialization
            fiedler = np.random.randn(n)
        
        # Create partition: vertices with positive Fiedler value go to group 0, negative to group 1
        # Use the median as threshold to balance the partition
        median_val = np.median(fiedler)
        labels = (fiedler > median_val).astype(int)
        return labels

    # Multi-start local search
    best_labels = None
    best_cut = -1

    # Start with spectral initialization
    labels = spectral_init()
    labels, cut = local_search(labels)
    if cut > best_cut:
        best_cut = cut
        best_labels = labels.copy()

    # Try a few random restarts if time permits
    rng = np.random.default_rng(42)
    max_restarts = 20
    for restart in range(max_restarts):
        if time.time() > deadline - 5.0:
            break
        
        # Random initialization
        labels = rng.integers(0, 2, size=n).astype(int)
        labels, cut = local_search(labels, max_iterations=50)
        if cut > best_cut:
            best_cut = cut
            best_labels = labels.copy()

    # Final check
    if best_labels is None:
        best_labels = np.arange(n) % 2

    return best_labels.astype(int)
# EVOLVE-BLOCK-END
