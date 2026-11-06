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
    E = 0.5 * adj.sum()  # total number of edges
    degrees = adj.sum(axis=1)

    # Deadline guard: 135 seconds (15s safety margin from 150s limit)
    deadline = time.time() + 135.0

    def get_cut_size(labels):
        """Compute the number of edges crossing the partition."""
        Z = 2.0 * labels - 1.0
        temp = adj @ Z
        zadjz = Z @ temp
        cut = 0.25 * (2.0 * E - zadjz)
        return cut

    def local_search(labels, max_iterations=100):
        """Greedy local search: flip a vertex if it improves the cut."""
        labels = labels.copy()
        current_cut = get_cut_size(labels)
        
        for iteration in range(max_iterations):
            if time.time() > deadline - 3.0:
                break
            improved = False
            for v in range(n):
                lv = labels[v]
                # Gain from flipping v = same_count - diff_count
                # same_count = neighbors with same label as v
                # diff_count = neighbors with different label as v
                
                if lv == 0:
                    # same = neighbors with label 0, diff = neighbors with label 1
                    diff_count = adj[v] @ labels
                    same_count = degrees[v] - diff_count
                else:
                    # same = neighbors with label 1, diff = neighbors with label 0
                    same_count = adj[v] @ labels
                    diff_count = degrees[v] - same_count
                
                gain = same_count - diff_count
                if gain > 0:
                    labels[v] = 1 - lv
                    current_cut += gain
                    improved = True
            
            if not improved:
                break
        
        return labels, current_cut

    # Spectral initialization using the Fiedler vector (second eigenvector of Laplacian)
    def spectral_init():
        """Create initial partition using the Fiedler vector."""
        D = sp.diags(degrees, format='csr')
        A = sp.csr_matrix(adj)
        L = D - A
        
        try:
            evals, evecs = spla.eigsh(L, k=2, sigma=0, which='LM', tol=1e-8)
            idx = np.argsort(evals)
            fiedler = evecs[:, idx[1]]
        except Exception:
            fiedler = np.random.randn(n)
        
        median_val = np.median(fiedler)
        labels = (fiedler > median_val).astype(int)
        return labels

    # Multi-start local search
    best_labels = None
    best_cut = -1

    # Start with spectral initialization
    labels = spectral_init()
    labels, cut = local_search(labels, max_iterations=150)
    if cut > best_cut:
        best_cut = cut
        best_labels = labels.copy()

    # Try random restarts if time permits
    rng = np.random.default_rng(42)
    max_restarts = 50
    for restart in range(max_restarts):
        if time.time() > deadline - 10.0:
            break
        
        labels = rng.integers(0, 2, size=n).astype(int)
        labels, cut = local_search(labels, max_iterations=100)
        if cut > best_cut:
            best_cut = cut
            best_labels = labels.copy()

    # Final high-precision polish pass on the best solution
    if best_labels is not None and time.time() < deadline - 5.0:
        labels, cut = local_search(best_labels, max_iterations=300)
        if cut > best_cut:
            best_cut = cut
            best_labels = labels.copy()

    # Final check
    if best_labels is None:
        best_labels = np.arange(n) % 2

    return best_labels.astype(int)
# EVOLVE-BLOCK-END
