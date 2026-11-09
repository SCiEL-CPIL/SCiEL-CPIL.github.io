"""Direct approach for Maximum Cut using spectral initialization + fast local search."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def partition(adjacency):
    n = len(adjacency)
    start_time = time.time()
    deadline = start_time + 130.0  # Safety margin before 150s limit
    
    # Convert to numpy int array
    adj = np.array(adjacency, dtype=np.int32)
    
    def compute_cut(labels):
        """Compute cut size using vectorized operation."""
        # Cut = number of edges (i,j) with adj[i,j]=1 and labels[i] != labels[j]
        # For symmetric matrix: sum over i<j
        # Vectorized: 0.5 * sum_{i,j} adj[i,j] * (labels[i] != labels[j])
        diff = (labels[:, None] != labels[None, :]).astype(np.int32)
        return int(np.sum(adj * diff) // 2)
    
    def flip_greedy(labels, max_passes=50):
        """Single-flip greedy improvement. Fast O(n^2) per pass."""
        labels = labels.copy()
        for _ in range(max_passes):
            if time.time() > deadline:
                break
            changed = False
            for v in range(n):
                if time.time() > deadline:
                    break
                # Count neighbors with same label vs different label
                same_side = adj[v, labels == labels[v]].sum()
                other_side = adj[v, labels != labels[v]].sum()
                if same_side > other_side:
                    labels[v] = 1 - labels[v]
                    changed = True
            if not changed:
                break
        return labels
    
    def pairwise_swap_search(labels, max_passes=20):
        """2-swap local search: swap labels of two vertices."""
        labels = labels.copy()
        current_cut = compute_cut(labels)
        
        for _ in range(max_passes):
            if time.time() > deadline:
                break
            improved = False
            # Shuffle vertex order for diversity
            order = np.random.permutation(n)
            for idx_i in range(n):
                if time.time() > deadline:
                    break
                i = order[idx_i]
                for idx_j in range(idx_i + 1, n):
                    if time.time() > deadline:
                        break
                    j = order[idx_j]
                    if labels[i] == labels[j]:
                        continue
                    
                    # Try swap
                    labels[i], labels[j] = labels[j], labels[i]
                    new_cut = compute_cut(labels)
                    if new_cut > current_cut:
                        current_cut = new_cut
                        improved = True
                    else:
                        labels[i], labels[j] = labels[j], labels[i]
            if not improved:
                break
        return labels, current_cut
    
    def spectral_init():
        """Spectral initialization: use Fiedler vector or leading eigenvector of L."""
        # Laplacian L = D - A
        degrees = adj.sum(axis=1)
        L = np.diag(degrees) - adj
        # Get eigenvectors; use the one corresponding to second-smallest eigenvalue
        # (Fiedler vector) for good initial partition
        try:
            eigvals, eigvecs = np.linalg.eigh(L)
            # Second smallest eigenvalue's eigenvector
            # Sort indices
            idx = np.argsort(eigvals)
            fiedler = eigvecs[:, idx[1]]
            # Partition: positive -> 1, negative -> 0
            labels = (fiedler > 0).astype(np.int32)
        except:
            labels = np.random.randint(0, 2, size=n)
        return labels
    
    def random_init():
        return np.random.randint(0, 2, size=n)
    
    def sdp_hyperplane_init():
        """
        Approximate SDP hyperplane initialization.
        Uses top eigenvectors of the adjacency matrix to create a subspace,
        then projects a random vector onto this subspace to generate a partition.
        This provides diverse, spectrally-informed initializations.
        """
        try:
            # Use top eigenvectors of the adjacency matrix
            # We only need a few (e.g., 5) to capture the main structure
            k = min(5, n)
            eigvals, eigvecs = np.linalg.eigh(adj)
            # Sort by largest eigenvalues
            idx = np.argsort(eigvals)[::-1]
            top_vecs = eigvecs[:, idx[:k]]
            
            # Generate a random vector in the subspace
            # This simulates the random hyperplane projection in SDP rounding
            random_coeff = np.random.randn(k)
            # Project to get scores for each node
            scores = top_vecs @ random_coeff
            
            # Partition based on sign
            labels = (scores > 0).astype(np.int32)
            
            # Ensure we don't get a trivial all-0 or all-1 solution
            if np.all(labels == 0) or np.all(labels == 1):
                # Flip a few random nodes or use random init as fallback
                labels = np.random.randint(0, 2, size=n)
                
        except Exception:
            labels = np.random.randint(0, 2, size=n)
            
        return labels
    
    def chebyshev_init():
        """Use top eigenvector of adjacency matrix."""
        try:
            eigvals, eigvecs = np.linalg.eigh(adj)
            # Largest eigenvalue's eigenvector
            idx = np.argmax(eigvals)
            vec = eigvecs[:, idx]
            labels = (vec > 0).astype(np.int32)
        except:
            labels = np.random.randint(0, 2, size=n)
        return labels
    
    best_labels = None
    best_cut = -1
    
    # Strategy: Multiple initialization methods + local search
    
    # 1. SDP hyperplane init (Goemans-Williamson inspired)
    if time.time() < deadline:
        labels = sdp_hyperplane_init()
        labels = flip_greedy(labels)
        labels, cut = pairwise_swap_search(labels)
        if cut > best_cut:
            best_cut = cut
            best_labels = labels.copy()
    
    # 2. Spectral init
    if time.time() < deadline:
        labels = spectral_init()
        labels = flip_greedy(labels)
        labels, cut = pairwise_swap_search(labels)
        if cut > best_cut:
            best_cut = cut
            best_labels = labels.copy()
    
    # 3. Chebyshev init
    if time.time() < deadline:
        labels = chebyshev_init()
        labels = flip_greedy(labels)
        labels, cut = pairwise_swap_search(labels)
        if cut > best_cut:
            best_cut = cut
            best_labels = labels.copy()
    
    # 4. Alternating parity init
    if time.time() < deadline:
        labels = np.arange(n) % 2
        labels = flip_greedy(labels)
        labels, cut = pairwise_swap_search(labels)
        if cut > best_cut:
            best_cut = cut
            best_labels = labels.copy()
    
    # 5-14. Random restarts with local search
    num_random_restarts = 10
    for r in range(num_random_restarts):
        if time.time() > deadline:
            break
        
        labels = random_init()
        # Quick flip to get near local minimum
        labels = flip_greedy(labels, max_passes=20)
        # Then 2-swap to escape local minima
        labels, cut = pairwise_swap_search(labels, max_passes=15)
        if cut > best_cut:
            best_cut = cut
            best_labels = labels.copy()
    
    # If we still have time, do more swaps on the best solution
    if best_labels is not None and time.time() < deadline - 5:
        labels, cut = pairwise_swap_search(best_labels, max_passes=30)
        if cut > best_cut:
            best_cut = cut
            best_labels = labels.copy()
    
    if best_labels is None:
        return np.arange(n) % 2
    return best_labels
# EVOLVE-BLOCK-END
