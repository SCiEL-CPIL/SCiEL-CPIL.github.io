"""Direct approach for Maximum Cut using spectral initialization + fast local search."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import random

def partition(adjacency):
    n = len(adjacency)
    start_time = time.time()
    deadline = start_time + 135.0  # Safety margin before 150s limit
    
    # Convert to numpy int array
    adj = np.array(adjacency, dtype=np.int32)
    
    def compute_cut(labels):
        """Compute cut size using vectorized operation."""
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
        """Spectral initialization using Fiedler vector."""
        degrees = adj.sum(axis=1)
        L = np.diag(degrees) - adj
        try:
            eigvals, eigvecs = np.linalg.eigh(L)
            idx = np.argsort(eigvals)
            fiedler = eigvecs[:, idx[1]]
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
        """
        try:
            k = min(5, n)
            eigvals, eigvecs = np.linalg.eigh(adj)
            idx = np.argsort(eigvals)[::-1]
            top_vecs = eigvecs[:, idx[:k]]
            
            random_coeff = np.random.randn(k)
            scores = top_vecs @ random_coeff
            
            labels = (scores > 0).astype(np.int32)
            
            if np.all(labels == 0) or np.all(labels == 1):
                labels = np.random.randint(0, 2, size=n)
                
        except Exception:
            labels = np.random.randint(0, 2, size=n)
            
        return labels
    
    def chebyshev_init():
        """Use top eigenvector of adjacency matrix."""
        try:
            eigvals, eigvecs = np.linalg.eigh(adj)
            idx = np.argmax(eigvals)
            vec = eigvecs[:, idx]
            labels = (vec > 0).astype(np.int32)
        except:
            labels = np.random.randint(0, 2, size=n)
        return labels
    
    def perturbation_search(labels, num_perturbations=30):
        """
        Apply random perturbations to escape local optima.
        Each perturbation flips a random subset of vertices, then
        runs local search to find a new local optimum.
        """
        labels = labels.copy()
        current_cut = compute_cut(labels)
        
        for _ in range(num_perturbations):
            if time.time() > deadline:
                break
            
            # Create a perturbed version
            perturbed = labels.copy()
            # Flip a random fraction of vertices (10-30%)
            flip_fraction = random.uniform(0.1, 0.3)
            num_flips = int(n * flip_fraction)
            flip_indices = random.sample(range(n), min(num_flips, n))
            for idx in flip_indices:
                perturbed[idx] = 1 - perturbed[idx]
            
            # Run local search on perturbed solution
            perturbed = flip_greedy(perturbed, max_passes=20)
            perturbed, cut = pairwise_swap_search(perturbed, max_passes=10)
            
            if cut > current_cut:
                current_cut = cut
                labels = perturbed.copy()
        
        return labels, current_cut
    
    best_labels = None
    best_cut = -1
    
    # Strategy: Multiple initialization methods + local search + perturbation
    
    # 1. SDP hyperplane init
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
    
    # 5. Random restarts with local search
    num_random_restarts = 10
    for r in range(num_random_restarts):
        if time.time() > deadline:
            break
        
        labels = random_init()
        labels = flip_greedy(labels, max_passes=20)
        labels, cut = pairwise_swap_search(labels, max_passes=15)
        if cut > best_cut:
            best_cut = cut
            best_labels = labels.copy()
    
    # 6. Perturbation search on best solution
    if best_labels is not None and time.time() < deadline - 10:
        labels, cut = perturbation_search(best_labels, num_perturbations=50)
        if cut > best_cut:
            best_cut = cut
            best_labels = labels.copy()
    
    # 7. Final polish with more aggressive local search
    if best_labels is not None and time.time() < deadline - 5:
        labels, cut = pairwise_swap_search(best_labels, max_passes=40)
        if cut > best_cut:
            best_cut = cut
            best_labels = labels.copy()
    
    if best_labels is None:
        return np.arange(n) % 2
    return best_labels
# EVOLVE-BLOCK-END
