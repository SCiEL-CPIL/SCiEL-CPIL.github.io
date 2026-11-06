"""Maximum Cut via batched vectorized local search with spectral k-means init and deadline guard."""
# EVOLVE-BLOCK-START
import time
import random
import numpy as np
from scipy.sparse.csgraph import laplacian as sp_laplacian
from scipy.sparse.linalg import eigsh
from scipy.sparse import csr_matrix


def partition(adjacency):
    """
    Assign each vertex a 0/1 label to maximize the number of cut edges.
    Uses batched vectorized local search (parallel non-conflicting flips),
    spectral k-means initialization for diverse starts, and a deadline guard.
    """
    adj = np.asarray(adjacency, dtype=np.int8)
    n = len(adj)

    if n <= 1:
        return np.zeros(n, dtype=np.int8)

    # Deadline: return best result before 145s (5s safety margin from 150s kill)
    deadline = time.time() + 145.0

    # Precompute degree for gain calculation
    degree = adj.sum(axis=1)
    total_edges = adj.sum() // 2

    def compute_cut(labels):
        """Number of edges crossing the partition. Fast vectorized version."""
        diff = (labels[None, :] != labels[:, None])
        cut2 = (adj * diff).sum()
        return cut2 // 2

    def batched_local_search(labels, max_rounds=500):
        """
        Batched greedy 1-flip local search.
        In each round, find ALL vertices with positive gain, then apply
        a maximal set of non-conflicting flips (no two flipped vertices are adjacent).
        This parallelizes the improvement step significantly.
        """
        current_cut = compute_cut(labels)
        
        # Precompute same_count for each vertex
        same_mask = (labels[None, :] == labels[:, None])
        same_count = (adj * same_mask).sum(axis=1)
        gains = 2 * same_count - degree
        
        for rnd in range(max_rounds):
            if time.time() > deadline:
                break
            
            # Find all vertices with positive gain
            positive_mask = gains > 0
            if not positive_mask.any():
                break
            
            # Greedily select a maximal independent set among positive-gain vertices
            # Process in order of decreasing gain
            pos_indices = np.where(positive_mask)[0]
            pos_gains = gains[pos_indices]
            order = np.argsort(-pos_gains)
            pos_indices = pos_indices[order]
            
            selected = []
            blocked = np.zeros(n, dtype=bool)
            
            for idx in pos_indices:
                if blocked[idx]:
                    continue
                selected.append(idx)
                # Block idx and all its neighbors
                blocked[idx] = True
                neighbors = np.where(adj[idx] == 1)[0]
                blocked[neighbors] = True
            
            if not selected:
                break
            
            # Apply all selected flips simultaneously
            for v in selected:
                old_label = labels[v]
                labels[v] = 1 - old_label
                current_cut += gains[v]
            
            # Update same_count and gains for all affected vertices
            # Recompute for simplicity and correctness after batch update
            same_mask = (labels[None, :] == labels[:, None])
            same_count = (adj * same_mask).sum(axis=1)
            gains = 2 * same_count - degree
        
        return labels, current_cut

    def two_flip_local_search(labels, max_rounds=30):
        """
        Try 2-flip moves to escape local optima.
        Uses incremental gain computation for speed.
        """
        current_cut = compute_cut(labels)
        
        # Precompute same_count
        same_mask = (labels[None, :] == labels[:, None])
        same_count = (adj * same_mask).sum(axis=1)
        
        for _ in range(max_rounds):
            if time.time() > deadline:
                break
            
            improved = False
            # Try random pairs
            n_pairs_to_try = 200
            for _ in range(n_pairs_to_try):
                if time.time() > deadline:
                    break
                i = np.random.randint(n)
                j = np.random.randint(n)
                if i == j:
                    continue
                
                # Compute 2-flip gain analytically
                # gain_2flip = sum over neighbors of i and j of the change in cut contribution
                # For vertex v, flipping changes its contribution by:
                #   delta_v = 2*(#same_label_neighbors_v) - degree_v  (for single flip)
                # For two flips i,j:
                #   If i and j are adjacent: special case
                #   If not adjacent: gain = gain_i + gain_j - 2*adj[i,j]*(labels[i]==labels[j] ? 1 : -1)
                
                gi = 2 * same_count[i] - degree[i]
                gj = 2 * same_count[j] - degree[j]
                
                if adj[i, j] == 1:
                    # i and j are adjacent
                    if labels[i] == labels[j]:
                        # Same side: flipping both keeps them on same side (both flip)
                        # Edge i-j was not cut, still not cut: no change from this edge
                        # But their other neighbors change
                        gain_2flip = gi + gj
                    else:
                        # Different sides: edge i-j was cut, after flip both change side
                        # Edge i-j: was cut, now not cut: lose 1
                        # But wait, both flip so relative relationship stays same
                        # Actually if labels[i]!=labels[j], after both flip, still different
                        # So edge i-j contribution unchanged
                        gain_2flip = gi + gj
                else:
                    # Not adjacent: independent
                    gain_2flip = gi + gj
                
                if gain_2flip > 0:
                    # Apply the flip
                    labels[i] = 1 - labels[i]
                    labels[j] = 1 - labels[j]
                    current_cut += gain_2flip
                    improved = True
                    # Update same_count
                    same_mask = (labels[None, :] == labels[:, None])
                    same_count = (adj * same_mask).sum(axis=1)
                    break
            
            if not improved:
                break
        
        return labels, current_cut

    def simulated_annealing(labels, initial_temp=15.0, cooling=0.9995, max_iters=20000):
        """
        Simulated Annealing with 1-flip moves.
        Uses incremental gain computation for speed.
        """
        current_cut = compute_cut(labels)
        best_labels = labels.copy()
        best_cut = current_cut
        
        # Precompute same_count
        same_mask = (labels[None, :] == labels[:, None])
        same_count = (adj * same_mask).sum(axis=1)
        
        temp = initial_temp
        no_improve_count = 0
        
        for it in range(max_iters):
            if time.time() > deadline:
                break
            
            # Random vertex
            v = np.random.randint(n)
            
            # Compute gain of flipping v
            same_count_v = same_count[v]
            gain = 2 * same_count_v - degree[v]
            
            # Accept if gain > 0 or with probability exp(gain / temp) for gain < 0
            accepted = False
            if gain >= 0:
                accepted = True
            else:
                if np.random.rand() < np.exp(gain / temp):
                    accepted = True
            
            if accepted:
                old_label = labels[v]
                labels[v] = 1 - old_label
                current_cut += gain
                
                # Update same_count for neighbors
                new_label = 1 - old_label
                neighbors = np.where(adj[v] == 1)[0]
                if len(neighbors) > 0:
                    sign = 2 * new_label - 1
                    delta_same = (2 * labels[neighbors] - 1) * sign
                    same_count[neighbors] += delta_same
                
                # Update same_count for v
                same_count[v] = degree[v] - same_count[v]
                
                if current_cut > best_cut:
                    best_cut = current_cut
                    best_labels = labels.copy()
                    no_improve_count = 0
                else:
                    no_improve_count += 1
            
            temp *= cooling
            
            if no_improve_count > 500:
                temp *= 0.9
                no_improve_count = 0
        
        return best_labels, best_cut

    def greedy_init(adj, n, rng):
        """Greedy initialization: add vertices one by one to the side that maximizes cut."""
        labels = np.zeros(n, dtype=int)
        order = np.arange(n)
        rng.shuffle(order)
        
        for v in order:
            left_count = (adj[v] * (labels == 0)).sum()
            right_count = (adj[v] * (labels == 1)).sum()
            if right_count > left_count:
                labels[v] = 0
            else:
                labels[v] = 1
        return labels

    def kmeans_2d(points, k=2, max_iter=10, rng=None):
        """Simple k-means for 2D points, returns cluster labels."""
        if rng is None:
            rng = np.random.RandomState(0)
        m = len(points)
        # Initialize centroids: pick k random points
        init_idx = rng.choice(m, size=k, replace=False)
        centroids = points[init_idx].copy()
        
        labels = np.zeros(m, dtype=int)
        for _ in range(max_iter):
            # Assign each point to nearest centroid
            dists = np.zeros((m, k))
            for c in range(k):
                dists[:, c] = np.sum((points - centroids[c]) ** 2, axis=1)
            labels = np.argmin(dists, axis=1)
            
            # Update centroids
            new_centroids = centroids.copy()
            for c in range(k):
                mask = labels == c
                if mask.sum() > 0:
                    new_centroids[c] = points[mask].mean(axis=0)
                else:
                    # Reinitialize empty cluster
                    new_centroids[c] = points[rng.randint(m)]
            
            if np.allclose(centroids, new_centroids, atol=1e-6):
                break
            centroids = new_centroids
        
        return labels

    # Multiple restarts with different initializations
    best_labels_overall = None
    best_cut_overall = -1
    rng = np.random.RandomState(42)
    
    def try_restart(labels_init, tag=""):
        nonlocal best_labels_overall, best_cut_overall
        if time.time() > deadline:
            return
        
        labels = labels_init.copy()
        # Batched local search
        labels, cut = batched_local_search(labels, max_rounds=500)
        
        # Try 2-flip escape
        if time.time() < deadline - 5:
            labels2, cut2 = two_flip_local_search(labels, max_rounds=30)
            if cut2 > cut:
                labels, cut = labels2, cut2
                # Polish after 2-flip
                labels, cut = batched_local_search(labels, max_rounds=200)
        
        if cut > best_cut_overall:
            best_cut_overall = cut
            best_labels_overall = labels.copy()
    
    # Spectral initialization: use top-k Laplacian eigenvectors + k-means
    spectral_initializations = []
    try:
        # Build sparse adjacency for Laplacian
        adj_sparse = csr_matrix(adj)
        L = sp_laplacian(adj_sparse, normed=False)
        
        # Get several smallest eigenvalues (skip the trivial zero eigenvalue)
        k_eig = min(6, n - 1)
        if k_eig >= 2:
            try:
                eigvals, eigvecs = eigsh(L, k=k_eig, which='SM')
                idx = np.argsort(eigvals)
                eigvecs_sorted = eigvecs[:, idx]
                
                # Use eigenvectors 1..k (skip index 0 which is constant)
                # For each subset of eigenvectors, do k-means to get a partition
                
                # Option 1: Use just the Fiedler vector (index 1)
                fiedler_vec = eigvecs_sorted[:, 1]
                median_val = np.median(fiedler_vec)
                spectral_initializations.append((fiedler_vec > median_val).astype(np.int8))
                
                # Option 2: Use first 2 eigenvectors (indices 1,2) with k-means
                if k_eig >= 3:
                    pts2d = eigvecs_sorted[:, 1:3]
                    km_labels = kmeans_2d(pts2d, k=2, max_iter=15, rng=rng)
                    spectral_initializations.append(km_labels.astype(np.int8))
                
                # Option 3: Use first 3 eigenvectors (indices 1,2,3) with k-means
                if k_eig >= 4:
                    pts3d = eigvecs_sorted[:, 1:4]
                    # Project to 2D via PCA-like: use first 2 principal components
                    # Simple approach: just use first 2 columns
                    pts3d_2d = pts3d[:, :2]
                    km_labels3 = kmeans_2d(pts3d_2d, k=2, max_iter=15, rng=rng)
                    spectral_initializations.append(km_labels3.astype(np.int8))
                    
                    # Option 4: Use eigenvectors 2,3 (skip Fiedler)
                    if k_eig >= 5:
                        pts_alt = eigvecs_sorted[:, 2:4]
                        km_labels_alt = kmeans_2d(pts_alt, k=2, max_iter=15, rng=rng)
                        spectral_initializations.append(km_labels_alt.astype(np.int8))
                        
                        # Option 5: Random hyperplane in eigenspace
                        # Project random direction onto eigenspace, threshold
                        rand_dir = rng.randn(k_eig - 1)  # skip constant
                        rand_dir /= np.linalg.norm(rand_dir)
                        projection = eigvecs_sorted[:, 1:] @ rand_dir
                        med = np.median(projection)
                        spectral_initializations.append((projection > med).astype(np.int8))
                        
                        # Option 6: Another random hyperplane
                        rand_dir2 = rng.randn(k_eig - 1)
                        rand_dir2 /= np.linalg.norm(rand_dir2)
                        projection2 = eigvecs_sorted[:, 1:] @ rand_dir2
                        med2 = np.median(projection2)
                        spectral_initializations.append((projection2 > med2).astype(np.int8))
            except Exception:
                pass
    except Exception:
        pass
    
    # Restart 1: spectral initializations (highest priority)
    for i, spec_labels in enumerate(spectral_initializations):
        if time.time() > deadline:
            break
        try_restart(spec_labels, f"spectral_{i}")
    
    # Restart: alternating
    try_restart(np.arange(n) % 2, "alt")
    
    # Restart: greedy
    try_restart(greedy_init(adj, n, rng), "greedy")
    
    # Restart: random
    try_restart(rng.randint(0, 2, size=n), "rand")
    
    # Restart: random with bias
    try_restart(rng.randint(0, 2, size=n), "rand2")
    
    # Restart: all zeros then flip random
    labels_init = np.zeros(n, dtype=np.int8)
    flip_mask = rng.rand(n) < 0.5
    labels_init[flip_mask] = 1
    try_restart(labels_init, "half")
    
    # More random starts
    for _ in range(4):
        if time.time() > deadline:
            break
        try_restart(rng.randint(0, 2, size=n), "rand_extra")
    
    # If we have time, do SA from best solution
    if best_labels_overall is not None and time.time() < deadline - 10:
        labels_sa = best_labels_overall.copy()
        labels_sa, cut_sa = simulated_annealing(labels_sa, initial_temp=20.0, cooling=0.999, max_iters=30000)
        
        # Polish
        labels_sa, cut_sa = batched_local_search(labels_sa, max_rounds=300)
        
        if cut_sa > best_cut_overall:
            best_cut_overall = cut_sa
            best_labels_overall = labels_sa.copy()
    
    # Final polish
    if best_labels_overall is not None and time.time() < deadline - 2:
        labels_final = best_labels_overall.copy()
        labels_final, cut_final = batched_local_search(labels_final, max_rounds=500)
        if cut_final > best_cut_overall:
            best_cut_overall = cut_final
            best_labels_overall = labels_final.copy()
    
    return best_labels_overall
# EVOLVE-BLOCK-END
