import numpy as np
import time

def heuristics(distance_matrix, coordinates, demands, capacity):
    start_time = time.time()
    deadline = start_time + 250.0  # Safety margin
    
    n = distance_matrix.shape[0]
    dm = distance_matrix
    d = demands
    customers = np.arange(1, n)
    
    # 1. Base Heuristic: Inverse Distance
    eps = 1e-9
    base_heur = 1.0 / (dm + eps)
    np.fill_diagonal(base_heur, 0.0)
    
    # 2. Savings Heuristic Signal
    # s[i, j] = d[i, 0] + d[0, j] - d[i, j]
    d_to_depot = dm[:, 0]
    savings = d_to_depot[:, None] + d_to_depot[None, :] - dm
    
    # Only for customer-customer edges
    cust_savings = savings[np.ix_(customers, customers)]
    cust_savings = np.clip(cust_savings, 0, None)
    
    max_savings = np.max(cust_savings)
    if max_savings > 0:
        norm_savings = cust_savings / max_savings
    else:
        norm_savings = np.zeros_like(cust_savings)
    
    # 3. Spectral Clustering for Structural Prior (Vectorized)
    total_demand = np.sum(demands[1:])
    num_clusters = max(1, int(np.ceil(total_demand / capacity)))
    num_clusters = min(num_clusters, n - 1)
    
    cluster_match = np.zeros((n, n), dtype=float)
    
    if num_clusters > 1 and n > 2:
        try:
            # Build affinity matrix for customers
            cust_dist = dm[np.ix_(customers, customers)]
            affinity = 1.0 / (cust_dist + eps)
            np.fill_diagonal(affinity, 0.0)
            
            # KNN sparsification for stability and speed
            k_knn = max(5, min(15, len(customers) // 3))
            
            # Compute KNN mask
            knn_mask = np.zeros_like(affinity, dtype=bool)
            for i in range(len(customers)):
                row = affinity[i]
                idx = np.argpartition(row, -k_knn)[-k_knn:]
                knn_mask[i, idx] = True
            knn_mask = knn_mask | knn_mask.T
            
            W = affinity * knn_mask
            np.fill_diagonal(W, 0.0)
            
            # Normalized Laplacian
            degree = W.sum(axis=1)
            degree_safe = np.where(degree > 0, degree, 1.0)
            d_inv_sqrt = 1.0 / np.sqrt(degree_safe)
            D_inv_sqrt = np.diag(d_inv_sqrt)
            L_sym = np.eye(len(customers)) - D_inv_sqrt @ W @ D_inv_sqrt
            L_sym = (L_sym + L_sym.T) / 2.0
            
            # Compute eigenvectors
            # Use full eigendecomposition for small n (50 customers)
            eigenvalues, eigenvectors = np.linalg.eigh(L_sym)
            idx_sort = np.argsort(eigenvalues)
            eigenvectors = eigenvectors[:, idx_sort]
            
            # Take first k eigenvectors (excluding the trivial constant one)
            # Actually, for k-means on spectral embedding, we usually take the first k eigenvectors
            # corresponding to the smallest eigenvalues.
            # The first eigenvector is constant.
            # We want k clusters, so we take eigenvectors 1 to k (indices 1 to k)
            # Or indices 0 to k-1?
            # Standard spectral clustering: use eigenvectors corresponding to the k smallest eigenvalues.
            # Usually the first one is constant.
            # Let's take eigenvectors[:, 1:k+1] if we want k dimensions?
            # Or just eigenvectors[:, :k]?
            # If k=1, we take eigenvectors[:, 1] (Fiedler vector)
            # If k=2, we take eigenvectors[:, 1:3]
            
            if k_knn > 1:
                k_vecs = min(k_knn, len(customers) - 1)
                # Skip the first eigenvector (constant)
                vecs = eigenvectors[:, 1:1+k_vecs]
            else:
                vecs = eigenvectors[:, 1:2]
                
            # Normalize rows for k-means
            norms = np.linalg.norm(vecs, axis=1, keepdims=True)
            norms[norms == 0] = 1
            vecs_norm = vecs / norms
            
            # Simple K-Means implementation to avoid scipy dependency issues or just use scipy if available
            # Let's try scipy.cluster.vq.kmeans2
            try:
                from scipy.cluster.vq import kmeans2
                centroids, labels = kmeans2(vecs_norm, k_knn, minit='++', seed=0)
            except ImportError:
                # Fallback to a very simple k-means
                # Initialize with random points
                rng = np.random.default_rng(0)
                centroids = vecs_norm[rng.choice(len(customers), k_knn, replace=False)]
                for _ in range(10):
                    # Assign
                    dists = np.linalg.norm(vecs_norm[:, None, :] - centroids[None, :, :], axis=2)
                    labels = np.argmin(dists, axis=1)
                    # Update
                    new_centroids = []
                    for j in range(k_knn):
                        if np.any(labels == j):
                            new_centroids.append(vecs_norm[labels == j].mean(axis=0))
                        else:
                            new_centroids.append(centroids[j])
                    new_centroids = np.array(new_centroids)
                    if np.allclose(centroids, new_centroids):
                        break
                    centroids = new_centroids
            
            # Create cluster match matrix
            # Vectorized: labels[:, None] == labels[None, :]
            match_matrix = (labels[:, None] == labels[None, :]).astype(float)
            
            # Map back to full matrix
            cluster_match[np.ix_(customers, customers)] = match_matrix
            
        except Exception:
            cluster_match = np.zeros((n, n), dtype=float)
    
    # 4. Combine Heuristics
    # H = Base * (1 + alpha * Savings + beta * Cluster)
    # We want to boost edges that are close (Base), have high savings (Savings), and are within the same cluster (Cluster)
    
    alpha = 1.0
    beta = 1.0
    
    final_heur = base_heur.copy()
    
    # Create boost matrix
    full_boost = np.ones((n, n))
    
    # Apply boosts to customer-customer edges
    boost = 1.0 + alpha * norm_savings + beta * cluster_match[np.ix_(customers, customers)]
    full_boost[np.ix_(customers, customers)] = boost
    
    final_heur = final_heur * full_boost
    
    # Ensure positivity and finiteness
    final_heur = np.where(final_heur > 0, final_heur, 1e-9)
    final_heur = np.where(np.isfinite(final_heur), final_heur, 1e-9)
    
    # Diagonal handling
    # The task says "Heuristic entries that are not positive become 1e-9".
    # Diagonal is 0, so it becomes 1e-9. This is fine.
    
    if final_heur.shape != distance_matrix.shape:
        final_heur = 1.0 / (distance_matrix + eps)
        
    return final_heur
