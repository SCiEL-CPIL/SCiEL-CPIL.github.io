import numpy as np
import time
from scipy.spatial import Delaunay

def _fast_aco_simulate(distance_matrix, heuristic, n_ants=5, n_iterations=10, rho=0.9):
    """
    Fast internal ACO simulation to evaluate heuristic quality.
    Uses fewer ants/iterations for speed during search.
    """
    n = distance_matrix.shape[0]
    tau = np.ones((n, n))
    best_tour_len = float('inf')
    
    for it in range(n_iterations):
        tour_lengths = []
        for ant in range(n_ants):
            visited = np.zeros(n, dtype=bool)
            visited[0] = True
            current = 0
            tour_len = 0.0
            tour_path = [0]
            
            for _ in range(n - 1):
                probs = tau[current, :] * heuristic[current, :]
                probs[visited] = 0.0
                
                total_prob = probs.sum()
                if total_prob < 1e-10:
                    unvisited_indices = np.where(~visited)[0]
                    next_node = unvisited_indices[np.random.randint(len(unvisited_indices))]
                else:
                    probs = probs / total_prob
                    next_node = np.random.choice(n, p=probs)
                
                tour_len += distance_matrix[current, next_node]
                visited[next_node] = True
                tour_path.append(next_node)
                current = next_node
            
            tour_len += distance_matrix[current, 0]
            tour_lengths.append(tour_len)
            
            deposit = 1.0 / tour_len
            for i in range(n):
                u = tour_path[i]
                v = tour_path[(i + 1) % n]
                tau[u, v] += deposit
        
        tau *= rho
        
        if tour_lengths:
            current_best = min(tour_lengths)
            if current_best < best_tour_len:
                best_tour_len = current_best
                
    return best_tour_len

def heuristics(distance_matrix):
    start_time = time.time()
    deadline = start_time + 280.0  # Safety margin for 300s limit
    
    n = distance_matrix.shape[0]
    
    # Extract coordinates from distance matrix
    # We assume the distance matrix is Euclidean. We can recover coordinates up to rotation/translation.
    # However, for Delaunay we need the actual 2D points.
    # Since the problem states "50 uniform random points in the unit square", we can try to recover them.
    # A simpler approach: if we can't recover coordinates easily, we can approximate Delaunay edges 
    # using a k-NN graph or just rely on the distance matrix structure.
    # But scipy.spatial.Delaunay requires points.
    # Let's try to recover points. If it fails, fallback to distance-based heuristic.
    
    points = None
    try:
        # MDS (Multidimensional Scaling) to recover 2D points
        from scipy.spatial.distance import squareform
        from sklearn.manifold import MDS # Might not be available, let's stick to scipy/numpy
        
        # Actually, we can use a simpler method:
        # The distance matrix is symmetric.
        # We can use PCA on the double-centered distance matrix.
        D = distance_matrix.copy()
        np.fill_diagonal(D, 0)
        
        # Double centering
        J = np.eye(n) - np.ones((n, n)) / n
        B = -0.5 * J @ (D ** 2) @ J
        
        # PCA to get 2D coordinates
        eigvals, eigvecs = np.linalg.eigh(B)
        # Sort in descending order
        idx = np.argsort(eigvals)[::-1]
        eigvals = eigvals[idx]
        eigvecs = eigvecs[:, idx]
        
        # Take top 2 eigenvalues
        # Handle negative eigenvalues due to numerical noise
        eigvals = np.maximum(eigvals, 0)
        coords = eigvecs[:, :2] * np.sqrt(eigvals[:2])
        
        # Normalize to unit square? The original points are in unit square.
        # The scale might be off. Let's check the range.
        min_x, min_y = coords.min(axis=0)
        max_x, max_y = coords.max(axis=0)
        
        # If the recovered points are degenerate or very small, something is wrong.
        if max_x - min_x < 1e-6 or max_y - min_y < 1e-6:
            points = None
        else:
            # Normalize to [0, 1]
            points = (coords - min_x) / (max_x - min_x)
            # Note: This preserves relative geometry but might distort if aspect ratio is wrong.
            # However, Delaunay triangulation is invariant to affine transformations? No, it's invariant to similarity transformations.
            # If aspect ratio is distorted, Delaunay edges might be slightly different.
            # But it's a good approximation.
            
            # Verify by comparing distances
            D_recovered = np.sqrt(((points[:, np.newaxis] - points[np.newaxis, :]) ** 2).sum(axis=2))
            # Check correlation
            corr = np.corrcoef(D.flatten(), D_recovered.flatten())[0, 1]
            if corr < 0.9:
                points = None
            else:
                points = points
    except:
        points = None
    
    if points is None:
        # Fallback: Use k-NN as a proxy for Delaunay edges
        # This is less accurate but robust
        k = 6
        knn_mask = np.zeros((n, n), dtype=bool)
        for i in range(n):
            dists = distance_matrix[i].copy()
            dists[i] = np.inf
            k_indices = np.argsort(dists)[:k]
            knn_mask[i, k_indices] = True
            knn_mask[k_indices, i] = True # Symmetric
            
        delaunay_mask = knn_mask
    else:
        # Compute Delaunay triangulation
        try:
            tri = Delaunay(points)
            delaunay_mask = np.zeros((n, n), dtype=bool)
            for simplex in tri.simplices:
                for i in range(3):
                    for j in range(3):
                        if i != j:
                            delaunay_mask[simplex[i], simplex[j]] = True
        except:
            # Fallback to k-NN
            k = 6
            delaunay_mask = np.zeros((n, n), dtype=bool)
            for i in range(n):
                dists = distance_matrix[i].copy()
                dists[i] = np.inf
                k_indices = np.argsort(dists)[:k]
                delaunay_mask[i, k_indices] = True
                delaunay_mask[k_indices, i] = True

    # Compute local circumradius ratio for Delaunay edges
    # For each edge (i, j) in Delaunay, find a triangle containing it.
    # Circumradius R = abc / (4 * Area)
    # Quality score = R / d_ij ? Or d_ij / R?
    # Smaller d_ij/R means the edge is "short" relative to the triangle size, which is good.
    # Let's use score = 1 / (d_ij / R) = R / d_ij.
    
    edge_quality = np.zeros((n, n))
    
    if points is not None:
        try:
            tri = Delaunay(points)
            for simplex in tri.simplices:
                a, b, c = simplex
                # Compute side lengths
                d_ab = distance_matrix[a, b]
                d_bc = distance_matrix[b, c]
                d_ca = distance_matrix[c, a]
                
                # Compute Area using Heron's formula or cross product
                # Cross product is easier with coordinates
                p_a = points[a]
                p_b = points[b]
                p_c = points[c]
                
                # 2D cross product magnitude for area
                area = 0.5 * abs((p_b[0] - p_a[0]) * (p_c[1] - p_a[1]) - (p_c[0] - p_a[0]) * (p_b[1] - p_a[1]))
                
                if area < 1e-10:
                    continue
                
                # Circumradius R = abc / (4 * Area)
                R = (d_ab * d_bc * d_ca) / (4 * area)
                
                # Assign quality to edges
                # We want high heuristic for good edges.
                # Good edge: small d, large R (relative to d).
                # Let's use quality = R / d_edge
                edge_quality[a, b] = max(edge_quality[a, b], R / d_ab)
                edge_quality[b, a] = max(edge_quality[b, a], R / d_ab)
                
                edge_quality[b, c] = max(edge_quality[b, c], R / d_bc)
                edge_quality[c, b] = max(edge_quality[c, b], R / d_bc)
                
                edge_quality[c, a] = max(edge_quality[c, a], R / d_ca)
                edge_quality[a, c] = max(edge_quality[a, c], R / d_ca)
        except:
            pass
    
    # Normalize edge_quality
    max_q = edge_quality.max()
    if max_q > 0:
        edge_quality = edge_quality / max_q
    else:
        edge_quality = np.ones((n, n))
        np.fill_diagonal(edge_quality, 0)

    # Base heuristic: inverse distance
    safe_dist = distance_matrix.copy()
    np.fill_diagonal(safe_dist, 1.0)
    safe_dist[safe_dist < 1e-9] = 1e-9
    base_heur = 1.0 / safe_dist
    
    # Blend: h = base_heur^beta * (1 + boost * edge_quality)
    # Search over beta and boost
    
    best_score = float('inf')
    best_beta = 2.0
    best_boost = 1.0
    best_heuristic = None
    
    betas = np.linspace(1.0, 4.0, 10)
    boosts = np.linspace(0.5, 3.0, 10)
    
    for beta in betas:
        if time.time() > deadline - 20.0:
            break
        for boost in boosts:
            if time.time() > deadline - 20.0:
                break
            
            h = (base_heur ** beta) * (1.0 + boost * edge_quality)
            h[h <= 0] = 1e-9
            h[~np.isfinite(h)] = 1e-9
            
            score = _fast_aco_simulate(distance_matrix, h, n_ants=5, n_iterations=10)
            
            if score < best_score:
                best_score = score
                best_beta = beta
                best_boost = boost
                best_heuristic = h.copy()
                
    # Fine-grained search
    if time.time() < deadline - 30.0 and best_heuristic is not None:
        lower_b = max(0.5, best_beta - 0.5)
        upper_b = min(5.0, best_beta + 0.5)
        fine_betas = np.linspace(lower_b, upper_b, 5)
        
        lower_boost = max(0.1, best_boost - 0.5)
        upper_boost = min(5.0, best_boost + 0.5)
        fine_boosts = np.linspace(lower_boost, upper_boost, 5)
        
        for beta in fine_betas:
            if time.time() > deadline - 20.0:
                break
            for boost in fine_boosts:
                if time.time() > deadline - 20.0:
                    break
                
                h = (base_heur ** beta) * (1.0 + boost * edge_quality)
                h[h <= 0] = 1e-9
                h[~np.isfinite(h)] = 1e-9
                
                score = _fast_aco_simulate(distance_matrix, h, n_ants=5, n_iterations=10)
                
                if score < best_score:
                    best_score = score
                    best_beta = beta
                    best_boost = boost
                    best_heuristic = h.copy()
                    
    # Final construction
    if best_heuristic is None:
        # Fallback
        final_heuristic = base_heur ** 2.0
    else:
        final_heuristic = best_heuristic
        
    final_heuristic[final_heuristic <= 0] = 1e-9
    final_heuristic[~np.isfinite(final_heuristic)] = 1e-9
    
    return final_heuristic
