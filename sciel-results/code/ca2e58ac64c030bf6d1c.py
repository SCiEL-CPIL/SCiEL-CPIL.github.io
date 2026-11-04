import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import eigsh
from scipy.optimize import minimize
from scipy.spatial.distance import cdist

def construct():
    N = 50
    
    # 1. Define a function to generate a hexagonal grid layout
    # This creates the "contact graph" topology implicitly
    def make_hex_grid(row_counts):
        points = []
        y_pos = 0.0
        for i, count in enumerate(row_counts):
            # Alternate offset for hex packing
            offset = 0.5 if i % 2 == 1 else 0.0
            # Center the row horizontally relative to the whole grid
            # We'll center later, but for now, just lay out
            for j in range(count):
                x = j + offset
                y = y_pos
                points.append([x, y])
            y_pos += np.sqrt(3) / 2
        
        centers = np.array(points)
        # Center the grid
        centers -= np.mean(centers, axis=0)
        
        # Scale to fit within [-0.5, 0.5] with some margin
        max_ext = np.max(np.abs(centers))
        if max_ext > 0:
            # Scale so that the max extent is slightly less than 0.5
            # to allow for optimization room
            scale = 0.48 / max_ext
            centers = centers * scale
        return centers

    # 2. Define candidate row configurations for 50 circles
    # Common hexagonal packings for N=50
    # Total must be 50.
    # 6+7+8+8+8+7+6 = 50
    # 7+8+8+7+8+7+5 = 50
    # 8+8+7+8+7+6+6 = 50
    grid_configs = [
        [6, 7, 8, 8, 8, 7, 6],
        [7, 8, 8, 7, 8, 7, 5],
        [8, 8, 7, 8, 7, 6, 6],
        [7, 7, 8, 8, 8, 7, 5],
        [8, 7, 8, 8, 7, 7, 5],
    ]
    
    # 3. Spectral Embedding Function
    def spectral_embed(centers):
        """
        Given initial centers (from grid), build a graph where edges connect
        close neighbors, then use Laplacian eigenvectors to refine positions.
        """
        # Determine neighbors based on initial distances
        dist = cdist(centers, centers)
        np.fill_diagonal(dist, np.inf)
        
        # Define a threshold for "contact" or "neighbor"
        # In a hex grid, nearest neighbors are at distance d, next at d*sqrt(3), etc.
        # Let's pick a threshold that captures the first shell of neighbors.
        # Estimate average nearest neighbor distance
        min_dists = np.min(dist, axis=1)
        avg_min_dist = np.mean(min_dists)
        
        # Threshold should be between 1st and 2nd shell
        # 2nd shell is sqrt(3) * 1st shell approx
        threshold = 1.2 * avg_min_dist
        
        adj = np.zeros((N, N))
        for i in range(N):
            neighbors = np.where(dist[i] < threshold)[0]
            for j in neighbors:
                adj[i, j] = 1
                adj[j, i] = 1
        
        # Compute Laplacian
        degree = adj.sum(axis=1)
        L = csr_matrix(np.diag(degree) - adj)
        
        # Get eigenvectors
        # We want the eigenvectors corresponding to the smallest non-zero eigenvalues
        # eigsh with sigma=0 finds eigenvalues closest to 0
        try:
            vals, vecs = eigsh(L, k=3, sigma=0, which='LM')
            # Sort by eigenvalue
            idx = np.argsort(vals)
            vecs = vecs[:, idx]
            
            # The first eigenvector is constant (for connected graph), so we use the next two
            x_spec = vecs[:, 1]
            y_spec = vecs[:, 2]
            
            # Center
            x_spec -= np.mean(x_spec)
            y_spec -= np.mean(y_spec)
            
            # Scale to fit in box
            max_ext_spec = np.max(np.abs(np.column_stack([x_spec, y_spec])))
            if max_ext_spec > 0:
                scale_spec = 0.48 / max_ext_spec
                centers_spec = np.column_stack([x_spec * scale_spec, y_spec * scale_spec])
            else:
                centers_spec = centers.copy()
                
            return centers_spec
        except Exception:
            return centers

    # 4. Objective function
    # We use a soft-min approximation to the minimum distance to make it differentiable
    # r = min( min_ij d_ij, min_i wall_dist_i )
    # We want to maximize r.
    # Equivalent to minimizing -r.
    # To handle the min, we use log-sum-exp trick on negative distances.
    
    def objective(x, lam=100.0):
        x = x.reshape(N, 2)
        
        # Wall distances
        wx = 0.5 - np.abs(x[:, 0])
        wy = 0.5 - np.abs(x[:, 1])
        walls = np.vstack([wx, wy]).flatten()
        
        # Inter-circle distances
        d = cdist(x, x)
        np.fill_diagonal(d, np.inf)
        tri_indices = np.triu_indices(N, 1)
        d_inter = d[tri_indices]
        
        all_d = np.concatenate([d_inter, walls])
        
        # Clip to avoid log(0) or negative values causing issues
        # If any distance is very small, the penalty should be high
        all_d = np.clip(all_d, 1e-8, None)
        
        # We want to maximize the minimum distance.
        # min(all_d) approx -1/lam * log(sum(exp(-lam * d)))
        # So we want to minimize: 1/lam * log(sum(exp(-lam * d)))
        # This is the soft-min.
        
        u = -lam * all_d
        max_u = np.max(u)
        log_sum = np.log(np.sum(np.exp(u - max_u))) + max_u
        return log_sum / lam

    def get_radius(centers):
        centers = centers.reshape(N, 2)
        d = cdist(centers, centers)
        np.fill_diagonal(d, np.inf)
        min_dist = np.min(d)
        wx = 0.5 - np.abs(centers[:, 0])
        wy = 0.5 - np.abs(centers[:, 1])
        min_wall = np.min(np.vstack([wx, wy]).flatten())
        return min(min_dist / 2.0, min_wall)

    # 5. Generate initial points from multiple grid configs and spectral embed them
    np.random.seed(42)
    inits = []
    for config in grid_configs:
        grid_centers = make_hex_grid(config)
        spec_centers = spectral_embed(grid_centers)
        inits.append(grid_centers)
        inits.append(spec_centers)
        
        # Also add a jittered version of the spectral center
        noise = np.random.normal(0, 0.005, (N, 2))
        inits.append(spec_centers + noise)

    # 6. Optimization
    best_centers = inits[0]
    best_radius = get_radius(best_centers)
    
    bounds = [(-0.5, 0.5)] * (2 * N)
    
    # Multi-start local optimization
    for x0 in inits:
        x0_flat = x0.flatten()
        x0_flat = np.clip(x0_flat, -0.5, 0.5)
        
        # Use L-BFGS-B
        try:
            # First pass with lower lambda (smoother landscape)
            res = minimize(objective, x0_flat, args=(50.0,), method='L-BFGS-B', 
                           bounds=bounds, options={'maxiter': 500, 'ftol': 1e-12})
            centers_opt = res.x.reshape(N, 2)
            centers_opt = np.clip(centers_opt, -0.5, 0.5)
            
            # Second pass with higher lambda (sharper minimum)
            res2 = minimize(objective, centers_opt.flatten(), args=(200.0,), method='L-BFGS-B', 
                           bounds=bounds, options={'maxiter': 500, 'ftol': 1e-14})
            centers_opt2 = res2.x.reshape(N, 2)
            centers_opt2 = np.clip(centers_opt2, -0.5, 0.5)
            
            r = get_radius(centers_opt2)
            if r > best_radius:
                best_radius = r
                best_centers = centers_opt2
        except Exception:
            continue

    # 7. Final fine-tuning
    # Use a very high lambda for the final step to squeeze out maximum radius
    lam_final = 1000.0
    def objective_final(x):
        return objective(x, lam_final)

    try:
        res_final = minimize(objective_final, best_centers.flatten(), method='L-BFGS-B', 
                             bounds=bounds, options={'maxiter': 1000, 'ftol': 1e-16, 'gtol': 1e-15})
        centers_final = res_final.x.reshape(N, 2)
        centers_final = np.clip(centers_final, -0.5, 0.5)
        
        r_final = get_radius(centers_final)
        if r_final > best_radius:
            best_centers = centers_final
    except Exception:
        pass

    return best_centers
