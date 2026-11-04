import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import eigsh
from scipy.optimize import minimize
from scipy.spatial.distance import cdist

def construct():
    N = 50
    
    # 1. Generate a hexagonal grid layout for 50 circles
    # Try multiple row configurations
    def make_hex_grid(row_counts):
        points = []
        y_pos = 0.0
        for i, count in enumerate(row_counts):
            offset = 0.5 if i % 2 == 1 else 0.0
            for j in range(count):
                x = j + offset
                y = y_pos
                points.append([x, y])
            y_pos += np.sqrt(3) / 2
        centers = np.array(points)
        centers -= np.mean(centers, axis=0)
        max_ext = np.max(np.abs(centers))
        if max_ext > 0:
            centers = centers * (0.45 / max_ext)
        return centers
    
    # Multiple grid configurations
    grid_configs = [
        [8, 7, 8, 7, 8, 7, 5],
        [7, 8, 7, 8, 7, 8, 5],
        [6, 8, 7, 8, 7, 8, 6],
        [7, 7, 8, 8, 8, 7, 5],
    ]
    
    centers_grid = make_hex_grid(grid_configs[0])
    
    # 2. Spectral Embedding from grid
    centers_spec = centers_grid.copy()
    try:
        dist = cdist(centers_grid, centers_grid)
        np.fill_diagonal(dist, np.inf)
        adj = np.zeros((N, N))
        tol = 1.15
        for i in range(N):
            neighbors = np.where(dist[i] < tol)[0]
            for j in neighbors:
                adj[i, j] = 1
                adj[j, i] = 1
                
        degree = adj.sum(axis=1)
        L = csr_matrix(np.diag(degree) - adj)
        
        vals, vecs = eigsh(L, k=3, sigma=0, which='LM')
        idx = np.argsort(vals)
        vecs = vecs[:, idx]
        
        x_spec = vecs[:, 1]
        y_spec = vecs[:, 2]
        
        x_spec -= np.mean(x_spec)
        y_spec -= np.mean(y_spec)
        
        max_ext_spec = np.max(np.abs(np.column_stack([x_spec, y_spec])))
        if max_ext_spec > 0:
            scale_spec = 0.45 / max_ext_spec
            centers_spec = np.column_stack([x_spec * scale_spec, y_spec * scale_spec])
    except Exception:
        pass

    # 3. Objective function using soft-min (log-sum-exp)
    lam = 200.0

    def objective(x):
        x = x.reshape(N, 2)
        wx = 0.5 - np.abs(x[:, 0])
        wy = 0.5 - np.abs(x[:, 1])
        walls = np.vstack([wx, wy]).flatten()
        
        d = cdist(x, x)
        np.fill_diagonal(d, np.inf)
        tri_indices = np.triu_indices(N, 1)
        d_inter = d[tri_indices]
        
        all_d = np.concatenate([d_inter, walls])
        all_d = np.clip(all_d, 1e-8, None)
        
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

    # 4. Multi-Start Optimization with multiple grid configs
    best_centers = centers_grid
    best_radius = get_radius(best_centers)
    
    np.random.seed(42)
    inits = [centers_grid, centers_spec]
    
    # Add other grid configurations
    for config in grid_configs[1:]:
        inits.append(make_hex_grid(config))
    
    # Add jittered versions of the main grid
    for i in range(6):
        noise = np.random.normal(0, 0.008, (N, 2))
        inits.append(centers_grid + noise)
    
    # Add jittered spectral
    for i in range(3):
        noise = np.random.normal(0, 0.01, (N, 2))
        inits.append(centers_spec + noise)
    
    # Add random starts
    for i in range(2):
        inits.append(np.random.uniform(-0.4, 0.4, (N, 2)))

    bounds = [(-0.5, 0.5)] * (2 * N)

    # First pass: coarse optimization
    for x0 in inits:
        x0_flat = x0.flatten()
        x0_flat = np.clip(x0_flat, -0.5, 0.5)
        
        try:
            res = minimize(objective, x0_flat, method='L-BFGS-B', bounds=bounds, 
                           options={'maxiter': 1500, 'ftol': 1e-14, 'gtol': 1e-12})
            centers_opt = res.x.reshape(N, 2)
            centers_opt = np.clip(centers_opt, -0.5, 0.5)
            
            r = get_radius(centers_opt)
            if r > best_radius:
                best_radius = r
                best_centers = centers_opt
        except Exception:
            continue

    # Second pass: fine-tuning from best solution with higher lambda
    lam = 500.0
    def objective_fine(x):
        x = x.reshape(N, 2)
        wx = 0.5 - np.abs(x[:, 0])
        wy = 0.5 - np.abs(x[:, 1])
        walls = np.vstack([wx, wy]).flatten()
        
        d = cdist(x, x)
        np.fill_diagonal(d, np.inf)
        tri_indices = np.triu_indices(N, 1)
        d_inter = d[tri_indices]
        
        all_d = np.concatenate([d_inter, walls])
        all_d = np.clip(all_d, 1e-8, None)
        
        u = -lam * all_d
        max_u = np.max(u)
        log_sum = np.log(np.sum(np.exp(u - max_u))) + max_u
        return log_sum / lam

    # Fine-tune from best and nearby points
    fine_inits = [best_centers]
    for i in range(4):
        noise = np.random.normal(0, 0.003, (N, 2))
        fine_inits.append(best_centers + noise)
    
    for x0 in fine_inits:
        x0_flat = x0.flatten()
        x0_flat = np.clip(x0_flat, -0.5, 0.5)
        
        try:
            res = minimize(objective_fine, x0_flat, method='L-BFGS-B', bounds=bounds, 
                           options={'maxiter': 2000, 'ftol': 1e-16, 'gtol': 1e-14})
            centers_opt = res.x.reshape(N, 2)
            centers_opt = np.clip(centers_opt, -0.5, 0.5)
            
            r = get_radius(centers_opt)
            if r > best_radius:
                best_radius = r
                best_centers = centers_opt
        except Exception:
            continue

    # Final verification and cleanup
    best_centers = np.clip(best_centers, -0.5, 0.5)
    
    return best_centers
