import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import eigsh
from scipy.optimize import minimize
from scipy.spatial.distance import cdist

def construct():
    N = 50
    
    # 1. Generate a specific hexagonal grid layout for 50 circles
    # Row counts: 8, 7, 8, 7, 8, 7, 5 = 50
    # This topology approximates the known near-optimal packing structure.
    
    points = []
    row_counts = [8, 7, 8, 7, 8, 7, 5]
    y_pos = 0.0
    for i, count in enumerate(row_counts):
        # Offset every other row by 0.5 to form a hexagonal lattice
        offset = 0.5 if i % 2 == 1 else 0.0
        for j in range(count):
            x = j + offset
            y = y_pos
            points.append([x, y])
        y_pos += np.sqrt(3) / 2
        
    centers_grid = np.array(points)
    
    # Center the grid
    centers_grid -= np.mean(centers_grid, axis=0)
    
    # 2. Build the Contact Graph (Adjacency Matrix)
    # Connect points that are nearest neighbors in the hexagonal lattice (distance ~ 1.0)
    dist = cdist(centers_grid, centers_grid)
    np.fill_diagonal(dist, np.inf)
    
    adj = np.zeros((N, N))
    tol = 1.1  # Tolerance for neighbor detection
    for i in range(N):
        neighbors = np.where(dist[i] < tol)[0]
        for j in neighbors:
            adj[i, j] = 1
            adj[j, i] = 1
            
    # 3. Spectral Embedding Initialization
    # Compute the Graph Laplacian L = D - A
    degree = adj.sum(axis=1)
    L = csr_matrix(np.diag(degree) - adj)
    
    centers_init = centers_grid # Fallback
    
    try:
        # Find the 3 smallest eigenvalues/vectors of the Laplacian
        # eigsh with sigma=0 finds eigenvalues closest to 0.
        # The first is the trivial constant vector (lambda=0).
        # The next two are the Fiedler vectors, which provide a 2D embedding.
        vals, vecs = eigsh(L, k=3, sigma=0, which='LM')
        
        # Sort by eigenvalue to ensure order
        idx = np.argsort(vals)
        vecs = vecs[:, idx]
        
        # Extract Fiedler vectors (indices 1 and 2)
        x_spec = vecs[:, 1]
        y_spec = vecs[:, 2]
        
        # Center the spectral coordinates
        x_spec -= np.mean(x_spec)
        y_spec -= np.mean(y_spec)
        
        # Scale to fit within the unit square [-0.5, 0.5]
        max_ext = np.max(np.abs(np.column_stack([x_spec, y_spec])))
        if max_ext > 0:
            scale = 0.45 / max_ext
            centers_init = np.column_stack([x_spec * scale, y_spec * scale])
            
    except Exception:
        # Fallback to scaled grid if spectral embedding fails
        max_ext = np.max(np.abs(centers_grid))
        if max_ext > 0:
            scale = 0.45 / max_ext
            centers_init = centers_grid * scale

    # 4. Optimization
    # Objective: Maximize the minimum distance between circles and to the walls.
    # We use a smooth approximation: minimize -1/lambda * log(sum(exp(-lambda * d)))
    # This is equivalent to minimizing the "soft minimum" of the distances.
    
    lam = 100.0 # High lambda for a sharp approximation of the min
    
    def objective(x):
        x = x.reshape(N, 2)
        
        # Wall distances: min(0.5 - |x|, 0.5 - |y|) for each point
        wx = 0.5 - np.abs(x[:, 0])
        wy = 0.5 - np.abs(x[:, 1])
        walls = np.vstack([wx, wy]).flatten()
        
        # Inter-point distances
        d = cdist(x, x)
        np.fill_diagonal(d, np.inf)
        tri_indices = np.triu_indices(N, 1)
        d_inter = d[tri_indices]
        
        all_d = np.concatenate([d_inter, walls])
        
        # Clip to avoid numerical issues (log of 0)
        all_d = np.clip(all_d, 1e-6, None)
        
        # LogSumExp calculation for stability
        u = -lam * all_d
        max_u = np.max(u)
        log_sum = np.log(np.sum(np.exp(u - max_u))) + max_u
        
        return log_sum / lam

    x0 = centers_init.flatten()
    bounds = [(-0.5, 0.5)] * (2 * N)
    
    # Optimize using L-BFGS-B
    res = minimize(objective, x0, method='L-BFGS-B', bounds=bounds, 
                   options={'maxiter': 2000, 'ftol': 1e-14, 'gtol': 1e-12})
    
    centers_opt = res.x.reshape(N, 2)
    
    # Final safety clip to ensure validity
    centers_opt = np.clip(centers_opt, -0.5, 0.5)
    
    return centers_opt
