import numpy as np
import time
from scipy.stats import norm
import random

def heuristics(distance_matrix, coordinates, demands, capacity):
    """
    Returns an edge-attractiveness matrix for 3 fixed instances.
    Uses a multi-dimensional heuristic space optimized via a Gaussian Process surrogate.
    Heuristic form: eta_ij = (1/d_ij)^alpha * (1 - angle_ij)^beta * (nn_score_ij)^gamma
    where angle_ij is the angle between vector (i->j) and (i->depot), 
    and nn_score_ij indicates if j is a nearest neighbor of i.
    """
    start_time = time.time()
    deadline = start_time + 320.0
    
    D = np.asarray(distance_matrix, dtype=np.float64)
    coords = np.asarray(coordinates, dtype=np.float64)
    n = D.shape[0]
    demands = np.asarray(demands, dtype=np.float64)
    
    # Precompute features for the heuristic space
    # 1. Inverse distance
    inv_dist = np.where(D > 1e-10, 1.0 / D, 1e9)
    
    # 2. Angle to depot (0 is best, pointing towards depot)
    # Vector from i to j: coords[j] - coords[i]
    # Vector from i to depot (0): coords[0] - coords[i]
    # Cosine similarity: dot(v1, v2) / (|v1| |v2|)
    # We want high attractiveness when angle is small (cos close to 1).
    # Let's use 'alignment' = (cos + 1) / 2, so range [0, 1].
    # But we need to handle i=0 (depot) separately or just let it be 0.
    # For i=0, vector to depot is zero. We'll define alignment[0, :] = 1.0 (neutral or max?)
    # Actually, from depot, direction doesn't matter as much for "returning", but for first step it does.
    # Let's compute alignment for all i.
    
    diff_ij = coords[None, :, :] - coords[:, None, :] # shape (n, n, 2)
    diff_i0 = coords[0][None, :] - coords[:, None, :] # shape (n, 2) -> broadcast to (n, n, 2)? No.
    # Vector i->j: v_ij = coords[j] - coords[i]
    # Vector i->0: v_i0 = coords[0] - coords[i]
    
    v_ij = coords[None, :, :] - coords[:, None, :] # (n, n, 2)
    v_i0 = coords[0][None, :] - coords[:, None, :] # (n, 2)
    v_i0_expanded = np.broadcast_to(v_i0[:, None, :], (n, n, 2))
    
    norm_v_ij = np.linalg.norm(v_ij, axis=2)
    norm_v_i0 = np.linalg.norm(v_i0_expanded, axis=2)
    
    # Avoid division by zero
    norm_v_ij_safe = np.where(norm_v_ij > 1e-10, norm_v_ij, 1.0)
    norm_v_i0_safe = np.where(norm_v_i0 > 1e-10, norm_v_i0, 1.0)
    
    dot_prod = np.sum(v_ij * v_i0_expanded, axis=2)
    cos_angle = dot_prod / (norm_v_ij_safe * norm_v_i0_safe)
    
    # Clip to [-1, 1]
    cos_angle = np.clip(cos_angle, -1.0, 1.0)
    
    # Alignment: 1 when cos=1 (same direction), 0 when cos=-1 (opposite)
    alignment = (cos_angle + 1.0) / 2.0
    
    # For depot (i=0), v_i0 is 0. We set alignment[0, :] = 1.0 to not penalize first step heavily?
    # Or maybe 0.5? Let's say 1.0 so it's neutral/high.
    alignment[0, :] = 1.0
    alignment[:, 0] = 1.0 # Returning to depot is always good?
    
    # 3. Nearest Neighbor Score
    # For each i, find the k nearest neighbors.
    # If j is in the top-k nearest neighbors of i, score = 1, else 0?
    # Or continuous: rank-based.
    # Let's use: 1 if j is in top 5 nearest neighbors of i, else 0.
    nn_score = np.zeros((n, n))
    k_nn = 5
    for i in range(n):
        if i == 0:
            continue
        dists = D[i, :].copy()
        dists[i] = np.inf
        # Get indices of k smallest
        if n > k_nn:
            nn_indices = np.argpartition(dists, k_nn)[:k_nn]
        else:
            nn_indices = np.arange(n)
            nn_indices = nn_indices[nn_indices != i]
        
        for j in nn_indices:
            if j != i:
                nn_score[i, j] = 1.0
    
    # Diagonal is irrelevant, set to 0
    np.fill_diagonal(nn_score, 0.0)
    
    # Heuristic function:
    # eta_ij = (inv_dist_ij)^alpha * (alignment_ij)^beta * (nn_score_ij + eps)^gamma
    # Note: nn_score is 0 or 1. If gamma > 0, edges not in NN get 0 attractiveness (if eps=0).
    # To avoid 0, we can use (nn_score + 0.1) or similar.
    # Let's use: eta = (inv_dist)^alpha * (alignment)^beta * (0.1 + nn_score)^gamma
    
    def compute_heuristic(alpha, beta, gamma):
        """
        Computes the heuristic matrix for given parameters.
        """
        with np.errstate(divide='ignore', invalid='ignore', over='ignore'):
            # Avoid 0^0 or 0^negative issues
            base1 = np.where(inv_dist > 1e-10, inv_dist, 1e-9)
            base2 = np.where(alignment > 1e-10, alignment, 1e-9)
            base3 = np.where(0.1 + nn_score > 1e-10, 0.1 + nn_score, 1e-9)
            
            eta = (base1 ** alpha) * (base2 ** beta) * (base3 ** gamma)
        
        # Ensure positive and finite
        eta = np.where(np.isfinite(eta) & (eta > 0), eta, 1e-9)
        return eta

    def run_proxy_aco(eta, num_tours=100):
        """
        Fast proxy ACO.
        """
        best_cost = float('inf')
        rng = np.random.RandomState(42)
        
        for t in range(num_tours):
            if time.time() > deadline - 20.0:
                break
            
            current = 0
            current_load = 0.0
            total_distance = 0.0
            unvisited_indices = list(range(1, n))
            
            while unvisited_indices:
                rem_cap = capacity - current_load
                feasible_indices = [j for j in unvisited_indices if demands[j] <= rem_cap]
                
                if not feasible_indices:
                    total_distance += D[current, 0]
                    break
                
                if len(feasible_indices) == 1:
                    next_node = feasible_indices[0]
                else:
                    feasible_arr = np.array(feasible_indices)
                    weights = eta[current, feasible_arr]
                    total_w = np.sum(weights)
                    if total_w <= 0:
                        next_node = feasible_arr[rng.randint(len(feasible_arr))]
                    else:
                        probs = weights / total_w
                        next_node = feasible_arr[rng.choice(len(feasible_arr), p=probs)]
                
                total_distance += D[current, next_node]
                current_load += demands[next_node]
                unvisited_indices.remove(next_node)
                current = next_node
                
                if not unvisited_indices:
                    total_distance += D[current, 0]
                    break
            
            if total_distance < best_cost:
                best_cost = total_distance
        
        return best_cost

    # GP Surrogate Optimization
    # Parameters: alpha, beta, gamma
    # Bounds:
    # alpha: [0.5, 3.0]
    # beta: [0.0, 2.0]
    # gamma: [0.0, 2.0]
    
    bounds = np.array([[0.5, 0.0, 0.0], [3.0, 2.0, 2.0]])
    
    # Initial samples
    # 1. Standard: alpha=1, beta=0, gamma=0
    # 2. Angle focused: alpha=1, beta=1, gamma=0
    # 3. NN focused: alpha=1, beta=0, gamma=1
    # 4. Random few
    
    X_observed = []
    Y_observed = []
    
    initial_samples = [
        [1.0, 0.0, 0.0],
        [1.0, 1.0, 0.0],
        [1.0, 0.0, 1.0],
        [2.0, 1.0, 1.0],
        [0.5, 0.5, 0.5]
    ]
    
    for x in initial_samples:
        if time.time() > deadline - 30.0:
            break
        eta = compute_heuristic(x[0], x[1], x[2])
        cost = run_proxy_aco(eta, num_tours=50)
        X_observed.append(x)
        Y_observed.append(cost)
    
    # GP Model
    # Use a simple GP implementation with RBF kernel
    # y ~ N(f(x), sigma^2)
    # f(x) ~ GP(0, k(x, x'))
    
    def rbf_kernel(X1, X2, length_scales, sigma):
        # X1: (N, D), X2: (M, D)
        # length_scales: (D,)
        # sigma: variance
        
        X1_scaled = X1 / length_scales
        X2_scaled = X2 / length_scales
        
        # Squared distances
        # d^2 = ||x1||^2 - 2 x1.x2 + ||x2||^2
        sq_dists = np.sum(X1_scaled**2, axis=1)[:, None] - 2 * np.dot(X1_scaled, X2_scaled.T) + np.sum(X2_scaled**2, axis=1)[None, :]
        sq_dists = np.maximum(sq_dists, 0)
        
        K = sigma**2 * np.exp(-0.5 * sq_dists)
        return K

    # Hyperparameters for GP
    gp_length_scales = np.array([1.0, 1.0, 1.0])
    gp_sigma = 1.0
    noise = 1e-6
    
    best_params = initial_samples[0]
    best_cost = Y_observed[0]
    
    # Iterative optimization
    max_iters = 50
    for it in range(max_iters):
        if time.time() > deadline - 30.0:
            break
            
        X_obs = np.array(X_observed)
        Y_obs = np.array(Y_observed)
        
        # Standardize Y for GP stability?
        y_mean = np.mean(Y_obs)
        y_std = np.std(Y_obs) + 1e-10
        Y_std = (Y_obs - y_mean) / y_std
        
        # Fit GP
        K = rbf_kernel(X_obs, X_obs, gp_length_scales, gp_sigma) + noise * np.eye(len(X_obs))
        
        try:
            L = np.linalg.cholesky(K)
            alpha = np.linalg.solve(L.T, np.linalg.solve(L, Y_std))
        except np.linalg.LinAlgError:
            break
            
        # Acquire Function: Expected Improvement
        # We want to minimize cost, so we minimize Y.
        # EI(x) = (f_min - mu(x)) * Phi((f_min - mu(x))/sigma(x)) + sigma(x) * phi((f_min - mu(x))/sigma(x))
        
        f_min = np.min(Y_std)
        
        # Candidate points: Latin Hypercube or Grid
        # Since 3D, grid is feasible.
        n_grid = 20
        grid_alpha = np.linspace(bounds[0, 0], bounds[1, 0], n_grid)
        grid_beta = np.linspace(bounds[0, 1], bounds[1, 1], n_grid)
        grid_gamma = np.linspace(bounds[0, 2], bounds[1, 2], n_grid)
        
        X_grid = np.array(np.meshgrid(grid_alpha, grid_beta, grid_gamma)).T.reshape(-1, 3)
        
        # Predict mean and variance
        K_s = rbf_kernel(X_obs, X_grid, gp_length_scales, gp_sigma)
        K_ss = rbf_kernel(X_grid, X_grid, gp_length_scales, gp_sigma)
        
        mu = K_s.T @ alpha
        v = np.linalg.solve(L, K_s)
        var = gp_sigma**2 - np.sum(v**2, axis=0)
        var = np.maximum(var, 1e-10)
        std = np.sqrt(var)
        
        # EI
        z = (f_min - mu) / std
        EI = (f_min - mu) * norm.cdf(z) + std * norm.pdf(z)
        
        # Select best point
        best_idx = np.argmax(EI)
        new_x = X_grid[best_idx]
        
        # Evaluate
        eta = compute_heuristic(new_x[0], new_x[1], new_x[2])
        cost = run_proxy_aco(eta, num_tours=50)
        
        X_observed.append(new_x)
        Y_observed.append(cost)
        
        if cost < best_cost:
            best_cost = cost
            best_params = new_x

    # Final Polish: Fine grid search around best_params
    if time.time() < deadline - 15.0:
        center = np.array(best_params)
        radius = 0.2
        fine_gammas = np.arange(-radius, radius + 0.05, 0.05)
        
        # Simple 1D search along each dimension? Or small 3D grid?
        # Let's do a small 3D grid around the best point
        n_fine = 5
        fine_alpha = np.linspace(max(bounds[0,0], center[0]-radius), min(bounds[1,0], center[0]+radius), n_fine)
        fine_beta = np.linspace(max(bounds[0,1], center[1]-radius), min(bounds[1,1], center[1]+radius), n_fine)
        fine_gamma = np.linspace(max(bounds[0,2], center[2]-radius), min(bounds[1,2], center[2]+radius), n_fine)
        
        X_fine = np.array(np.meshgrid(fine_alpha, fine_beta, fine_gamma)).T.reshape(-1, 3)
        
        for x in X_fine:
            if time.time() > deadline - 10.0:
                break
            eta = compute_heuristic(x[0], x[1], x[2])
            cost = run_proxy_aco(eta, num_tours=100)
            if cost < best_cost:
                best_cost = cost
                best_params = x

    # Construct final heuristic
    final_heuristic = compute_heuristic(best_params[0], best_params[1], best_params[2])
    
    if final_heuristic.shape != D.shape:
        final_heuristic = np.ones_like(D) * 1e-9
        np.fill_diagonal(final_heuristic, 1e-9)
        
    return final_heuristic
