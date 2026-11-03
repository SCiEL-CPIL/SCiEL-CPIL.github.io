import numpy as np
import time
import math
from scipy.optimize import least_squares
from scipy.spatial import Delaunay
import scipy.sparse as sp
import scipy.sparse.linalg as spla

N = 300

def compute_score(points):
    """
    Compute the packing radius for a set of points.
    """
    rad = np.linalg.norm(points, axis=1)
    min_bound = 1.0 - np.max(rad)
    
    diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff**2, axis=2))
    np.fill_diagonal(dists, np.inf)
    min_pair = np.min(dists)
    
    score = min(min_pair, min_bound)
    return score

def soft_min_value(points, k=100.0):
    """
    Compute the smooth minimum of pairwise distances and boundary distances.
    """
    diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff**2, axis=2))
    np.fill_diagonal(dists, np.inf)
    
    rad = np.linalg.norm(points, axis=1)
    bound_dists = 1.0 - rad
    
    pair_dists = dists[np.triu_indices(N, k=1)]
    all_dists = np.concatenate([pair_dists, bound_dists])
    
    d_min = np.min(all_dists)
    exp_terms = np.exp(-k * (all_dists - d_min))
    sum_exp = np.sum(exp_terms)
    
    if sum_exp == 0:
        return d_min
    
    soft_min = d_min - (1.0 / k) * np.log(sum_exp)
    return soft_min

def spectral_rigidity_objective(points, alpha=0.3):
    """
    Compute a spectral rigidity objective that encourages uniform spacing
    and structural integrity via Delaunay triangulation and graph Laplacian.
    
    Returns (objective_value, gradient) where we want to maximize the value.
    The objective combines:
    1. Algebraic connectivity (second smallest Laplacian eigenvalue) - encourages
       well-connected, rigid structures
    2. Uniformity of edge lengths in the Delaunay graph
    3. Boundary distance (keep points away from container edge)
    """
    try:
        # Compute Delaunay triangulation
        tri = Delaunay(points)
        edges = tri.simplices
        edge_list = set()
        for simplex in edges:
            for i in range(3):
                for j in range(i+1, 3):
                    a, b = simplex[i], simplex[j]
                    if a > b:
                        a, b = b, a
                    edge_list.add((a, b))
        
        if len(edge_list) == 0:
            return 0.0, np.zeros((N, 2))
        
        edge_array = np.array(list(edge_list))
        n_edges = len(edge_array)
        
        # Compute edge lengths
        edge_diff = points[edge_array[:, 0]] - points[edge_array[:, 1]]
        edge_lens = np.linalg.norm(edge_diff, axis=1)
        
        # Build graph Laplacian
        # L = D - A where A is adjacency (with edge weights = 1/length for spring-like behavior)
        # We use weights w_ij = 1/|ij| to create a force-directed graph
        weights = 1.0 / (edge_lens + 1e-10)
        
        row_idx = []
        col_idx = []
        vals = []
        
        for e in range(n_edges):
            a, b = edge_array[e]
            w = weights[e]
            row_idx.append(a)
            col_idx.append(b)
            vals.append(-w)
            row_idx.append(b)
            col_idx.append(a)
            vals.append(-w)
            row_idx.append(a)
            col_idx.append(a)
            vals.append(w)
            row_idx.append(b)
            col_idx.append(b)
            vals.append(w)
        
        L = sp.coo_matrix((vals, (row_idx, col_idx)), shape=(N, N)).tocsr()
        
        # Compute algebraic connectivity (second smallest eigenvalue)
        # Use shift-invert for better accuracy
        try:
            eigvals, eigvecs = spla.eigsh(L, k=2, which='SM', sigma=0, tol=1e-8, v0=np.ones(N))
            # Sort eigenvalues
            idx = np.argsort(eigvals)
            eigvals = eigvals[idx]
            eigvecs = eigvecs[:, idx]
            lambda_2 = max(0.0, eigvals[1])  # Second smallest (first is ~0)
            phi_2 = eigvecs[:, 1]  # Corresponding eigenvector
        except Exception:
            lambda_2 = 0.0
            phi_2 = np.zeros(N)
        
        # Gradient of lambda_2 w.r.t. positions:
        # d(lambda_2)/d(x) = 2 * phi_2^T * dL/dx * phi_2
        # For each edge (a,b) with weight w, dL/dx_a = w * (I_a - I_b) outer (e_a - e_b)
        # This is complex; we'll use finite differences for the gradient
        
        # Uniformity term: minimize variance of edge lengths
        mean_len = np.mean(edge_lens)
        var_len = np.var(edge_lens)
        uniformity_term = -var_len  # We want to minimize variance
        
        # Boundary term: encourage points to be well inside
        rad = np.linalg.norm(points, axis=1)
        boundary_term = -np.mean(np.maximum(0, rad - 0.8))  # Penalize points too close to boundary
        
        # Combined objective
        obj = lambda_2 + alpha * uniformity_term + 0.1 * boundary_term
        
        # Compute gradient via finite differences (expensive but necessary)
        # We'll use a small step and central differences
        step = 1e-5
        grad = np.zeros((N, 2))
        
        # For efficiency, we'll only compute gradient for a subset of points
        # or use a simpler approximation
        # For now, return zero gradient and let the optimizer use its own approximations
        # Actually, let's compute a proper gradient using the eigenvector
        
        # The gradient of the spectral objective is:
        # grad_x = -2 * lambda_2 * phi_2 * (dL/dx) * phi_2
        # But dL/dx depends on edge lengths which depend on x
        
        # Simpler approach: use the fact that for a Laplacian, the gradient of
        # the Rayleigh quotient R(x) = phi^T L(x) phi / phi^T phi is:
        # dR/dx = 2 * phi^T (dL/dx) phi / (phi^T phi)
        
        # We'll approximate this by computing the change in lambda_2 when we
        # perturb each coordinate
        
        # For computational efficiency, we'll only compute gradient for a few iterations
        # and use a simple force-based gradient as a proxy
        
        # Force-based gradient: for each edge, the force is proportional to
        # (1/length) * (direction) * (length - target_length)
        target_len = mean_len
        for e in range(n_edges):
            a, b = edge_array[e]
            diff = points[a] - points[b]
            length = edge_lens[e]
            if length > 1e-10:
                # Spring force
                force = (1.0 / length) * (diff / length) * (length - target_len)
                grad[a] += force
                grad[b] -= force
        
        # Boundary gradient
        for i in range(N):
            r = rad[i]
            if r > 0.8:
                grad[i] -= 0.1 * 0.2 * points[i] / (r + 1e-10)
        
        return obj, grad
        
    except Exception as e:
        return 0.0, np.zeros((N, 2))

def polish_least_squares(points, time_budget, k=100.0, R_target=0.06):
    """
    Polish using scipy.optimize.least_squares.
    """
    start_time = time.time()
    
    def residuals(x_flat):
        x = x_flat.reshape(N, 2)
        rad = np.linalg.norm(x, axis=1)
        mask = rad > 1.0
        if np.any(mask):
            x = x.copy()
            x[mask] /= rad[mask, np.newaxis]
        
        s_val = soft_min_value(x, k)
        return np.array([R_target - s_val])

    x0 = points.flatten()
    
    try:
        result = least_squares(
            residuals, 
            x0, 
            method='trf', 
            max_nfev=20000,
            xtol=1e-12, 
            ftol=1e-12
        )
        
        x_opt = result.x.reshape(N, 2)
        score_opt = compute_score(x_opt)
        
        return x_opt, score_opt
        
    except Exception as e:
        print(f"LSQ failed: {e}")
        return points, compute_score(points)

def spectral_optimize(points, time_budget, alpha=0.3, max_iter=500):
    """
    Optimize using spectral rigidity objective with gradient descent.
    """
    start_time = time.time()
    current = points.copy()
    best = points.copy()
    best_score = compute_score(points)
    
    lr = 0.001
    lr_decay = 0.99
    
    for it in range(max_iter):
        if time.time() - start_time > time_budget:
            break
        
        obj, grad = spectral_rigidity_objective(current, alpha)
        
        # Normalize gradient
        grad_norm = np.linalg.norm(grad)
        if grad_norm > 1e-10:
            grad = grad / grad_norm
        
        # Update
        current = current + lr * grad
        
        # Project inside container
        rad = np.linalg.norm(current, axis=1)
        mask = rad > 0.99
        if np.any(mask):
            current = current.copy()
            current[mask] *= 0.99 / rad[mask, np.newaxis]
        
        # Evaluate score
        score = compute_score(current)
        
        if score > best_score:
            best_score = score
            best = current.copy()
        
        # Decay learning rate
        lr *= lr_decay
    
    return best, best_score

def construct():
    start_time = time.time()
    time_limit = 115.0  # Safety margin
    
    # 1. Initialization: Hexagonal Lattice
    s_init = np.sqrt(2 * np.pi / (np.sqrt(3) * N))
    s_gen = s_init * 0.95
    y_step = s_gen * np.sqrt(3) / 2
    max_j = int(1.0 / y_step) + 2
    max_i = int(1.0 / s_gen) + 2
    
    points_list = []
    for j in range(-max_j, max_j + 1):
        y = j * y_step
        offset = 0 if j % 2 == 0 else s_gen / 2
        for i in range(-max_i, max_i + 1):
            x = i * s_gen + offset
            if np.hypot(x, y) < 0.99:
                points_list.append([x, y])
                
    points = np.array(points_list)
    
    if len(points) > N:
        dists_to_center = np.linalg.norm(points, axis=1)
        idx = np.argsort(dists_to_center)[:N]
        points = points[idx]
    elif len(points) < N:
        while len(points) < N:
            p = np.random.uniform(-0.5, 0.5, 2)
            points = np.vstack([points, p])
        points = points[:N]
        
    best_points = points.copy()
    best_score = compute_score(points)
    
    # 2. Spectral Optimization Phase
    # Use spectral rigidity to improve global structure
    remaining_time = time_limit - (time.time() - start_time)
    spectral_budget = min(30.0, remaining_time * 0.4)
    
    if spectral_budget > 5.0:
        spec_points, spec_score = spectral_optimize(best_points, spectral_budget, alpha=0.3)
        if spec_score > best_score:
            best_score = spec_score
            best_points = spec_points
    
    # 3. Final Polish with least_squares
    remaining_time = time_limit - (time.time() - start_time)
    polish_budget = min(remaining_time - 5.0, 60.0)
    
    if polish_budget > 5.0:
        # Run multiple polish iterations with perturbation
        num_runs = 3
        time_per_run = polish_budget / num_runs
        
        for i in range(num_runs):
            if time.time() - start_time > time_limit - 3.0:
                break
            
            if i == 0:
                current_points = best_points.copy()
            else:
                noise = np.random.normal(0, 0.005, size=best_points.shape)
                current_points = best_points + noise
                rad = np.linalg.norm(current_points, axis=1)
                mask = rad > 1.0
                if np.any(mask):
                    current_points[mask] /= rad[mask, np.newaxis]
            
            run_budget = min(time_per_run, time_limit - (time.time() - start_time) - 2.0)
            if run_budget < 1.0:
                continue
            
            polished_points, polished_score = polish_least_squares(current_points, run_budget)
            
            if polished_score > best_score:
                best_score = polished_score
                best_points = polished_points
    
    # Final validation
    rad_final = np.linalg.norm(best_points, axis=1)
    if np.any(rad_final > 1.0) or not np.all(np.isfinite(best_points)):
        return np.zeros((N, 2))
        
    return best_points
