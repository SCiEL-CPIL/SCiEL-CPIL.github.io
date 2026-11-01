import numpy as np
import time
from scipy.optimize import minimize
from scipy.spatial.distance import pdist

N = 300

def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # 5s safety margin
    
    # 1. Initialization: Hexagonal Lattice
    # We want a dense hexagonal lattice. 
    # The optimal spacing is roughly 2*r_opt. 
    # r_opt ~ 0.053, so spacing ~ 0.106.
    # Let's generate a grid and pick the best N points.
    
    def generate_hex_grid(spacing):
        # Estimate number of rows/cols needed
        # Area of unit circle = pi. Area of hex cell = spacing^2 * sqrt(3)/2.
        # N_cells ~ pi / (spacing^2 * sqrt(3)/2)
        # We need at least N points.
        # Let's just go from -1 to 1 in steps of spacing.
        points = []
        # Hex lattice: 
        # x = (i + 0.5*(j%2)) * spacing
        # y = j * spacing * sqrt(3)/2
        # We need to cover the unit circle.
        # Max j such that |y| < 1 => |j| * spacing * sqrt(3)/2 < 1 => |j| < 2/(spacing*sqrt(3))
        max_j = int(2.0 / (spacing * np.sqrt(3.0))) + 2
        max_i = int(2.0 / spacing) + 2
        
        for j in range(-max_j, max_j + 1):
            for i in range(-max_i, max_i + 1):
                x = (i + 0.5 * (j % 2)) * spacing
                y = j * spacing * np.sqrt(3.0) / 2.0
                # Check if inside unit circle (with small margin)
                if x*x + y*y <= 1.0:
                    points.append([x, y])
        return np.array(points)
    
    # Try a few spacings to find the best initial configuration
    best_init = None
    best_init_score = -np.inf
    
    # Spacing around 0.106
    spacings = [0.10, 0.105, 0.106, 0.107, 0.11, 0.115]
    
    for sp in spacings:
        pts = generate_hex_grid(sp)
        if len(pts) < N:
            continue
        
        # We want to select N points that maximize the minimum distance.
        # A simple heuristic: select points closest to the center? 
        # Or just take the first N? 
        # Taking closest to center creates a more compact cluster, which is good.
        dists_to_center = np.sqrt(pts[:, 0]**2 + pts[:, 1]**2)
        idx = np.argsort(dists_to_center)[:N]
        init_pts = pts[idx].copy()
        
        # Calculate initial score
        d = pdist(init_pts)
        min_pair = np.min(d)
        min_bound = np.min(1.0 - dists_to_center[idx])
        score = min(min_pair / 2.0, min_bound)
        
        if score > best_init_score:
            best_init_score = score
            best_init = init_pts
    
    if best_init is None:
        # Fallback: Random
        best_init = np.random.randn(N, 2)
        best_init /= np.linalg.norm(best_init, axis=1, keepdims=True)
        best_init *= 0.9

    current_points = best_init
    
    # 2. Optimization using Soft-Min and L-BFGS-B
    
    # Soft-min function:
    # S_beta(x) = -1/beta * log(sum(exp(-beta * x_i)))
    # Gradient: dS/dx_i = exp(-beta * x_i) / sum(exp(-beta * x_j))
    
    def soft_min_and_grad(x, beta):
        # x is 1D array of distances
        # Numerical stability: subtract max
        x_max = np.max(x)
        exp_neg_beta_x = np.exp(-beta * (x - x_max))
        sum_exp = np.sum(exp_neg_beta_x)
        # Soft min value
        s_val = -1.0 / beta * (np.log(sum_exp) - beta * x_max)
        # Gradient
        grad = exp_neg_beta_x / sum_exp
        return s_val, grad

    def objective_and_grad(flat_pts, beta):
        # flat_pts is 1D array of size 2*N
        pts = flat_pts.reshape(N, 2)
        
        # Calculate pairwise distances
        # Using pdist for efficiency
        d = pdist(pts)
        
        # Calculate boundary distances
        dists_to_center = np.sqrt(pts[:, 0]**2 + pts[:, 1]**2)
        d_bound = 1.0 - dists_to_center
        
        # Combine all distances
        # We want to maximize min(all_distances)
        # So we minimize -min(all_distances)
        # Using soft-min: minimize -soft_min(all_distances, beta)
        
        # Note: The "radius" r is min(d_pair/2, d_bound).
        # So we want to maximize min(d_pair/2, d_bound).
        # Let y_i be the "constraints" for radius r.
        # Constraint 1: d_pair >= 2r  => d_pair/2 >= r
        # Constraint 2: d_bound >= r
        # So we want to maximize soft_min([d_pair/2, d_bound], beta).
        
        # Let's create the vector of "effective radii"
        # d_pair is of size M. d_bound is of size N.
        # effective_radii = [d_pair/2, d_bound]
        
        eff_rad = np.concatenate([d / 2.0, d_bound])
        
        # Calculate soft min and gradient with respect to eff_rad
        s_val, grad_eff = soft_min_and_grad(eff_rad, beta)
        
        # Objective is -s_val
        obj = -s_val
        
        # Gradient with respect to points
        # grad_eff has same shape as eff_rad
        # grad_pair = grad_eff[:len(d)]
        # grad_bound = grad_eff[len(d):]
        
        grad_pair = grad_eff[:len(d)]
        grad_bound = grad_eff[len(d):]
        
        # Backpropagate gradient to points
        # d = pdist(pts)
        # We need d(d)/d(pts)
        # This is complex to do analytically for pdist.
        # Alternative: Use finite differences for gradient? Too slow.
        # Alternative: Calculate pairwise distances manually to get gradients.
        
        # Let's rewrite using broadcasting to allow gradient calculation.
        # diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
        # d_matrix = np.sqrt(np.sum(diff**2, axis=2))
        # np.fill_diagonal(d_matrix, np.inf)
        # d_flat = d_matrix[d_matrix < np.inf] # This is not exactly pdist order, but we can map it.
        
        # Actually, let's just use the matrix form and sum gradients.
        
        diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
        d_matrix = np.sqrt(np.sum(diff**2, axis=2) + 1e-12)
        np.fill_diagonal(d_matrix, np.inf)
        
        # We need to map grad_pair (from pdist) to the matrix.
        # pdist returns distances in a specific order (i < j).
        # Let's create a mask for i < j.
        mask = np.triu(np.ones((N, N), dtype=bool), k=1)
        d_flat_matrix = d_matrix[mask]
        
        # Verify order matches pdist
        # pdist order is (0,1), (0,2), ..., (0,N-1), (1,2), ...
        # np.triu mask order is row-major: (0,1)...(0,N-1), (1,2)...
        # Yes, they match.
        
        # Now, grad_eff_pair corresponds to d_flat_matrix.
        # We need to distribute grad_eff_pair back to d_matrix.
        grad_d_matrix = np.zeros((N, N))
        grad_d_matrix[mask] = grad_pair
        
        # Gradient of d_matrix w.r.t pts
        # d(i,j) = ||p_i - p_j||
        # d(d(i,j))/d(p_i) = (p_i - p_j) / d(i,j)
        # d(d(i,j))/d(p_j) = (p_j - p_i) / d(i,j) = -(p_i - p_j) / d(i,j)
        
        # grad_pts[i] = sum_j grad_d_matrix[i,j] * (p_i - p_j) / d(i,j)
        #               - sum_j grad_d_matrix[j,i] * (p_i - p_j) / d(i,j)
        # Note: grad_d_matrix is symmetric? No, grad_pair is for i<j.
        # But the force is symmetric.
        # Let's compute the force vector.
        
        # safe_d = d_matrix
        # np.fill_diagonal(safe_d, 1.0) # Avoid div by 0
        
        # grad_d_matrix is 0 on diagonal.
        # We can compute:
        # F_i = sum_j (grad_d_matrix[i,j] * (p_i - p_j) / d(i,j))
        # Since grad_d_matrix is 0 for i>=j (in the upper triangle sense, but we filled it),
        # actually grad_d_matrix[i,j] is non-zero only if i<j.
        # Wait, the gradient of the objective w.r.t d(i,j) is symmetric in the sense that
        # the term for pair (i,j) contributes to both i and j.
        
        # Let's use the property:
        # grad_obj = sum_{i<j} g_{ij} * ( d d_{ij}/dp_i + d d_{ij}/dp_j )
        # where g_{ij} is the gradient of the soft min w.r.t d_{ij}/2 (note the /2).
        # Wait, eff_rad included d/2.
        # So grad_eff_pair is d(S)/d(d/2) = 2 * d(S)/d(d).
        # Let g_d = grad_eff_pair / 2.
        
        g_d = grad_pair / 2.0
        
        # Now, for each pair (i,j) with i<j:
        # Contribution to grad p_i: g_d * (p_i - p_j) / d_ij
        # Contribution to grad p_j: g_d * (p_j - p_i) / d_ij
        
        grad_pts = np.zeros_like(pts)
        
        # We can do this vectorized:
        # diff is p_i - p_j
        # d_matrix is distance
        # We need to sum g_d * diff / d_matrix
        # But g_d is only defined for i<j.
        # Let's create a full matrix G where G[i,j] = g_d if i<j, G[j,i] = g_d if i>j.
        # Actually, the force on i from j is proportional to (p_i - p_j).
        # The force on j from i is proportional to (p_j - p_i).
        
        # Let's define a matrix W where W[i,j] = g_d_{ij} / d_ij if i!=j else 0.
        # Then grad p_i = sum_j W[i,j] * (p_i - p_j)
        # This is equivalent to: grad p_i = (sum_j W[i,j] * p_i) - (sum_j W[i,j] * p_j)
        #                     = p_i * (sum_j W[i,j]) - (W @ p)_i
        
        W = np.zeros((N, N))
        W[mask] = g_d / d_matrix[mask]
        W = W + W.T # Symmetrize? 
        # Wait.
        # Term for pair (i,j) (i<j):
        # g_d * (p_i - p_j)/d_ij added to grad p_i.
        # g_d * (p_j - p_i)/d_ij added to grad p_j.
        # In the matrix formulation grad p_i = sum_j W[i,j] (p_i - p_j):
        # If W[i,j] = g_d/d_ij and W[j,i] = g_d/d_ij:
        # grad p_i gets term W[i,j](p_i-p_j) + W[i,k](p_i-p_k)...
        # The term for j is W[i,j](p_i-p_j). Correct.
        # grad p_j gets term W[j,i](p_j-p_i). Correct.
        # So yes, W must be symmetric with W[i,j] = g_d/d_ij.
        
        W = np.zeros((N, N))
        W[mask] = g_d / d_matrix[mask]
        W = W + W.T
        
        row_sums = np.sum(W, axis=1)
        grad_pts += row_sums[:, np.newaxis] * pts - (W @ pts)
        
        # Boundary Gradient
        # eff_rad_bound = 1 - ||p_i||
        # d(eff_rad_bound)/d(p_i) = -p_i / ||p_i||
        # grad_bound is d(S)/d(eff_rad_bound)
        # So contribution to grad p_i is grad_bound[i] * (-p_i / ||p_i||)
        
        dists_to_center = np.sqrt(pts[:, 0]**2 + pts[:, 1]**2)
        # Avoid div by 0
        dists_safe = np.where(dists_to_center < 1e-9, 1e-9, dists_to_center)
        
        grad_boundary = -grad_bound[:, np.newaxis] * pts / dists_safe[:, np.newaxis]
        
        total_grad = grad_pts + grad_boundary
        
        # Flatten gradient
        flat_grad = total_grad.flatten()
        
        return obj, flat_grad

    # Optimization Loop with Annealing
    # Start with low beta, increase to high beta.
    betas = [10, 50, 100, 200, 500, 1000, 2000]
    
    for beta in betas:
        if time.time() > deadline - 10.0:
            break
            
        # Define objective and gradient for current beta
        def obj_grad(x):
            return objective_and_grad(x, beta)
        
        # Use L-BFGS-B
        # Bounds: points must be inside unit circle?
        # L-BFGS-B supports bounds.
        # x_i, y_i must satisfy x_i^2 + y_i^2 <= 1.
        # This is a non-box constraint. L-BFGS-B only supports box constraints.
        # We can use box constraints [-1, 1] for x and y.
        # The objective function will penalize points outside via the boundary distance term.
        # However, if a point goes far outside, the gradient might be unstable.
        # Let's add a hard penalty if outside?
        # Or just rely on the soft-min boundary term.
        # If 1 - ||p|| becomes negative, it reduces the min.
        
        bounds = [(-1.0, 1.0)] * (2 * N)
        
        try:
            res = minimize(
                lambda x: obj_grad(x)[0],
                current_points.flatten(),
                jac=lambda x: obj_grad(x)[1],
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 100, 'ftol': 1e-8, 'gtol': 1e-8}
            )
            current_points = res.x.reshape(N, 2)
        except Exception as e:
            pass
        
        # Check time
        if time.time() > deadline - 5.0:
            break

    # Final Score Calculation
    # Ensure points are strictly inside
    dists_to_center = np.sqrt(current_points[:, 0]**2 + current_points[:, 1]**2)
    mask_out = dists_to_center > 1.0
    if np.any(mask_out):
        current_points[mask_out] /= dists_to_center[mask_out, np.newaxis]
        current_points[mask_out] *= 0.9999 # Pull in slightly

    d = pdist(current_points)
    min_pair = np.min(d)
    min_bound = np.min(1.0 - dists_to_center)
    final_r = min(min_pair / 2.0, min_bound)
    
    return current_points

if __name__ == "__main__":
    construct()
