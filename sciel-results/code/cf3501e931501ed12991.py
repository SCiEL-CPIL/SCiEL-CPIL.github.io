import numpy as np
import time

N = 26

def construct():
    start_time = time.time()
    deadline = 110.0  # Leave a safety margin
    
    def penalty_and_grad(x, y, r, lam):
        # x, y, r are arrays of shape (N,)
        # Compute pairwise distances
        dx = x[:, None] - x[None, :]
        dy = y[:, None] - y[None, :]
        d = np.sqrt(dx**2 + dy**2)
        np.fill_diagonal(d, np.inf)  # Avoid self-interaction
        
        # Compute radii sum for each pair
        r_sum = r[:, None] + r[None, :]
        
        # Collision penalty: max(0, r_sum - d)^2
        # Only consider pairs where i < j to avoid double counting
        triu_indices = np.triu_indices(N, k=1)
        d_pairs = d[triu_indices]
        r_sum_pairs = r_sum[triu_indices]
        
        # Soft penalty: use max(0, ...)^2
        # Gradient with respect to x_i, y_i, r_i
        # For each pair (i, j):
        # P_ij = max(0, r_i + r_j - d_ij)^2
        # dP_ij/dr_i = 2 * max(0, r_i + r_j - d_ij)
        # dP_ij/dx_i = -2 * max(0, r_i + r_j - d_ij) * (x_i - x_j) / d_ij
        # Similarly for y_i, and for j.
        
        # Initialize gradients
        grad_r = np.zeros(N)
        grad_x = np.zeros(N)
        grad_y = np.zeros(N)
        
        # Compute collision penalties and gradients
        viol = np.maximum(0, r_sum_pairs - d_pairs)
        penalty_pairs = np.sum(viol**2)
        
        # Gradients for pairs
        # dP_ij/dr_i = 2 * viol
        # dP_ij/dx_i = -2 * viol * (x_i - x_j) / d_ij
        # We need to map back to the full arrays
        # Create a matrix for the violation
        viol_mat = np.zeros((N, N))
        viol_mat[triu_indices] = viol
        viol_mat = viol_mat + viol_mat.T
        
        # Distance matrix (with inf on diagonal)
        # Avoid division by zero
        d_safe = np.where(d > 1e-10, d, 1.0)
        
        # Gradients for x and y
        # dP_ij/dx_i = -2 * viol_ij * (x_i - x_j) / d_ij
        # Sum over j for each i
        grad_x += -2 * np.sum(viol_mat * dx / d_safe, axis=1)
        grad_y += -2 * np.sum(viol_mat * dy / d_safe, axis=1)
        
        # Gradients for r
        # dP_ij/dr_i = 2 * viol_ij
        # Sum over j for each i
        grad_r += 2 * np.sum(viol_mat, axis=1)
        
        # Boundary penalties
        # P_i = max(0, r_i - x_i)^2 + max(0, r_i - (1 - x_i))^2 + max(0, r_i - y_i)^2 + max(0, r_i - (1 - y_i))^2
        viol_x_left = np.maximum(0, r - x)
        viol_x_right = np.maximum(0, r - (1 - x))
        viol_y_bottom = np.maximum(0, r - y)
        viol_y_top = np.maximum(0, r - (1 - y))
        
        penalty_boundary = np.sum(viol_x_left**2) + np.sum(viol_x_right**2) + np.sum(viol_y_bottom**2) + np.sum(viol_y_top**2)
        
        # Gradients for boundary
        grad_x += -2 * viol_x_left + 2 * viol_x_right
        grad_y += -2 * viol_y_bottom + 2 * viol_y_top
        grad_r += 2 * (viol_x_left + viol_x_right + viol_y_bottom + viol_y_top)
        
        total_penalty = penalty_pairs + penalty_boundary
        total_grad_r = grad_r
        total_grad_x = grad_x
        total_grad_y = grad_y
        
        return total_penalty, total_grad_r, total_grad_x, total_grad_y
    
    def objective(x, y, r):
        return np.sum(r)
    
    def optimize(x0, y0, r0, max_iter=2000, lr=0.01, lam=10.0):
        x = x0.copy()
        y = y0.copy()
        r = r0.copy()
        
        best_obj = -np.inf
        best_x, best_y, best_r = x.copy(), y.copy(), r.copy()
        
        for i in range(max_iter):
            if time.time() > deadline:
                break
            
            # Compute penalty and gradients
            penalty, grad_r, grad_x, grad_y = penalty_and_grad(x, y, r, lam)
            
            # Objective gradient
            obj_grad_r = np.ones(N) - lam * grad_r
            obj_grad_x = -lam * grad_x
            obj_grad_y = -lam * grad_y
            
            # Gradient ascent
            x += lr * obj_grad_x
            y += lr * obj_grad_y
            r += lr * obj_grad_r
            
            # Project to feasible region (approximate)
            # Ensure r > 0
            r = np.maximum(r, 1e-6)
            # Ensure circles are inside the square
            # x in [r, 1-r], y in [r, 1-r]
            x = np.clip(x, r, 1 - r)
            y = np.clip(y, r, 1 - r)
            
            # Update best
            obj = objective(x, y, r)
            if obj > best_obj:
                best_obj = obj
                best_x, best_y, best_r = x.copy(), y.copy(), r.copy()
                
            # Decay learning rate
            lr *= 0.999
        
        return best_x, best_y, best_r, best_obj
    
    # Multiple random starts
    best_result = None
    best_obj = -np.inf
    
    np.random.seed(42)
    
    # Try a few random initializations
    for restart in range(5):
        if time.time() > deadline:
            break
            
        # Random initial positions and radii
        x0 = np.random.uniform(0.1, 0.9, N)
        y0 = np.random.uniform(0.1, 0.9, N)
        r0 = np.full(N, 0.1)
        
        x, y, r, obj = optimize(x0, y0, r0)
        
        if obj > best_obj:
            best_obj = obj
            best_result = np.column_stack((x, y, r))
            
    # If no valid result, use the grid
    if best_result is None:
        side = int(np.ceil(np.sqrt(N)))
        cells = [((i + 0.5) / side, (j + 0.5) / side, 0.5 / side - 1e-3) for i in range(side) for j in range(side)]
        best_result = np.array(cells[:N])
        
    return best_result
