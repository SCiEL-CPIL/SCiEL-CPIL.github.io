import numpy as np
from scipy.optimize import minimize

def construct():
    N = 200
    
    # 1. Generate Initial Hexagonal Grid
    # Approximate spacing based on area
    s = np.sqrt(2 / (np.sqrt(3) * N))
    
    points = []
    # Generate enough rows to get at least N points
    # In a hex grid, average points per row is roughly N / n_rows
    # Let's just generate a large grid and filter
    # Or generate row by row
    row = 0
    while len(points) < N:
        y = row * s * np.sqrt(3) / 2
        # Offset for even/odd rows
        offset = (row % 2) * s / 2
        # Generate x points
        # We need to cover the square [-0.5, 0.5]
        # Let's generate from -1 to 1 and filter
        x_vals = np.arange(-1, 1, s) + offset
        for x in x_vals:
            points.append((x, y))
            if len(points) >= N:
                break
        row += 1
        
    # Take first N points
    points = np.array(points[:N])
    
    # Center the points
    center = points.mean(axis=0)
    points -= center
    
    # Scale/Clip to ensure they are roughly within the square
    # The hex grid might be slightly larger or smaller.
    # L-BFGS-B will handle the bounds, but starting inside is better.
    # Find the max extent
    max_extent = np.max(np.abs(points))
    if max_extent > 0.5:
        points *= 0.5 / max_extent
        
    x0 = points.flatten()
    
    # 2. Define Objective and Gradient
    def energy_and_grad(x_flat):
        x = x_flat.reshape(N, 2)
        
        # Pairwise distances
        # diff[i, j] = x[i] - x[j]
        diff = x[:, np.newaxis, :] - x[np.newaxis, :, :]
        dist_sq = np.sum(diff**2, axis=2)
        
        # Mask diagonal
        np.fill_diagonal(dist_sq, np.inf)
        
        # Inverse squared distances
        # Add small epsilon to avoid division by zero
        inv_dist_sq = 1.0 / (dist_sq + 1e-12)
        
        # Energy from pairs
        E_pair = np.sum(inv_dist_sq) / 2.0  # Divide by 2 because i<j and j<i
        
        # Gradient from pairs
        # d/dx_i (1/d^2) = -2 * (x_i - x_j) / d^4
        # grad_pair[i] = sum_j -2 * (x_i - x_j) * (1/d^2)^2
        # Note: inv_dist_sq is 1/d^2. So (1/d^2)^2 is inv_dist_sq**2.
        # diff is x_i - x_j.
        # We need to sum over j for each i.
        # term = diff * (inv_dist_sq**2)[:,:,np.newaxis]
        # grad_pair = -2 * np.sum(term, axis=1)
        
        # But we must be careful with the mask.
        # inv_dist_sq has inf on diagonal. 1/inf = 0. So it's fine.
        term = diff * (inv_dist_sq**2)[:, :, np.newaxis]
        grad_pair = -2.0 * np.sum(term, axis=1)
        
        # Boundary distances
        # Boundaries are x = -0.5, 0.5 and y = -0.5, 0.5
        # d_left = x + 0.5
        # d_right = 0.5 - x
        # d_bottom = y + 0.5
        # d_top = 0.5 - y
        
        d_left = x[:, 0] + 0.5
        d_right = 0.5 - x[:, 0]
        d_bottom = x[:, 1] + 0.5
        d_top = 0.5 - x[:, 1]
        
        # Ensure positive (they should be if inside, but clip for safety in calculation)
        d_left = np.maximum(d_left, 1e-6)
        d_right = np.maximum(d_right, 1e-6)
        d_bottom = np.maximum(d_bottom, 1e-6)
        d_top = np.maximum(d_top, 1e-6)
        
        inv_d_left_sq = 1.0 / (d_left**2)
        inv_d_right_sq = 1.0 / (d_right**2)
        inv_d_bottom_sq = 1.0 / (d_bottom**2)
        inv_d_top_sq = 1.0 / (d_top**2)
        
        E_bound = np.sum(inv_d_left_sq) + np.sum(inv_d_right_sq) + np.sum(inv_d_bottom_sq) + np.sum(inv_d_top_sq)
        
        E_total = E_pair + E_bound
        
        # Gradient from boundaries
        # d/dx (1/d^2) = -2/d^3 * (1 or -1)
        # d/dx_left (1/(x+0.5)^2) = -2/(x+0.5)^3
        # d/dx_right (1/(0.5-x)^2) = -2/(0.5-x)^3 * (-1) = 2/(0.5-x)^3
        
        grad_bound = np.zeros((N, 2))
        grad_bound[:, 0] += -2.0 / (d_left**3)
        grad_bound[:, 0] += 2.0 / (d_right**3)
        grad_bound[:, 1] += -2.0 / (d_bottom**3)
        grad_bound[:, 1] += 2.0 / (d_top**3)
        
        grad_total = grad_pair + grad_bound
        
        return E_total, grad_total.flatten()

    # 3. Optimize
    bounds = [(-0.5, 0.5)] * (2 * N)
    
    res = minimize(
        lambda x: energy_and_grad(x)[0],
        x0,
        jac=lambda x: energy_and_grad(x)[1],
        method='L-BFGS-B',
        bounds=bounds,
        options={'maxiter': 2000, 'ftol': 1e-12, 'gtol': 1e-8}
    )
    
    return res.x.reshape(N, 2)
