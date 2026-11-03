"""Improved program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 50

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize

def construct():
    """
    Uses Gradient-Based Local Search (BFGS) starting from a hexagonal lattice.
    This approach is distinct from the stochastic Simulated Annealing in the seed.
    It minimizes a repulsive energy function to maximize the minimum distance.
    """
    
    def hex_grid(n):
        """Generate a hexagonal grid of points centered in the square."""
        # Hexagonal packing has density ~0.9069
        # We need to fit n points.
        # Approximate radius of packing circle area: sqrt(n * pi / (0.9069 * 1^2)) / 2 ? No.
        # Let's just generate a dense hex grid and pick the center n points.
        
        # Hex grid parameters
        # Distance between centers in hex grid: d
        # Area per point: d^2 * sqrt(3)/2
        # Total area ~ 1. So d ~ sqrt(2 / (n * sqrt(3)))
        d = np.sqrt(2.0 / (n * np.sqrt(3.0)))
        
        # Generate grid
        rows = int(np.ceil(1.0 / (d * np.sqrt(3)/2))) + 2
        cols = int(np.ceil(1.0 / d)) + 2
        
        points = []
        for i in range(-rows, rows + 1):
            for j in range(-cols, cols + 1):
                x = j * d + (i % 2) * d / 2.0
                y = i * d * np.sqrt(3.0) / 2.0
                # Center the grid
                # We will collect all points and then select the best n
                points.append([x, y])
                
        points = np.array(points)
        
        # Center the grid around (0,0)
        cx = np.mean(points[:, 0])
        cy = np.mean(points[:, 1])
        points -= np.array([cx, cy])
        
        # Select the n points closest to the center (0,0) to ensure they are inside
        dists = np.sum(points**2, axis=1)
        idx = np.argsort(dists)
        points = points[idx[:n]]
        
        # Ensure they are strictly inside [-0.5, 0.5]
        # If some are outside, we might need to scale or shift.
        # For now, clip and hope BFGS fixes it. Or better, scale.
        max_coord = np.max(np.abs(points))
        if max_coord > 0.49:
            scale = 0.49 / max_coord
            points *= scale
            
        return points

    def energy(pts):
        """
        Repulsive energy: Sum of inverse 4th power distances.
        Includes boundary penalties.
        """
        pts = np.asarray(pts).reshape(-1, 2)
        
        # Pairwise distances
        diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
        dists_sq = np.sum(diff ** 2, axis=2)
        # Avoid division by zero on diagonal
        np.fill_diagonal(dists_sq, 1.0)
        
        # Use upper triangle to avoid double counting
        i_upper = np.triu_indices(N, k=1)
        pair_dists_sq = dists_sq[i_upper]
        
        # 1/d^4 = 1/(d^2)^2
        pair_energy = np.sum(1.0 / (pair_dists_sq ** 2))
        
        # Boundary distances
        # Distance to walls
        d_left = pts[:, 0] + 0.5
        d_right = 0.5 - pts[:, 0]
        d_bottom = pts[:, 1] + 0.5
        d_top = 0.5 - pts[:, 1]
        
        # Clip to avoid inf/nan
        eps = 1e-6
        d_left = np.clip(d_left, eps, None)
        d_right = np.clip(d_right, eps, None)
        d_bottom = np.clip(d_bottom, eps, None)
        d_top = np.clip(d_top, eps, None)
        
        bound_dists_sq = np.concatenate([d_left**2, d_right**2, d_bottom**2, d_top**2])
        bound_energy = np.sum(1.0 / (bound_dists_sq ** 2))
        
        # Weight boundary energy to keep circles well inside
        return pair_energy + 5.0 * bound_energy

    def gradient(pts):
        """
        Gradient of the energy function.
        """
        pts = np.asarray(pts).reshape(-1, 2)
        grad = np.zeros_like(pts)
        
        # Pairwise gradient
        diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
        dists_sq = np.sum(diff ** 2, axis=2)
        np.fill_diagonal(dists_sq, 1.0)
        
        # d(1/d^4)/d(x_i) = -4/d^6 * (x_i - x_j) * 2? No.
        # Let E_pair = sum 1/d^4 = sum d^-4
        # dE/dx_i = sum_j -4 d^-5 * (d/dd_ij d_ij)
        # d_ij = ||p_i - p_j||
        # d d_ij / d p_i = (p_i - p_j) / d_ij
        # So dE/dp_i = sum_j -4 d_ij^-5 * (p_i - p_j) / d_ij = sum_j -4 d_ij^-6 * (p_i - p_j)
        
        # d_ij^-6 = 1/(d_ij^2)^3
        inv_d6 = 1.0 / (dists_sq ** 3)
        
        # Sum over j for each i
        # grad_i = -4 * sum_j inv_d6[i,j] * (p_i - p_j)
        # Note: diff[i,j] = p_i - p_j
        # We need sum_j inv_d6[i,j] * diff[i,j]
        
        # Using matrix multiplication:
        # S = inv_d6 @ diff ? No, inv_d6 is (N,N), diff is (N,N,2).
        # We need to multiply element-wise and sum over axis 1 (j).
        # term = inv_d6[:,:,np.newaxis] * diff
        # grad_pair = -4.0 * np.sum(term, axis=1)
        
        term = inv_d6[:, :, np.newaxis] * diff
        grad_pair = -4.0 * np.sum(term, axis=1)
        
        # Boundary gradient
        # E_bound = sum 1/d^4
        # d(1/d^4)/d(p_x) for left wall (d = x + 0.5): -4 (x+0.5)^-5
        # d(1/d^4)/d(p_x) for right wall (d = 0.5 - x): -4 (0.5-x)^-5 * (-1) = 4 (0.5-x)^-5
        
        eps = 1e-6
        d_left = np.clip(pts[:, 0] + 0.5, eps, None)
        d_right = np.clip(0.5 - pts[:, 0], eps, None)
        d_bottom = np.clip(pts[:, 1] + 0.5, eps, None)
        d_top = np.clip(0.5 - pts[:, 1], eps, None)
        
        # Gradients for x
        grad_x_left = -4.0 * d_left ** -5
        grad_x_right = 4.0 * d_right ** -5
        grad_x = grad_x_left + grad_x_right
        
        # Gradients for y
        grad_y_bottom = -4.0 * d_bottom ** -5
        grad_y_top = 4.0 * d_top ** -5
        grad_y = grad_y_bottom + grad_y_top
        
        grad[:, 0] += 5.0 * grad_x
        grad[:, 1] += 5.0 * grad_y
        
        return grad + grad_pair

    # Initial points
    points = hex_grid(N)
    
    # Optimize
    res = minimize(energy, points, jac=gradient, method='BFGS', 
                   options={'maxiter': 500, 'gtol': 1e-8})
    
    return res.x

# EVOLVE-BLOCK-END
