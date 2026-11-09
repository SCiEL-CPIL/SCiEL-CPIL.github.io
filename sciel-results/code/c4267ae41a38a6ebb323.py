import numpy as np
import time
import sys
from scipy.optimize import minimize

def design(problem):
    start_time = time.time()
    deadline = start_time + 110.0
    
    size = problem["pixels"]
    eps_low = problem["eps_low"]
    eps_high = problem["eps_high"]
    
    M = 8 
    G_vals = np.arange(-M, M + 1)
    Gx, Gy = np.meshgrid(G_vals, G_vals, indexing='ij')
    G_flat = np.column_stack((Gx.ravel(), Gy.ravel()))
    N_pw = len(G_flat)
    
    # Precompute indices for eps_G and eps_diff
    idx_x_G = G_flat[:, 0] % size
    idx_y_G = G_flat[:, 1] % size
    
    # Precompute difference indices for eps_inv_diff
    # This is constant for all cells!
    Dx = G_flat[:, 0][:, None] - G_flat[:, 0][None, :]
    Dy = G_flat[:, 1][:, None] - G_flat[:, 1][None, :]
    idx_x_diff = Dx % size
    idx_y_diff = Dy % size
    
    # K-points
    # 30 points in IBZ (0 <= kx <= ky <= pi)
    k_points = []
    # Grid 6x6 in the triangle
    for i in range(6):
        for j in range(6 - i):
            kx = i / 5.0
            ky = (i + j) / 5.0
            k_points.append((kx, ky))
    # Add specific high symmetry points if not covered
    # (0,0), (1,0), (0.5, 0.5)
    # The grid above includes (0,0), (1,0) is (5,0)->kx=1,ky=1? No.
    # i=5, j=0 -> kx=1, ky=1. This is M? No, M is (0.5, 0.5).
    # Let's just use a fixed set of 30 points.
    k_points = [
        (0.0, 0.0),
        (0.2, 0.0), (0.4, 0.0), (0.6, 0.0), (0.8, 0.0), (1.0, 0.0),
        (0.1, 0.1), (0.3, 0.1), (0.5, 0.1), (0.7, 0.1), (0.9, 0.1),
        (0.2, 0.2), (0.4, 0.2), (0.6, 0.2), (0.8, 0.2),
        (0.3, 0.3), (0.5, 0.3), (0.7, 0.3),
        (0.4, 0.4), (0.6, 0.4),
        (0.5, 0.5),
        (0.5, 0.6), (0.5, 0.7), (0.5, 0.8), (0.5, 0.9),
        (0.6, 0.6), (0.6, 0.7), (0.6, 0.8),
        (0.7, 0.7), (0.7, 0.8),
        (0.8, 0.8), (0.8, 0.9), (0.9, 0.9)
    ]
    # Filter to ensure kx <= ky
    k_points = [(kx, ky) for kx, ky in k_points if kx <= ky + 1e-9]
    
    def generate_cell(R0, A):
        cell = np.zeros((size, size), dtype=int)
        xs = np.arange(size) - (size - 1) / 2.0
        ys = np.arange(size) - (size - 1) / 2.0
        XX, YY = np.meshgrid(xs, ys, indexing='ij')
        R = np.sqrt(XX**2 + YY**2)
        Theta = np.arctan2(YY, XX)
        r_shape = R0 * (1 + A * np.cos(4 * Theta))
        mask = R <= r_shape
        cell[mask] = 1
        return cell

    def compute_bands(cell):
        eps_grid = eps_low + (eps_high - eps_low) * cell.astype(float)
        eps_fft = np.fft.fft2(eps_grid)
        
        # Get eps_G
        eps_G = eps_fft[idx_x_G, idx_y_G]
        
        # Get eps_diff
        eps_diff = eps_fft[idx_x_diff, idx_y_diff]
        eps_inv_diff = 1.0 / eps_diff
        
        min_omega_5 = float('inf')
        max_omega_4 = 0.0
        
        for kx, ky in k_points:
            kx_rad = kx * np.pi
            ky_rad = ky * np.pi
            
            k_plus_G_sq = (kx_rad + G_flat[:, 0] * 2 * np.pi)**2 + (ky_rad + G_flat[:, 1] * 2 * np.pi)**2
            
            A = np.diag(k_plus_G_sq) - eps_inv_diff * k_plus_G_sq[None, :]
            
            try:
                eigvals = np.linalg.eigvalsh(A)
                if len(eigvals) >= 5:
                    w4 = eigvals[3]
                    w5 = eigvals[4]
                    omega4 = np.sqrt(w4)
                    omega5 = np.sqrt(w5)
                    
                    if omega5 < min_omega_5:
                        min_omega_5 = omega5
                    if omega4 > max_omega_4:
                        max_omega_4 = omega4
            except np.linalg.LinAlgError:
                pass
        
        if min_omega_5 == float('inf'):
            return -1.0
            
        return 2 * (min_omega_5 - max_omega_4) / (min_omega_5 + max_omega_4)

    best_score = -1.0
    best_params = (0.2, 0.0)
    
    # Initial evaluation
    cell0 = generate_cell(0.2, 0.0)
    score0 = compute_bands(cell0)
    best_score = score0
    
    def objective(params):
        if time.time() > deadline:
            return -1.0
        R0, A = params
        if R0 < 0.1 or R0 > 0.45:
            return -1.0
        if A < -0.5 or A > 0.5:
            return -1.0
        cell = generate_cell(R0, A)
        if np.all(cell == 0) or np.all(cell == 1):
            return -1.0
        return compute_bands(cell)

    start_points = [
        (0.2, 0.0),
        (0.3, 0.2),
        (0.3, -0.2),
        (0.25, 0.3),
        (0.25, -0.3),
        (0.35, 0.0)
    ]
    
    for sp in start_points:
        if time.time() > deadline:
            break
        res = minimize(objective, sp, method='Nelder-Mead', 
                       options={'xatol': 1e-3, 'fatol': 1e-4, 'maxiter': 100, 'maxfev': 100})
        if res.fun > best_score:
            best_score = res.fun
            best_params = res.x

    R0, A = best_params
    cell = generate_cell(R0, A)
    return cell.tolist()
