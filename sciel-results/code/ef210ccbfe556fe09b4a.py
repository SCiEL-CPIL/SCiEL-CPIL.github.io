import numpy as np
import time
from scipy.optimize import minimize

def design(problem):
    """
    Design a symmetric 32x32-pixel unit cell with a large TM band gap.
    """
    start_time = time.time()
    deadline = start_time + 110  # 10s margin
    
    size = problem["pixels"]
    eps_low = problem["eps_low"]
    eps_high = problem["eps_high"]
    
    # Fast plane-wave solver parameters
    N_pw = 11  # 121 plane waves
    k_points = [
        (0.0, 0.0),  # Gamma
        (0.5, 0.0),  # X
        (0.5, 0.5),  # M
        (0.25, 0.0),
        (0.25, 0.25),
        (0.5, 0.25),
    ]
    
    def generate_cell(params):
        amp, width = params
        cell = np.zeros((size, size))
        centers = [
            (8.5, 8.5),
            (23.5, 8.5),
            (8.5, 23.5),
            (23.5, 23.5)
        ]
        for cx, cy in centers:
            for i in range(size):
                for j in range(size):
                    x = i
                    y = j
                    dist_sq = (x - cx)**2 + (y - cy)**2
                    cell[i, j] += amp * np.exp(-dist_sq / (2 * width**2))
        
        threshold = 0.5
        cell = (cell > threshold).astype(int)
        return cell

    def compute_gap_ratio(cell):
        eps_inv = np.where(cell == 1, 1.0/eps_high, 1.0/eps_low)
        
        G_list = []
        for n1 in range(-N_pw//2, N_pw//2 + 1):
            for n2 in range(-N_pw//2, N_pw//2 + 1):
                G_list.append((n1, n2))
        
        G_x = np.array([g[0] for g in G_list])
        G_y = np.array([g[1] for g in G_list])
        
        r_x = (np.arange(size) + 0.5) / size
        r_y = (np.arange(size) + 0.5) / size
        RX, RY = np.meshgrid(r_x, r_y, indexing='ij')
        
        Gx = G_x[:, None, None] * 2 * np.pi
        Gy = G_y[:, None, None] * 2 * np.pi
        Phase = np.exp(-1j * (Gx * RX[None, :, :] + Gy * RY[None, :, :]))
        
        coeff = np.sum(eps_inv[None, :, :] * Phase, axis=(1, 2))
        
        G_map = {}
        for idx, g in enumerate(G_list):
            G_map[g] = idx
        
        max_band4 = 0
        min_band5 = np.inf
        
        for kx, ky in k_points:
            N = len(G_list)
            A_k = np.zeros((N, N), dtype=complex)
            for m in range(N):
                Gm = G_list[m]
                Gx_k = Gm[0] + kx
                Gy_k = Gm[1] + ky
                A_k[m, m] = Gx_k**2 + Gy_k**2
                for n in range(N):
                    Gn = G_list[n]
                    dG = (Gm[0] - Gn[0], Gm[1] - Gn[1])
                    if dG in G_map:
                        idx_dG = G_map[dG]
                        A_k[m, n] -= coeff[idx_dG]
            
            try:
                eigvals_k = np.linalg.eigvalsh(A_k)
            except:
                continue
                
            eigvals_k = np.sort(eigvals_k)
            
            if len(eigvals_k) >= 5:
                band4 = eigvals_k[3]
                band5 = eigvals_k[4]
                max_band4 = max(max_band4, band4)
                min_band5 = min(min_band5, band5)
                
        if min_band5 == np.inf:
            return 0.0
            
        midgap = (min_band5 + max_band4) / 2
        if midgap == 0:
            return 0.0
            
        gap = min_band5 - max_band4
        ratio = gap / midgap
        return ratio

    x0 = [1.0, 3.0]
    
    def objective(params):
        cell = generate_cell(params)
        ratio = compute_gap_ratio(cell)
        return -ratio
        
    best_params = x0
    best_score = -objective(x0)
    
    try:
        result = minimize(objective, x0, method='Nelder-Mead', options={'maxiter': 50, 'xatol': 1e-3, 'fatol': 1e-3})
        if result.fun < -best_score:
            best_params = result.x
            best_score = -result.fun
    except:
        pass
        
    if time.time() > deadline:
        cell = generate_cell(x0)
        return cell.tolist()
        
    cell = generate_cell(best_params)
    return cell.tolist()
