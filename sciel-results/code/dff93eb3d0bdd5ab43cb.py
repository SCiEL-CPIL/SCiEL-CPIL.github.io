import numpy as np
import time

def design(problem):
    size = 32
    t_start = time.time()
    deadline = t_start + 110.0
    
    # 1. Setup
    g_range = np.arange(-8, 9)
    G_vecs = np.array([(g1, g2) for g1 in g_range for g2 in g_range])
    n_pw = len(G_vecs)
    
    g_range_large = np.arange(-16, 17)
    G_vecs_large = np.array([(g1, g2) for g1 in g_range_large for g2 in g_range_large])
    
    # k-points
    k_points = []
    for i in range(10):
        k_points.append((np.pi * i / 9.0, 0.0))
    for i in range(1, 10):
        t = i / 9.0
        k_points.append((np.pi * (1 - t/2.0), np.pi * t/2.0))
    for i in range(1, 10):
        t = i / 9.0
        k_points.append((np.pi / 2.0 * (1 - t), np.pi / 2.0 * (1 - t)))
    k_points.append((np.pi/3.0, np.pi/3.0))
    
    # Precompute indices for eps extraction
    idx_g = (G_vecs[:, 0] + 16) * 33 + (G_vecs[:, 1] + 16)
    G_diff = G_vecs[:, None, :] - G_vecs[None, :, :]
    idx_eps_diff = (G_diff[:, :, 0] + 16) * 33 + (G_diff[:, :, 1] + 16)
    
    def symmetrize(cell):
        c = np.array(cell)
        for i in range(16):
            for j in range(16):
                val = c[i][j]
                c[i][j] = val
                c[j][i] = val
                c[31-i][j] = val
                c[31-i][31-j] = val
                c[i][31-j] = val
                c[31-j][i] = val
                c[31-j][31-i] = val
        return c.astype(int)

    def get_epsilon_G_large(cell):
        eps = np.where(cell == 1, 12.0, 1.0)
        # Precomputed cos/sin for large G
        # This is expensive to compute every time.
        # Let's precompute the cos/sin matrices for G_vecs_large once.
        pass

    # Precompute trig for large G
    cos_Gx_large = np.cos(G_vecs_large[:, 0][:, None] * np.arange(32)[None, :])
    sin_Gx_large = np.sin(G_vecs_large[:, 0][:, None] * np.arange(32)[None, :])
    cos_Gy_large = np.cos(G_vecs_large[:, 1][:, None] * np.arange(32)[None, :])
    sin_Gy_large = np.sin(G_vecs_large[:, 1][:, None] * np.arange(32)[None, :])
    
    c_gx_l = cos_Gx_large[:, :, None]
    s_gx_l = sin_Gx_large[:, :, None]
    c_gy_l = cos_Gy_large[:, None, :]
    s_gy_l = sin_Gy_large[:, None, :]
    term_cos_l = c_gx_l * c_gy_l - s_gx_l * s_gy_l

    def get_epsilon_G_large(cell):
        eps = np.where(cell == 1, 12.0, 1.0)
        return (eps[:, :, None] * term_cos_l).sum(axis=(1, 2))

    def compute_score(cell):
        eps_large = get_epsilon_G_large(cell)
        eps_289 = eps_large[idx_g]
        sqrt_eps = np.sqrt(eps_289)
        eps_diff = eps_large[idx_eps_diff]
        
        min_omega_5 = np.inf
        max_omega_4 = 0.0
        
        for kx, ky in k_points:
            kx_G = kx + G_vecs[:, 0]
            ky_G = ky + G_vecs[:, 1]
            dx = kx_G[:, None] + G_vecs[None, :, 0]
            dy = ky_G[:, None] + G_vecs[None, :, 1]
            kk_sq = dx**2 + dy**2
            
            M = kk_sq / (eps_diff * sqrt_eps[:, None] * sqrt_eps[None, :])
            vals, _ = np.linalg.eigh(M)
            vals = vals[vals > 0]
            if len(vals) >= 5:
                w4 = vals[3]
                w5 = vals[4]
                if w4 > max_omega_4:
                    max_omega_4 = w4
                if w5 < min_omega_5:
                    min_omega_5 = w5
        
        if min_omega_5 == np.inf:
            return -1.0
        gap = min_omega_5 - max_omega_4
        mid = (min_omega_5 + max_omega_4) / 2.0
        return gap / mid

    # Optimization
    cell = np.zeros((32, 32), dtype=int)
    for i in range(32):
        for j in range(32):
            x = (i - 15.5) / 16.0
            y = (j - 15.5) / 16.0
            if x*x + y*y < 0.35**2:
                cell[i][j] = 1
    cell = symmetrize(cell)
    
    best_cell = cell.copy()
    best_score = compute_score(cell)
    current_cell = cell.copy()
    current_score = best_score
    
    T = 0.05
    T_min = 0.001
    
    while time.time() < deadline:
        new_cell = current_cell.copy()
        i = np.random.randint(0, 32)
        j = np.random.randint(0, 32)
        new_cell[i][j] = 1 - new_cell[i][j]
        new_cell = symmetrize(new_cell)
        
        new_score = compute_score(new_cell)
        
        if new_score > current_score:
            current_cell = new_cell
            current_score = new_score
            if new_score > best_score:
                best_score = new_score
                best_cell = new_cell.copy()
        else:
            delta = new_score - current_score
            if np.random.rand() < np.exp(delta / T):
                current_cell = new_cell
                current_score = new_score
                
        T *= 0.99
        if T < T_min:
            T = T_min
            
    return best_cell.tolist()
