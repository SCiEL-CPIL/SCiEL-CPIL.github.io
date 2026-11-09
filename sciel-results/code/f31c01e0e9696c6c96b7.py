import numpy as np
import time

def design(problem):
    size = problem["pixels"]
    eps_low = problem["eps_low"]
    eps_high = problem["eps_high"]
    
    start_time = time.time()
    time_limit = 210.0 # Safety margin before 230s kill
    
    # Setup Plane Wave Basis
    # 289 plane waves: Gx, Gy in [-8, 8]
    N_G = 17
    g_range = np.arange(-N_G//2, N_G//2 + 1)
    Gx, Gy = np.meshgrid(g_range, g_range, indexing='ij')
    G_vectors = np.column_stack((Gx.ravel(), Gy.ravel()))
    N_pw = len(G_vectors)
    
    # K points (30 points covering irreducible BZ)
    # Irreducible BZ: 0 <= ky <= kx <= 0.5
    k_points = []
    for i in range(6):
        t = i/5.0
        k_points.append([t*0.5, 0.0]) # Gamma-M
        k_points.append([0.5, t*0.5]) # M-Corner
        k_points.append([t*0.5, t*0.5]) # Gamma-Corner
    # Interior points
    k_points.append([0.25, 0.1])
    k_points.append([0.4, 0.2])
    k_points.append([0.3, 0.15])
    k_points.append([0.45, 0.45])
    k_points.append([0.2, 0.1])
    k_points.append([0.35, 0.2])
    k_points = np.array(k_points)

    def get_eps_coeffs(cell):
        # cell is 32x32 0/1
        eps = np.where(cell == 1, eps_high, eps_low)
        eps_inv = 1.0 / eps
        # FFT
        coeffs = np.fft.fft2(eps_inv) / (size * size)
        # Extract subgrid for G in [-8, 8]
        c_sub = np.zeros((17, 17), dtype=complex)
        for gi in range(17):
            gx = gi - 8
            idx_x = size + gx if gx < 0 else gx
            for gj in range(17):
                gy = gj - 8
                idx_y = size + gy if gy < 0 else gy
                c_sub[gi, gj] = coeffs[idx_x, idx_y]
        return c_sub

    def evaluate(cell):
        c_sub = get_eps_coeffs(cell)
        min_omega_5 = float('inf')
        max_omega_4 = float('-inf')
        
        for k in k_points:
            k_plus_G = k[None, :] + G_vectors
            w = np.sum(k_plus_G**2, axis=1)
            
            H = np.zeros((N_pw, N_pw), dtype=complex)
            
            for m in range(N_pw):
                Gm = G_vectors[m]
                diffs = Gm[None, :] - G_vectors
                dx = diffs[:, 0]
                dy = diffs[:, 1]
                
                mask = (dx >= -8) & (dx <= 8) & (dy >= -8) & (dy <= 8)
                
                if np.any(mask):
                    idxs = np.where(mask)[0]
                    c_vals = c_sub[dx[idxs] + 8, dy[idxs] + 8]
                    H[m, idxs] = c_vals * w[idxs]
            
            evals = np.linalg.eigvals(H)
            evals = np.sort(np.real(evals))
            
            if len(evals) >= 5:
                w4 = evals[3]
                w5 = evals[4]
                max_omega_4 = max(max_omega_4, w4)
                min_omega_5 = min(min_omega_5, w5)
        
        if min_omega_5 == float('inf') or max_omega_4 == float('-inf'):
            return 0.0
            
        gap = min_omega_5 - max_omega_4
        midgap = (min_omega_5 + max_omega_4) / 2.0
        if midgap == 0:
            return 0.0
        return gap / midgap

    def make_symmetric(fundamental):
        cell = np.zeros((size, size), dtype=int)
        for i in range(size):
            for j in range(i + 1):
                val = fundamental[i, j]
                if val:
                    pts = [
                        (i, j), (j, i),
                        (size-1-i, j), (size-1-j, i),
                        (i, size-1-j), (j, size-1-i),
                        (size-1-i, size-1-j), (size-1-j, size-1-i)
                    ]
                    for r, c in pts:
                        cell[r, c] = 1
        return cell

    def init_pattern(type):
        c = np.zeros((size, size), dtype=int)
        if type == 'rod':
            for i in range(size):
                for j in range(size):
                    dx = i - 15.5
                    dy = j - 15.5
                    if dx*dx + dy*dy < 4.0**2:
                        c[i, j] = 1
        elif type == '4rods':
            centers = [(8, 8), (8, 24), (24, 8), (24, 24)]
            for cx, cy in centers:
                for i in range(size):
                    for j in range(size):
                        dx = i - cx
                        dy = j - cy
                        if dx*dx + dy*dy < 3.0**2:
                            c[i, j] = 1
        elif type == 'ring':
            for i in range(size):
                for j in range(size):
                    dx = i - 15.5
                    dy = j - 15.5
                    r2 = dx*dx + dy*dy
                    if 6.0**2 < r2 < 10.0**2:
                        c[i, j] = 1
        elif type == 'air_holes':
            c[:] = 1
            for i in range(size):
                for j in range(size):
                    dx = i - 15.5
                    dy = j - 15.5
                    if dx*dx + dy*dy < 4.0**2:
                        c[i, j] = 0
        elif type == 'sq_ring':
            for i in range(size):
                for j in range(size):
                    dx = abs(i - 15.5)
                    dy = abs(j - 15.5)
                    r_inf = max(dx, dy)
                    if 6.0 < r_inf < 10.0:
                        c[i, j] = 1
                        
        f = np.zeros((size, size), dtype=int)
        for i in range(size):
            for j in range(i+1):
                f[i, j] = c[i, j]
        return f

    candidates = [
        init_pattern('rod'),
        init_pattern('4rods'),
        init_pattern('ring'),
        init_pattern('air_holes'),
        init_pattern('sq_ring')
    ]
    
    best_score = -1.0
    best_fund = None
    
    for f in candidates:
        c = make_symmetric(f)
        score = evaluate(c)
        if score > best_score:
            best_score = score
            best_fund = f.copy()
            
    current_fund = best_fund.copy()
    current_score = best_score
    
    fund_coords = [(i, j) for i in range(size) for j in range(i+1)]
    np.random.seed(42)
    
    # Simulated Annealing
    T = 0.1
    T_min = 0.001
    cooling = 0.99
    
    while time.time() - start_time < time_limit - 5.0:
        idx = np.random.randint(len(fund_coords))
        i, j = fund_coords[idx]
        
        current_fund[i, j] = 1 - current_fund[i, j]
        c = make_symmetric(current_fund)
        score = evaluate(c)
        
        delta = score - current_score
        
        if delta > 0:
            current_score = score
            if score > best_score:
                best_score = score
                best_fund = current_fund.copy()
        else:
            # Accept with probability exp(delta/T)
            if np.random.rand() < np.exp(delta / T):
                current_score = score
            else:
                current_fund[i, j] = 1 - current_fund[i, j]
                
        T *= cooling
        if T < T_min:
            T = T_min

    final_cell = make_symmetric(best_fund)
    return final_cell.tolist()
