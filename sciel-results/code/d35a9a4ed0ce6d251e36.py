"""Photonic crystal design: maximize TM band gap between bands 4 and 5."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def design(problem):
    size = problem["pixels"]  # 32
    eps_low = problem["eps_low"]    # 1
    eps_high = problem["eps_high"]  # 12
    lower_band = problem["lower_band"]  # 4
    deadline = time.time() + 195  # safety margin
    
    # Symmetry constraints: cell == cell.T and cell == cell[::-1] (rows and cols)
    # Independent pixels: top-left 16x16 upper triangle (including diagonal)
    indep_indices = []
    for i in range(16):
        for j in range(i, 16):
            indep_indices.append((i, j))
    n_indep = len(indep_indices)  # 136
    
    # Fast PW evaluation with small basis
    N_pw_half = 3  # G indices from -3 to 3, so 7x7 = 49 PWs
    g_range = np.arange(-N_pw_half, N_pw_half + 1)
    G_vectors = [(gx, gy) for gx in g_range for gy in g_range]
    n_pw = len(G_vectors)
    G_arr = np.array(G_vectors, dtype=float)
    
    # Precompute position grid
    ii = np.arange(size, dtype=float)
    jj = np.arange(size, dtype=float)
    II, JJ = np.meshgrid(ii, jj, indexing='ij')
    pos_x = (II + 0.5) / size
    pos_y = (JJ + 0.5) / size
    pos_flat_x = pos_x.flatten()
    pos_flat_y = pos_y.flatten()
    
    # Precompute phase factors for each G vector
    # phase[k, idx] = -2*pi*(gx*pos_x[idx] + gy*pos_y[idx])
    phases = np.zeros((n_pw, size * size), dtype=float)
    for k, (gx, gy) in enumerate(G_vectors):
        phases[k] = -2 * np.pi * (gx * pos_flat_x + gy * pos_flat_y)
    
    # Precompute k-points
    k_points = []
    # Boundary of irreducible BZ
    for ky in np.linspace(0, np.pi, 6):
        k_points.append((0.0, ky))
    for kx in np.linspace(0, np.pi / np.sqrt(2), 6):
        k_points.append((kx, kx))
    # Interior points
    k_points.extend([
        (np.pi/4, np.pi/2),
        (np.pi/3, 2*np.pi/3),
        (np.pi/6, np.pi/2),
        (np.pi/3, np.pi),
        (np.pi/4, np.pi),
        (0, np.pi/2),
        (np.pi/6, np.pi),
    ])
    # Deduplicate
    unique_k = []
    for kp in k_points:
        if not any(abs(kp[0]-u[0]) < 1e-9 and abs(kp[1]-u[1]) < 1e-9 for u in unique_k):
            unique_k.append(kp)
    k_points = np.array(unique_k[:30])
    
    def compute_gap_fast(cell_np):
        """Fast approximate TM band gap computation."""
        eps = np.where(cell_np == 1, eps_high, eps_low)
        inv_eps = 1.0 / eps
        inv_eps_flat = inv_eps.flatten()
        
        # Compute Fourier coefficients: c_G = (1/N) * sum inv_eps * exp(i*phase)
        # exp(i*phase) where phase = -2*pi*(gx*x + gy*y)
        exp_phases = np.exp(1j * phases)  # (n_pw, N_pix)
        coeffs = (exp_phases @ inv_eps_flat) / (size * size)  # (n_pw,)
        
        # For each k-point, build and solve eigenvalue problem
        max_w4 = 0.0
        min_w5 = np.inf
        
        for kx, ky in k_points:
            k = np.array([kx, ky])
            # kg = k + G for each PW
            kg = k[np.newaxis, :] + G_arr  # (n_pw, 2)
            kg2 = np.sum(kg * kg, axis=1)  # |k+G|^2, (n_pw,)
            
            # Build matrix A[ig, ig2] = kg2[ig]*delta_ig,ig2 - (k+G_ig)·(k+G_ig2) * c_{G_ig - G_ig2}
            # Precompute dot products
            kg_dot = kg @ kg.T  # (n_pw, n_pw), [ig, ig2] = (k+G_ig)·(k+G_ig2)
            
            # Need c_{G_ig - G_ig2} for all pairs
            # G_ig - G_ig2 = (gx_ig - gx_ig2, gy_ig - gy_ig2)
            dG = G_arr[:, np.newaxis, :] - G_arr[np.newaxis, :, :]  # (n_pw, n_pw, 2)
            dGx = dG[:, :, 0]
            dGy = dG[:, :, 1]
            
            # For each (ig, ig2), find if (dGx, dGy) is in G_vectors
            # Precompute a lookup
            G_lookup = {}
            for idx, (gx, gy) in enumerate(G_vectors):
                G_lookup[(gx, gy)] = idx
            
            c_diff = np.zeros((n_pw, n_pw), dtype=complex)
            for ig in range(n_pw):
                for ig2 in range(n_pw):
                    key = (int(dGx[ig, ig2]), int(dGy[ig, ig2]))
                    if key in G_lookup:
                        c_diff[ig, ig2] = coeffs[G_lookup[key]]
            
            A = np.diag(kg2).astype(complex) - kg_dot.astype(complex) * c_diff
            
            try:
                eigvals = np.linalg.eigvalsh(A)
                eigvals = np.sort(np.real(eigvals))
                eigvals_pos = eigvals[eigvals > 1e-8]
                if len(eigvals_pos) >= 5:
                    omega = np.sqrt(eigvals_pos)
                    max_w4 = max(max_w4, omega[3])
                    min_w5 = min(min_w5, omega[4])
            except:
                pass
        
        if min_w5 == np.inf or min_w5 <= 0:
            return -1.0
        gap = min_w5 - max_w4
        midgap = (min_w5 + max_w4) / 2.0
        if midgap <= 0:
            return -1.0
        return gap / midgap
    
    def build_full_cell(indep_vals):
        full = np.zeros((size, size), dtype=np.int8)
        for idx, (i, j) in enumerate(indep_indices):
            val = indep_vals[idx]
            full[i, j] = val
            full[j, i] = val
            full[31-i, 31-j] = val
            full[31-j, 31-i] = val
        return full
    
    def get_indep_vals(cell_np):
        vals = np.zeros(n_indep, dtype=np.int8)
        for idx, (i, j) in enumerate(indep_indices):
            vals[idx] = cell_np[i, j]
        return vals
    
    # Initialize with seed (dielectric rod)
    cell = np.zeros((size, size), dtype=np.int8)
    for i in range(size):
        for j in range(size):
            x = (i + 0.5) / size - 0.5
            y = (j + 0.5) / size - 0.5
            if x * x + y * y < 0.2 ** 2:
                cell[i, j] = 1
    
    indep_vals = get_indep_vals(cell)
    best_gap = compute_gap_fast(cell)
    best_indep = indep_vals.copy()
    
    # Simulated annealing
    T = 0.3
    T_min = 0.005
    cooling = 0.997
    n_iter = 0
    current_gap = best_gap
    
    while time.time() < deadline - 3:
        n_iter += 1
        idx = np.random.randint(n_indep)
        new_indep = indep_vals.copy()
        new_indep[idx] = 1 - new_indep[idx]
        new_cell = build_full_cell(new_indep)
        new_gap = compute_gap_fast(new_cell)
        
        if new_gap > best_gap:
            best_gap = new_gap
            best_indep = new_indep.copy()
        
        delta = new_gap - current_gap
        if delta > 0 or np.random.random() < np.exp(delta / max(T, 1e-10)):
            indep_vals = new_indep
            current_gap = new_gap
        
        T = max(T_min, T * cooling)
        if n_iter % 200 == 0 and T <= T_min:
            T = 0.15  # reheat
    
    best_cell = build_full_cell(best_indep)
    result = [[int(best_cell[i, j]) for j in range(size)] for i in range(size)]
    return result
# EVOLVE-BLOCK-END
