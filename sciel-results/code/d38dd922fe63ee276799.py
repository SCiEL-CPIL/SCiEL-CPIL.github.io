import numpy as np
import time

def design(problem):
    start_time = time.time()
    deadline = 210.0  # Safety margin from 230s limit
    
    size = problem["pixels"]
    eps_high = problem["eps_high"]
    eps_low = problem["eps_low"]
    
    # Fundamental domain for 4-fold symmetric 32x32 cell
    # Symmetries: cell[i][j] == cell[i][31-j], cell[i][j] == cell[31-i][j], cell[i][j] == cell[j][i]
    # Fundamental domain: i in [0, 15], j in [i, 15]
    fd_indices = []
    for i in range(16):
        for j in range(i, 16):
            fd_indices.append((i, j))
    n_fd = len(fd_indices)
    
    # Build full 32x32 cell from fundamental domain
    def build_cell(fd_values):
        cell = np.zeros((size, size), dtype=np.int8)
        for k, (i, j) in enumerate(fd_indices):
            val = fd_values[k]
            positions = [
                (i, j),
                (j, i),
                (i, 31-j),
                (31-j, i),
                (31-i, j),
                (j, 31-i),
                (31-i, 31-j),
                (31-j, 31-i)
            ]
            for (r, c) in positions:
                cell[r, c] = val
        return cell
    
    # Fast evaluator using plane-wave method
    # Use a moderate number of G vectors for speed
    N_w = 9  # 9x9 grid of G vectors = 81 modes
    g_range = np.arange(-N_w//2, N_w//2 + 1)
    g_vectors = [(m, n) for m in g_range for n in g_range]
    num_g = len(g_vectors)
    
    # K-points in irreducible BZ (triangle: Gamma to X to M)
    k_points = [
        (0.0, 0.0),      # Gamma
        (np.pi, 0.0),    # X
        (np.pi, np.pi),  # M
        (0.5*np.pi, 0.0),
        (0.5*np.pi, 0.5*np.pi),
        (0.8*np.pi, 0.0),
        (0.0, 0.5*np.pi),
        (0.2*np.pi, 0.2*np.pi),
        (0.6*np.pi, 0.4*np.pi),
        (0.4*np.pi, 0.6*np.pi),
    ]
    
    # Precompute G vector components
    g_mags = np.array(g_vectors, dtype=float)
    g_x = 2.0 * np.pi * g_mags[:, 0] / size
    g_y = 2.0 * np.pi * g_mags[:, 1] / size
    
    # Precompute FFT index mapping for chi coefficients
    g_fft_idx = [(m % size, n % size) for m, n in g_vectors]
    
    def evaluate_gap(cell_2d):
        """Evaluate the gap-midgap ratio between TM bands 4 and 5."""
        chi_low = 1.0 / eps_low
        chi_high = 1.0 / eps_high
        delta_chi = chi_high - chi_low
        
        # Compute FFT of cell for chi coefficients
        fft_cell = np.fft.fftn(cell_2d.astype(float))
        fft_chi = delta_chi * fft_cell
        fft_chi[0, 0] += chi_low * size * size
        
        min_omega_5 = float('inf')
        max_omega_4 = 0.0
        
        for kx, ky in k_points:
            # Build H matrix: H[i,j] = (G_i+K) dot (G_j+K) * chi_{G_i-G_j}
            H = np.zeros((num_g, num_g), dtype=complex)
            
            for i in range(num_g):
                Kx_i = g_x[i] + kx
                Ky_i = g_y[i] + ky
                mi, ni = g_vectors[i]
                
                for j in range(num_g):
                    Kx_j = g_x[j] + kx
                    Ky_j = g_y[j] + ky
                    mj, nj = g_vectors[j]
                    
                    # Chi index: (mi-mj, ni-nj)
                    chi_i = (mi - mj) % size
                    chi_j = (ni - nj) % size
                    chi_val = fft_chi[chi_i, chi_j]
                    
                    dot_prod = Kx_i * Kx_j + Ky_i * Ky_j
                    H[i, j] = dot_prod * chi_val
            
            # Symmetrize to ensure Hermitian
            H = 0.5 * (H + H.conj().T)
            
            try:
                eigvals = np.linalg.eigvalsh(H)
                eigvals = eigvals[eigvals > 1e-10]
                if len(eigvals) >= 5:
                    omega_4 = np.sqrt(eigvals[3])
                    omega_5 = np.sqrt(eigvals[4])
                    max_omega_4 = max(max_omega_4, omega_4)
                    min_omega_5 = min(min_omega_5, omega_5)
            except Exception:
                pass
        
        if min_omega_5 == float('inf'):
            return -1.0
        
        gap = 2.0 * (min_omega_5 - max_omega_4)
        midgap = min_omega_5 + max_omega_4
        return gap / midgap if midgap > 0 else 0.0
    
    # Parameterized superellipse family
    # |x/a|^n + |y/b|^n = 1, centered at (15.5, 15.5)
    def build_superellipse(n_exp, a, b):
        fd = np.zeros(n_fd, dtype=np.int8)
        for k, (i, j) in enumerate(fd_indices):
            dx = abs(i - 15.5)
            dy = abs(j - 15.5)
            if dx < a and dy < b:
                val = (dx/a)**n_exp + (dy/b)**n_exp
                if val <= 1.0:
                    fd[k] = 1
        return fd
    
    # Generate initial configurations
    np.random.seed(42)
    
    initial_configs = []
    
    # Superellipse grid
    for n_exp in [2.0, 3.0, 4.0, 6.0, 8.0]:
        for a in np.arange(5.0, 13.0, 1.0):
            for b in np.arange(5.0, 13.0, 1.0):
                initial_configs.append(('superellipse', n_exp, a, b))
    
    # Some random starts
    for _ in range(5):
        rd_fd = np.random.randint(0, 2, n_fd).astype(np.int8)
        initial_configs.append(('random', rd_fd))
    
    # Evaluate initial configs and find best
    best_score = -1.0
    best_fd = None
    
    for config in initial_configs:
        if time.time() > deadline:
            break
        
        if config[0] == 'superellipse':
            fd = build_superellipse(config[1], config[2], config[3])
        else:
            fd = config[1].copy()
        
        cell = build_cell(fd)
        score = evaluate_gap(cell)
        
        if score > best_score:
            best_score = score
            best_fd = fd.copy()
    
    # If we have time, do local refinement
    if best_fd is not None and time.time() < deadline - 30:
        current_fd = best_fd.copy()
        current_score = best_score
        
        # Hill climbing with random flips
        max_steps = 2000
        for step in range(max_steps):
            if time.time() > deadline - 10:
                break
            
            # Pick random index to flip
            idx = np.random.randint(n_fd)
            current_fd[idx] = 1 - current_fd[idx]
            
            new_cell = build_cell(current_fd)
            new_score = evaluate_gap(new_cell)
            
            if new_score > current_score:
                current_score = new_score
                if current_score > best_score:
                    best_score = current_score
                    best_fd = current_fd.copy()
            else:
                current_fd[idx] = 1 - current_fd[idx]
    
    # Fallback
    if best_fd is None:
        best_fd = build_superellipse(2.0, 8.0, 8.0)
    
    return build_cell(best_fd).tolist()
