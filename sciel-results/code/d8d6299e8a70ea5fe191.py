import numpy as np
import time

def _apply_d4_symmetry(quarter, size=32):
    """Apply full D4 symmetry to a top-left quarter to produce the full cell."""
    cell = np.zeros((size, size), dtype=int)
    half = size // 2
    for r in range(half):
        for c in range(half):
            val = quarter[r, c]
            cell[r, c] = val
            cell[r, size-1-c] = val
            cell[size-1-r, c] = val
            cell[size-1-r, size-1-c] = val
            cell[c, r] = val
            cell[c, size-1-r] = val
            cell[size-1-r, c] = val
            cell[size-1-r, size-1-c] = val
    return cell

def _compute_gap_fast(cell, n_pw=9, n_kpts=5):
    """Fast proxy computation of the 4th-5th TM band gap ratio."""
    size = cell.shape[0]
    eps_high = 12.0
    eps_low = 1.0
    
    # Compute Fourier coefficients of 1/epsilon
    inv_eps = np.where(cell == 1, 1.0 / eps_high, 1.0 / eps_low)
    fft_2d = np.fft.fft2(inv_eps)
    
    n_half = n_pw // 2
    G_indices = np.arange(-n_half, n_half + 1)
    
    f_G = np.zeros((n_pw, n_pw), dtype=complex)
    for i, m in enumerate(G_indices):
        for j, n in enumerate(G_indices):
            f_G[i, j] = fft_2d[m % size, n % size] / (size * size)
    
    # G-vector list
    G_list = []
    for i in range(n_pw):
        for j in range(n_pw):
            G_list.append((G_indices[i] * 2.0 * np.pi, G_indices[j] * 2.0 * np.pi))
    
    # Key k-points in IBZ (in units of 2pi/a)
    k_points = [
        (0.0, 0.0),    # Gamma
        (0.5, 0.0),    # mid Gamma-X
        (1.0, 0.0),    # X
        (0.5, 0.5),    # mid X-M
        (1.0, 1.0),    # M
    ]
    
    min_omega_5 = np.inf
    max_omega_4 = -np.inf
    
    for kx, ky in k_points:
        k_actual = np.array([kx * 2.0 * np.pi, ky * 2.0 * np.pi])
        
        H = np.zeros((n_pw*n_pw, n_pw*n_pw), dtype=complex)
        
        for a in range(n_pw*n_pw):
            Ga = G_list[a]
            k_plus_Ga = k_actual + np.array(Ga)
            kp2 = k_plus_Ga[0]**2 + k_plus_Ga[1]**2
            for b in range(n_pw*n_pw):
                Gb = G_list[b]
                dGx = Ga[0] - Gb[0]
                dGy = Ga[1] - Gb[1]
                mg = int(round(dGx / (2.0 * np.pi)))
                ng = int(round(dGy / (2.0 * np.pi)))
                if -n_half <= mg <= n_half and -n_half <= ng <= n_half:
                    idx_m = G_indices.tolist().index(mg)
                    idx_n = G_indices.tolist().index(ng)
                    f_val = f_G[idx_m, idx_n]
                else:
                    f_val = 0.0
                H[a, b] = kp2 * f_val
        
        try:
            eigenvalues = np.sort(np.real(np.linalg.eigvalsh(H)))
        except Exception:
            continue
        
        if len(eigenvalues) >= 5:
            omega_4 = eigenvalues[3]
            omega_5 = eigenvalues[4]
            max_omega_4 = max(max_omega_4, omega_4)
            min_omega_5 = min(min_omega_5, omega_5)
    
    if max_omega_4 == -np.inf or min_omega_5 == np.inf:
        return -1.0
    if min_omega_5 <= max_omega_4:
        return -1.0
    
    gap = min_omega_5 - max_omega_4
    midgap = (min_omega_5 + max_omega_4) / 2.0
    if midgap <= 0:
        return -1.0
    
    return 2.0 * gap / midgap

def _generate_designs(size=32):
    """Generate a diverse set of candidate designs."""
    coords = np.arange(size)
    x = (coords + 0.5) / size - 0.5
    y = (coords + 0.5) / size - 0.5
    X, Y = np.meshgrid(x, y)
    
    designs = []
    
    # 2x2 rod array with varying parameters
    for dx in [0.18, 0.22, 0.25, 0.28, 0.30]:
        for R in [0.08, 0.10, 0.12, 0.14, 0.16]:
            cell = np.zeros((size, size), dtype=int)
            for sx in [-dx, dx]:
                for sy in [-dx, dx]:
                    dist = np.sqrt((X - sx)**2 + (Y - sy)**2)
                    cell[dist <= R] = 1
            designs.append(cell)
    
    # 2x2 ring array
    for dx in [0.20, 0.24, 0.28]:
        for R_in in [0.06, 0.08, 0.10]:
            for R_out in [0.16, 0.20, 0.24]:
                cell = np.zeros((size, size), dtype=int)
                for sx in [-dx, dx]:
                    for sy in [-dx, dx]:
                        dist = np.sqrt((X - sx)**2 + (Y - sy)**2)
                        mask = (dist >= R_in) & (dist <= R_out)
                        cell[mask] = 1
                designs.append(cell)
    
    # 4x4 rod grid
    for dx in [0.10, 0.12, 0.14]:
        for R in [0.05, 0.06, 0.08]:
            cell = np.zeros((size, size), dtype=int)
            for sx in [-1.5*dx, -0.5*dx, 0.5*dx, 1.5*dx]:
                for sy in [-1.5*dx, -0.5*dx, 0.5*dx, 1.5*dx]:
                    dist = np.sqrt((X - sx)**2 + (Y - sy)**2)
                    cell[dist <= R] = 1
            designs.append(cell)
    
    # Single large ring at center
    for R_in in [0.15, 0.20, 0.25]:
        for R_out in [0.30, 0.35, 0.40]:
            cell = np.zeros((size, size), dtype=int)
            dist = np.sqrt(X**2 + Y**2)
            mask = (dist >= R_in) & (dist <= R_out)
            cell[mask] = 1
            designs.append(cell)
    
    # Cross pattern
    for w in [0.08, 0.10, 0.12]:
        cell = np.zeros((size, size), dtype=int)
        cell[np.abs(X) <= w] = 1
        cell[np.abs(Y) <= w] = 1
        designs.append(cell)
    
    # Checkerboard-like patterns
    for period in [4, 6, 8]:
        cell = np.zeros((size, size), dtype=int)
        for i in range(size):
            for j in range(size):
                if (i // period + j // period) % 2 == 0:
                    cell[i, j] = 1
        designs.append(cell)
    
    return designs

def design(problem):
    """Design a symmetric 32x32 photonic crystal unit cell."""
    start_time = time.time()
    deadline = start_time + 210.0  # Safety margin before 230s kill
    
    size = problem["pixels"]
    half = size // 2
    
    # Generate initial diverse designs
    designs = _generate_designs(size)
    
    best_score = -np.inf
    best_cell = None
    
    # Evaluate all initial designs
    for cell in designs:
        if time.time() > deadline - 30:
            break
        score = _compute_gap_fast(cell, n_pw=9, n_kpts=5)
        if score > best_score:
            best_score = score
            best_cell = cell.copy()
    
    # If no design found, use a default
    if best_cell is None:
        best_cell = np.zeros((size, size), dtype=int)
        best_score = -1.0
    
    # Local search on the best design
    # Extract quarter and optimize
    quarter = best_cell[:half, :half].copy()
    best_quarter = quarter.copy()
    
    # Greedy local search with time budget
    improved = True
    no_improve_count = 0
    max_no_improve = 3
    
    while improved and no_improve_count < max_no_improve and time.time() < deadline - 40:
        improved = False
        no_improve_count = 0
        
        # Try flipping each pixel in the quarter
        for r in range(half):
            for c in range(half):
                if time.time() > deadline - 40:
                    break
                
                new_quarter = quarter.copy()
                new_quarter[r, c] = 1 - new_quarter[r, c]
                new_cell = _apply_d4_symmetry(new_quarter, size)
                
                score = _compute_gap_fast(new_cell, n_pw=9, n_kpts=5)
                
                if score > best_score + 1e-6:
                    best_score = score
                    best_quarter = new_quarter.copy()
                    quarter = new_quarter
                    improved = True
                    no_improve_count = 0
                else:
                    no_improve_count += 1
        
        if no_improve_count >= max_no_improve * half * half:
            break
    
    final_cell = _apply_d4_symmetry(best_quarter, size)
    
    return final_cell.tolist()
