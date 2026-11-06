import numpy as np
import time
import scipy.linalg

def _apply_symmetry(cell_quarter, size=32):
    """Apply 4-fold symmetry (D4) to a quarter-cell to produce full cell."""
    cell = np.zeros((size, size), dtype=int)
    half = size // 2  # 16
    
    for r in range(half):
        for c in range(half):
            val = cell_quarter[r, c]
            # Apply all 8 symmetry operations
            cell[r, c] = val
            cell[r, size-1-c] = val
            cell[size-1-r, c] = val
            cell[size-1-r, size-1-c] = val
            cell[c, r] = val
            cell[c, size-1-r] = val
            cell[size-1-r, c] = val
            cell[size-1-r, size-1-c] = val
            
    return cell

def _compute_gap(cell, n_pw=17, n_bands=6, n_kpts=5):
    """Compute the gap between bands 4 and 5 for TM polarization.
    
    Returns the gap-midgap ratio: 2*(min_omega_5 - max_omega_4) / (min_omega_5 + max_omega_4)
    """
    try:
        size = cell.shape[0]
        eps_high = 12.0
        eps_low = 1.0
        
        # Compute Fourier coefficients of 1/epsilon
        inv_eps = np.where(cell == 1, 1.0 / eps_high, 1.0 / eps_low)
        fft_2d = np.fft.fft2(inv_eps)
        
        n_half = n_pw // 2
        G_indices = np.arange(-n_half, n_half + 1)
        
        # Precompute B matrix indices
        G_list = []
        for i in range(n_pw):
            for j in range(n_pw):
                G_list.append((G_indices[i], G_indices[j]))
        
        # Build B matrix
        B = np.zeros((n_pw*n_pw, n_pw*n_pw), dtype=complex)
        for a in range(n_pw*n_pw):
            ma, na = G_list[a]
            for b in range(n_pw*n_pw):
                mb, nb = G_list[b]
                dm = ma - mb
                dn = na - nb
                if -n_half <= dm <= n_half and -n_half <= dn <= n_half:
                    fft_idx_i = dm % size
                    fft_idx_j = dn % size
                    B[a, b] = fft_2d[fft_idx_i, fft_idx_j] / (size * size)
        
        k_points = [
            (0.0, 0.0),
            (0.5, 0.0),
            (1.0, 0.0),
            (0.5, 0.5),
            (1.0, 1.0),
            (0.25, 0.25),
            (0.75, 0.25),
        ]
        
        min_omega_5 = np.inf
        max_omega_4 = -np.inf
        
        for kx, ky in k_points:
            A_diag = np.zeros(n_pw*n_pw)
            for a in range(n_pw*n_pw):
                ma, na = G_list[a]
                A_diag[a] = (kx + ma)**2 + (ky + na)**2
            
            A_mat = np.diag(A_diag)
            
            try:
                eigenvalues, _ = scipy.linalg.eigh(A_mat, B)
                eigenvalues = np.sort(np.real(eigenvalues))
            except Exception:
                continue
            
            if len(eigenvalues) >= n_bands:
                omega_4 = eigenvalues[3]
                omega_5 = eigenvalues[4]
                
                if omega_4 > max_omega_4:
                    max_omega_4 = omega_4
                if omega_5 < min_omega_5:
                    min_omega_5 = omega_5
        
        if max_omega_4 == -np.inf or min_omega_5 == np.inf:
            return -1.0
        
        if min_omega_5 <= max_omega_4:
            return -1.0
        
        gap = min_omega_5 - max_omega_4
        midgap = (min_omega_5 + max_omega_4) / 2.0
        
        if midgap <= 0:
            return -1.0
        
        return 2.0 * gap / midgap

    except Exception:
        return -1.0

def design(problem):
    """
    Design a symmetric 32x32 photonic crystal unit cell.
    Uses Simulated Annealing to optimize the 4th-5th TM band gap.
    """
    start_time = time.time()
    deadline = start_time + 115.0
    
    size = problem["pixels"]
    half = size // 2
    
    # Initial candidates
    coords = np.arange(size)
    x = (coords + 0.5) / size - 0.5
    y = (coords + 0.5) / size - 0.5
    X, Y = np.meshgrid(x, y)
    
    cell1 = np.zeros((size, size), dtype=int)
    dx = 0.25
    dy = 0.25
    R_rod = 0.12
    dist1 = np.sqrt((X - dx)**2 + (Y - dy)**2)
    cell1[dist1 <= R_rod] = 1
    dist2 = np.sqrt((X + dx)**2 + (Y - dy)**2)
    cell1[dist2 <= R_rod] = 1
    dist3 = np.sqrt((X - dx)**2 + (Y + dy)**2)
    cell1[dist3 <= R_rod] = 1
    dist4 = np.sqrt((X + dx)**2 + (Y + dy)**2)
    cell1[dist4 <= R_rod] = 1
    
    cell2 = np.zeros((size, size), dtype=int)
    for r in range(size):
        for c in range(size):
            if (r // 4 + c // 4) % 2 == 0:
                cell2[r, c] = 1
    
    candidates = [cell1, cell2]
    
    best_score = -1.0
    best_cell = None
    for c in candidates:
        score = _compute_gap(c, n_pw=13, n_bands=6, n_kpts=5)
        if score > best_score:
            best_score = score
            best_cell = c
    
    quarter = best_cell[:half, :half].copy()
    
    T = 1.0
    T_min = 0.01
    cooling = 0.99
    
    independent_positions = [(r, c) for r in range(half) for c in range(half)]
    
    while time.time() < deadline and T > T_min:
        idx = np.random.randint(len(independent_positions))
        r, c = independent_positions[idx]
        
        new_quarter = quarter.copy()
        new_quarter[r, c] = 1 - new_quarter[r, c]
        new_cell = _apply_symmetry(new_quarter, size)
        
        new_score = _compute_gap(new_cell, n_pw=13, n_bands=6, n_kpts=5)
        
        delta = new_score - best_score
        
        if delta > 0 or np.random.random() < np.exp(delta / T):
            best_score = new_score
            best_cell = new_cell
            quarter = new_quarter
        
        T *= cooling
    
    final_cell = _apply_symmetry(quarter, size)
    
    return final_cell.tolist()
