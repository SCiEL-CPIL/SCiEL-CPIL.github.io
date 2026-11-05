# EVOLVE-BLOCK-START
def design(problem):
    """
    Design a 32x32 unit cell to maximize the TM band gap between bands 4 and 5.
    """
    size = problem["pixels"]
    eps_high = problem.get("eps_high", 12)
    lower_band = problem.get("lower_band", 4)
    
    start_time = time.time()
    deadline = start_time + 210.0

    # Precompute G vectors and k-points
    n_pw = 13
    m_vals = np.arange(-n_pw//2, n_pw//2 + 1)
    if len(m_vals) > n_pw:
        m_vals = m_vals[:n_pw]
    elif len(m_vals) < n_pw:
        m_vals = np.concatenate([m_vals, [m_vals[-1] + 1] * (n_pw - len(m_vals))])
        
    G_vectors = []
    for m in m_vals:
        for n in m_vals:
            G_vectors.append((m, n))
    G_vectors = np.array(G_vectors)
    n_total = len(G_vectors)
    
    # Precompute G_diff indices for S matrix
    G_diff = G_vectors[:, np.newaxis, :] - G_vectors[np.newaxis, :, :]
    idx_m_diff = np.mod(G_diff[:, :, 0], size)
    idx_n_diff = np.mod(G_diff[:, :, 1], size)
    
    # Precompute k-points
    k_points = []
    n_seg = 10 # 30 points total roughly
    for t in np.linspace(0, 1, n_seg + 1):
        k_points.append((0.0, t))
    for t in np.linspace(0, 1, n_seg + 1):
        k_points.append((t, 1.0))
    for t in np.linspace(1, 0, n_seg + 1):
        k_points.append((t, t))
        
    k_points_unique = []
    for kp in k_points:
        if not any(np.allclose(kp, kp2, atol=1e-10) for kp2 in k_points_unique):
            k_points_unique.append(kp)
    k_points = k_points_unique
    
    # Precompute G_sq for each k-point? No, G_sq depends on k.
    # But we can precompute Gx, Gy parts.
    # Gx = 2*m + kx, Gy = 2*n + ky
    # G_sq = (2*m + kx)^2 + (2*n + ky)^2
    # We can vectorize this over k-points and G-vectors.
    # G_vectors shape (n_total, 2)
    # k_points shape (n_kpts, 2)
    
    # Let's create a function to score a cell given its fft_coeffs
    def score_from_fft(fft_coeffs):
        S = fft_coeffs[idx_m_diff, idx_n_diff] / (size * size)
        S = S.real
        
        n_kpts = len(k_points)
        omega = np.zeros((n_kpts, n_total))
        n_bands_needed = lower_band + 1
        
        # Vectorize G_sq computation
        # G_vectors[:, 0] shape (n_total, 1)
        # k_points[:, 0] shape (1, n_kpts)
        Gx = 2 * G_vectors[:, 0][:, np.newaxis] + k_points[:, 0][np.newaxis, :]
        Gy = 2 * G_vectors[:, 1][:, np.newaxis] + k_points[:, 1][np.newaxis, :]
        G_sq = Gx**2 + Gy**2 # shape (n_total, n_kpts)
        
        for ki in range(n_kpts):
            G_sq_k = G_sq[:, ki]
            A = S * G_sq_k[np.newaxis, :]
            B = S
            try:
                from scipy.linalg import eigh
                eigenvalues = eigh(A, B, eigvals_only=True)
                eigenvalues = np.sort(eigenvalues)
                omega[ki, :n_bands_needed] = np.sqrt(np.maximum(eigenvalues[:n_bands_needed], 0))
            except:
                return -np.inf
                
        if np.any(np.isnan(omega)):
            return -np.inf
            
        max_omega_4 = np.max(omega[:, 3])
        min_omega_5 = np.min(omega[:, 4])
        gap = min_omega_5 - max_omega_4
        midgap = (min_omega_5 + max_omega_4) / 2.0
        if midgap > 0:
            return 2.0 * gap / midgap
        else:
            return -np.inf

    def make_cell_from_params(r1, r2, shape_type):
        cell = np.zeros((size, size), dtype=int)
        coords = (np.arange(size) + 0.5) / size - 0.5
        x, y = np.meshgrid(coords, coords, indexing='ij')
        
        if shape_type == 'disk':
            mask = (x**2 + y**2) <= r1**2
            cell[mask] = 1
        elif shape_type == 'ring':
            mask = ((x**2 + y**2) >= r1**2) & ((x**2 + y**2) <= r2**2)
            cell[mask] = 1
        elif shape_type == 'four_rods':
            centers = [(0.25, 0.25), (-0.25, 0.25), (0.25, -0.25), (-0.25, -0.25)]
            mask = np.zeros_like(x, dtype=bool)
            for cx, cy in centers:
                mask |= ((x - cx)**2 + (y - cy)**2) <= r1**2
            cell[mask] = 1
        elif shape_type == 'cross':
            width = r1
            mask = (np.abs(x) <= width) | (np.abs(y) <= width)
            cell[mask] = 1
        elif shape_type == 'square_ring':
            mask = (np.maximum(np.abs(x), np.abs(y)) >= r1) & (np.maximum(np.abs(x), np.abs(y)) <= r2)
            cell[mask] = 1
            
        return cell

    candidates = []
    for r in [0.1, 0.15, 0.2, 0.25]:
        candidates.append(make_cell_from_params(r, 0, 'disk'))
    for r_in in [0.1, 0.15]:
        for r_out in [0.2, 0.25, 0.3]:
            if r_out > r_in:
                candidates.append(make_cell_from_params(r_in, r_out, 'ring'))
    for r in [0.08, 0.1, 0.12, 0.15]:
        candidates.append(make_cell_from_params(r, 0, 'four_rods'))
    for w in [0.05, 0.08, 0.1]:
        candidates.append(make_cell_from_params(w, 0, 'cross'))
    for r_in in [0.1, 0.15]:
        for r_out in [0.2, 0.25]:
            if r_out > r_in:
                candidates.append(make_cell_from_params(r_in, r_out, 'square_ring'))

    best_cell = candidates[0]
    best_score = -np.inf
    
    # Screen candidates
    for cell in candidates:
        if time.time() > deadline - 30:
            break
        eps = 1.0 + (eps_high - 1.0) * cell
        inv_eps = 1.0 / eps
        fft_coeffs = np.fft.fft2(inv_eps)
        s = score_from_fft(fft_coeffs)
        if s > best_score:
            best_score = s
            best_cell = cell
            best_fft = fft_coeffs.copy()
            best_inv_eps = inv_eps.copy()
            
    # Local search
    current_cell = np.array(best_cell)
    current_score = best_score
    current_fft = best_fft.copy()
    current_inv_eps = best_inv_eps.copy()
    
    # Precompute symmetric partners for each pixel
    partners_map = {}
    for i in range(size):
        for j in range(size):
            partners = [(i, j), (size-1-i, size-1-j), (j, i), (size-1-j, size-1-i)]
            partners_map[(i,j)] = list(set(partners))
            
    n_iterations = 200
    for it in range(n_iterations):
        if time.time() > deadline - 5:
            break
            
        i, j = np.random.randint(0, size, 2)
        partners = partners_map[(i,j)]
        
        trial_cell = current_cell.copy()
        val = trial_cell[partners[0][0], partners[0][1]]
        new_val = 1 - val
        
        for pi, pj in partners:
            trial_cell[pi, pj] = new_val
            
        # Update FFT incrementally
        # delta_inv_eps = new_val - old_val for the changed pixels
        # But wait, inv_eps is not linear in cell value.
        # cell: 0 -> eps 1 -> inv_eps 1
        # cell: 1 -> eps 12 -> inv_eps 1/12
        # So if we flip 0->1, delta = 1/12 - 1 = -11/12
        # If we flip 1->0, delta = 1 - 1/12 = 11/12
        
        delta_val = 0
        if new_val == 1:
            delta_val = (1.0/eps_high) - 1.0
        else:
            delta_val = 1.0 - (1.0/eps_high)
            
        trial_fft = current_fft.copy()
        for pi, pj in partners:
            # Update FFT coefficients
            # F[m,n] += delta * exp(-2pi i (m pi + n pj) / N)
            # m, n range from -N/2 to N/2? No, FFT indices are 0..N-1.
            # The term is exp(-2pi i (m pi + n pj) / N)
            # We can vectorize this update over all m,n
            m_grid, n_grid = np.meshgrid(np.arange(size), np.arange(size), indexing='ij')
            phase = np.exp(-2j * np.pi * (m_grid * pi + n_grid * pj) / size)
            trial_fft += delta_val * phase
            
        s_trial = score_from_fft(trial_fft)
        
        if s_trial > current_score:
            current_cell = trial_cell
            current_score = s_trial
            current_fft = trial_fft
            best_cell = current_cell
            best_score = current_score
            
    return best_cell.tolist()

# EVOLVE-BLOCK-END
