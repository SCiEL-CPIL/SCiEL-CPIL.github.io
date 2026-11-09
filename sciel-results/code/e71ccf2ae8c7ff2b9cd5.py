import numpy as np
import time
import math

def design(problem):
    """
    Designs a 32x32 unit cell for a TM band gap between bands 4 and 5.
    Strategy:
    1. Parameterize the dielectric field using C4v-symmetric Fourier coefficients.
    2. Optimize these coefficients using a fast local search to maximize the gap.
    3. Binarize the resulting field using an optimal threshold search.
    4. Enforce C4v symmetry exactly on the final pixel grid.
    """
    size = problem["pixels"]  # 32
    eps_low = problem.get("eps_low", 1.0)
    eps_high = problem.get("eps_high", 12.0)
    
    # Time management
    start_time = time.time()
    deadline = start_time + 195.0  # Safety margin for 200s limit

    # --- 1. Fast Evaluator Setup ---
    # Use 7x7 plane waves (49 modes) for speed, similar to seed
    pw_half = 3
    G_vectors = [(h, k) for h in range(-pw_half, pw_half + 1) 
                 for k in range(-pw_half, pw_half + 1)]
    N_pw = len(G_vectors)
    
    Gx_arr = np.array([g[0] for g in G_vectors])
    Gy_arr = np.array([g[1] for g in G_vectors])
    
    G_diffs = np.zeros((N_pw, N_pw, 2), dtype=int)
    for i in range(N_pw):
        for j in range(N_pw):
            G_diffs[i, j, 0] = G_vectors[i][0] - G_vectors[j][0]
            G_diffs[i, j, 1] = G_vectors[i][1] - G_vectors[j][1]
    
    G_diff_idx_h = G_diffs[:, :, 0] % size
    G_diff_idx_k = G_diffs[:, :, 1] % size
    
    # K-points in irreducible BZ (subset for speed)
    k_points = []
    k_points.append((0.0, 0.0))
    for t in [0.1, 0.2, 0.3, 0.4, 0.5]:
        k_points.append((0.0, t))
    for t in [0.1, 0.2, 0.3, 0.4, 0.5]:
        k_points.append((t, 0.5 - t))
    k_points.append((0.5, 0.5))
    
    def evaluate_gap_from_eps(eps_grid):
        """
        Computes the approximate band gap between bands 4 and 5 from a continuous eps grid.
        """
        inv_eps = 1.0 / eps_grid
        fft_1eps = np.fft.fft2(inv_eps)
        coeffs = fft_1eps[G_diff_idx_h, G_diff_idx_k] / (size * size)
        
        max_omega_4 = 0.0
        min_omega_5 = float('inf')
        
        for kx, ky in k_points:
            Kx = Gx_arr + kx
            Ky = Gy_arr + ky
            
            dots = np.outer(Kx, Kx) + np.outer(Ky, Ky)
            H = dots * coeffs
            H_real = H.real
            
            eigenvalues = np.linalg.eigvalsh(H_real)
            
            if len(eigenvalues) >= 5:
                w4_sq = eigenvalues[3]
                w5_sq = eigenvalues[4]
                
                w4 = np.sqrt(max(0, w4_sq))
                w5 = np.sqrt(max(0, w5_sq))
                
                if w4 > max_omega_4:
                    max_omega_4 = w4
                if w5 < min_omega_5:
                    min_omega_5 = w5
        
        if min_omega_5 == float('inf'):
            return 0.0
            
        gap = min_omega_5 - max_omega_4
        midgap = (min_omega_5 + max_omega_4) / 2.0
        
        if midgap == 0:
            return 0.0
            
        ratio = 2 * gap / midgap
        return ratio

    def evaluate_gap_from_cell(cell_grid):
        """
        Computes the gap from a binary 0/1 cell grid.
        """
        eps_grid = np.where(cell_grid == 1, eps_high, eps_low)
        return evaluate_gap_from_eps(eps_grid)

    # --- 2. Fourier Synthesis Parameterization ---
    # C4v symmetric basis: cos(2*pi*m*(x-0.5)) * cos(2*pi*n*(y-0.5)) with m<=n
    max_harmonic = 4
    params_idx = []
    for m in range(max_harmonic + 1):
        for n in range(m, max_harmonic + 1):
            params_idx.append((m, n))
    num_params = len(params_idx)
    
    x = np.linspace(0.5/size, 1 - 0.5/size, size)
    X, Y = np.meshgrid(x, x, indexing='ij')
    Xc = X - 0.5
    Yc = Y - 0.5
    
    basis_funcs = []
    for (m, n) in params_idx:
        if m == 0 and n == 0:
            basis_funcs.append(np.ones((size, size)))
        else:
            cm = np.cos(2 * np.pi * m * Xc)
            cn = np.cos(2 * np.pi * n * Yc)
            basis_funcs.append(cm * cn)
    
    def build_eps_from_coeffs(coeffs):
        """
        Reconstructs a continuous epsilon grid from Fourier coefficients.
        Maps linear combination of basis functions to [eps_low, eps_high].
        """
        field = np.zeros((size, size))
        for i, (m, n) in enumerate(params_idx):
            field += coeffs[i] * basis_funcs[i]
        
        # Normalize field to [0, 1]
        f_min = np.min(field)
        f_max = np.max(field)
        if f_max - f_min < 1e-6:
            return np.full((size, size), (eps_low + eps_high) / 2.0)
        
        field_norm = (field - f_min) / (f_max - f_min)
        
        # Map to epsilon range
        eps_grid = eps_low + (eps_high - eps_low) * field_norm
        return eps_grid

    def binarize_eps(eps_grid):
        """
        Finds the best threshold to binarize the continuous eps grid to maximize the gap.
        """
        # Try a range of thresholds
        best_score = -np.inf
        best_cell = None
        
        # Sample thresholds
        thresholds = np.linspace(0.1, 0.9, 17)
        
        for t in thresholds:
            # Threshold relative to the mean
            mean_eps = np.mean(eps_grid)
            std_eps = np.std(eps_grid)
            if std_eps < 1e-6:
                continue
            
            # Threshold value
            thresh = mean_eps + t * std_eps
            
            cell = (eps_grid > thresh).astype(int)
            
            # Enforce symmetry on the binarized cell
            cell = enforce_symmetry(cell)
            
            score = evaluate_gap_from_cell(cell)
            
            if score > best_score:
                best_score = score
                best_cell = cell
        
        if best_cell is None:
            # Fallback: simple threshold at mean
            cell = (eps_grid > np.mean(eps_grid)).astype(int)
            cell = enforce_symmetry(cell)
            return cell, evaluate_gap_from_cell(cell)
            
        return best_cell, best_score

    def enforce_symmetry(c):
        sym = np.zeros_like(c)
        for i in range(size):
            for j in range(size):
                points = [
                    (i, j),
                    (size - 1 - i, j),
                    (i, size - 1 - j),
                    (size - 1 - i, size - 1 - j),
                    (j, i),
                    (size - 1 - j, j),
                    (j, size - 1 - i),
                    (size - 1 - j, size - 1 - i)
                ]
                s = sum(c[pi, pj] for pi, pj in points)
                sym[i, j] = 1 if s >= 4 else 0
        return sym

    # --- 3. Optimization: Local Search on Fourier Coefficients ---
    
    # Initial guess: Start with a simple ring-like structure
    # Coeffs: [DC, (1,0), (0,1), (1,1), (2,0), (0,2), ...]
    # (1,0) and (0,1) create variation in x and y
    # (1,1) creates diagonal variation
    init_coeffs = np.zeros(num_params)
    init_coeffs[0] = 0.0  # DC
    init_coeffs[1] = 1.0  # (1,0)
    init_coeffs[2] = 1.0  # (0,1) is same as (1,0) due to symmetry in our param? No, (0,1) is not in list if m<=n. 
    # List: (0,0), (1,0), (1,1), (2,0), (2,1), (2,2), (3,0)...
    # Index 0: (0,0)
    # Index 1: (1,0)
    # Index 2: (1,1)
    # Index 3: (2,0)
    # Index 4: (2,1)
    # Index 5: (2,2)
    
    # Let's try to create a "hole" in the center or a ring.
    # A ring is often good for TM gaps.
    # We want high eps on the outside, low in the center? Or vice versa?
    # Usually, dielectric rods in air (1=dielectric) are standard.
    # Let's start with a configuration that looks like 4 disks.
    
    # Random initialization for robustness
    np.random.seed(42)
    
    best_global_score = -np.inf
    best_global_cell = None
    
    num_restarts = 10
    
    for r in range(num_restarts):
        if time.time() > deadline:
            break
            
        # Initialize coefficients
        if r == 0:
            coeffs = init_coeffs.copy()
            # Try a simple ring: high frequency component
            coeffs[1] = 1.0 # (1,0)
            coeffs[2] = -0.5 # (1,1)
            coeffs[3] = 0.5 # (2,0)
        else:
            coeffs = np.random.normal(0, 1, num_params)
            # Add some structure
            coeffs[0] = 0
            
        # Local search (Hill Climbing with momentum-like steps)
        step_size = 0.5
        current_coeffs = coeffs.copy()
        
        # Evaluate initial
        eps_grid = build_eps_from_coeffs(current_coeffs)
        current_cell, current_score = binarize_eps(eps_grid)
        
        if current_score > best_global_score:
            best_global_score = current_score
            best_global_cell = current_cell.copy()
            
        no_improve_count = 0
        max_no_improve = 50
        
        while time.time() < deadline and no_improve_count < max_no_improve:
            # Pick a random coefficient to perturb
            idx = np.random.randint(num_params)
            
            # Try positive and negative steps
            best_local_score = current_score
            best_local_coeffs = current_coeffs.copy()
            
            for sign in [1, -1]:
                trial_coeffs = current_coeffs.copy()
                trial_coeffs[idx] += sign * step_size
                
                trial_eps = build_eps_from_coeffs(trial_coeffs)
                trial_cell, trial_score = binarize_eps(trial_eps)
                
                if trial_score > best_local_score:
                    best_local_score = trial_score
                    best_local_coeffs = trial_coeffs
            
            if best_local_score > current_score:
                current_coeffs = best_local_coeffs
                current_score = best_local_score
                no_improve_count = 0
                
                if current_score > best_global_score:
                    best_global_score = current_score
                    best_global_cell = current_cell.copy()
            else:
                no_improve_count += 1
                # Reduce step size if stuck
                step_size *= 0.95
                if step_size < 0.01:
                    break

    # --- 4. Final Polish: Pixel-level refinement on the best cell ---
    if best_global_cell is not None:
        current_cell = best_global_cell.copy()
        current_score = best_global_score
        
        # Simple local pixel flipping to fix symmetry or minor issues
        # We'll do a few iterations of flipping random orbits if it improves
        steps = 50
        for _ in range(steps):
            if time.time() > deadline:
                break
                
            i = np.random.randint(size)
            j = np.random.randint(size)
            
            orbit = [
                (i, j),
                (size - 1 - i, j),
                (i, size - 1 - j),
                (size - 1 - i, size - 1 - j),
                (j, i),
                (size - 1 - j, j),
                (j, size - 1 - i),
                (size - 1 - j, size - 1 - i)
            ]
            orbit = list(set(orbit))
            
            new_cell = current_cell.copy()
            for pi, pj in orbit:
                new_cell[pi, pj] = 1 - new_cell[pi, pj]
                
            new_score = evaluate_gap_from_cell(new_cell)
            
            if new_score > current_score:
                current_cell = new_cell
                current_score = new_score
                if current_score > best_global_score:
                    best_global_score = current_score
                    best_global_cell = current_cell.copy()
    
    # Ensure we have a valid cell
    if best_global_cell is None:
        # Fallback to a simple disk
        cell = np.zeros((size, size), dtype=int)
        centers = [(8, 8), (8, 24), (24, 8), (24, 24)]
        radius = 4.0
        for cy, cx in centers:
            for i in range(size):
                for j in range(size):
                    dist = math.sqrt((i - cy)**2 + (j - cx)**2)
                    if dist < radius:
                        cell[i, j] = 1
        best_global_cell = enforce_symmetry(cell)
    
    final_cell = enforce_symmetry(best_global_cell)
    return final_cell.tolist()
