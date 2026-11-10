"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time
import math

# EVOLVE-BLOCK-START
def design(problem):
    """
    Designs a 32x32 unit cell for a TM band gap between bands 4 and 5.
    Strategy:
    1. Initialize with a symmetric superlattice configuration (4 dielectric disks).
    2. Use direct pixel-level optimization with a fast plane-wave evaluator.
    3. Enforce C4v symmetry throughout.
    4. Return the best design found within the time budget.
    """
    size = problem["pixels"]  # 32
    eps_low = problem.get("eps_low", 1.0)
    eps_high = problem.get("eps_high", 12.0)
    
    # Time management
    start_time = time.time()
    deadline = start_time + 185.0  # Safety margin for 200s limit

    # --- 1. Initialize with 4 dielectric disks at quarter-points ---
    # This is a known configuration that can open band gaps
    cell = np.zeros((size, size), dtype=int)
    
    # Disk centers at (8,8), (8,24), (24,8), (24,24) with radius ~4
    centers = [(8, 8), (8, 24), (24, 8), (24, 24)]
    radius = 4.0
    
    for cy, cx in centers:
        for i in range(size):
            for j in range(size):
                dist = math.sqrt((i - cy)**2 + (j - cx)**2)
                if dist < radius:
                    cell[i, j] = 1
    
    # Enforce C4v symmetry exactly
    def enforce_symmetry(c):
        sym = np.zeros_like(c)
        for i in range(size):
            for j in range(size):
                # All 8 symmetric points
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
    
    cell = enforce_symmetry(cell)

    # --- 2. Fast Evaluator ---
    # Use 7x7 plane waves (49 modes) for speed
    pw_half = 3
    G_vectors = [(h, k) for h in range(-pw_half, pw_half + 1) 
                 for k in range(-pw_half, pw_half + 1)]
    N_pw = len(G_vectors)
    
    # Precompute G vector arrays
    Gx_arr = np.array([g[0] for g in G_vectors])
    Gy_arr = np.array([g[1] for g in G_vectors])
    
    # Precompute G differences
    G_diffs = np.zeros((N_pw, N_pw, 2), dtype=int)
    for i in range(N_pw):
        for j in range(N_pw):
            G_diffs[i, j, 0] = G_vectors[i][0] - G_vectors[j][0]
            G_diffs[i, j, 1] = G_vectors[i][1] - G_vectors[j][1]
    
    # FFT index lookup
    G_diff_idx_h = G_diffs[:, :, 0] % size
    G_diff_idx_k = G_diffs[:, :, 1] % size
    
    # K-points in irreducible BZ (subset for speed)
    k_points = []
    # Gamma (0,0)
    k_points.append((0.0, 0.0))
    # Along Gamma-M: (0, t)
    for t in [0.1, 0.2, 0.3, 0.4, 0.5]:
        k_points.append((0.0, t))
    # Along M-X: (t, 0.5-t)
    for t in [0.1, 0.2, 0.3, 0.4, 0.5]:
        k_points.append((t, 0.5 - t))
    # X (0.5, 0.5)
    k_points.append((0.5, 0.5))
    
    def evaluate_gap(cell_grid):
        """
        Computes the approximate band gap between bands 4 and 5.
        cell_grid: 32x32 array of 0/1 (0=air, 1=dielectric)
        """
        # Build epsilon grid
        eps_grid = np.where(cell_grid == 1, eps_high, eps_low)
        inv_eps = 1.0 / eps_grid
        
        # FFT for Fourier coefficients
        fft_1eps = np.fft.fft2(inv_eps)
        coeffs = fft_1eps[G_diff_idx_h, G_diff_idx_k] / (size * size)
        
        max_omega_4 = 0.0
        min_omega_5 = float('inf')
        
        for kx, ky in k_points:
            Kx = Gx_arr + kx
            Ky = Gy_arr + ky
            
            # H = (K_i · K_j) * coeffs_{G_i - G_j}
            dots = np.outer(Kx, Kx) + np.outer(Ky, Ky)
            H = dots * coeffs
            H_real = H.real
            
            # Get eigenvalues (we need indices 3 and 4 for bands 4 and 5)
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

    # --- 3. Optimization: Simulated Annealing on pixels ---
    current_cell = cell.copy()
    current_score = evaluate_gap(current_cell)
    best_cell = current_cell.copy()
    best_score = current_score
    
    temperature = 0.5
    cooling_rate = 0.995
    max_iterations = 50000
    iteration = 0
    
    while time.time() < deadline and iteration < max_iterations:
        iteration += 1
        
        # Pick a random symmetric orbit to flip
        # An orbit consists of up to 8 pixels related by C4v symmetry
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
        # Remove duplicates
        orbit = list(set(orbit))
        
        # Create new cell by flipping the orbit
        new_cell = current_cell.copy()
        for pi, pj in orbit:
            new_cell[pi, pj] = 1 - new_cell[pi, pj]
        
        new_score = evaluate_gap(new_cell)
        
        # Accept or reject based on Metropolis criterion
        delta = new_score - current_score
        if delta > 0 or np.random.random() < math.exp(delta / max(temperature, 1e-6)):
            current_cell = new_cell
            current_score = new_score
            if current_score > best_score:
                best_score = current_score
                best_cell = current_cell.copy()
        
        # Cool down
        temperature *= cooling_rate
        if temperature < 0.01:
            temperature = 0.01

    # --- 4. Finalize ---
    final_cell = enforce_symmetry(best_cell)
    return final_cell.tolist()

# EVOLVE-BLOCK-END
