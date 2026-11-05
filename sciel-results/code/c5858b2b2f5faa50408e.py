import time
import math
import numpy as np

def design(problem):
    start_time = time.time()
    deadline = start_time + 215  # 15s safety margin before 230s kill

    size = problem["pixels"]  # 32
    eps_low = problem["eps_low"]   # 1
    eps_high = problem["eps_high"] # 12

    # --- Fast PWE solver setup ---
    # We use a moderate number of plane waves for speed.
    # N_pw_half = 7 -> 15x15 = 225 plane waves.
    # N_pw_half = 8 -> 17x17 = 289 plane waves (matches evaluator exactly).
    # Let's use N_pw_half = 8 to match the evaluator's 289 PWs for accuracy,
    # but we'll optimize the matrix construction.
    N_pw_half = 8
    N_pw = 2 * N_pw_half + 1  # 17
    num_pw = N_pw * N_pw      # 289

    # Plane wave vectors (nx, ny)
    pw_list = []
    for nx in range(-N_pw_half, N_pw_half + 1):
        for ny in range(-N_pw_half, N_pw_half + 1):
            pw_list.append((nx, ny))
    pw_list = np.array(pw_list)  # shape (num_pw, 2)

    # Precompute |k_i - k_j|^2 for all pairs (num_pw, num_pw)
    # k_i - k_j = (nx_i - nx_j, ny_i - ny_j)
    diff_x = pw_list[:, 0:1] - pw_list[:, 0:1].T  # (num_pw, num_pw)
    diff_y = pw_list[:, 1:2] - pw_list[:, 1:2].T
    k_diff_sq = diff_x**2 + diff_y**2  # (num_pw, num_pw)

    # Precompute the integer q indices for each pair (i, j)
    # q_x = nx_i - nx_j, q_y = ny_i - ny_j
    # We need to map these to FFT indices.
    # FFT grid is size x size = 32x32.
    # fftshifted: index 0 is -16, index 31 is +15.
    # So idx = q + 16.
    # But q can be from -16 to +16 (since nx in -8..8, diff in -16..16).
    # If |q| > 15, the coefficient is 0 (outside FFT range).
    q_x_idx = (diff_x + 16).astype(int)  # 0..32
    q_y_idx = (diff_y + 16).astype(int)
    
    # Valid mask: 0 <= idx < 32
    valid_q = (q_x_idx >= 0) & (q_x_idx < size) & (q_y_idx >= 0) & (q_y_idx < size)

    # 30 k-points in the irreducible BZ (Gamma-M-X-Gamma)
    # Gamma: (0,0), M: (0.5, 0), X: (0.5, 0.5)
    # We'll sample 30 points along the path.
    k_points = []
    # 15 points from Gamma to M
    for i in range(15):
        t = i / 14.0
        k_points.append((0.5 * t, 0.0))
    # 15 points from M to X
    for i in range(15):
        t = i / 14.0
        k_points.append((0.5, 0.5 * t))
    # Remove duplicate M point? The evaluator uses 30 distinct points.
    # Let's just use these 30.
    k_points = np.array(k_points)

    def compute_eps_inv_fft(cell):
        """Compute FFT of 1/eps for the given 0/1 cell."""
        eps = np.where(cell == 1, eps_high, eps_low)
        inv_eps = 1.0 / eps
        fft_inv_eps = np.fft.fft2(inv_eps)
        fft_inv_eps /= (size * size)
        fft_inv_eps = np.fft.fftshift(fft_inv_eps)
        return fft_inv_eps

    def evaluate_gap_fast(cell):
        """Evaluate the gap-midgap ratio using the fast precomputed solver."""
        fft_inv_eps = compute_eps_inv_fft(cell)
        
        max_omega_4 = 0.0
        min_omega_5 = float('inf')
        
        for kx, ky in k_points:
            # Build H matrix: H_ij = (1/eps)_{k_i - k_j} * |k_i - k_j|^2
            # (1/eps)_{q} is fft_inv_eps[q_y + 16, q_x + 16] if valid, else 0.
            
            # Extract the coefficients for all pairs
            # Use advanced indexing
            coeff_matrix = np.zeros((num_pw, num_pw), dtype=complex)
            coeff_matrix[valid_q] = fft_inv_eps[q_y_idx[valid_q], q_x_idx[valid_q]]
            
            H = coeff_matrix * k_diff_sq
            
            # Solve eigenvalue problem
            # H is Hermitian
            try:
                vals = np.linalg.eigvalsh(H)
            except:
                continue
                
            # Filter out negative or tiny eigenvalues
            vals = vals[vals > 1e-6]
            if len(vals) < 5:
                continue
                
            omegas = np.sqrt(vals)
            # We need bands 4 and 5 (0-indexed: 3 and 4)
            if len(omegas) >= 5:
                max_omega_4 = max(max_omega_4, omegas[3])
                min_omega_5 = min(min_omega_5, omegas[4])
                
        if min_omega_5 == float('inf'):
            return -1.0
            
        gap = min_omega_5 - max_omega_4
        midgap = (min_omega_5 + max_omega_4) / 2.0
        if midgap == 0:
            return 0.0
        return gap / midgap

    # --- Parameterized Design Family ---
    # We use a structured superlattice: 4 concentric square rings centered at the cell center.
    # Each ring has a fill fraction (0 to 1) and a thickness.
    # Parameters: [fill_0, fill_1, fill_2, fill_3] where fill_i in [0, 1]
    # Ring 0: innermost (core), Ring 3: outermost.
    
    # Precompute the ring masks
    y, x = np.mgrid[0:size, 0:size]
    y = y + 0.5
    x = x + 0.5
    cx, cy = size/2.0, size/2.0
    
    # Define ring boundaries (Chebyshev distance for square rings)
    dist = np.maximum(np.abs(x - cx), np.abs(y - cy))
    
    # Ring 0: dist < 4
    # Ring 1: 4 <= dist < 8
    # Ring 2: 8 <= dist < 12
    # Ring 3: 12 <= dist < 16
    ring_masks = []
    ring_masks.append(dist < 4)
    ring_masks.append((dist >= 4) & (dist < 8))
    ring_masks.append((dist >= 8) & (dist < 12))
    ring_masks.append((dist >= 12) & (dist < 16))
    
    # Convert to float arrays for weighted sum
    ring_masks = [m.astype(float) for m in ring_masks]

    def params_to_cell(params):
        """Convert 4 fill fractions to a 32x32 cell."""
        cell = np.zeros((size, size), dtype=float)
        for i in range(4):
            cell += params[i] * ring_masks[i]
        # Threshold
        cell = (cell > 0.5).astype(int)
        # Enforce symmetry (should be inherent, but just in case)
        cell = (cell + cell[::-1, ::-1]) // 2
        cell = (cell + cell.T) // 2
        return cell

    # --- Differential Evolution ---
    np.random.seed(42)
    pop_size = 20
    n_params = 4
    
    # Initialize population
    population = np.random.rand(pop_size, n_params)
    # Ensure some diversity: mix of high and low fill
    population[0] = [0.5, 0.5, 0.5, 0.5]
    population[1] = [1.0, 0.0, 1.0, 0.0]
    population[2] = [0.0, 1.0, 0.0, 1.0]
    
    scores = np.full(pop_size, -1.0)
    
    # Evaluate initial population
    for i in range(pop_size):
        cell = params_to_cell(population[i])
        scores[i] = evaluate_gap_fast(cell)
    
    best_idx = np.argmax(scores)
    best_params = population[best_idx].copy()
    best_score = scores[best_idx]
    best_cell = params_to_cell(best_params)
    
    # DE loop
    F = 0.8  # Scale factor
    CR = 0.9  # Crossover rate
    
    gen = 0
    while time.time() < deadline - 10:
        gen += 1
        new_population = np.zeros_like(population)
        new_scores = np.zeros(pop_size)
        
        for i in range(pop_size):
            # Select 3 distinct individuals
            a, b, c = np.random.choice(pop_size, 3, replace=False)
            
            # Mutation
            v = population[a] + F * (population[b] - population[c])
            v = np.clip(v, 0, 1)
            
            # Crossover
            trial = np.zeros(n_params)
            j_rand = np.random.randint(n_params)
            for j in range(n_params):
                if np.random.rand() < CR or j == j_rand:
                    trial[j] = v[j]
                else:
                    trial[j] = population[i][j]
            
            # Evaluate
            cell = params_to_cell(trial)
            score = evaluate_gap_fast(cell)
            
            # Selection
            if score >= scores[i]:
                new_population[i] = trial
                new_scores[i] = score
            else:
                new_population[i] = population[i]
                new_scores[i] = scores[i]
                
        population = new_population
        scores = new_scores
        
        best_idx = np.argmax(scores)
        if scores[best_idx] > best_score:
            best_score = scores[best_idx]
            best_params = population[best_idx].copy()
            best_cell = params_to_cell(best_params)
            
        # Check convergence
        if gen % 10 == 0:
            spread = np.std(scores)
            if spread < 1e-4:
                break

    # If we have time, do a local refinement around the best solution
    if time.time() < deadline - 20:
        # Simple local search: perturb each parameter slightly
        for _ in range(50):
            if time.time() > deadline - 10:
                break
            # Randomly pick a parameter to perturb
            j = np.random.randint(n_params)
            trial = best_params.copy()
            trial[j] += np.random.uniform(-0.05, 0.05)
            trial[j] = np.clip(trial[j], 0, 1)
            
            cell = params_to_cell(trial)
            score = evaluate_gap_fast(cell)
            
            if score > best_score:
                best_score = score
                best_params = trial
                best_cell = cell

    return best_cell.tolist()
