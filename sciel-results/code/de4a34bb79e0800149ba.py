<<<<<<< SEARCH
import numpy as np
import time
import math
import random

def design(problem):
    start_time = time.time()
    deadline = start_time + 210  # Safety margin before 230s wall-clock limit
    
    size = problem["pixels"]
    eps_low = problem["eps_low"]
    eps_high = problem["eps_high"]
    
    # Fast PWE solver for evaluation
    def compute_gap_fast(cell, n_g=6, n_k=12):
        """
        Fast PWE gap computation with reduced G-vectors and k-points.
        """
        def get_fourier_coefficients(cell, n_g):
            coeffs = np.zeros((2*n_g+1, 2*n_g+1), dtype=complex)
            term_x = np.zeros((size, 2*n_g+1), dtype=complex)
            term_y = np.zeros((size, 2*n_g+1), dtype=complex)
            
            for n in range(-n_g, n_g+1):
                idx = n + n_g
                G = 2 * np.pi * n
                if G == 0:
                    term_x[:, idx] = 1.0 / size
                    term_y[:, idx] = 1.0 / size
                else:
                    sinc_factor = (2.0 / G) * np.sin(G / (2.0 * size))
                    base_exp = np.exp(-1j * G * 0.5 / size)
                    i_vals = np.arange(size)
                    exp_terms = np.exp(i_vals * (-1j * G / size)) * base_exp
                    term_x[:, idx] = sinc_factor * exp_terms
                    term_y[:, idx] = sinc_factor * exp_terms
            
            M = np.array(cell, dtype=float)
            f_coeffs = term_x.T @ (M @ term_y)
            
            delta_eps = eps_high - eps_low
            eps_coeffs = f_coeffs * delta_eps
            eps_coeffs[n_g, n_g] += eps_low
            
            return eps_coeffs

        eps_coeffs = get_fourier_coefficients(cell, n_g)
        
        G_vecs = []
        for nx in range(-n_g, n_g+1):
            for ny in range(-n_g, n_g+1):
                G_vecs.append((2 * np.pi * nx, 2 * np.pi * ny, nx, ny))
        
        N_G = len(G_vecs)
        
        # K points in IBZ
        k_points = []
        for i in range(n_k):
            for j in range(n_k):
                kx = i / (n_k-1) * np.pi
                ky = j / (n_k-1) * np.pi
                if kx <= ky:
                    k_points.append((kx, ky))
        
        all_omegas = []
        
        # Pre-build E indices
        E_indices = np.zeros((N_G, N_G), dtype=int)
        for idx_g, (Gx, Gy, nx, ny) in enumerate(G_vecs):
            for idx_gp, (Gxp, Gyp, nxp, nyp) in enumerate(G_vecs):
                dx = nx - nxp
                dy = ny - nyp
                if -n_g <= dx <= n_g and -n_g <= dy <= n_g:
                    E_indices[idx_g, idx_gp] = (dx + n_g) * (2*n_g+1) + (dy + n_g)
                else:
                    E_indices[idx_g, idx_gp] = -1
        
        eps_flat = eps_coeffs.flatten()
        
        for kx, ky in k_points:
            D = np.zeros(N_G, dtype=float)
            for idx_g, (Gx, Gy, nx, ny) in enumerate(G_vecs):
                kx_g = kx + Gx
                ky_g = ky + Gy
                D[idx_g] = kx_g * kx_g + ky_g * ky_g
            
            valid_mask = E_indices != -1
            E = np.zeros((N_G, N_G), dtype=complex)
            E[valid_mask] = eps_flat[E_indices[valid_mask]]
            
            try:
                sqrt_D = np.sqrt(D)
                inv_sqrt_D = 1.0 / sqrt_D
                H = (sqrt_D[:, None] * E) * inv_sqrt_D[None, :]
                H = (H + H.conj().T) / 2.0
                
                mu_vals = np.linalg.eigvalsh(H)
                mu_real = np.real(mu_vals)
                
                mu_valid = mu_real[mu_real > 1e-10]
                if len(mu_valid) < 5:
                    continue
                
                mu_sorted = np.sort(mu_valid)
                omegas = 1.0 / np.sqrt(mu_sorted)
                omegas = np.sort(omegas)
                
                if len(omegas) >= 5:
                    all_omegas.append(omegas)
            except Exception:
                continue
        
        if not all_omegas:
            return 0.0
            
        max_band4 = max(o[3] for o in all_omegas)
        min_band5 = min(o[4] for o in all_omegas)
        
        gap = min_band5 - max_band4
        midgap = (min_band5 + max_band4) / 2.0
        
        if midgap == 0:
            return 0.0
            
        ratio = gap / midgap
        return ratio
    
    # Generate a symmetric cell from a quadrant representation
    def quadrant_to_cell(quadrant):
        """
        Convert a 16x16 quadrant to a full 32x32 symmetric cell.
        The quadrant represents the top-right quadrant (i=0..15, j=16..31).
        We enforce: cell[i][j] = cell[31-i][j] = cell[i][31-j] = cell[31-i][31-j]
        """
        cell = [[0] * size for _ in range(size)]
        for i in range(16):
            for j in range(16):
                val = quadrant[i][j]
                # Top-right quadrant
                cell[i][31 - j] = val
                # Bottom-right quadrant (vertical flip)
                cell[31 - i][31 - j] = val
                # Top-left quadrant (horizontal flip)
                cell[i][j] = val
                # Bottom-left quadrant (both flips)
                cell[31 - i][j] = val
        return cell
    
    def cell_to_quadrant(cell):
        """Extract the top-right quadrant from a full cell."""
        quadrant = [[0] * 16 for _ in range(16)]
        for i in range(16):
            for j in range(16):
                quadrant[i][j] = cell[i][31 - j]
        return quadrant
    
    # Generate initial diverse designs
    def generate_initial_designs():
        designs = []
        
        # Design 1: Four circular rods
        for r in [0.08, 0.1, 0.12, 0.15]:
            for sp in [0.15, 0.2, 0.25]:
                cell = [[0] * size for _ in range(size)]
                for i in range(size):
                    for j in range(size):
                        x = (i + 0.5) / size - 0.5
                        y = (j + 0.5) / size - 0.5
                        is_diel = False
                        for sx in [1, -1]:
                            for sy in [1, -1]:
                                dx = x - sx * sp
                                dy = y - sy * sp
                                if dx * dx + dy * dy < r * r:
                                    is_diel = True
                        cell[i][j] = 1 if is_diel else 0
                designs.append(cell)
        
        # Design 2: Square ring
        for inner in [0.1, 0.15, 0.2]:
            for outer in [0.3, 0.35, 0.4]:
                cell = [[0] * size for _ in range(size)]
                for i in range(size):
                    for j in range(size):
                        x = (i + 0.5) / size - 0.5
                        y = (j + 0.5) / size - 0.5
                        r_max = max(abs(x), abs(y))
                        if inner < r_max < outer:
                            cell[i][j] = 1
                designs.append(cell)
        
        # Design 3: Cross + corner blocks
        for cross_w in [0.05, 0.1, 0.15]:
            for corner_w in [0.05, 0.1, 0.15]:
                cell = [[0] * size for _ in range(size)]
                for i in range(size):
                    for j in range(size):
                        x = (i + 0.5) / size - 0.5
                        y = (j + 0.5) / size - 0.5
                        ax, ay = abs(x), abs(y)
                        is_diel = False
                        if ax < cross_w / 2 or ay < cross_w / 2:
                            is_diel = True
                        if ax > 0.5 - corner_w and ay > 0.5 - corner_w:
                            is_diel = True
                        cell[i][j] = 1 if is_diel else 0
                designs.append(cell)
        
        # Design 4: Central dielectric block
        for w in [0.1, 0.15, 0.2, 0.25]:
            cell = [[0] * size for _ in range(size)]
            for i in range(size):
                for j in range(size):
                    x = (i + 0.5) / size - 0.5
                    y = (j + 0.5) / size - 0.5
                    if abs(x) < w / 2 and abs(y) < w / 2:
                        cell[i][j] = 1
            designs.append(cell)
        
        # Design 5: Random symmetric designs
        for _ in range(10):
            quadrant = [[random.randint(0, 1) for _ in range(16)] for _ in range(16)]
            cell = quadrant_to_cell(quadrant)
            designs.append(cell)
        
        return designs
    
    # Mutation: flip pixels in the quadrant
    def mutate_quadrant(quadrant, rate=0.05):
        new_quadrant = [row[:] for row in quadrant]
        for i in range(16):
            for j in range(16):
                if random.random() < rate:
                    new_quadrant[i][j] = 1 - new_quadrant[i][j]
        return new_quadrant
    
    # Crossover: combine two quadrants
    def crossover_quadrant(q1, q2):
        new_quadrant = [[0] * 16 for _ in range(16)]
        for i in range(16):
            for j in range(16):
                if random.random() < 0.5:
                    new_quadrant[i][j] = q1[i][j]
                else:
                    new_quadrant[i][j] = q2[i][j]
        return new_quadrant
    
    # Phase 1: Evaluate initial designs and select best
    initial_designs = generate_initial_designs()
    scored_designs = []
    
    for cell in initial_designs:
        if time.time() > deadline - 60:
            break
        score = compute_gap_fast(cell, n_g=6, n_k=12)
        scored_designs.append((cell, score))
    
    scored_designs.sort(key=lambda x: x[1], reverse=True)
    
    # Keep top N for evolution
    top_n = min(20, len(scored_designs))
    population = [scored_designs[i][0] for i in range(top_n)]
    best_cell = scored_designs[0][0]
    best_score = scored_designs[0][1]
    
    # Phase 2: Evolutionary optimization
    population_size = 15
    mutation_rate = 0.03
    crossover_rate = 0.5
    
    for gen in range(50):
        if time.time() > deadline - 30:
            break
        
        new_population = []
        
        # Elitism: keep best 3
        for i in range(min(3, len(population))):
            new_population.append(population[i])
        
        # Generate offspring
        while len(new_population) < population_size:
            if time.time() > deadline - 30:
                break
            
            # Select parents via tournament selection
            parent1_idx = random.randint(0, len(population) - 1)
            parent2_idx = random.randint(0, len(population) - 1)
            parent1 = population[parent1_idx]
            parent2 = population[parent2_idx]
            
            # Convert to quadrants
            q1 = cell_to_quadrant(parent1)
            q2 = cell_to_quadrant(parent2)
            
            # Crossover or mutation
            if random.random() < crossover_rate and len(population) > 1:
                new_q = crossover_quadrant(q1, q2)
            else:
                new_q = [row[:] for row in q1]
            
            # Mutation
            new_q = mutate_quadrant(new_q, mutation_rate)
            
            # Convert back to cell
            new_cell = quadrant_to_cell(new_q)
            new_population.append(new_cell)
        
        # Evaluate new population
        scored_new = []
        for cell in new_population:
            if time.time() > deadline - 30:
                break
            score = compute_gap_fast(cell, n_g=6, n_k=12)
            scored_new.append((cell, score))
        
        scored_new.sort(key=lambda x: x[1], reverse=True)
        
        # Update population
        population = [scored_new[i][0] for i in range(min(population_size, len(scored_new)))]
        
        # Update best
        if scored_new[0][1] > best_score:
            best_score = scored_new[0][1]
            best_cell = scored_new[0][0]
    
    # Phase 3: Local refinement around best solution
    if time.time() < deadline - 15:
        best_quadrant = cell_to_quadrant(best_cell)
        
        # Try more aggressive local search
        for trial in range(30):
            if time.time() > deadline - 15:
                break
            
            # Random pixel flips
            new_quadrant = [row[:] for row in best_quadrant]
            n_flips = random.randint(1, 5)
            for _ in range(n_flips):
                i = random.randint(0, 15)
                j = random.randint(0, 15)
                new_quadrant[i][j] = 1 - new_quadrant[i][j]
            
            new_cell = quadrant_to_cell(new_quadrant)
            score = compute_gap_fast(new_cell, n_g=6, n_k=12)
            
            if score > best_score:
                best_score = score
                best_cell = new_cell
                best_quadrant = new_quadrant
    
    return best_cell
=======
import numpy as np
import time
import math
import random

def design(problem):
    start_time = time.time()
    deadline = start_time + 220  # Safety margin before 230s wall-clock limit
    
    size = problem["pixels"]
    eps_low = problem["eps_low"]
    eps_high = problem["eps_high"]
    
    # Fast FFT-based PWE surrogate for evaluation
    def compute_gap_fft(cell, n_g=7, n_k=10):
        """
        Fast PWE gap computation using FFT for Fourier coefficients.
        """
        M = np.array(cell, dtype=float)
        
        # Compute Fourier coefficients of eps(r) = eps_low + delta_eps * M
        delta_eps = eps_high - eps_low
        N = 2 * n_g + 1
        eps_coeffs = np.zeros((N, N), dtype=complex)
        
        # Use FFT to get Fourier coefficients efficiently
        # Pad to power of 2 for FFT speed
        fft_size = 1 << (size - 1).bit_length()
        M_padded = np.zeros((fft_size, fft_size), dtype=float)
        M_padded[:size, :size] = M
        
        eps_padded = eps_low + delta_eps * M_padded
        fft_result = np.fft.fft2(eps_padded) / (fft_size * fft_size)
        
        # Extract the needed Fourier coefficients
        for nx in range(-n_g, n_g + 1):
            for ny in range(-n_g, n_g + 1):
                idx_x = nx % fft_size
                idx_y = ny % fft_size
                eps_coeffs[nx + n_g, ny + n_g] = fft_result[idx_x, idx_y]
        
        # G vectors
        G_vecs = []
        for nx in range(-n_g, n_g + 1):
            for ny in range(-n_g, n_g + 1):
                G_vecs.append((2 * np.pi * nx, 2 * np.pi * ny, nx, ny))
        
        N_G = len(G_vecs)
        
        # K points in IBZ (triangular region: 0 <= ky <= kx <= pi)
        k_points = []
        for i in range(n_k + 1):
            for j in range(n_k + 1):
                kx = i / float(n_k) * np.pi
                ky = j / float(n_k) * np.pi
                if ky <= kx:
                    k_points.append((kx, ky))
        
        all_omegas = []
        
        # Pre-build E indices
        E_indices = np.full((N_G, N_G), -1, dtype=int)
        for idx_g, (Gx, Gy, nx, ny) in enumerate(G_vecs):
            for idx_gp, (Gxp, Gyp, nxp, nyp) in enumerate(G_vecs):
                dx = nx - nxp
                dy = ny - nyp
                if -n_g <= dx <= n_g and -n_g <= dy <= n_g:
                    E_indices[idx_g, idx_gp] = (dx + n_g) * N + (dy + n_g)
        
        eps_flat = eps_coeffs.flatten()
        
        for kx, ky in k_points:
            D = np.zeros(N_G, dtype=float)
            for idx_g, (Gx, Gy, nx, ny) in enumerate(G_vecs):
                kx_g = kx + Gx
                ky_g = ky + Gy
                D[idx_g] = kx_g * kx_g + ky_g * ky_g
            
            valid_mask = E_indices != -1
            E = np.zeros((N_G, N_G), dtype=complex)
            E[valid_mask] = eps_flat[E_indices[valid_mask]]
            
            try:
                sqrt_D = np.sqrt(D)
                inv_sqrt_D = 1.0 / sqrt_D
                H = (sqrt_D[:, None] * E) * inv_sqrt_D[None, :]
                H = (H + H.conj().T) / 2.0
                
                mu_vals = np.linalg.eigvalsh(H)
                mu_real = np.real(mu_vals)
                
                mu_valid = mu_real[mu_real > 1e-10]
                if len(mu_valid) < 5:
                    continue
                
                mu_sorted = np.sort(mu_valid)
                omegas = 1.0 / np.sqrt(mu_sorted)
                omegas = np.sort(omegas)
                
                if len(omegas) >= 5:
                    all_omegas.append(omegas)
            except Exception:
                continue
        
        if not all_omegas:
            return 0.0
            
        max_band4 = max(o[3] for o in all_omegas)
        min_band5 = min(o[4] for o in all_omegas)
        
        gap = min_band5 - max_band4
        midgap = (min_band5 + max_band4) / 2.0
        
        if midgap == 0:
            return 0.0
            
        ratio = gap / midgap
        return ratio
    
    # Generate a symmetric cell with four circular rods
    def make_rod_cell(radius, offset, size=32):
        """
        Create a 32x32 cell with four circular rods at (±offset, ±offset).
        Ensures C4v symmetry.
        """
        cell = [[0] * size for _ in range(size)]
        for i in range(size):
            for j in range(size):
                x = (i + 0.5) / size - 0.5
                y = (j + 0.5) / size - 0.5
                is_diel = False
                for sx in [1, -1]:
                    for sy in [1, -1]:
                        dx = x - sx * offset
                        dy = y - sy * offset
                        if dx * dx + dy * dy < radius * radius:
                            is_diel = True
                cell[i][j] = 1 if is_diel else 0
        return cell
    
    # Verify and enforce symmetry
    def enforce_symmetry(cell):
        """
        Enforce C4v symmetry: cell == cell[::-1] row-wise, column-wise, and transpose.
        """
        size = len(cell)
        sym_cell = [[0] * size for _ in range(size)]
        for i in range(size):
            for j in range(size):
                # Average over symmetry operations (majority vote for binary)
                vals = [
                    cell[i][j],
                    cell[size-1-i][j],
                    cell[i][size-1-j],
                    cell[size-1-i][size-1-j],
                    cell[j][i],
                    cell[size-1-j][i],
                    cell[j][size-1-i],
                    cell[size-1-j][size-1-i],
                ]
                sym_cell[i][j] = 1 if sum(vals) >= 4 else 0
        return sym_cell
    
    # Phase 1: Coarse grid search over rod parameters
    best_score = -1.0
    best_cell = None
    
    radius_vals = [0.06, 0.08, 0.10, 0.12, 0.14, 0.16, 0.18, 0.20]
    offset_vals = [0.12, 0.15, 0.18, 0.21, 0.24, 0.27, 0.30]
    
    for r in radius_vals:
        if time.time() > deadline - 30:
            break
        for off in offset_vals:
            if time.time() > deadline - 30:
                break
            cell = make_rod_cell(r, off, size)
            cell = enforce_symmetry(cell)
            score = compute_gap_fft(cell, n_g=7, n_k=10)
            
            if score > best_score:
                best_score = score
                best_cell = cell
    
    # Phase 2: Local refinement around best
    if best_cell is not None and time.time() < deadline - 20:
        # Find the best parameters from Phase 1
        best_r = None
        best_off = None
        best_diff = float('inf')
        
        for r in radius_vals:
            for off in offset_vals:
                cell_test = make_rod_cell(r, off, size)
                cell_test = enforce_symmetry(cell_test)
                diff = sum(1 for i in range(size) for j in range(size) if cell_test[i][j] != best_cell[i][j])
                if diff < best_diff:
                    best_diff = diff
                    best_r = r
                    best_off = off
        
        if best_r is not None:
            # Fine grid around best
            r_steps = [best_r + d for d in [-0.02, -0.01, 0.0, 0.01, 0.02, 0.03, 0.04]]
            off_steps = [best_off + d for d in [-0.02, -0.01, 0.0, 0.01, 0.02, 0.03, 0.04]]
            
            for r in r_steps:
                if time.time() > deadline - 20:
                    break
                if r < 0.03 or r > 0.3:
                    continue
                for off in off_steps:
                    if time.time() > deadline - 20:
                        break
                    if off < 0.05 or off > 0.4:
                        continue
                    cell = make_rod_cell(r, off, size)
                    cell = enforce_symmetry(cell)
                    score = compute_gap_fft(cell, n_g=7, n_k=10)
                    
                    if score > best_score:
                        best_score = score
                        best_cell = cell
    
    # Phase 3: Try a few alternative topologies if time permits
    if time.time() < deadline - 15:
        # Central ring
        for inner in [0.05, 0.08, 0.12]:
            if time.time() > deadline - 15:
                break
            for outer in [0.15, 0.20, 0.25, 0.30]:
                if time.time() > deadline - 15:
                    break
                cell = [[0] * size for _ in range(size)]
                for i in range(size):
                    for j in range(size):
                        x = (i + 0.5) / size - 0.5
                        y = (j + 0.5) / size - 0.5
                        r = math.sqrt(x*x + y*y)
                        if inner < r < outer:
                            cell[i][j] = 1
                cell = enforce_symmetry(cell)
                score = compute_gap_fft(cell, n_g=7, n_k=10)
                if score > best_score:
                    best_score = score
                    best_cell = cell
    
    # Phase 4: Final local pixel optimization on best cell
    if best_cell is not None and time.time() < deadline - 10:
        current = [row[:] for row in best_cell]
        current_score = best_score
        
        for iteration in range(20):
            if time.time() > deadline - 10:
                break
            
            improved = False
            # Try flipping each pixel in the top-right quadrant and its symmetric partners
            for i in range(16):
                if time.time() > deadline - 10:
                    break
                for j in range(16):
                    # Get symmetric partners
                    partners = [
                        (i, j),
                        (size-1-i, j),
                        (i, size-1-j),
                        (size-1-i, size-1-j),
                        (j, i),
                        (size-1-j, i),
                        (j, size-1-i),
                        (size-1-j, size-1-i),
                    ]
                    
                    # Flip all partners
                    for (pi, pj) in partners:
                        current[pi][pj] = 1 - current[pi][pj]
                    
                    test_score = compute_gap_fft(current, n_g=7, n_k=10)
                    
                    if test_score > current_score:
                        current_score = test_score
                        improved = True
                    else:
                        # Revert
                        for (pi, pj) in partners:
                            current[pi][pj] = 1 - current[pi][pj]
            
            if not improved:
                break
        
        if current_score > best_score:
            best_score = current_score
            best_cell = current
    
    # Final symmetry enforcement
    if best_cell is not None:
        best_cell = enforce_symmetry(best_cell)
    
    # Fallback if no valid cell was found
    if best_cell is None:
        best_cell = make_rod_cell(0.12, 0.20, size)
        best_cell = enforce_symmetry(best_cell)
    
    return best_cell
>>>>>>> REPLACE

