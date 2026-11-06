"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time

# EVOLVE-BLOCK-START
def design(problem):
    """
    Design a 32x32 unit cell to maximize the TM band gap between bands 4 and 5.
    
    Strategy:
    1. Fast internal PWE solver using vectorized matrix construction.
    2. Parametric family: two concentric superellipse rings (inner rod + outer ring)
       with D4 symmetry by construction.
    3. Differential evolution global optimizer over the parametric space.
    4. Final local pixel-flip refinement.
    5. Maintain D4 symmetry and stay within time budget.
    """
    size = problem["pixels"]
    eps_high = problem.get("eps_high", 12)
    lower_band = problem.get("lower_band", 4)
    
    start_time = time.time()
    deadline = start_time + 210.0
    
    def check_time():
        return time.time() > deadline - 15.0
    
    # Fast internal PWE solver for TM modes using vectorized operations
    def compute_tm_bands(cell, n_pw=13, n_kpts=15):
        """
        Compute TM band structure using plane-wave expansion (vectorized).
        Returns gap-midgap ratio between bands 4 and 5.
        """
        cell = np.array(cell, dtype=float)
        eps = 1.0 + (eps_high - 1.0) * cell
        inv_eps = 1.0 / eps
        
        N = size
        fft_coeffs = np.fft.fft2(inv_eps) / (N * N)
        
        half = n_pw // 2
        m_vals = np.arange(-half, half + 1)
        n_total = len(m_vals) ** 2
        
        g_indices = [(m, n) for m in m_vals for n in m_vals]
        g_m = np.array([g[0] for g in g_indices])
        g_n = np.array([g[1] for g in g_indices])
        
        dm = g_m[:, None] - g_m[None, :]
        dn = g_n[:, None] - g_n[None, :]
        idx_m = dm % N
        idx_n = dn % N
        S = fft_coeffs[idx_m, idx_n]
        
        # k-points in irreducible BZ
        k_points = []
        n_per_seg = max(1, n_kpts // 3)
        for t in np.linspace(0, 0.5, n_per_seg):
            k_points.append((t, 0.0))
        for t in np.linspace(0.5, 0, n_per_seg):
            k_points.append((0.5, t))
        for t in np.linspace(0.5, 0, n_per_seg):
            k_points.append((t, t))
        
        k_points_unique = []
        for kp in k_points:
            if not any(np.allclose(kp, kp2, atol=1e-10) for kp2 in k_points_unique):
                k_points_unique.append(kp)
        
        max_omega4 = -np.inf
        min_omega5 = np.inf
        
        for kx, ky in k_points_unique:
            if check_time():
                break
            
            gk_m = g_m + kx
            gk_n = g_n + ky
            gk_sq = gk_m**2 + gk_n**2
            
            A = S * gk_sq[None, :]
            B = S
            
            try:
                from scipy.linalg import eig
                vals, _ = eig(A, B)
                vals_real = np.real(vals)
                vals_real = vals_real[np.isfinite(vals_real)]
                vals_sorted = np.sort(vals_real)
                
                if len(vals_sorted) >= 5:
                    omega4 = vals_sorted[3]
                    omega5 = vals_sorted[4]
                    if omega5 > 0 and omega4 > 0:
                        max_omega4 = max(max_omega4, omega4)
                        min_omega5 = min(min_omega5, omega5)
            except Exception:
                pass
        
        if max_omega4 > -np.inf and min_omega5 > 0:
            gap = min_omega5 - max_omega4
            midgap = (min_omega5 + max_omega4) / 2.0
            if midgap > 0:
                return 2.0 * gap / midgap
            else:
                return -np.inf
        else:
            return -np.inf
    
    # Parametric cell generator: two concentric superellipse rings
    # Parameters: [inner_r, inner_n, outer_r, outer_n, center_r, center_n]
    # inner_r: radius of inner ring (0 = no inner ring, just solid rod)
    # inner_n: superellipse exponent for inner boundary
    # outer_r: radius of outer boundary
    # outer_n: superellipse exponent for outer boundary
    # center_r: radius of central hole (0 = no central hole)
    # center_n: superellipse exponent for central hole
    def make_parametric_cell(params):
        """
        Generate a D4-symmetric cell from parameters.
        params = [inner_r, inner_n, outer_r, outer_n, center_r, center_n]
        
        The cell is: dielectric where (r_outer <= 1) AND NOT (r_inner <= 1) AND NOT (r_center <= 1)
        If inner_r == 0, the inner ring condition is skipped (solid from center outward).
        If center_r == 0, the center hole condition is skipped.
        """
        inner_r = max(0.0, params[0])
        inner_n = max(1.5, params[1])
        outer_r = max(0.05, params[2])
        outer_n = max(1.5, params[3])
        center_r = max(0.0, params[4])
        center_n = max(1.5, params[5])
        
        cell = np.zeros((size, size), dtype=int)
        coords = (np.arange(size) + 0.5) / size - 0.5
        x_grid, y_grid = np.meshgrid(coords, coords, indexing='ij')
        
        # Outer boundary: must be inside
        r_outer = np.abs(x_grid / outer_r)**outer_n + np.abs(y_grid / outer_r)**outer_n
        mask = (r_outer <= 1.0 + 1e-9)
        
        # Inner ring: must be outside (if inner_r > 0)
        if inner_r > 0.01:
            r_inner = np.abs(x_grid / inner_r)**inner_n + np.abs(y_grid / inner_r)**inner_n
            mask &= (r_inner > 1.0 + 1e-9)
        
        # Center hole: must be outside (if center_r > 0)
        if center_r > 0.01:
            r_center = np.abs(x_grid / center_r)**center_n + np.abs(y_grid / center_r)**center_n
            mask &= (r_center > 1.0 + 1e-9)
        
        cell[mask] = 1
        
        # Enforce D4 symmetry
        cell = ((cell + cell.T) // 2).astype(int)
        cell = ((cell + cell[::-1, ::-1]) // 2).astype(int)
        
        return cell.tolist()
    
    # Parameter bounds for differential evolution
    # [inner_r, inner_n, outer_r, outer_n, center_r, center_n]
    bounds = [
        (0.0, 0.35),    # inner_r: 0 means solid center
        (1.5, 8.0),     # inner_n: superellipse exponent
        (0.10, 0.45),   # outer_r: outer radius
        (1.5, 8.0),     # outer_n: superellipse exponent
        (0.0, 0.25),    # center_r: 0 means no center hole
        (1.5, 8.0),     # center_n: superellipse exponent
    ]
    
    n_params = len(bounds)
    
    # Differential Evolution optimizer
    def differential_evolution(pop_size=24, max_generations=80, F=0.8, CR=0.9):
        """
        Differential evolution for maximizing the band gap.
        """
        rng = np.random.RandomState(42)
        
        # Initialize population
        population = np.zeros((pop_size, n_params))
        for i in range(pop_size):
            for j in range(n_params):
                population[i, j] = rng.uniform(bounds[j][0], bounds[j][1])
        
        # Evaluate initial population
        scores = np.zeros(pop_size)
        for i in range(pop_size):
            if check_time():
                break
            cell = make_parametric_cell(population[i])
            scores[i] = compute_tm_bands(cell)
        
        best_idx = np.argmax(scores)
        best_score = scores[best_idx]
        best_params = population[best_idx].copy()
        
        for gen in range(max_generations):
            if check_time():
                break
            
            new_population = population.copy()
            new_scores = scores.copy()
            
            for i in range(pop_size):
                if check_time():
                    break
                
                # Select three distinct random individuals
                while True:
                    a, b, c = rng.choice(pop_size, 3, replace=False)
                    if len(set([a, b, c, i])) == 4:
                        break
                
                # Mutation: v = a + F * (b - c)
                v = population[a] + F * (population[b] - population[c])
                
                # Clamp to bounds
                for j in range(n_params):
                    v[j] = np.clip(v[j], bounds[j][0], bounds[j][1])
                
                # Crossover
                trial = population[i].copy()
                j_rand = rng.randint(n_params)
                for j in range(n_params):
                    if rng.random() < CR or j == j_rand:
                        trial[j] = v[j]
                
                # Evaluate trial
                cell = make_parametric_cell(trial)
                trial_score = compute_tm_bands(cell)
                
                # Selection
                if trial_score >= new_scores[i]:
                    new_population[i] = trial
                    new_scores[i] = trial_score
            
            population = new_population
            scores = new_scores
            
            gen_best_idx = np.argmax(scores)
            if scores[gen_best_idx] > best_score:
                best_score = scores[gen_best_idx]
                best_params = population[gen_best_idx].copy()
        
        return best_params, best_score
    
    # Phase 1: Differential evolution
    best_params, best_score = differential_evolution(pop_size=20, max_generations=60)
    
    # Phase 2: Local refinement with pixel flips
    if not check_time():
        best_cell = make_parametric_cell(best_params)
        best_arr = np.array(best_cell)
        
        # Multiple rounds of local search
        for round_num in range(3):
            if check_time():
                break
            
            improved = False
            max_flips = 40
            
            for attempt in range(max_flips):
                if check_time():
                    break
                
                # Randomly pick a pixel in the upper-left quadrant (for D4 symmetry)
                i = np.random.randint(0, size // 2)
                j = np.random.randint(0, size // 2)
                
                # Get the symmetric orbit (up to 4 pixels)
                orbit = [(i, j), (j, i), (size-1-i, size-1-j), (size-1-j, size-1-i)]
                # Remove duplicates
                orbit = list(set(orbit))
                
                # Check if all pixels in orbit have the same value
                vals = set(best_arr[p, q] for p, q in orbit)
                if len(vals) != 1:
                    continue
                
                # Try flipping
                new_val = 1 - best_arr[i, j]
                for p, q in orbit:
                    best_arr[p, q] = new_val
                
                new_cell = best_arr.tolist()
                new_score = compute_tm_bands(new_cell)
                
                if new_score > best_score:
                    best_score = new_score
                    best_cell = new_cell
                    improved = True
                else:
                    # Revert
                    for p, q in orbit:
                        best_arr[p, q] = 1 - new_val
            
            if not improved:
                break
    
    return best_cell

# EVOLVE-BLOCK-END
