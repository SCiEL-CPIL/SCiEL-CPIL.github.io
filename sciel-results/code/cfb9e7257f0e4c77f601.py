import numpy as np
import time
import scipy.linalg as sla

def design(problem):
    """
    Designs a 32x32 photonic crystal unit cell maximizing the TM band gap 
    between the 4th and 5th bands using a level-set/spline boundary optimization.
    """
    size = problem["pixels"]
    eps_low = problem["eps_low"]
    eps_high = problem["eps_high"]
    
    # Time management
    start_time = time.time()
    timeout = 210.0 # Safety margin before 230s kill (evaluator budget is 200s, but let's be conservative)

    # Pre-compute grid coordinates
    # x, y in [0, 1)
    x_coords = (np.arange(size) + 0.5) / size
    y_coords = (np.arange(size) + 0.5) / size
    X, Y = np.meshgrid(x_coords, y_coords, indexing='ij')
    
    # Distance from center (0.5, 0.5)
    dx = X - 0.5
    dy = Y - 0.5
    R = np.sqrt(dx**2 + dy**2)
    Theta = np.arctan2(dy, dx)
    
    # Surrogate Solver Setup
    # Use 289 PWs (17x17) to closely match the evaluator for accuracy.
    # 17x17 = 289. Matrix size 289x289.
    # 30 k-points as specified.
    # 289^3 is ~24 million ops for inversion/diagonalization.
    # 30 points * 24M ~ 720M ops. In NumPy/SciPy this is fast (<1s per eval if optimized).
    # Let's use 17x17 for fidelity.
    N_pw = 17
    half = N_pw // 2
    G_list = []
    for i in range(-half, half + 1):
        for j in range(-half, half + 1):
            G_list.append((i, j))
    G = np.array(G_list)
    N_G = len(G)
    
    # Key K-points in the irreducible BZ
    # The evaluator uses 30 wave vectors. We should use a representative set.
    # Gamma (0,0), X (0.5, 0), M (0.5, 0.5) are critical.
    # Interior points are important for band edges.
    # We will generate a grid of points in the irreducible wedge: 0 <= kx <= ky, kx+ky <= 0.5? 
    # No, irreducible wedge for square is 0 <= kx <= ky, kx, ky >= 0? 
    # Standard irreducible wedge: 0 <= kx <= ky, kx+ky <= 0.5 is for hexagonal?
    # For square: 0 <= kx <= ky, 0 <= ky <= 0.5? No.
    # The full BZ is [-0.5, 0.5]^2. Irreducible is 1/8th: 0 <= kx <= ky, kx+ky <= 0.5? 
    # Actually, for square symmetry, the irreducible wedge is 0 <= kx <= ky, kx >= 0, ky >= 0, and kx+ky <= 0.5 is not right.
    # It's the triangle with vertices (0,0), (0.5, 0), (0.5, 0.5).
    # So 0 <= kx <= ky <= 0.5.
    
    # Let's define a fixed set of 30 k-points covering this wedge.
    k_points = []
    # High symmetry
    k_points.append((0.0, 0.0))
    k_points.append((0.5, 0.0))
    k_points.append((0.5, 0.5))
    k_points.append((0.25, 0.0))
    k_points.append((0.25, 0.25))
    k_points.append((0.5, 0.25))
    
    # Interior grid
    # We want to catch band edges.
    # Let's add a 5x5 grid in the wedge.
    # kx in [0, 0.25], ky in [kx, 0.5]
    # This is getting complicated to generate exactly 30.
    # Let's just use a dense grid and pick 30.
    # Or just use a fixed set that works well.
    
    # Common points:
    k_points.extend([
        (0.1, 0.1), (0.1, 0.2), (0.1, 0.3), (0.1, 0.4),
        (0.2, 0.2), (0.2, 0.3), (0.2, 0.4), (0.2, 0.5),
        (0.3, 0.3), (0.3, 0.4), (0.3, 0.5),
        (0.4, 0.4), (0.4, 0.5),
        (0.0, 0.1), (0.0, 0.2), (0.0, 0.3), (0.0, 0.4),
        (0.05, 0.15), (0.15, 0.25), (0.25, 0.35),
        (0.1, 0.15), (0.2, 0.25), (0.3, 0.35),
        (0.1, 0.25), (0.2, 0.4),
        (0.15, 0.2), (0.25, 0.3),
        (0.05, 0.1), (0.15, 0.1),
    ])
    
    # Remove duplicates and ensure within wedge
    unique_kpts = []
    seen = set()
    for k in k_points:
        kx, ky = k
        if kx < 0 or ky < 0 or kx > 0.5 or ky > 0.5: continue
        if kx > ky + 1e-9: continue # Enforce kx <= ky
        key = (round(kx, 4), round(ky, 4))
        if key not in seen:
            seen.add(key)
            unique_kpts.append(k)
    k_points = unique_kpts[:30] # Take first 30

    def calculate_gap(mask):
        """
        Calculates the gap ratio using the surrogate solver.
        mask: 32x32 array of 0/1.
        Returns: gap_ratio
        """
        # Compute 1/eps mask
        inv_eps_mask = np.zeros_like(mask, dtype=float)
        inv_eps_mask[mask == 1] = 1.0 / eps_high
        inv_eps_mask[mask == 0] = 1.0 / eps_low
        
        # Compute Fourier coefficients of 1/eps
        # c_{n, m} = (1/N^2) sum_{i,j} f(i,j) exp(-i 2pi (ni + mj)/N)
        # We need c for G in -8..8 (since N_pw=17, half=8)
        
        # Using FFT
        fft_inv_eps = np.fft.fft2(inv_eps_mask)
        fft_inv_eps = np.fft.fftshift(fft_inv_eps)
        
        # Build M matrix (1/eps Fourier coefficients)
        # M_{G, G'} = c_{G - G'}
        M = np.zeros((N_G, N_G), dtype=complex)
        
        # Precompute indices
        # G is (i, j). Index in shifted FFT: 16 + i, 16 + j.
        # Wait, size is 32. fftshift puts 0 at 16.
        # So freq f is at index f + 16.
        
        for idx1, G1 in enumerate(G):
            for idx2, G2 in enumerate(G):
                dG = G1 - G2
                # dG components are in -16..16
                # Index: 16 + dG[0], 16 + dG[1]
                # Check bounds: 16 +/- 8 = 8..24. Within 0..31.
                c = fft_inv_eps[16 + int(dG[0]), 16 + int(dG[1])] / (size * size)
                M[idx1, idx2] = c
        
        # Solve for each k-point
        min_omega_5 = float('inf')
        max_omega_4 = 0.0
        
        for k in k_points:
            kx, ky = k
            # D matrix values: (k + G)^2
            D_vals = np.array([ (kx + gx)**2 + (ky + gy)**2 for gx, gy in G ])
            
            # Generalized eigenvalue problem: D a = omega^2 M a
            # Use scipy.linalg.eigh
            try:
                # M is Hermitian positive definite
                eigvals, eigvecs = sla.eigh(D_vals, M)
                eigvals = np.sort(eigvals)
                
                if len(eigvals) >= 5:
                    w4_sq = eigvals[3]
                    w5_sq = eigvals[4]
                    
                    if w4_sq > 0 and w5_sq > 0:
                        w4 = np.sqrt(w4_sq)
                        w5 = np.sqrt(w5_sq)
                        
                        max_omega_4 = max(max_omega_4, w4)
                        min_omega_5 = min(min_omega_5, w5)
            except Exception:
                pass
        
        if max_omega_4 == 0.0 or min_omega_5 == float('inf'):
            return -1.0
            
        gap = min_omega_5 - max_omega_4
        midgap = (min_omega_5 + max_omega_4) / 2.0
        if midgap == 0:
            return 0.0
        return 2.0 * gap / midgap

    best_score = -1.0
    best_mask = None
    
    # Level-set / Spline representation
    # Define boundary R(Theta) using Fourier series to allow arbitrary shapes.
    # R(Theta) = R0 + sum_{n} a_n cos(4n Theta) + b_n sin(4n Theta)
    # Since we need 4-fold symmetry, only multiples of 4 are allowed.
    # Let's use n = 1, 2, 3, 4.
    # Params: R0, a1, b1, a2, b2, a3, b3, a4, b4
    # 9 parameters.
    
    # Initial guess: Circle
    # R0 = 0.2
    init_params = [0.2, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    
    def generate_mask_from_params(params):
        R0 = params[0]
        a1, b1, a2, b2, a3, b3, a4, b4 = params[1:9]
        
        # R(Theta) = R0 + a1*cos(4T) + b1*sin(4T) + ...
        # Note: sin(4T) breaks reflection symmetry?
        # Reflection x->-x: T -> -T. cos(-4T)=cos(4T), sin(-4T)=-sin(4T).
        # So b terms break reflection symmetry.
        # The problem requires: cell == cell[::-1] row-wise and col-wise.
        # This implies reflection symmetry.
        # So we MUST set b_n = 0.
        # Only cosine terms are allowed for reflection symmetry.
        
        # Let's enforce b=0 in the search or just not use them.
        # We will search only over R0 and a_n.
        
        R_target = R0 + a1 * np.cos(4*Theta) + a2 * np.cos(8*Theta) + a3 * np.cos(12*Theta) + a4 * np.cos(16*Theta)
        
        mask = (R < R_target).astype(int)
        
        # Enforce exact symmetry
        mask = np.minimum(mask, mask[::-1, :])
        mask = np.minimum(mask, mask[:, ::-1])
        mask = np.minimum(mask, mask.T)
        
        return mask

    def objective(params):
        mask = generate_mask_from_params(params)
        score = calculate_gap(mask)
        return -score # Minimize negative score
    
    # Nelder-Mead optimization
    # Parameters: R0, a1, a2, a3, a4
    # Initial: [0.2, 0, 0, 0, 0]
    
    # Let's do a coarse grid search first to find a good region, then refine.
    # Or just start Nelder-Mead with a reasonable initial simplex.
    
    # Initial simplex
    x0 = np.array([0.2, 0.0, 0.0, 0.0, 0.0])
    # Perturbations
    dx = 0.05
    simplex = [x0.copy()]
    for i in range(5):
        p = x0.copy()
        p[i] += dx
        simplex.append(p)
    
    # Nelder-Mead implementation (simple version)
    def nelder_mead(f, x0, max_iter=100, alpha=1.0, gamma=2.0, rho=0.5, sigma=0.5):
        n = len(x0)
        simplex = [x0.copy()]
        for i in range(n):
            p = x0.copy()
            p[i] += 0.05
            simplex.append(p)
        
        fvals = [f(p) for p in simplex]
        
        for it in range(max_iter):
            if time.time() - start_time > timeout - 5:
                break
                
            # Sort
            order = np.argsort(fvals)
            simplex = [simplex[i] for i in order]
            fvals = [fvals[i] for i in order]
            
            # Centroid
            centroid = np.mean(simplex[:-1], axis=0)
            
            # Reflection
            xr = centroid + alpha * (centroid - simplex[-1])
            fr = f(xr)
            
            if fr < fvals[0]:
                # Expansion
                xe = centroid + gamma * (xr - centroid)
                fe = f(xe)
                if fe < fr:
                    simplex[-1] = xe
                    fvals[-1] = fe
                else:
                    simplex[-1] = xr
                    fvals[-1] = fr
            elif fr < fvals[-2]:
                simplex[-1] = xr
                fvals[-1] = fr
            else:
                # Contraction
                if fr < fvals[-1]:
                    # Outside contraction
                    xc = centroid + rho * (xr - centroid)
                else:
                    # Inside contraction
                    xc = centroid - rho * (xr - centroid)
                fc = f(xc)
                if fc < min(fr, fvals[-1]):
                    simplex[-1] = xc
                    fvals[-1] = fc
                else:
                    # Shrink
                    for i in range(1, n+1):
                        simplex[i] = simplex[0] + sigma * (simplex[i] - simplex[0])
                        fvals[i] = f(simplex[i])
        
        # Return best
        best_idx = np.argmin(fvals)
        return simplex[best_idx], fvals[best_idx]

    # Run Nelder-Mead
    # Note: f is objective (negative gap)
    x_opt, f_opt = nelder_mead(objective, x0, max_iter=50)
    
    # Evaluate best
    best_mask = generate_mask_from_params(x_opt)
    best_score = calculate_gap(best_mask)
    
    # Try a few other starting points if time permits
    if time.time() - start_time < timeout - 20:
        # Try different R0
        for R0_try in [0.15, 0.25, 0.3]:
            if time.time() - start_time > timeout - 10:
                break
            x0_try = np.array([R0_try, 0.0, 0.0, 0.0, 0.0])
            x_opt_try, f_opt_try = nelder_mead(objective, x0_try, max_iter=20)
            mask_try = generate_mask_from_params(x_opt_try)
            score_try = calculate_gap(mask_try)
            if score_try > best_score:
                best_score = score_try
                best_mask = mask_try

    # If no valid mask found, return baseline
    if best_mask is None:
        best_mask = (R < 0.2).astype(int)

    # Convert to list of lists
    return best_mask.tolist()
