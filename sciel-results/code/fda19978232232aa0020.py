import math
import time
import numpy as np

def design(problem):
    """
    Design a symmetric 32x32-pixel unit cell to maximize the TM band gap
    between bands 4 and 5 using a low-frequency Fourier parameterization
    with Nelder-Mead optimization in coefficient space.
    """
    size = problem["pixels"]  # 32
    eps_low = problem["eps_low"]   # 1
    eps_high = problem["eps_high"] # 12
    
    # Deadline guard
    start_time = time.time()
    deadline = start_time + 115  # 115s out of 120s budget (with margin)
    
    # We'll work with the 16x16 top-left quadrant due to 4-fold symmetry
    def quadrant_to_full(q):
        """Convert a 16x16 quadrant to a full 32x32 symmetric cell."""
        q = np.array(q, dtype=int)
        top = np.hstack([q, q[:, ::-1]])
        bottom = top[::-1, :]
        return np.vstack([top, bottom])
    
    def full_to_quadrant(cell):
        """Extract the top-left 16x16 quadrant from a full 32x32 cell."""
        return np.array(cell[:16, :16], dtype=int)
    
    # Fast band gap evaluation using plane wave method
    def compute_gap(cell_2d):
        """
        Compute the gap-midgap ratio between TM bands 4 and 5.
        Uses a plane-wave expansion with 289 basis functions.
        """
        N = 32
        eps = np.where(np.array(cell_2d) == 1, eps_high, eps_low)
        
        # Use 289 plane waves: (2*8+1)^2 = 17^2 = 289
        npw = 8
        G = []
        for mx in range(-npw, npw+1):
            for my in range(-npw, npw+1):
                G.append((2*np.pi*mx, 2*np.pi*my))
        G = np.array(G)
        nG = len(G)
        
        # Compute inverse dielectric Fourier coefficients
        inv_eps = 1.0 / eps
        x = (np.arange(N) + 0.5) / N
        y = (np.arange(N) + 0.5) / N
        X, Y = np.meshgrid(x, y, indexing='ij')
        
        inv_eps_coeffs = np.zeros(nG, dtype=complex)
        for idx, (gx, gy) in enumerate(G):
            inv_eps_coeffs[idx] = np.sum(inv_eps * np.exp(-1j * (gx * X + gy * Y))) / (N * N)
        
        # Key k-points in the irreducible Brillouin zone
        k_points = [
            (0.0, 0.0),
            (np.pi, 0.0),
            (np.pi/2, np.pi/2),
            (np.pi/2, 0.0),
            (np.pi/2, np.pi/4),
            (3*np.pi/8, np.pi/8),
            (np.pi/4, np.pi/4),
            (np.pi/3, np.pi/6),
            (2*np.pi/5, np.pi/5),
            (np.pi/2, np.pi/6),
        ]
        
        max_omega4 = 0.0
        min_omega5 = np.inf
        
        for kx, ky in k_points:
            kvec = np.array([kx, ky])
            k_plus_G = kvec + G
            
            kG_sq = np.sum(k_plus_G**2, axis=1)
            
            G_dict = {}
            for idx, (gx, gy) in enumerate(G):
                G_dict[(gx, gy)] = idx
            
            A = np.zeros((nG, nG), dtype=complex)
            for i in range(nG):
                for j in range(nG):
                    G_diff = G[i] - G[j]
                    key = (G_diff[0], G_diff[1])
                    if key in G_dict:
                        A[i, j] = inv_eps_coeffs[G_dict[key]]
            
            H = A * kG_sq[np.newaxis, :]
            H = (H + H.conj().T) / 2
            
            eigvals = np.linalg.eigvalsh(H)
            eigvals = np.sort(eigvals.real)
            
            if len(eigvals) >= 6:
                omega4 = eigvals[4]
                omega5 = eigvals[5]
                max_omega4 = max(max_omega4, omega4)
                min_omega5 = min(min_omega5, omega5)
        
        if min_omega5 == np.inf or max_omega4 == 0:
            return 0.0
        
        gap = min_omega5 - max_omega4
        midgap = (min_omega5 + max_omega4) / 2
        
        if midgap <= 0:
            return 0.0
        
        return gap / midgap
    
    # Fourier parameterization: use D4-symmetric low-frequency coefficients
    # For a 32x32 cell with D4 symmetry, we can represent it as:
    # cell[i,j] = 0.5 + sum_{(mx,my) in D4_orbits} [a_{mx,my} cos(2*pi*mx*i/32 + phase) * ...]
    
    # Let's use a simpler approach: parameterize the quadrant directly with
    # a small number of radial/angular modes that respect D4 symmetry.
    
    # Alternative: use a low-resolution grid and upscale
    # Use a 4x4 grid for the quadrant, then upscale to 16x16
    
    def params_to_quadrant(params):
        """Convert continuous parameters to a 16x16 quadrant (0-1 values)."""
        # params has 16 values for a 4x4 grid
        grid = np.array(params).reshape(4, 4)
        # Upsample to 16x16 using bilinear interpolation
        from scipy.ndimage import zoom
        factor = 16 / 4.0
        quadrant = zoom(grid, factor, order=1)
        # Clip to [0, 1]
        quadrant = np.clip(quadrant, 0, 1)
        return quadrant
    
    def quadrant_to_params(quadrant):
        """Convert 16x16 quadrant to 4x4 grid params."""
        # Downsample using averaging
        q = np.array(quadrant, dtype=float)
        # Resize to 4x4
        from scipy.ndimage import zoom
        factor = 4 / 16.0
        grid = zoom(q, factor, order=1)
        return grid.flatten().tolist()
    
    def params_to_cell(params):
        """Convert continuous params to binary 32x32 cell."""
        quadrant = params_to_quadrant(params)
        # Threshold at 0.5
        quadrant_bin = (quadrant >= 0.5).astype(int)
        full_cell = quadrant_to_full(quadrant_bin)
        return full_cell
    
    # Initial design: 2x2 rods
    def initial_params():
        """Create initial parameter vector for 2x2 rod design."""
        # Create 32x32 cell with 2x2 rods
        radius = 0.14
        centers = [
            (0.25, 0.25),
            (0.25, 0.75),
            (0.75, 0.25),
            (0.75, 0.75)
        ]
        
        cell = np.zeros((size, size), dtype=int)
        for i in range(size):
            y = (i + 0.5) / size
            for j in range(size):
                x = (j + 0.5) / size
                for cx, cy in centers:
                    dx = x - cx
                    dy = y - cy
                    if dx*dx + dy*dy < radius*radius:
                        cell[i, j] = 1
                        break
        
        quadrant = full_to_quadrant(cell)
        params = quadrant_to_params(quadrant)
        return params
    
    # Objective function: negative gap (we minimize to maximize gap)
    def objective(params):
        cell = params_to_cell(params)
        gap = compute_gap(cell)
        return -gap
    
    # Run Nelder-Mead optimization
    init_params = initial_params()
    best_score = -objective(init_params)
    best_params = init_params.copy()
    
    # Multiple restarts with different initial conditions
    rng = np.random.RandomState(42)
    
    max_restarts = 8
    for restart in range(max_restarts):
        if time.time() > deadline - 10:
            break
        
        if restart == 0:
            x0 = init_params.copy()
        else:
            # Perturb from best found so far
            x0 = best_params + rng.randn(len(best_params)) * 0.3
        
        try:
            from scipy.optimize import minimize
            result = minimize(
                objective, x0,
                method='Nelder-Mead',
                options={'maxiter': 300, 'xatol': 1e-4, 'fatol': 1e-5, 'adaptive': True}
            )
            score = -result.fun
            if score > best_score:
                best_score = score
                best_params = result.x.copy()
        except Exception:
            pass
    
    # Final refinement: local pixel search around the best Fourier design
    best_cell = params_to_cell(best_params)
    best_q_score = compute_gap(best_cell)
    
    # Quick local pixel flips for fine-tuning
    quadrant = full_to_quadrant(best_cell)
    best_quadrant = quadrant.copy()
    
    while time.time() < deadline - 5:
        improved = False
        for i in range(16):
            if time.time() > deadline - 5:
                break
            for j in range(16):
                if time.time() > deadline - 5:
                    break
                quadrant[i, j] = 1 - quadrant[i, j]
                full_cell = quadrant_to_full(quadrant)
                s = compute_gap(full_cell)
                if s > best_q_score + 1e-6:
                    best_q_score = s
                    best_quadrant = quadrant.copy()
                    improved = True
                else:
                    quadrant[i, j] = 1 - quadrant[i, j]
        if not improved:
            break
    
    best_cell = quadrant_to_full(best_quadrant)
    best_cell = (best_cell > 0.5).astype(int)
    
    return best_cell.tolist()
