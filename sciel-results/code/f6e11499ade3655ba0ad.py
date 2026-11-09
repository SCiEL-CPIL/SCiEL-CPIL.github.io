"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import itertools

def _make_cell_from_params(params, size=32):
    """Generate a symmetric 32x32 cell from continuous parameters.
    params: list of (cx, cy, r, sign) for up to N circles in the upper-right quadrant.
    We mirror to all 4 quadrants for 4-fold symmetry.
    """
    cell = np.zeros((size, size), dtype=np.float64)
    half = size // 2
    # Work in the upper-right quadrant (i < half, j < half)
    # Pixel centers: (i+0.5)/size - 0.5 for i in [0, half)
    # For upper-right quadrant, x >= 0, y >= 0
    # Actually let's just compute for all pixels using symmetry
    
    # Simpler: compute for all 32x32 but enforce symmetry by construction
    # Use radial basis in 4 quadrants
    # For each pixel (i,j), determine which quadrant it's in, map to UR quadrant
    # and check circles
    
    # Actually, let's just build it directly with 4-fold symmetry
    # A pixel at (i,j) is dielectric if any circle in UR quadrant covers the mapped position
    # Map (i,j) to UR quadrant: 
    #   x = abs((i + 0.5)/size - 0.5)
    #   y = abs((j + 0.5)/size - 0.5)
    # Then check if (x,y) is inside any circle in UR quadrant
    
    # But circles are defined in UR quadrant coordinates where x,y in [0, 0.5]
    # Circle at (cx, cy) with radius r covers (x,y) if (x-cx)^2 + (y-cy)^2 < r^2
    
    for idx, (cx, cy, r, sign) in enumerate(params):
        # Create mask for this circle
        # x_coords for all pixels
        coords = (np.arange(size) + 0.5) / size - 0.5
        # For symmetry, use absolute value to fold into UR quadrant
        x_abs = np.abs(coords)
        # Create 2D grid
        X, Y = np.meshgrid(x_abs, x_abs, indexing='ij')
        # Distance to circle center
        dist2 = (X - cx)**2 + (Y - cy)**2
        mask = dist2 < r**2
        if sign > 0:
            cell += mask.astype(np.float64)
        else:
            cell -= mask.astype(np.float64)
    
    # Threshold to binary
    cell = (cell > 0).astype(np.float64)
    
    # Ensure exact symmetry: average with flipped and transposed versions
    cell = (cell + cell[::-1].copy() + cell[:, ::-1].copy() + cell[::-1].copy()[:, ::-1].copy() + cell.T + cell[::-1].copy().T + cell[:, ::-1].copy().T + cell[::-1].copy()[:, ::-1].copy().T) / 8.0
    cell = (cell > 0.5).astype(np.float64)
    
    return cell

def _pw_solver(cell, n_pw=17, n_kpts=10):
    """Fast plane-wave solver for TM bands.
    cell: 32x32 array of 0/1
    n_pw: number of plane waves in each direction (odd)
    Returns: (max_omega4, min_omega5) over sampled k-points
    """
    size = cell.shape[0]
    eps_low = 1.0
    eps_high = 12.0
    
    # Compute Fourier coefficients of 1/eps
    # 1/eps = 1 for air (eps=1), 1/12 for dielectric (eps=12)
    inv_eps = np.where(cell > 0.5, 1.0/12.0, 1.0)
    
    # Fourier coefficients: G = 2*pi*(m, n) / a, with a=1
    # For a 32x32 pixel grid, pixel size = 1/32
    # Fourier coeff: epsilon_G = (1/32^2) * sum_{i,j} epsilon(i,j) * exp(-i*G*(i,j))
    # But we need 1/eps for TM mode
    
    # Plane wave indices
    n = n_pw
    indices = list(range(-n//2, n//2 + 1))
    
    # Precompute Fourier coefficients for all G vectors
    # G = 2*pi*(m, n)
    # inv_eps_G = (1/N^2) * sum_{x,y} inv_eps(x,y) * exp(-i*G*x)
    # where x = (i+0.5)/32, y = (j+0.5)/32
    
    x_coords = (np.arange(size) + 0.5) / size
    X, Y = np.meshgrid(x_coords, x_coords, indexing='ij')
    
    # Compute all needed Fourier coefficients
    G_indices = []
    for m in indices:
        for n_idx in indices:
            G_indices.append((m, n_idx))
    
    # Compute Fourier transform efficiently using FFT
    # Pad inv_eps to size 32, but we need specific G values
    # Actually, let's just compute directly since 32x32 is small
    
    fourier_coeffs = {}
    for m, n_idx in G_indices:
        Gx = 2 * np.pi * m
        Gy = 2 * np.pi * n_idx
        phase = np.exp(-1j * (Gx * X + Gy * Y))
        fourier_coeffs[(m, n_idx)] = np.sum(inv_eps * phase) / (size * size)
    
    # K-points in irreducible BZ
    # Square lattice BZ: [-pi, pi] x [-pi, pi]
    # Irreducible: 0 <= kx <= ky <= pi (for square symmetry)
    # Key points: Gamma (0,0), M (pi, 0), X (pi/2, pi/2)
    # Let's sample along Gamma-X-M-Gamma path and some interior points
    
    k_points = []
    # Gamma to X: (t*pi/2, t*pi/2) for t in [0,1]
    for t in np.linspace(0, 1, 5):
        k_points.append((t * np.pi / 2, t * np.pi / 2))
    # X to M: (pi/2 + t*pi/2, pi/2 - t*pi/2) for t in [0,1]
    for t in np.linspace(0, 1, 5):
        k_points.append((np.pi/2 + t * np.pi/2, np.pi/2 - t * np.pi/2))
    # M to Gamma: (pi - t*pi, 0) for t in [0,1]
    for t in np.linspace(0, 1, 5):
        k_points.append((np.pi - t * np.pi, 0))
    
    # Deduplicate
    k_points = list(set([(round(kx, 6), round(ky, 6)) for kx, ky in k_points]))
    
    max_omega4 = 0.0
    min_omega5 = np.inf
    
    for kx, ky in k_points:
        # Build eigenvalue problem for TM mode
        # For TM: (1/eps(k-G)) * |k-G|^2 * c_G = omega^2 * c_G
        # This is a generalized eigenvalue problem
        # A * c = omega^2 * B * c where A_GG' = delta_GG' * |k-G|^2, B_GG' = inv_eps(G-G')
        
        n_pw_total = len(G_indices)
        G_list = G_indices
        
        # Build matrix A (diagonal) and B (full)
        A = np.zeros((n_pw_total, n_pw_total), dtype=np.complex128)
        B = np.zeros((n_pw_total, n_pw_total), dtype=np.complex128)
        
        for i, (mi, ni) in enumerate(G_list):
            kx_G = kx - 2 * np.pi * mi
            ky_G = ky - 2 * np.pi * ni
            A[i, i] = kx_G**2 + ky_G**2
        
        for i, (mi, ni) in enumerate(G_list):
            for j, (mj, nj) in enumerate(G_list):
                dm = mi - mj
                dn = ni - nj
                if (dm, dn) in fourier_coeffs:
                    B[i, j] = fourier_coeffs[(dm, dn)]
        
        # Solve generalized eigenvalue problem: A * c = omega^2 * B * c
        # Equivalent to: B^{-1} * A * c = omega^2 * c
        # But B may not be well-conditioned, so use scipy
        
        try:
            from scipy.linalg import eig
            # Solve B * c = (1/omega^2) * A * c
            # Or: A * c = omega^2 * B * c
            # Use eig(A, B)
            vals, vecs = eig(A, B)
            # Take real parts of eigenvalues
            vals_real = np.real(vals)
            # Filter positive eigenvalues
            vals_pos = vals_real[vals_real > 0]
            vals_pos.sort()
            
            if len(vals_pos) >= 5:
                omega4 = np.sqrt(vals_pos[3])  # 4th band (0-indexed: 3)
                omega5 = np.sqrt(vals_pos[4])  # 5th band (0-indexed: 4)
                max_omega4 = max(max_omega4, omega4)
                min_omega5 = min(min_omega5, omega5)
        except Exception:
            continue
    
    return max_omega4, min_omega5

def _score_cell(cell):
    """Compute the gap-midgap ratio for a given cell."""
    try:
        max_omega4, min_omega5 = _pw_solver(cell, n_pw=15, n_kpts=10)
        if min_omega5 > max_omega4:
            gap = min_omega5 - max_omega4
            midgap = (min_omega5 + max_omega4) / 2.0
            if midgap > 0:
                return gap / midgap
            else:
                return 0.0
        else:
            return -1.0
    except Exception:
        return -1.0

def design(problem):
    """Search for optimal photonic crystal unit cell using continuous parameters."""
    size = problem["pixels"]
    start_time = time.time()
    deadline = start_time + 100.0  # Leave margin before 230s limit
    
    # Initial guess: simple rod
    best_score = -1.0
    best_cell = None
    
    # Generate initial rod cell
    cell_rod = np.zeros((size, size), dtype=np.float64)
    for i in range(size):
        for j in range(size):
            x = (i + 0.5) / size - 0.5
            y = (j + 0.5) / size - 0.5
            if x*x + y*y < 0.2**2:
                cell_rod[i, j] = 1.0
    best_cell = cell_rod.copy()
    best_score = _score_cell(cell_rod)
    
    # Parameterized search: circles in UR quadrant
    # Parameters: (cx, cy, r, sign) for up to 4 circles
    # cx, cy in [0, 0.5], r in [0.05, 0.4], sign in {+1, -1}
    
    np.random.seed(42)
    
    n_params = 3  # Number of circles to use
    param_ranges = {
        'cx': (0.0, 0.5),
        'cy': (0.0, 0.5),
        'r': (0.05, 0.4),
    }
    
    # Start with random initial parameters
    current_params = []
    for _ in range(n_params):
        cx = np.random.uniform(*param_ranges['cx'])
        cy = np.random.uniform(*param_ranges['cy'])
        r = np.random.uniform(*param_ranges['r'])
        sign = 1
        current_params.append((cx, cy, r, sign))
    
    current_cell = _make_cell_from_params(current_params, size)
    current_score = _score_cell(current_cell)
    
    if current_score > best_score:
        best_score = current_score
        best_cell = current_cell.copy()
    
    # Hill climbing with random restarts
    step_size = 0.1
    max_steps_per_restart = 50
    
    while time.time() < deadline:
        # Random restart
        for _ in range(n_params):
            cx = np.random.uniform(*param_ranges['cx'])
            cy = np.random.uniform(*param_ranges['cy'])
            r = np.random.uniform(*param_ranges['r'])
            sign = 1
            current_params.append((cx, cy, r, sign))
        
        current_cell = _make_cell_from_params(current_params, size)
        current_score = _score_cell(current_cell)
        
        if current_score > best_score:
            best_score = current_score
            best_cell = current_cell.copy()
        
        # Hill climbing
        for step in range(max_steps_per_restart):
            if time.time() > deadline - 10:
                break
            
            # Perturb a random parameter
            idx = np.random.randint(n_params)
            cx, cy, r, sign = current_params[idx]
            
            # Choose which parameter to perturb
            ptype = np.random.choice(['cx', 'cy', 'r'])
            
            new_cx = cx + np.random.normal(0, step_size)
            new_cy = cy + np.random.normal(0, step_size)
            new_r = r + np.random.normal(0, step_size * 0.5)
            
            # Clamp
            new_cx = np.clip(new_cx, *param_ranges['cx'])
            new_cy = np.clip(new_cy, *param_ranges['cy'])
            new_r = np.clip(new_r, *param_ranges['r'])
            
            new_params = list(current_params)
            new_params[idx] = (new_cx, new_cy, new_r, sign)
            
            new_cell = _make_cell_from_params(new_params, size)
            new_score = _score_cell(new_cell)
            
            if new_score > current_score:
                current_params = new_params
                current_cell = new_cell
                current_score = new_score
                
                if current_score > best_score:
                    best_score = current_score
                    best_cell = current_cell.copy()
            
            # Adaptive step size
            step_size *= 0.95
        
        # Reset step size for next restart
        step_size = 0.1
    
    # Convert to nested list of 0/1
    result = []
    for i in range(size):
        row = []
        for j in range(size):
            row.append(1 if best_cell[i, j] > 0.5 else 0)
        result.append(row)
    
    return result
# EVOLVE-BLOCK-END
