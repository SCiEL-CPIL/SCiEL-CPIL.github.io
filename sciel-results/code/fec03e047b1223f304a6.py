"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import math
import numpy as np

def _build_cell(size, params):
    """Build a symmetric 32x32 cell from symmetric parameters.
    
    params is a dict with keys describing a parametric family.
    We enforce 4-fold symmetry by constructing the top-left 16x16 quadrant
    and mirroring it.
    """
    half = size // 2
    # Build top-left quadrant
    quad = np.zeros((half, half), dtype=np.uint8)
    
    # Get the top-left pixel indices
    # For a 32x32 cell, top-left quadrant is rows 0-15, cols 0-15
    # Center of cell is at (16, 16)
    # For pixel (i, j) in top-left quadrant:
    #   x = (j + 0.5) / size - 0.5
    #   y = (i + 0.5) / size - 0.5
    
    # We'll use a parametric family: combination of rings, crosses, and corner fills
    # Parameters:
    #   inner_radius: radius of inner air hole (0 = no hole)
    #   outer_radius: radius of outer dielectric region
    #   ring_width: width of dielectric ring
    #   cross_width: width of cross arms
    #   corner_fill: fraction of corner to fill with dielectric
    
    inner_r = params.get('inner_r', 0.0)
    outer_r = params.get('outer_r', 0.4)
    ring_r = params.get('ring_r', 0.0)  # center of ring
    ring_w = params.get('ring_w', 0.0)  # width of ring
    cross_w = params.get('cross_w', 0.0)  # half-width of cross arms
    corner_frac = params.get('corner_frac', 0.0)  # fraction of corner filled
    
    for i in range(half):
        for j in range(half):
            x = (j + 0.5) / size - 0.5
            y = (i + 0.5) / size - 0.5
            r = math.sqrt(x * x + y * y)
            
            dielectric = False
            
            # Base: dielectric disk of radius outer_r
            if r < outer_r:
                dielectric = True
            
            # Subtract inner air hole
            if inner_r > 0 and r < inner_r:
                dielectric = False
            
            # Add dielectric ring
            if ring_w > 0:
                if abs(r - ring_r) < ring_w / 2:
                    dielectric = True
            
            # Add cross arms (along x and y axes)
            if cross_w > 0:
                # Cross along x-axis (horizontal arm)
                if abs(y) < cross_w and abs(x) < 0.5:
                    dielectric = True
                # Cross along y-axis (vertical arm)
                if abs(x) < cross_w and abs(y) < 0.5:
                    dielectric = True
            
            # Corner fill: fill pixels in the corner region
            if corner_frac > 0:
                # Corner region is where both |x| and |y| are large
                # In top-left quadrant, x < 0, y < 0
                # Corner is at x ~ -0.5, y ~ -0.5
                # Define corner as region where x < -0.5 + corner_frac and y < -0.5 + corner_frac
                corner_thresh = 0.5 - corner_frac
                if x < -corner_thresh and y < -corner_thresh:
                    dielectric = True
            
            quad[i, j] = 1 if dielectric else 0
    
    # Mirror to create full cell
    cell = np.zeros((size, size), dtype=np.uint8)
    # Top-left quadrant
    cell[:half, :half] = quad
    # Top-right quadrant (mirror horizontally)
    cell[:half, half:] = quad[:, ::-1]
    # Bottom-left quadrant (mirror vertically)
    cell[half:, :half] = quad[::-1, :]
    # Bottom-right quadrant (mirror both)
    cell[half:, half:] = quad[::-1, ::-1]
    
    return cell

def _coarse_evaluate(cell, size=32, n_pw=17):
    """Coarse evaluation using plane-wave method with reduced number of plane waves.
    
    This is a fast approximation to the full 289-plane-wave evaluator.
    We compute the TM band structure on a few key k-points and estimate
    the gap between bands 4 and 5.
    """
    # Build dielectric function
    eps = np.ones((size, size))
    eps[cell == 1] = 12.0
    
    # For TM polarization, we solve:
    #   (k^2 - eps^{-1} * (k - G)^2) |A> = 0
    # where k is the wavevector and G are reciprocal lattice vectors
    
    # Use a reduced set of plane waves for speed
    # n_pw = 17 means we use a (2*8+1) x (2*8+1) grid = 17x17 = 289 plane waves
    # But for coarse eval, use fewer: n_pw=9 means 9x9=81 plane waves
    
    half_n = n_pw // 2
    
    # Generate reciprocal lattice vectors
    G_vectors = []
    for mg in range(-half_n, half_n + 1):
        for ng in range(-half_n, half_n + 1):
            G_vectors.append((mg, ng))
    G_vectors = np.array(G_vectors, dtype=float)
    
    # Compute Fourier coefficients of 1/eps
    # 1/eps(x) = 1 for air, 1/12 for dielectric
    inv_eps = np.zeros((size, size))
    inv_eps[cell == 0] = 1.0
    inv_eps[cell == 1] = 1.0 / 12.0
    
    # Compute Fourier coefficients
    N = len(G_vectors)
    eps_G = np.zeros((N, N), dtype=complex)
    
    # For speed, compute 1/eps Fourier coefficients directly
    # 1/eps_G = (1/size^2) * sum_{x,y} (1/eps(x,y)) * exp(-i G . (x,y))
    
    # Precompute positions
    x_pos = np.arange(size) / size - 0.5
    y_pos = np.arange(size) / size - 0.5
    
    # Use FFT for speed
    # Pad inv_eps to a larger grid and use FFT
    # Actually, let's just compute directly for small grids
    for gi, (mg, ng) in enumerate(G_vectors):
        for gj, (mh, nh) in enumerate(G_vectors):
            Gx = (mg - mh) * 2 * np.pi
            Gy = (ng - nh) * 2 * np.pi
            # Compute (1/size^2) * sum_{i,j} inv_eps[i,j] * exp(-i (Gx*i/size + Gy*j/size))
            # Use FFT
            # Actually, let's use a direct sum for now since size is small
            phase = np.exp(-1j * (Gx * x_pos[np.newaxis, :] + Gy * y_pos[:, np.newaxis]))
            eps_G[gi, gj] = np.sum(inv_eps * phase) / (size * size)
    
    # Evaluate on a few k-points in the irreducible BZ
    # Irreducible BZ for square lattice: 0 <= kx <= ky, kx + ky <= pi
    # Key points: Gamma (0,0), X (pi/2, 0), M (pi/2, pi/2)
    # And some interior points
    
    k_points = [
        (0.0, 0.0),       # Gamma
        (np.pi / 2, 0.0), # X
        (np.pi / 2, np.pi / 2), # M
        (np.pi / 4, 0.0),
        (np.pi / 4, np.pi / 4),
        (np.pi / 2, np.pi / 4),
        (np.pi / 6, 0.0),
        (np.pi / 3, 0.0),
    ]
    
    max_omega_4 = 0.0
    min_omega_5 = float('inf')
    
    for kx, ky in k_points:
        k = np.array([kx, ky])
        
        # Build the eigenvalue matrix
        # For TM: (k^2 - (1/eps_G) * (k-G)^2) |A> = 0
        # This is a generalized eigenvalue problem, but we can reformulate
        
        # Actually, the standard formulation is:
        #   sum_G' (k-G)^2 / eps_{G-G'} |A_{G'}> = omega^2 |A_G>
        # where eps_G is the Fourier coefficient of 1/eps
        
        # So we need to solve: M |A> = omega^2 |A>
        # where M_{G,G'} = (k-G)^2 / eps_{G-G'}
        
        # Build matrix M
        M = np.zeros((N, N), dtype=complex)
        for gi in range(N):
            kg = G_vectors[gi]
            k_minus_g = k - kg
            k_minus_g_sq = k_minus_g[0]**2 + k_minus_g[1]**2
            for gj in range(N):
                kgp = G_vectors[gj]
                G_diff = kg - kgp
                # Find index of G_diff in G_vectors
                # G_diff = (mg - mh, ng - nh)
                idx = None
                for ii in range(N):
                    if G_vectors[ii][0] == G_diff[0] and G_vectors[ii][1] == G_diff[1]:
                        idx = ii
                        break
                if idx is not None and eps_G[idx, 0] != 0:  # Check if this G_diff is in our set
                    # Actually, eps_G is indexed by [gi, gj] where gi is G and gj is G'
                    # We need eps_{G-G'} which is the Fourier coefficient at G-G'
                    # eps_G[gi, gj] = Fourier coefficient at G_i - G_j
                    # So eps_{G-G'} = eps_G[gi, gj] where G_i = G and G_j = G'
                    M[gi, gj] = k_minus_g_sq / eps_G[gi, gj]
        
        # Solve eigenvalue problem
        try:
            eigvals = np.linalg.eigvalsh(M)
            eigvals = np.real(eigvals)
            eigvals = np.sort(eigvals)
            
            # Take the 5th and 6th eigenvalues (0-indexed: 4 and 5)
            if len(eigvals) >= 6:
                omega_4 = eigvals[4]
                omega_5 = eigvals[5]
                if omega_4 > max_omega_4:
                    max_omega_4 = omega_4
                if omega_5 < min_omega_5:
                    min_omega_5 = omega_5
        except:
            pass
    
    if min_omega_5 == float('inf') or max_omega_4 == 0:
        return -1.0
    
    gap = min_omega_5 - max_omega_4
    midgap = min_omega_5 + max_omega_4
    if midgap <= 0:
        return -1.0
    
    return 2 * gap / midgap

def _hill_climb(params, best_score, max_iter=200, step_sizes=None):
    """Simple hill climbing on the parameter space."""
    if step_sizes is None:
        step_sizes = {
            'inner_r': 0.01,
            'outer_r': 0.01,
            'ring_r': 0.01,
            'ring_w': 0.01,
            'cross_w': 0.01,
            'corner_frac': 0.01,
        }
    
    current_params = dict(params)
    current_score = best_score
    best_params = dict(params)
    best = best_score
    
    for it in range(max_iter):
        # Pick a random parameter to perturb
        param_name = list(step_sizes.keys())[it % len(step_sizes)]
        old_val = current_params[param_name]
        new_val = old_val + step_sizes[param_name] * (1 if it % 2 == 0 else -1)
        
        # Clamp to valid range
        if param_name == 'inner_r':
            new_val = max(0.0, min(0.4, new_val))
        elif param_name == 'outer_r':
            new_val = max(0.1, min(0.49, new_val))
        elif param_name == 'ring_r':
            new_val = max(0.0, min(0.45, new_val))
        elif param_name == 'ring_w':
            new_val = max(0.0, min(0.2, new_val))
        elif param_name == 'cross_w':
            new_val = max(0.0, min(0.2, new_val))
        elif param_name == 'corner_frac':
            new_val = max(0.0, min(0.3, new_val))
        
        current_params[param_name] = new_val
        
        # Evaluate
        cell = _build_cell(32, current_params)
        score = _coarse_evaluate(cell)
        
        if score > current_score:
            current_score = score
            if score > best:
                best = score
                best_params = dict(current_params)
        else:
            # Revert
            current_params[param_name] = old_val
    
    return best_params, best

def design(problem):
    """Design a symmetric 32x32 photonic crystal unit cell to maximize TM band gap between bands 4 and 5."""
    start_time = time.time()
    deadline = start_time + 210  # Leave 20s margin
    
    size = problem["pixels"]
    
    # Initial parameters: start with a ring + cross design
    initial_params = {
        'inner_r': 0.05,
        'outer_r': 0.35,
        'ring_r': 0.2,
        'ring_w': 0.08,
        'cross_w': 0.05,
        'corner_frac': 0.0,
    }
    
    # Build initial cell and evaluate
    cell = _build_cell(size, initial_params)
    best_score = _coarse_evaluate(cell)
    best_params = dict(initial_params)
    
    # Hill climbing
    params, score = _hill_climb(initial_params, best_score, max_iter=100)
    if score > best_score:
        best_score = score
        best_params = params
    
    # Check time
    if time.time() > deadline - 60:
        # Not enough time for more, return best
        cell = _build_cell(size, best_params)
        return cell.tolist()
    
    # Try different starting points
    start_points = [
        {'inner_r': 0.0, 'outer_r': 0.3, 'ring_r': 0.15, 'ring_w': 0.1, 'cross_w': 0.08, 'corner_frac': 0.05},
        {'inner_r': 0.1, 'outer_r': 0.4, 'ring_r': 0.25, 'ring_w': 0.05, 'cross_w': 0.03, 'corner_frac': 0.1},
        {'inner_r': 0.0, 'outer_r': 0.35, 'ring_r': 0.0, 'ring_w': 0.0, 'cross_w': 0.1, 'corner_frac': 0.15},
        {'inner_r': 0.08, 'outer_r': 0.38, 'ring_r': 0.18, 'ring_w': 0.12, 'cross_w': 0.04, 'corner_frac': 0.0},
    ]
    
    for sp in start_points:
        if time.time() > deadline - 30:
            break
        
        cell = _build_cell(size, sp)
        sc = _coarse_evaluate(cell)
        
        if sc > best_score:
            best_score = sc
            best_params = dict(sp)
        
        # Refine
        params, score = _hill_climb(sp, sc, max_iter=50)
        if score > best_score:
            best_score = score
            best_params = params
    
    # Final cell
    cell = _build_cell(size, best_params)
    return cell.tolist()
# EVOLVE-BLOCK-END
