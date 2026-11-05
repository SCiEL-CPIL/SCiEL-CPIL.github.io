"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time

# EVOLVE-BLOCK-START
def design(problem):
    """
    Design a 32x32 unit cell to maximize the TM band gap between bands 4 and 5.
    
    Strategy:
    1. Implement a fast internal PWE solver using vectorized matrix construction.
    2. Generate candidate cells using four-quadrant rod configurations.
    3. Score each candidate with the internal solver and return the best.
    4. Maintain D4 symmetry and stay within the time budget.
    """
    size = problem["pixels"]
    eps_high = problem.get("eps_high", 12)
    
    start_time = time.time()
    deadline = start_time + 200.0

    # Fast internal PWE solver for TM modes
    def compute_tm_bands(cell, n_pw=13, n_kpts=15):
        """
        Compute TM band structure using plane-wave expansion.
        Returns min_omega5 and max_omega4 over the BZ path.
        """
        cell = np.array(cell, dtype=float)
        eps = 1.0 + (eps_high - 1.0) * cell
        inv_eps = 1.0 / eps
        N = size
        
        # Get Fourier coefficients of inv_eps via FFT
        fft_coeffs = np.fft.fft2(inv_eps) / (N * N)
        
        # Plane wave indices: use hexagonal-like cutoff for speed
        # m_vals, n_vals such that m^2 + n^2 <= (n_pw/2)^2
        half = n_pw // 2
        g_indices = []
        for m in range(-half, half + 1):
            for n in range(-half, half + 1):
                if m * m + n * n <= half * half + half:
                    g_indices.append((m, n))
        
        n_total = len(g_indices)
        
        # Precompute S matrix: S[i,j] = inv_eps(G_i - G_j)
        S = np.zeros((n_total, n_total), dtype=complex)
        for i in range(n_total):
            mi, ni = g_indices[i]
            for j in range(n_total):
                mj, nj = g_indices[j]
                gm = mi - mj
                gn = ni - nj
                idx_m = gm % N
                idx_n = gn % N
                S[i, j] = fft_coeffs[idx_m, idx_n]
        
        # k-points along the irreducible BZ boundary
        # Gamma(0,0) -> X(0.5,0) -> M(0.5,0.5) -> Gamma(0,0)
        k_points = []
        n_seg = n_kpts // 3
        # Gamma to X
        for t in np.linspace(0, 0.5, n_seg + 1):
            k_points.append((t, 0.0))
        # X to M
        for t in np.linspace(0, 0.5, n_seg + 1):
            k_points.append((0.5, t))
        # M to Gamma
        for t in np.linspace(0.5, 0.0, n_seg + 1):
            k_points.append((t, t))
        
        # Remove duplicates
        k_unique = []
        for kp in k_points:
            if not any(abs(kp[0]-k2[0]) < 1e-10 and abs(kp[1]-k2[1]) < 1e-10 for k2 in k_unique):
                k_unique.append(kp)
        
        min_omega5 = np.inf
        max_omega4 = -np.inf
        
        for (kx, ky) in k_unique:
            # Build the generalized eigenvalue matrix
            # A[i,j] = S[i,j] * |G_j + k|^2
            # B[i,j] = S[i,j]
            # Eigenvalue lambda = (omega*a/2pi*c)^2
            # Actually for TM: sum_j S(Gi-Gj) |Gj+k|^2 cj = omega^2 * sum_j S(Gi-Gj) cj
            
            kvec = np.array([kx, ky])
            Gk = np.array([(m + kx, n + ky) for m, n in g_indices])
            Gk_sq = Gk[:, 0]**2 + Gk[:, 1]**2
            
            A = S * Gk_sq[None, :]  # A[i,j] = S[i,j] * |Gj+k|^2
            B = S
            
            # Solve generalized eigenvalue problem A v = lambda B v
            # Use scipy if available, otherwise use np.linalg.eig on A @ inv(B)
            try:
                from scipy.linalg import eig as sp_eig
                vals, vecs = sp_eig(A, B)
                vals = np.real(vals)
                vals = vals[vals > 1e-6]
                vals = np.sort(vals)
            except:
                # Fallback: solve A v = lambda B v by converting to standard form
                # B^{-1} A v = lambda v
                try:
                    B_inv_A = np.linalg.solve(B, A)
                    vals = np.linalg.eigvals(B_inv_A)
                    vals = np.real(vals)
                    vals = vals[vals > 1e-6]
                    vals = np.sort(vals)
                except:
                    continue
            
            if len(vals) < 5:
                continue
            
            omega4 = vals[3]  # 4th band (0-indexed: index 3)
            omega5 = vals[4]  # 5th band (0-indexed: index 4)
            
            max_omega4 = max(max_omega4, omega4)
            min_omega5 = min(min_omega5, omega5)
        
        if min_omega5 == np.inf or max_omega4 == -np.inf:
            return -1.0
        
        # Gap-midgap ratio: 2*(min_omega5 - max_omega4) / (min_omega5 + max_omega4)
        if min_omega5 + max_omega4 <= 0:
            return -1.0
        return 2.0 * (min_omega5 - max_omega4) / (min_omega5 + max_omega4)
    
    def make_four_quadrant_cell(rod_radius, exponent=2.0):
        """Create a cell with four rods centered in each quadrant."""
        cell = np.zeros((size, size), dtype=int)
        coords = (np.arange(size) + 0.5) / size - 0.5
        centers = [(-0.25, -0.25), (0.25, -0.25), (-0.25, 0.25), (0.25, 0.25)]
        
        # Vectorized: for each pixel, check if it's in any rod
        xx, yy = np.meshgrid(coords, coords, indexing='ij')
        for (cx, cy) in centers:
            dx = xx - cx
            dy = yy - cy
            if rod_radius > 0:
                val = np.abs(dx / rod_radius)**exponent + np.abs(dy / rod_radius)**exponent
            else:
                val = np.zeros_like(xx)
            cell[val <= 1.0 + 1e-9] = 1
        
        # Enforce D4 symmetry: cell == cell[::-1] rows, cols, and transpose
        # The four-quadrant design should already be symmetric, but enforce it
        cell = np.maximum(cell, cell[::-1, :])
        cell = np.maximum(cell, cell[:, ::-1])
        cell = np.maximum(cell, cell.T)
        # Ensure 180 rotation symmetry
        cell = np.maximum(cell, cell[::-1, ::-1])
        
        return cell.tolist()
    
    # Generate and evaluate candidates
    best_score = -np.inf
    best_candidate = make_four_quadrant_cell(0.12, 2.0)
    
    # Coarse grid search over radius and exponent
    radii = [0.06, 0.08, 0.10, 0.11, 0.12, 0.13, 0.14, 0.15, 0.16, 0.18]
    exponents = [2.0, 3.0, 4.0]
    
    candidates = []
    for r in radii:
        for e in exponents:
            candidates.append((r, e))
    
    # Sort by priority: start with likely good values
    candidates.sort(key=lambda x: abs(x[0] - 0.12) + abs(x[1] - 2.0))
    
    for (r, e) in candidates:
        if time.time() > deadline - 30:
            break
        cell = make_four_quadrant_cell(r, e)
        score = compute_tm_bands(cell)
        if score > best_score:
            best_score = score
            best_candidate = cell
            # Fine-tune around the best
            if best_score > 5.0 and time.time() < deadline - 60:
                # Do a local refinement
                for dr in [-0.01, 0.01]:
                    for de in [-0.5, 0.5]:
                        r2 = r + dr
                        e2 = e + de
                        if r2 < 0.03 or e2 < 1.5:
                            continue
                        cell2 = make_four_quadrant_cell(r2, e2)
                        score2 = compute_tm_bands(cell2)
                        if score2 > best_score:
                            best_score = score2
                            best_candidate = cell2
    
    return best_candidate

# EVOLVE-BLOCK-END
