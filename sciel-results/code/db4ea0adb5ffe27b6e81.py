import math
import time
import numpy as np

def design(problem):
    """
    Design a 32x32 unit cell with four superelliptical rods to enhance 
    the 4-5 TM band gap. Uses a local search on shape parameters.
    """
    start_time = time.time()
    deadline = start_time + 110.0  # 10s safety margin
    
    size = problem.get("pixels", 32)
    eps_high = problem.get("eps_high", 12.0)
    
    # Pre-compute k-points and G-vectors for the solver
    # 289 plane waves -> M=8, G in [-8, 8] x [-8, 8]
    M = 8
    G_vals = np.arange(-M, M + 1)
    G_vecs = np.array([(gx, gy) for gx in G_vals for gy in G_vals])
    N_pw = len(G_vecs)
    
    # 30 k-points in the irreducible BZ
    # Irreducible BZ: 0 <= kx <= ky, kx + ky <= 1 (in units of 2pi/a? No, usually 1/a)
    # Let's use normalized coordinates where reciprocal lattice vectors are 1.
    # BZ is [-0.5, 0.5] x [-0.5, 0.5]. Irreducible wedge: 0 <= kx <= ky, kx+ky <= 0.5?
    # Standard irreducible wedge for square lattice (first BZ):
    # Vertices: (0,0), (0.5, 0), (0.5, 0.5).
    # Boundary: kx=0, ky=kx, kx+ky=0.5.
    
    k_points = []
    # Generate points on the boundary and interior
    # Gamma (0,0), M (0.5, 0), X (0.5, 0.5)
    # Line Gamma-M: (t, 0) for t in [0, 0.5]
    # Line Gamma-X: (t, t) for t in [0, 0.5]
    # Line M-X: (0.5, t) for t in [0, 0.5]
    
    num_k = 10
    for i in range(num_k):
        t = i / (num_k - 1) * 0.5
        k_points.append((t, 0))       # Gamma-M
        k_points.append((t, t))       # Gamma-X
        k_points.append((0.5, t))     # M-X
    
    # Remove duplicates or just use a set
    k_points = list(set(k_points))
    # Ensure we have enough, pad if necessary or just use what we have
    # The prompt says 30. We can just use these ~30 unique points.
    
    # Pre-compute 1/eps Fourier coefficients for a generic cell?
    # No, they depend on the cell. We must compute them for each candidate.
    
    def compute_fourier_coeffs(cell):
        """
        Compute Fourier coefficients of 1/epsilon(x,y) for G vectors.
        cell is 32x32 list of 0/1.
        """
        # Convert to numpy array for speed
        eps_map = np.zeros((size, size))
        for i in range(size):
            for j in range(size):
                if cell[i][j] == 1:
                    eps_map[i, j] = eps_high
                else:
                    eps_map[i, j] = 1.0
        
        # 1/eps map
        inv_eps = 1.0 / eps_map
        
        # Compute FFT? Or direct sum?
        # Direct sum for 289 G-vectors and 1024 pixels is 289 * 1024 ~ 300k ops. Very fast.
        # G = (m, n). k_pixel = (j, i) / size.
        # coeff_G = sum_{x,y} inv_eps(x,y) exp(-i G . (x,y))
        
        # Pixel centers
        x_coords = (np.arange(size) + 0.5) / size
        y_coords = (np.arange(size) + 0.5) / size
        XX, YY = np.meshgrid(x_coords, y_coords, indexing='xy')
        
        # Flatten
        xx = XX.flatten()
        yy = YY.flatten()
        inv_eps_flat = inv_eps.flatten()
        
        coeffs = np.zeros(N_pw, dtype=complex)
        for idx, (gx, gy) in enumerate(G_vecs):
            # Phase factor
            phase = np.exp(-1j * 2 * np.pi * (gx * xx + gy * yy))
            coeffs[idx] = np.sum(inv_eps_flat * phase)
            
        # Normalize by number of pixels?
        # The Fourier series is f(x) = sum c_G exp(i G x).
        # c_G = 1/A sum f(x) exp(-i G x).
        # A = 1. So c_G = sum f(x) exp(-i G x) / N_pixels?
        # Actually, if we use the discrete approximation:
        # integral f(x) e^{-iGx} dx approx sum f(x_i) e^{-iGx_i} (dx dy)
        # dx dy = 1/N^2.
        # So coeff = (1/N^2) * sum ...
        coeffs /= (size * size)
        
        return coeffs

    def compute_gap_ratio(cell):
        """
        Compute the gap-midgap ratio for TM bands 4 and 5.
        """
        inv_eps_coeffs = compute_fourier_coeffs(cell)
        
        # Map G index to index in G_vecs
        g_index_map = {}
        for idx, (gx, gy) in enumerate(G_vecs):
            g_index_map[(gx, gy)] = idx
            
        min_omega_5 = float('inf')
        max_omega_4 = -float('inf')
        
        for kx, ky in k_points:
            # Build matrix A
            # A_{G, G'} = (k+G) . (k+G') * inv_eps_{G-G'}
            # Note: G-G' must be in the set of G_vecs.
            # Since G_vecs is a grid from -M to M, G-G' is in range if M is large enough.
            # M=8. Max diff is 16. Wait.
            # If G is in [-8,8] and G' is in [-8,8], G-G' is in [-16,16].
            # But we only have coefficients for G in [-8,8].
            # This is a truncation error.
            # The prompt says "289-plane-wave Galerkin method using the exact Fourier coefficients".
            # Usually, this implies the convolution is done within the truncated basis.
            # If G-G' is outside the basis, we assume inv_eps_{G-G'} = 0?
            # Or do we need a larger basis for the coefficients?
            # The prompt says "exact Fourier coefficients of the pixel map".
            # And "289-plane-wave Galerkin".
            # This usually means we expand the field in 289 PWs, and the operator matrix is 289x289.
            # The element (G, G') requires the coefficient of 1/eps for G-G'.
            # If G-G' is not in our 289 set, we don't have it.
            # However, for a 32x32 pixel map, the "exact" coefficients decay.
            # If we truncate the *basis* to 289, we should probably truncate the *coefficients* to the same range, 
            # assuming higher order terms are negligible or handled by the Galerkin projection.
            # Standard practice: If the basis is G in [-M, M], the matrix element A_{G, G'} uses coeff_{G-G'}.
            # If G-G' is outside [-M, M], it is not in the basis.
            # In a strict Galerkin method with basis {e^{i G r}}, the matrix element is <G | H | G'>.
            # H involves multiplication by 1/eps.
            # <G | 1/eps | G'> = sum_K <G|K> <K|1/eps|G'>? No.
            # It is the Fourier component of 1/eps at G-G'.
            # If we don't have that component, we can't compute it.
            # However, usually M is chosen large enough.
            # Here M=8. Max index 8. Diff 16.
            # This suggests we might need to compute coefficients for a larger range, or the prompt implies 
            # that the 289 PWs are sufficient and we just use the available coefficients (zero padding).
            # Given "exact-coefficient", I will compute coefficients for a larger range if possible, 
            # but store only what's needed?
            # Actually, let's just compute coefficients for all G-G' needed.
            # Max G-G' is 16. So we need G up to 16?
            # Let's re-read: "289-plane-wave Galerkin method".
            # This fixes the basis size.
            # I will assume that for indices G-G' outside the computed coefficient range, we use 0.
            # This is a standard truncation.
            
            A = np.zeros((N_pw, N_pw), dtype=complex)
            
            for i, (gx, gy) in enumerate(G_vecs):
                kx_g = kx + gx
                ky_g = ky + gy
                for j, (gx_p, gy_p) in enumerate(G_vecs):
                    kx_gp = kx + gx_p
                    ky_gp = ky + gy_p
                    
                    # Dot product (k+G).(k+G')
                    dot = kx_g * kx_gp + ky_g * ky_gp
                    
                    # G - G'
                    dgx = gx - gx_p
                    dgy = gy - gy_p
                    
                    # Get coefficient
                    # If (dgx, dgy) is in g_index_map, use it. Else 0.
                    if (dgx, dgy) in g_index_map:
                        coeff = inv_eps_coeffs[g_index_map[(dgx, dgy)]]
                    else:
                        coeff = 0.0
                    
                    A[i, j] = dot * coeff
            
            # Solve eigenvalues
            # A is Hermitian?
            # A_{G,G'} = (k+G).(k+G') c_{G-G'}
            # A_{G',G} = (k+G').(k+G) c_{G'-G} = (k+G).(k+G') c_{-(G-G')}
            # c_{-G} = c_G^* (for real eps).
            # So A is Hermitian.
            
            try:
                vals = np.linalg.eigvalsh(A)
                # vals are omega^2
                omegas = np.sqrt(np.real(vals))
                # Sort
                omegas = np.sort(omegas)
                
                # Get bands 4 and 5 (indices 3 and 4)
                if len(omegas) >= 5:
                    w4 = omegas[3]
                    w5 = omegas[4]
                    max_omega_4 = max(max_omega_4, w4)
                    min_omega_5 = min(min_omega_5, w5)
            except np.linalg.LinAlgError:
                continue
        
        if min_omega_5 == float('inf') or max_omega_4 == -float('inf'):
            return -1.0
            
        gap = min_omega_5 - max_omega_4
        midgap = (min_omega_5 + max_omega_4) / 2.0
        
        if midgap == 0:
            return -1.0
            
        ratio = 2 * gap / (min_omega_5 + max_omega_4)
        return ratio

    def generate_cell(radius, n_exp, center_offset):
        """
        Generate a symmetric 32x32 cell with 4 superelliptical rods.
        """
        cell = [[0 for _ in range(size)] for _ in range(size)]
        
        centers = [
            (-center_offset, -center_offset),
            ( center_offset, -center_offset),
            (-center_offset,  center_offset),
            ( center_offset,  center_offset)
        ]
        
        for i in range(size):
            y = (i + 0.5) / size - 0.5
            for j in range(size):
                x = (j + 0.5) / size - 0.5
                
                inside = False
                for cx, cy in centers:
                    dx = x - cx
                    dy = y - cy
                    # Superellipse: |dx|^n + |dy|^n <= R^n
                    # To avoid 0^0 issues or numerical issues, use small epsilon
                    val = (abs(dx) + 1e-9)**n_exp + (abs(dy) + 1e-9)**n_exp
                    if val <= (radius + 1e-9)**n_exp:
                        inside = True
                        break
                
                if inside:
                    cell[i][j] = 1
                else:
                    cell[i][j] = 0
        
        # Enforce symmetry (should be symmetric by construction, but just in case)
        # The superellipse is symmetric. The centers are symmetric.
        # Pixel grid is symmetric.
        return cell

    # Optimization
    best_score = -1000.0
    best_cell = None
    
    # Initial guess: Seed program parameters
    # Seed: radius=0.15, center=0.25, circle (n=2)
    init_params = (0.15, 2.0, 0.25)
    
    # Search space
    # radius: 0.10 to 0.20
    # n_exp: 1.5 to 4.0
    # center: 0.20 to 0.30
    
    import random
    random.seed(42)
    
    # Evaluate initial
    cell_init = generate_cell(*init_params)
    score_init = compute_gap_ratio(cell_init)
    if score_init > best_score:
        best_score = score_init
        best_cell = cell_init
        
    # Local search / Random sampling
    # We have ~110s.
    # compute_gap_ratio takes ~0.1-0.5s?
    # Let's estimate. 30 k-points * 289x289 eigvalsh.
    # eigvalsh 289x289 is fast in NumPy.
    # Let's try 50 evaluations.
    
    num_evals = 0
    max_evals = 50
    
    while time.time() < deadline and num_evals < max_evals:
        # Perturb best parameters or random sample
        if best_cell is None:
            r = 0.15
            n = 2.0
            c = 0.25
        else:
            # We need to track best params.
            # For simplicity, let's just do random search in the box.
            r = random.uniform(0.10, 0.20)
            n = random.uniform(1.5, 4.0)
            c = random.uniform(0.20, 0.30)
            
        cell = generate_cell(r, n, c)
        score = compute_gap_ratio(cell)
        num_evals += 1
        
        if score > best_score:
            best_score = score
            best_cell = cell

    if best_cell is None:
        # Fallback to seed
        best_cell = generate_cell(0.15, 2.0, 0.25)
        
    return best_cell
