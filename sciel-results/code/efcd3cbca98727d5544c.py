import numpy as np
import time
import sys
from scipy.optimize import minimize
from scipy.linalg import eigh

def design(problem):
    """
    Designs a symmetric 32x32 photonic crystal unit cell to maximize the TM band gap
    between the 4th and 5th bands using a symmetry-adapted Fourier series optimization.
    """
    size = problem.get("pixels", 32)
    eps_low = problem.get("eps_low", 1.0)
    eps_high = problem.get("eps_high", 12.0)
    
    # Deadline guard
    start_time = time.time()
    deadline = start_time + 180  # 180s buffer out of 230s limit

    # 1. Define Symmetry-Adapted Fourier Basis
    # For C4v symmetry in a square lattice, we use cosines with wave vectors (nx, ny)
    # such that the pattern is invariant under rotation by 90 degrees and reflection.
    # We select a set of low-order harmonics.
    # The condition for C4v invariance for a term cos(nx*x + ny*y) + cos(ny*x - ny*y) etc.
    # Simplified: We use a basis of functions that are explicitly symmetric.
    
    # Basis functions: f_i(x, y)
    # We will represent epsilon(x,y) = eps_avg + sum( c_i * phi_i(x,y) )
    # phi_i are chosen to be C4v invariant.
    
    # Let's use a grid of Fourier coefficients for a 5x5 or 7x7 supercell of modes,
    # but constrained to symmetry.
    # A simpler approach: Use radial-like or specific harmonic combinations.
    
    # Let's define a set of harmonic indices (n, m) that generate C4v symmetric patterns when summed appropriately.
    # For a pattern to be C4v symmetric, the Fourier coefficients must satisfy:
    # c_{n,m} = c_{m,-n} = c_{-n,-m} = c_{-m,n}
    # And for real epsilon, c_{-n,-m} = c_{n,m}^*.
    # So we just need to define coefficients for a fundamental domain of (n,m) and mirror them.
    
    # We will optimize a vector of coefficients for a truncated set of (n,m).
    # Let N_max be the maximum harmonic index.
    N_max = 4  # Max index for n, m
    
    # Generate unique (n,m) pairs for the fundamental domain of C4v
    # Domain: n >= 0, m >= 0, and n >= m (to avoid duplicates from rotation)
    # Actually, for C4v, the fundamental domain in reciprocal space is 1/8 of the BZ.
    # Let's just pick a list of (n,m) that are "symmetric" enough or enforce symmetry on the grid.
    
    # Alternative robust approach:
    # Optimize a small set of coefficients for a 3x3 or 5x5 Fourier grid.
    # Then construct the full symmetric Fourier series.
    
    # Let's use a 5x5 grid of coefficients (n,m) from -2 to 2.
    # We will optimize the coefficients for n>=0, m>=0, n>=m (fundamental domain).
    
    indices = []
    for n in range(0, N_max + 1):
        for m in range(0, n + 1):
            # Exclude (0,0) as it's the average
            if n == 0 and m == 0:
                continue
            indices.append((n, m))
            
    # Number of coefficients to optimize
    num_coeffs = len(indices)
    
    # Initial guess: Small random or zero
    # Let's start with a configuration similar to a rod, which has significant low-frequency content.
    # A rod is roughly a top-hat function. Its Fourier transform decays as 1/k.
    # We'll initialize with a decaying amplitude.
    x0 = np.array([0.1 * np.exp(-0.5 * (n**2 + m**2)) for n, m in indices])
    
    # 2. Define the PWE Solver for Optimization
    # We need a fast solver to evaluate the gap for many iterations.
    # Use a coarse plane-wave expansion (e.g., 9x9 or 11x11 modes).
    # And a small number of k-points (e.g., 5-10 points).
    
    def build_epsilon_grid(coeffs, grid_size):
        """
        Reconstructs the epsilon grid from the optimized coefficients.
        coeffs correspond to the fundamental domain indices.
        We must expand them to the full symmetric set.
        """
        # Create the full coefficient matrix for indices -N_max to N_max
        C_full = np.zeros((2 * N_max + 1, 2 * N_max + 1), dtype=float)
        
        # Map optimized coeffs to the full grid
        for i, (n, m) in enumerate(indices):
            c_val = coeffs[i]
            # Symmetry expansions for C4v:
            # (n, m) -> (m, n), (-n, m), (-m, n), (n, -m), (m, -n), (-n, -m), (-m, -n)
            # Since we use cosines, we can just add c_val to all symmetric positions.
            # C_full[n+N_max, m+N_max] += c_val
            # C_full[m+N_max, n+N_max] += c_val
            # ... etc.
            
            # It is easier to sum the cosine terms directly or fill the matrix.
            # Let's fill the matrix for the DFT.
            positions = [
                (n, m), (m, n), (-n, m), (-m, n),
                (n, -m), (m, -n), (-n, -m), (-m, -n)
            ]
            # Avoid double counting if n==m or n==0 or m==0
            # We'll just sum them up.
            for pn, pm in positions:
                C_full[pn + N_max, pm + N_max] += c_val
                
        # Note: (0,0) is not in indices, so C_full[N_max, N_max] is 0.
        # We need to add the average epsilon.
        # Let's fix eps_avg to be the midpoint? Or optimize it?
        # The problem asks for 0/1 pixels. The average density will determine the center.
        # Let's optimize eps_avg as well? No, let's keep it fixed to (eps_low+eps_high)/2
        # or optimize a "bias" term.
        # Actually, the DC term (0,0) is crucial for the average dielectric constant.
        # Let's add a bias term to the optimization vector.
        
        # For now, let's assume the optimized coefficients define the modulation.
        # We will compute the grid.
        
        x = np.linspace(-0.5, 0.5, grid_size, endpoint=False)
        y = np.linspace(-0.5, 0.5, grid_size, endpoint=False)
        X, Y = np.meshgrid(x, y)
        
        epsilon = np.zeros((grid_size, grid_size))
        
        # Add DC term (Average)
        # Let's estimate a reasonable average. A rod of radius 0.2 has area ~0.12.
        # So eps_avg ~ 1 + 0.12*(12-1) ~ 2.3.
        # Let's optimize a 'bias' parameter that shifts the mean.
        # To keep it simple, we'll add a bias variable to the optimization.
        
        # Re-define optimization vector: [bias, coeffs...]
        # But we already defined x0. Let's modify the solver to include bias.
        
        pass # Placeholder, logic moved to solver

    # Let's refine the optimization setup to include a bias term.
    
    def get_epsilon_from_params(params, grid_size):
        bias = params[0]
        coeffs = params[1:]
        
        x = np.linspace(-0.5, 0.5, grid_size, endpoint=False)
        y = np.linspace(-0.5, 0.5, grid_size, endpoint=False)
        X, Y = np.meshgrid(x, y)
        
        epsilon = np.full((grid_size, grid_size), bias)
        
        for i, (n, m) in enumerate(indices):
            c_val = coeffs[i]
            # Sum all symmetric cosine terms
            # cos(nx + my) + cos(mx - ny) + ...
            # Using the property that the sum of cosines over the orbit is 8 * cos(nx+my) * cos(mx-ny) ? No.
            # Let's just sum the 8 terms explicitly for clarity and correctness, 
            # though it's slower. For N_max=4, num_coeffs is small (~10-15).
            
            # Terms:
            # cos( n*x + m*y )
            # cos( m*x - n*y )  (Rotation by 90 deg: x->y, y->-x => n*y - m*x = -(m*x - n*y))
            # cos( -n*x + m*y ) (Reflection x->-x)
            # cos( -m*x - n*y ) (Rotation 90 of Refl)
            # cos( n*x - m*y ) (Reflection y->-y)
            # cos( m*x + n*y ) (Rotation 90 of Refl Y)
            # cos( -n*x - m*y ) (Inversion)
            # cos( -m*x + n*y ) (Rotation 90 of Inversion)
            
            # Since cos is even, cos(-A) = cos(A).
            # The set of arguments is:
            # n*x + m*y
            # m*x - n*y
            # -n*x + m*y  == n*x - m*y
            # -m*x - n*y  == m*x + n*y
            
            # So we sum:
            # cos(nx+my) + cos(mx-ny) + cos(nx-my) + cos(mx+ny)
            
            epsilon += c_val * (
                np.cos(n * X + m * Y) +
                np.cos(m * X - n * Y) +
                np.cos(n * X - m * Y) +
                np.cos(m * X + n * Y)
            )
            
        return epsilon

    def solve_pwe_fast(epsilon_grid, k_points, num_modes):
        """
        Solves the PWE eigenvalue problem for a given epsilon grid.
        Returns the max of band 4 and min of band 5.
        """
        # Precompute the inverse dielectric matrix elements
        # 1/epsilon(k, k') = sum_q (1/N_q) * (1/epsilon(q)) * delta_{k-q, k'}
        # For a pixelated grid, we can use the exact Fourier coefficients.
        # However, for speed in optimization, we can approximate 1/eps using a truncated series
        # or just use the pixel grid's DFT.
        
        # Since we are evaluating on a coarse grid of k-points and modes, 
        # and the epsilon grid is 32x32, we can compute the DFT of 1/epsilon once per epsilon update.
        
        eps_inv = 1.0 / epsilon_grid
        # Compute DFT of eps_inv
        # The grid is 32x32. The DFT will have 32x32 coefficients.
        # We need coefficients for the plane waves in our expansion.
        
        # Plane wave indices
        # k = G + k0
        # G are reciprocal lattice vectors.
        # For a square lattice of period 1, G = (2*pi*n, 2*pi*m).
        # In our dimensionless units (x in [-0.5, 0.5]), the basis functions are exp(i 2*pi n x).
        # So the Fourier coefficients are just the DFT of the grid, scaled.
        
        # DFT of eps_inv
        # numpy.fft.fft2
        eps_inv_fft = np.fft.fft2(eps_inv)
        eps_inv_fft /= (32 * 32) # Normalization
        
        # We need to map the plane wave indices (n, m) to the FFT array indices.
        # FFT index [i, j] corresponds to frequency (i, j) if i < N/2, else (i-N, j-N) etc.
        
        def get_coeff(n, m):
            # Map (n, m) to FFT index
            idx_n = n % 32
            idx_m = m % 32
            return eps_inv_fft[idx_n, idx_m]
        
        # Define the plane waves
        # num_modes x num_modes grid of G vectors
        # G range: -num_modes//2 to num_modes//2
        n_max_pw = num_modes // 2
        g_vectors = [(n, m) for n in range(-n_max_pw, n_max_pw + 1) for m in range(-n_max_pw, n_max_pw + 1)]
        # Ensure num_modes is odd, so n_max_pw is integer
        if num_modes % 2 == 0:
             num_modes += 1
             n_max_pw = num_modes // 2
             g_vectors = [(n, m) for n in range(-n_max_pw, n_max_pw + 1) for m in range(-n_max_pw, n_max_pw + 1)]
        
        N_pw = len(g_vectors)
        
        # Build the matrix A
        # A_{ij} = delta_{ij} * |k_i|^2 + (1/|k_i| |k_j|) * <1/eps>_{k_i - k_j}
        # For TM polarization:
        # (k^2) * c_i - sum_j (k_i x k_j) ... wait, TM is scalar.
        # TM: (k^2) c_i - sum_j (k_i . k_j) / (|k_i| |k_j|) * <1/eps>_{G_i - G_j} * c_j = 0 ?
        # Standard PWE for TM:
        # |k|^2 c_k - sum_{k'} (k . k') / (|k| |k'|) * (1/eps)_{k-k'} c_{k'} = 0
        # Actually, the coupling term is (k_i . k_j) / (|k_i| |k_j|) * <1/eps>_{G_i - G_j}
        
        # Let's build the matrix for each k_point
        max_omega_4 = 0
        min_omega_5 = np.inf
        
        for k0 in k_points:
            # k0 is a tuple (kx, ky)
            # Construct the matrix
            A = np.zeros((N_pw, N_pw), dtype=float)
            
            for i, (gi, gj) in enumerate(g_vectors):
                ki = (k0[0] + 2 * np.pi * gi, k0[1] + 2 * np.pi * gj)
                ki_sq = ki[0]**2 + ki[1]**2
                
                # Diagonal term
                A[i, i] = ki_sq
                
                for j, (gh, gm) in enumerate(g_vectors):
                    if i == j:
                        continue
                    kj = (k0[0] + 2 * np.pi * gh, k0[1] + 2 * np.pi * gm)
                    kj_sq = kj[0]**2 + kj[1]**2
                    
                    # Dot product
                    dot = ki[0]*kj[0] + ki[1]*kj[1]
                    
                    # Normalization
                    norm = np.sqrt(ki_sq * kj_sq)
                    if norm < 1e-12:
                        continue
                    
                    # Fourier coefficient of 1/eps for G_i - G_j
                    Gn = gi - gh
                    Gm = gj - gm
                    coeff = get_coeff(Gn, Gm)
                    
                    A[i, j] -= (dot / norm) * coeff
            
            # Solve eigenvalue problem
            # A c = omega^2 c
            # Use eigh for symmetric matrices (A is symmetric if 1/eps is real and symmetric)
            # Note: A is real symmetric.
            try:
                vals, vecs = eigh(A)
                # Sort eigenvalues
                vals = np.sort(vals)
                
                # We need bands 4 and 5. 
                # The lowest bands correspond to the lowest eigenvalues.
                # Band 1 is index 0.
                # So Band 4 is index 3, Band 5 is index 4.
                
                if len(vals) >= 5:
                    omega_4 = np.sqrt(vals[3])
                    omega_5 = np.sqrt(vals[4])
                    
                    max_omega_4 = max(max_omega_4, omega_4)
                    min_omega_5 = min(min_omega_5, omega_5)
            except:
                # If solver fails, return bad score
                return -np.inf

        if max_omega_4 == 0 or min_omega_5 == np.inf:
            return -np.inf
            
        gap = min_omega_5 - max_omega_4
        midgap = (min_omega_5 + max_omega_4) / 2.0
        if midgap < 1e-9:
            return -np.inf
        return gap / midgap

    # 3. Optimization Loop
    
    # Define k-points for fast evaluation
    # Irreducible BZ for square lattice:
    # (0,0) to (pi,0) to (pi,pi) to (0,pi)
    # High symmetry points: Gamma(0,0), X(pi,0), M(pi,pi)
    # We'll sample a few points along the boundaries and interior.
    k_points = [
        (0.0, 0.0),
        (np.pi, 0.0),
        (np.pi, np.pi),
        (0.0, np.pi),
        (np.pi/2, 0.0),
        (np.pi/2, np.pi/2),
        (np.pi/4, np.pi/4),
        (3*np.pi/4, np.pi/4)
    ]
    
    num_modes_fast = 9  # Coarse grid for speed
    
    def objective(params):
        # params[0] is bias, params[1:] are coeffs
        # Check time
        if time.time() > deadline:
            return -np.inf
            
        epsilon_grid = get_epsilon_from_params(params, 32)
        
        # Clip epsilon to valid range to avoid division by zero or negative
        # The problem implies 0/1 pixels, but during optimization we have continuous values.
        # We should map the continuous epsilon to [eps_low, eps_high]
        # A simple sigmoid or clipping works.
        # Let's clip to [1, 12]
        epsilon_grid = np.clip(epsilon_grid, eps_low, eps_high)
        
        score = solve_pwe_fast(epsilon_grid, k_points, num_modes_fast)
        return -score  # Minimize negative score

    # Initial parameters
    # Bias: Start with a value that gives ~30% dielectric
    # eps = 1 + 0.3*(11) = 4.3
    bias_init = 4.0
    x0_opt = np.concatenate(([bias_init], x0))
    
    # Bounds
    # Bias should be between 1 and 12
    # Coeffs can be positive or negative, but bounded to prevent huge values
    bounds = [(eps_low, eps_high)] + [(-2.0, 2.0)] * num_coeffs
    
    print(f"Starting optimization with {num_coeffs} coeffs", file=sys.stderr)
    
    best_params = x0_opt
    best_score = -np.inf
    
    try:
        res = minimize(objective, x0_opt, method='L-BFGS-B', bounds=bounds,
                       options={'maxiter': 50, 'ftol': 1e-4, 'gtol': 1e-4})
        
        if res.fun < -best_score:
            best_score = -res.fun
            best_params = res.x
            
    except Exception as e:
        print(f"Optimization error: {e}", file=sys.stderr)

    # 4. Project to Pixels
    
    # Use the best parameters found
    epsilon_final = get_epsilon_from_params(best_params, 32)
    
    # Map to 0/1
    # Threshold at the midpoint?
    # Or simply: if epsilon > (eps_low + eps_high)/2 then 1 else 0
    mid_val = (eps_low + eps_high) / 2.0
    cell = (epsilon_final > mid_val).astype(int)
    
    # 5. Enforce Symmetry
    # The Fourier method should produce a symmetric field, but discretization and thresholding
    # might break it slightly.
    # Enforce C4v symmetry on the grid.
    
    # Symmetrize by averaging all 8 symmetric transformations
    # R90: (i,j) -> (j, N-1-i)
    # R180: (i,j) -> (N-1-i, N-1-j)
    # R270: (i,j) -> (N-1-j, i)
    # RefX: (i,j) -> (N-1-i, j)
    # RefY: (i,j) -> (i, N-1-j)
    # RefDiag: (i,j) -> (j, i)
    # RefAntiDiag: (i,j) -> (N-1-j, N-1-i)
    
    N = size
    cell_float = cell.astype(float)
    sym_cell = np.zeros_like(cell_float)
    
    for i in range(N):
        for j in range(N):
            # Get all symmetric positions
            pos = [
                (i, j),
                (j, N-1-i),
                (N-1-i, N-1-j),
                (N-1-j, i),
                (N-1-i, j),
                (i, N-1-j),
                (j, i),
                (N-1-j, N-1-i)
            ]
            # Average the values
            val = np.mean([cell_float[p[0], p[1]] for p in pos])
            sym_cell[i, j] = val
            
    # Threshold the symmetrized cell
    cell_final = (sym_cell > 0.5).astype(int)
    
    # Verify symmetry
    # cell == cell[::-1] row-wise
    # cell == cell.T
    # This should hold by construction.
    
    # Convert to list of lists
    cell_list = cell_final.tolist()
    
    return cell_list

