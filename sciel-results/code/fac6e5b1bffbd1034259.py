"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
from scipy.optimize import minimize
import itertools

# EVOLVE-BLOCK-START
def design(problem):
    """
    Design a 32x32 unit cell for a 2D square-lattice photonic crystal
    that maximizes the gap between TM bands 4 and 5.
    
    Strategy: Surrogate-Assisted Optimization.
    We define a parametric family of symmetric patterns (sum of Fourier modes).
    We use a fast plane-wave method (reduced basis) as a surrogate objective
    function to optimize the pattern parameters.
    """
    size = problem["pixels"]
    eps_high = problem.get("eps_high", 12.0)
    eps_low = problem.get("eps_low", 1.0)
    
    # Coordinates
    x = np.arange(size)
    y = np.arange(size)
    X, Y = np.meshgrid(x, y)
    # Centered coordinates for symmetry
    x_c = (X - (size - 1) / 2.0) / size
    y_c = (Y - (size - 1) / 2.0) / size

    def generate_pattern(params):
        """
        Generate a binary 32x32 pattern from parameters.
        params: [A10, A11, A20, A22]
        A10 controls (1,0) and (0,1) modes.
        A11 controls (1,1) mode.
        A20 controls (2,0) and (0,2) modes.
        A22 controls (2,2) mode.
        """
        A10, A11, A20, A22 = params
        
        V = np.zeros((size, size))
        V += A10 * (np.cos(2 * np.pi * x_c) + np.cos(2 * np.pi * y_c))
        V += A11 * np.cos(2 * np.pi * x_c) * np.cos(2 * np.pi * y_c)
        V += A20 * (np.cos(4 * np.pi * x_c) + np.cos(4 * np.pi * y_c))
        V += A22 * np.cos(4 * np.pi * x_c) * np.cos(4 * np.pi * y_c)
        
        # Normalize V to zero mean
        V = V - np.mean(V)
        
        # Threshold to 50% filling
        # Using median ensures ~50% ones
        threshold = np.median(V)
        cell = (V > threshold).astype(float)
        
        # Apply epsilon values
        # 1 -> eps_high, 0 -> eps_low
        eps_map = np.where(cell == 1, eps_high, eps_low)
        
        return eps_map

    def fast_pwe_solver(eps_map, kx, ky, n_pw=5):
        """
        Fast Plane Wave Expansion solver for TM mode.
        n_pw: number of G-vectors in each dimension (e.g., 5 means -2 to 2).
        Returns the first few eigenfrequencies (in units of a/lambda).
        """
        N = n_pw * n_pw
        # G vectors
        g_vals = np.arange(-n_pw//2, n_pw//2 + 1)
        Gx, Gy = np.meshgrid(g_vals, g_vals, indexing='ij')
        Gx = Gx.flatten()
        Gy = Gy.flatten()
        
        # k vector
        k = np.array([kx, ky])
        
        # Fourier coefficients of 1/epsilon
        # eps_map is 32x32. We need the discrete Fourier transform of 1/eps.
        # The plane wave method uses the continuous Fourier series coefficients.
        # For a pixelated map, we approximate the integral with a sum.
        # f_mn = (1/A) * sum_ij (1/eps_ij) * exp(-i (G_mn . r_ij))
        
        inv_eps = 1.0 / eps_map
        
        # Compute FFT of inv_eps
        # Note: The standard FFT gives coefficients for frequencies 0..N-1.
        # We need coefficients for G vectors in -n_pw//2 .. n_pw//2.
        # The pixel grid is size=32. The fundamental frequency is 2*pi/32.
        # G vector (m, n) corresponds to frequency m*2*pi/32, n*2*pi/32.
        # So we can use FFT and pick the appropriate indices.
        
        fft_inv_eps = np.fft.fft2(inv_eps)
        # Shift so zero frequency is at center? 
        # fft2 output index [i, j] corresponds to freq i, j.
        # We want freq m, n where m, n are in g_vals.
        # g_vals are integers.
        
        # Construct the matrix
        # H[m, n] = (k - G_m)^2 * delta_mn + (1/A) * sum_l (1/eps_l) * exp(i (G_m - G_n) . r_l) * (k-G_m).(k-G_n) / |k-G_m|^2 ?
        # Standard TM PWE equation:
        # sum_n [ (k - G_m) . (k - G_n) / |k - G_m|^2 * epsilon^{-1}_{mn} ] C_n = omega^2 C_m
        # Wait, the standard form is:
        # sum_n (k - G_m) . (k - G_n) / |k - G_m|^2 * epsilon^{-1}_{m-n} C_n = omega^2 C_m
        # where epsilon^{-1}_{m-n} is the Fourier coefficient of 1/eps.
        
        # Let's compute the Fourier coefficients of 1/eps for the required G differences.
        # G_diff = G_m - G_n.
        # We need coeff[dx, dy] where dx, dy are differences of g_vals.
        
        # Get the full FFT coefficients
        # The FFT of a 32x32 array gives coefficients for k=0..31.
        # Negative frequencies are at 32-k.
        def get_coeff(dx, dy):
            # dx, dy are integers in range -4..4 (for n_pw=5)
            # Map to FFT indices
            idx_x = dx % size
            idx_y = dy % size
            return fft_inv_eps[idx_x, idx_y] / (size * size) # Normalization by Area (1 pixel = 1/32 x 1/32 area? No, area of cell is 1. Pixel area is 1/1024. Sum is 1024 * 1/1024 * val = val. So divide by 1024? 
            # Actually, f_mn = (1/A_cell) * sum_pixel (1/eps_p) * exp(-i G_mn . r_p) * A_pixel
            # A_cell = 1. A_pixel = 1/1024.
            # sum_pixel ... = (1/1024) * sum_val * exp...
            # So f_mn = (1/1024) * FFT_val / (1/1024)? No.
            # FFT_val = sum_val * exp...
            # So f_mn = (1/1024) * FFT_val.
            # Let's check: If eps is constant, 1/eps is constant. FFT[0,0] = 1024 * val.
            # f_00 should be val.
            # (1/1024) * 1024 * val = val. Correct.
            pass
        
        # Precompute coefficients for all needed differences
        # The differences G_m - G_n range from -4 to 4.
        # We can just compute the full FFT and index it.
        
        H = np.zeros((N, N), dtype=complex)
        
        for i in range(N):
            km = k - np.array([Gx[i], Gy[i]])
            km2 = km[0]**2 + km[1]**2
            if km2 < 1e-12:
                # Handle k=0 case carefully, though we usually avoid it or it's well-defined
                H[i, i] = 0 
                continue
            for j in range(N):
                kn = k - np.array([Gx[j], Gy[j]])
                dot = km[0]*kn[0] + km[1]*kn[1]
                
                dx = Gx[i] - Gx[j]
                dy = Gy[i] - Gy[j]
                
                # Get Fourier coeff of 1/eps for (dx, dy)
                idx_x = dx % size
                idx_y = dy % size
                coeff = fft_inv_eps[idx_x, idx_y] / (size * size)
                
                H[i, j] = (dot / km2) * coeff
        
        # Solve eigenvalue problem
        # H is Hermitian?
        # H[i,j] = (km.kn / |km|^2) * eps_inv_{mn}
        # eps_inv_{mn} = eps_inv_{-mn}^*
        # H[j,i] = (kn.km / |kn|^2) * eps_inv_{nm}
        # This is not obviously Hermitian in this form. 
        # The standard PWE matrix is constructed to be Hermitian.
        # Let's use the standard formulation:
        # A_{mn} = (k-G_m) . (k-G_n) / |k-G_m|^2 * epsilon^{-1}_{m-n}
        # This A is Hermitian if epsilon is real.
        # Let's verify H[i,j] and H[j,i].
        # H[i,j] = (km.kn / |km|^2) * C_{mn}
        # H[j,i] = (kn.km / |kn|^2) * C_{nm}
        # C_{nm} = C_{-mn} = C_{mn}^* (if real eps)
        # So H[j,i] = (km.kn / |kn|^2) * C_{mn}^*
        # This is NOT the conjugate of H[i,j] unless |km|=|kn|.
        # 
        # Correction: The eigenvalue equation is:
        # sum_n [ (k-G_m) . (k-G_n) / |k-G_m|^2 * epsilon^{-1}_{m-n} ] C_n = omega^2 C_m
        # This matrix is NOT Hermitian in general.
        # However, it is similar to a Hermitian matrix.
        # Or, we can use the alternative form:
        # sum_n [ (k-G_m) . (k-G_n) * epsilon^{-1}_{m-n} ] C_n = omega^2 |k-G_m|^2 C_m ? No.
        #
        # Let's use the standard Hermitian form:
        # Define vector V_m = C_m * (k-G_m)
        # Then the equation is:
        # sum_n epsilon^{-1}_{m-n} (k-G_m) . (k-G_n) V_n / |k-G_n|^2 ? No.
        #
        # Actually, the standard PWE matrix for TM is:
        # M_{mn} = (k-G_m) . (k-G_n) / |k-G_m|^2 * epsilon^{-1}_{m-n}
        # This matrix is Hermitian?
        # Let's check the literature.
        # Joannopoulos et al., "Photonic Crystals".
        # The matrix is:
        # H_{mn} = (k - G_m) \cdot (k - G_n) / |k - G_m|^2 \epsilon^{-1}_{m-n}
        # This matrix is Hermitian because:
        # H_{nm}^* = [ (k - G_n) \cdot (k - G_m) / |k - G_n|^2 \epsilon^{-1}_{n-m} ]^*
        # = (k - G_n) \cdot (k - G_m) / |k - G_n|^2 \epsilon^{-1}_{m-n}
        # This is not equal to H_{mn} unless |k-G_m| = |k-G_n|.
        # 
        # Wait, the eigenvalue equation is:
        # \sum_n \frac{(k-G_m) \cdot (k-G_n)}{|k-G_m|^2} \epsilon^{-1}_{m-n} a_n = \omega^2 a_m
        # This is a generalized eigenvalue problem? No, it's a standard one.
        # But the matrix is not Hermitian.
        # 
        # Alternative formulation:
        # \sum_n (k-G_m) \cdot (k-G_n) \epsilon^{-1}_{m-n} a_n = \omega^2 |k-G_m|^2 a_m
        # This is also not standard.
        #
        # Let's use the Hermitian formulation:
        # \mathbf{E}(r) = \sum_G a_G (k-G) e^{i(k-G)r}
        # The wave equation leads to:
        # \sum_G \epsilon^{-1}_{G-G'} (k-G) \cdot (k-G') a_{G'} = \omega^2 (k-G) a_G
        # This is a generalized eigenvalue problem:
        # A a = \omega^2 B a
        # Where A_{GG'} = \epsilon^{-1}_{G-G'} (k-G) \cdot (k-G')
        # And B_{GG'} = \delta_{GG'} |k-G|^2
        # Both A and B are Hermitian and positive definite (for B).
        # We can solve this using scipy.linalg.eigh.
        
        A = np.zeros((N, N), dtype=complex)
        B = np.zeros((N, N), dtype=complex)
        
        for i in range(N):
            km = k - np.array([Gx[i], Gy[i]])
            B[i, i] = km[0]**2 + km[1]**2
            for j in range(N):
                kn = k - np.array([Gx[j], Gy[j]])
                dot = km[0]*kn[0] + km[1]*kn[1]
                dx = Gx[i] - Gx[j]
                dy = Gy[i] - Gy[j]
                idx_x = dx % size
                idx_y = dy % size
                coeff = fft_inv_eps[idx_x, idx_y] / (size * size)
                A[i, j] = coeff * dot
                
        # Solve A a = omega^2 B a
        # This is a Hermitian generalized eigenvalue problem.
        try:
            vals, vecs = np.linalg.eigh(A, B)
            # Filter out negative or zero values
            vals = vals[vals > 1e-10]
            return vals
        except Exception:
            return np.array([])

    def objective(params):
        """
        Calculate the gap ratio for the pattern defined by params.
        """
        eps_map = generate_pattern(params)
        
        # Sample K-points in the irreducible Brillouin zone
        # Gamma: (0, 0)
        # X: (0.5, 0)
        # M: (0.5, 0.5)
        # Line Gamma-X: (t, 0)
        # Line X-M: (0.5, t)
        
        k_points = []
        # Gamma to X
        for t in np.linspace(0, 0.5, 10):
            k_points.append((t, 0))
        # X to M
        for t in np.linspace(0, 0.5, 10):
            k_points.append((0.5, t))
            
        min_omega_5 = np.inf
        max_omega_4 = 0.0
        
        for kx, ky in k_points:
            omegas = fast_pwe_solver(eps_map, kx, ky, n_pw=5)
            if len(omegas) < 5:
                continue
            # omegas are omega^2. We need omega.
            omegas = np.sqrt(omegas)
            # We need bands 4 and 5. Indices 3 and 4.
            if len(omegas) >= 5:
                w4 = omegas[3]
                w5 = omegas[4]
                max_omega_4 = max(max_omega_4, w4)
                min_omega_5 = min(min_omega_5, w5)
                
        if min_omega_5 == np.inf or max_omega_4 == 0:
            return -1.0
            
        gap = min_omega_5 - max_omega_4
        midgap = (min_omega_5 + max_omega_4) / 2.0
        
        if midgap == 0:
            return -1.0
            
        ratio = gap / midgap
        return ratio

    # Initial guess
    # A10, A11, A20, A22
    x0 = np.array([0.5, 1.0, 0.1, 0.0])
    
    # Optimize
    # Use Nelder-Mead as it doesn't require gradients
    res = minimize(objective, x0, method='Nelder-Mead', options={'maxiter': 200, 'xatol': 1e-3, 'fatol': 1e-4})
    
    best_params = res.x
    # Ensure symmetry of parameters? The pattern generation handles it.
    
    # Generate final pattern
    eps_map = generate_pattern(best_params)
    cell = (eps_map > (eps_high + eps_low) / 2.0).astype(int)
    
    # Enforce symmetry explicitly
    cell = (cell + cell[::-1, ::-1]) // 2
    cell = (cell + cell.T) // 2
    
    return cell.tolist()
# EVOLVE-BLOCK-END
