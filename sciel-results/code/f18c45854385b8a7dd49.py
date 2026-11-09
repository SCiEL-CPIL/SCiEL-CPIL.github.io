import numpy as np
import time
import math

def design(problem):
    """
    Designs a 32x32 photonic crystal unit cell to maximize the TM band gap
    between bands 4 and 5.
    Uses a parametric Fourier representation and Nelder-Mead optimization.
    """
    size = problem.get("pixels", 32)
    eps_low = problem.get("eps_low", 1.0)
    eps_high = problem.get("eps_high", 12.0)
    
    # Time budget management
    start_time = time.time()
    time_limit = 220.0  # Safety margin before 230s kill
    
    # 1. Define the parametrized continuous permittivity field
    # We use a small set of symmetric Fourier modes.
    # Basis: cos(2*pi*x/L)*cos(2*pi*y/L) is symmetric under C4 if coefficients match?
    # Actually, cos(nx)cos(ny) is symmetric under x->-x, y->-y, x<->y.
    # Let's use modes: (1,1), (2,2), (1,3) is not C4 symmetric alone?
    # cos(x)cos(3y) is not symmetric under x<->y.
    # To maintain C4 symmetry (transpose invariance), we need terms where n=m or combined symmetrically.
    # Let's stick to n=m terms: (1,1), (2,2), (3,3).
    # epsilon(x,y) = eps_avg + A1*cos(2*pi*x)*cos(2*pi*y) + A2*cos(4*pi*x)*cos(4*pi*y) + ...
    # x,y in [-0.5, 0.5]
    
    def get_eps_field(params):
        # params: [eps_avg, A1, A2, A3]
        eps_avg = params[0]
        A1 = params[1]
        A2 = params[2]
        A3 = params[3]
        
        # Create grid
        # x, y in [-0.5, 0.5]
        x = (np.arange(size) + 0.5) / size - 0.5
        y = (np.arange(size) + 0.5) / size - 0.5
        X, Y = np.meshgrid(x, y, indexing='ij')
        
        # Fourier terms
        # Note: cos(2*pi*n*x)
        term1 = A1 * np.cos(2*np.pi*X) * np.cos(2*np.pi*Y)
        term2 = A2 * np.cos(4*np.pi*X) * np.cos(4*np.pi*Y)
        term3 = A3 * np.cos(6*np.pi*X) * np.cos(6*np.pi*Y)
        
        eps_cont = eps_avg + term1 + term2 + term3
        
        # Clip to valid range [eps_low, eps_high]
        eps_cont = np.clip(eps_cont, eps_low, eps_high)
        
        # Binarize: The task requires 0/1 pixels (air/dielectric).
        # We can interpret the continuous field as a probability or just threshold.
        # Thresholding at the mean is standard.
        # However, to get more control, we can add a bias or just threshold at (eps_low+eps_high)/2?
        # If eps_avg is optimized, the threshold can be fixed.
        # Let's threshold at the global mean of the field? Or just a fixed value?
        # If we threshold at (eps_low+eps_high)/2, the fill fraction depends on the amplitude.
        # Let's use a threshold T. We can add T as a parameter?
        # For simplicity, threshold at (eps_low + eps_high) / 2.
        threshold = (eps_low + eps_high) / 2.0
        eps_bin = (eps_cont > threshold).astype(int)
        
        return eps_bin

    # 2. Plane Wave Solver
    # TM polarization: eigenvalue problem for omega^2
    # Matrix M_{G,G'} = (k+G')^2 * eps_inv[G-G']
    # We need eps_inv Fourier coefficients.
    
    # Precompute G vectors
    # N_pw = 17 (odd number for symmetry, -8 to 8)
    N_pw = 17
    G_vals = np.arange(-N_pw//2, N_pw//2 + 1)
    Gx, Gy = np.meshgrid(G_vals, G_vals, indexing='ij')
    G_vectors = np.column_stack((Gx.ravel(), Gy.ravel()))
    G_indices = np.arange(len(G_vectors))
    
    # Map G to index
    def g_index(gx, gy):
        # Find index in G_vectors
        # This is slow. Precompute a dict.
        pass

    # Precompute mapping for difference G - G'
    # G_diff = G_i - G_j
    # We need eps_inv[G_diff]
    
    def compute_eps_inv_coeffs(eps_bin):
        # Compute Fourier coefficients of 1/eps
        # eps is 0/1. 1/eps is inf where eps=0.
        # Wait, the problem says "air and dielectric".
        # If eps=0 (air), 1/eps is undefined.
        # Standard PWM for TM uses 1/epsilon.
        # If there is air (eps=1), 1/eps = 1.
        # If there is dielectric (eps=12), 1/eps = 1/12.
        # The prompt says: "1 = dielectric with epsilon 12, 0 = air".
        # So eps(x) is 12 if pixel is 1, 1 if pixel is 0.
        # So 1/eps(x) is 1/12 if pixel is 1, 1 if pixel is 0.
        
        eps_actual = np.where(eps_bin == 1, eps_high, eps_low)
        inv_eps = 1.0 / eps_actual
        
        # Compute Fourier coefficients of inv_eps
        # C_G = 1/N^2 sum_{i,j} inv_eps[i,j] exp(-i (Gx*i + Gy*j) * 2*pi / N)
        # Using FFT:
        # fft2(inv_eps) gives coefficients for G in range [-N/2, N/2]
        # We need coefficients for G in G_vals.
        
        fft_inv_eps = np.fft.fft2(inv_eps)
        fft_inv_eps = np.fft.fftshift(fft_inv_eps)
        
        coeffs = np.zeros(len(G_vectors), dtype=complex)
        for idx, (gx, gy) in enumerate(G_vectors):
            # Map G to FFT index
            # FFT indices range from -N/2 to N/2-1
            # G_vals range from -8 to 8
            # If N=32, FFT indices are -16..15.
            # Our G range -8..8 is within -16..15.
            ix = int(gx)
            iy = int(gy)
            # fftshifted array: index 0 corresponds to -N/2.
            # So index = ix + N/2
            idx_x = ix + size // 2
            idx_y = iy + size // 2
            if 0 <= idx_x < size and 0 <= idx_y < size:
                coeffs[idx] = fft_inv_eps[idx_x, idx_y] / (size * size)
            else:
                coeffs[idx] = 0.0
                
        return coeffs

    # Precompute G-G' matrix indices
    # For each pair (i, j), we need the index of G_i - G_j in our G_vectors list.
    # If G_i - G_j is not in our list, the coefficient is 0 (neglecting aliasing).
    # With N_pw=17 and size=32, the max G is 8. Max diff is 16.
    # Our G list only goes to 8. So we miss some terms.
    # This is standard truncation.
    
    g_to_idx = {}
    for idx, (gx, gy) in enumerate(G_vectors):
        g_to_idx[(gx, gy)] = idx
        
    M_diff_indices = np.zeros((len(G_vectors), len(G_vectors)), dtype=int)
    for i in range(len(G_vectors)):
        for j in range(len(G_vectors)):
            gx_diff = G_vectors[i][0] - G_vectors[j][0]
            gy_diff = G_vectors[i][1] - G_vectors[j][1]
            if (gx_diff, gy_diff) in g_to_idx:
                M_diff_indices[i, j] = g_to_idx[(gx_diff, gy_diff)]
            else:
                M_diff_indices[i, j] = -1 # Mark as missing

    def solve_bands(eps_bin):
        inv_eps_coeffs = compute_eps_inv_coeffs(eps_bin)
        
        # Define k-points in irreducible BZ
        # Triangle: (0,0) to (pi,0) to (pi,pi)
        # kx in [0, pi], ky in [0, kx]
        # Let's sample 30 points.
        k_points = []
        # Line Gamma-M: ky=0, kx in [0, pi]
        # Line M-X: kx=pi, ky in [0, pi] ? No, M is (pi,0), X is (pi/2, pi/2)?
        # Square BZ: [-pi, pi]x[-pi, pi].
        # Irreducible: 0 <= kx <= ky, kx+ky <= pi?
        # Let's use the triangle (0,0), (pi,0), (pi/2, pi/2) ??
        # The 1/8th BZ is bounded by kx=0, ky=kx, and kx+ky=pi?
        # Vertices: (0,0), (pi/2, pi/2), (pi, 0).
        
        num_k = 30
        # Generate points in the triangle
        for i in range(num_k):
            # Use barycentric coordinates or simple grid
            # Let's just use a grid in the square [0,pi]x[0,pi] and filter
            pass
        
        # Simpler: 10 points on Gamma-M, 10 on M-X, 10 on X-Gamma?
        # Gamma (0,0), M (pi, 0), X (pi/2, pi/2)
        
        k_list = []
        # Gamma to M
        for t in np.linspace(0, 1, 10):
            k_list.append((t * np.pi, 0))
        # M to X
        for t in np.linspace(0, 1, 10):
            k_list.append((np.pi, t * np.pi / 2))
        # X to Gamma
        for t in np.linspace(0, 1, 10):
            k_list.append(((1-t) * np.pi / 2, (1-t) * np.pi / 2))
            
        # Remove duplicates
        unique_k = []
        seen = set()
        for k in k_list:
            key = (round(k[0], 4), round(k[1], 4))
            if key not in seen:
                seen.add(key)
                unique_k.append(k)
        
        max_omega4 = 0
        min_omega5 = np.inf
        
        for kx, ky in unique_k:
            k_vec = np.array([kx, ky])
            
            # Construct Matrix M
            # M[i,j] = (k + G_j)^2 * inv_eps_coeffs[diff_idx(i,j)]
            # Note: The equation is sum_j M_{ij} c_j = omega^2 c_i
            # M_{ij} = |k + G_j|^2 * (1/eps)_{G_i - G_j}
            
            M = np.zeros((len(G_vectors), len(G_vectors)), dtype=complex)
            
            for i in range(len(G_vectors)):
                for j in range(len(G_vectors)):
                    k_plus_G = k_vec + G_vectors[j]
                    k2 = np.dot(k_plus_G, k_plus_G)
                    diff_idx = M_diff_indices[i, j]
                    if diff_idx != -1:
                        M[i, j] = k2 * inv_eps_coeffs[diff_idx]
                    # else 0
            
            # Solve eigenvalue problem
            # M is Hermitian?
            # inv_eps_coeffs are complex. But since eps is real and symmetric, coeffs should be real?
            # eps(x) is real. 1/eps is real. Fourier coeffs are complex conjugates.
            # M should be Hermitian.
            eigvals, eigvecs = np.linalg.eigh(M)
            
            # Sort eigenvalues
            eigvals = np.sort(eigvals.real)
            
            # Bands 4 and 5 are indices 3 and 4 (0-based)
            # We need the 4th and 5th positive eigenvalues?
            # Usually bands are indexed 1, 2, 3...
            # The gap is between band 4 and band 5.
            # So we look at eigvals[3] and eigvals[4].
            
            # Check if they are valid (positive)
            if eigvals[3] > 0 and eigvals[4] > 0:
                max_omega4 = max(max_omega4, eigvals[3])
                min_omega5 = min(min_omega5, eigvals[4])
        
        if min_omega5 == np.inf:
            return -1.0
            
        gap = min_omega5 - max_omega4
        midgap = (min_omega5 + max_omega4) / 2.0
        
        if midgap == 0:
            return 0.0
            
        return 2.0 * gap / midgap

    # 3. Optimization Loop
    # Parameters: [eps_avg, A1, A2, A3]
    # Initial guess: eps_avg = (1+12)/2 = 6.5, A1=0, A2=0, A3=0 (Uniform)
    # But uniform has no gap.
    # Let's start with a rod-like structure.
    # A large A1 creates a 4-lobe pattern? cos(x)cos(y) is positive in corners, negative in center?
    # cos(2pi x) is 1 at x=0, -1 at x=0.5.
    # So A1 > 0 makes center high (dielectric), corners low (air).
    # This is a rod.
    
    initial_params = [6.5, 5.0, 0.0, 0.0]
    
    # Nelder-Mead
    # Implement a simple version or use scipy?
    # "Python standard library, NumPy and SciPy are available."
    # Use scipy.optimize.minimize
    
    from scipy.optimize import minimize
    
    def objective(params):
        if time.time() - start_time > time_limit:
            return 1000.0 # Penalty
        try:
            eps_bin = get_eps_field(params)
            score = solve_bands(eps_bin)
            return -score # Maximize score -> Minimize -score
        except Exception:
            return 1000.0

    # Bounds for parameters
    # eps_avg in [1, 12]
    # Amplitudes in [-5, 5]
    bounds = [(1, 12), (-5, 5), (-5, 5), (-5, 5)]
    
    # Run optimization
    # Max iterations to keep within time
    max_iter = 50
    options = {'maxiter': max_iter, 'xatol': 1e-3, 'fatol': 1e-3}
    
    res = minimize(objective, initial_params, method='Nelder-Mead', bounds=bounds, options=options)
    
    # Get best params
    best_params = res.x
    best_eps_bin = get_eps_field(best_params)
    
    # Final check: Ensure symmetry
    # The construction guarantees symmetry, but let's verify and enforce if needed.
    # cell == cell[::-1] and cell == cell.T
    # Our field is cos(x)cos(y) which is symmetric.
    # Binarization preserves symmetry.
    
    return best_eps_bin.tolist()
