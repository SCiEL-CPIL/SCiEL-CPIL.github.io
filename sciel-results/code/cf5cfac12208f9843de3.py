import numpy as np
import time

def _chebyshev_basis(size):
    """
    Generate a set of C4v-symmetric basis functions on a 32x32 grid.
    We use Chebyshev polynomials of the first kind.
    To ensure C4v symmetry, we construct basis functions that are invariant 
    under the symmetry group, or we average the final field.
    Averaging is simpler: we generate a field F(x,y) from arbitrary coefficients,
    then compute F_sym(x,y) = (F(x,y) + F(-x,y) + F(x,-y) + F(-x,-y) + F(y,x) + F(-y,x) + F(y,-x) + F(-y,-x)) / 8.
    However, for optimization efficiency, it's better to restrict the basis to symmetric terms.
    Let's use a set of symmetric monomials/Chebyshev combinations.
    
    Simpler approach: 
    Use a small number of parameters that directly define a symmetric shape.
    e.g., Superellipse: |x|^(2/n) + |y|^(2/n) = R^2.
    Or a level set: f(x,y) = sum c_i * T_i(x) T_j(y) where T are Chebyshev.
    To guarantee symmetry, we only use terms that are symmetric under x->-x, y->-y, x<->y.
    This implies only even powers of x and y in x and y separately, and symmetric in x,y.
    Terms like T_{2k}(x) T_{2l}(y) + T_{2l}(x) T_{2k}(y).
    """
    size = 32
    x = (np.arange(size) + 0.5) / size - 0.5
    y = (np.arange(size) + 0.5) / size - 0.5
    xx, yy = np.meshgrid(x, y)
    
    # Precompute Chebyshev polynomials up to order 8
    # T_0 = 1, T_1 = x, T_2 = 2x^2-1, ...
    # We will construct a set of symmetric basis functions.
    
    # Basis 0: Constant (background)
    basis = []
    basis.append(np.ones_like(xx))
    
    # Basis 1: Radial-like (superellipse n=2, circle)
    # x^2 + y^2
    basis.append(xx**2 + yy**2)
    
    # Basis 2: Superellipse n=4 (squarish)
    # x^4 + y^4 + x^2y^2 (to keep it smooth and symmetric)
    basis.append(xx**4 + yy**4 + xx**2 * yy**2)
    
    # Basis 3: Four-fold rosette
    # x^2 y^2
    basis.append(xx**2 * yy**2)
    
    # Basis 4: Higher order radial
    # x^6 + y^6 + x^4y^2 + x^2y^4
    basis.append(xx**6 + yy**6 + xx**4*yy**2 + xx**2*yy**4)
    
    # Basis 5: Cross-like
    # x^2 - y^2 (breaks C4, so skip or combine)
    # We need C4 invariant. x^2+y^2 is C4. x^4+y^4 is C4.
    # x^2 y^2 is C4.
    # What about (x^2 - y^2)^2 = x^4 - 2x^2y^2 + y^4? This is C4 invariant.
    basis.append((xx**2 - yy**2)**2)
    
    # Basis 6: More complex
    # (x^2 + y^2)^2 = x^4 + 2x^2y^2 + y^4
    basis.append((xx**2 + yy**2)**2)
    
    return np.array(basis)

def _generate_cell(coeffs, size=32):
    """
    Generate a binary cell from Chebyshev-like coefficients.
    coeffs[0] is the threshold/offset.
    """
    basis = _chebyshev_basis(size)
    # Field = sum c_i * basis_i
    # We want to find the boundary where Field = 0.
    # Let's define the field F(x,y) = coeffs[1]*B1 + ... - coeffs[0]
    # Dielectric if F(x,y) > 0 ? Or < 0?
    # Let's say Dielectric if F(x,y) > 0.
    
    field = np.zeros((size, size))
    for i in range(1, len(basis)):
        field += coeffs[i] * basis[i]
    field -= coeffs[0]
    
    # Threshold
    cell = (field > 0).astype(int)
    
    # Ensure strict C4v symmetry by averaging with symmetries
    # This corrects any numerical asymmetry in the basis or thresholding
    cell_avg = (cell + cell[::-1, ::] + cell[::-1, :] + cell[:, ::] + 
                cell.T + cell.T[::-1, ::] + cell.T[::-1, :] + cell.T[:, ::]) / 8.0
    cell_final = (cell_avg >= 0.5).astype(int)
    
    return cell_final

def _fast_gap_eval(cell, size=32):
    """
    Fast approximate band structure calculation.
    Uses a small number of plane waves and k-points.
    """
    eps_low = 1.0
    eps_high = 12.0
    
    # Create epsilon grid
    eps = np.where(cell == 1, eps_high, eps_low)
    
    # Calculate inverse epsilon
    eps_inv = 1.0 / eps
    
    # Discretize for FFT to get Fourier coefficients
    # Use a slightly larger grid for FFT to avoid aliasing? 
    # Standard plane wave method uses exact coefficients.
    # For speed, we approximate coefficients using FFT on the 32x32 grid.
    # This is an approximation but should be consistent for ranking.
    
    # Get Fourier coefficients of eps_inv
    # G vectors: kx, ky. We need a set of G vectors.
    # Let's use a 5x5 grid of G vectors (indices -2 to 2)
    N_G = 5
    g_indices = np.arange(-N_G//2, N_G//2 + 1)
    G_vectors = np.array([(gx, gy) for gx in g_indices for gy in g_indices])
    
    # Compute eps_inv(G)
    # eps_inv(G) = 1/(32*32) * sum_{i,j} eps_inv(i,j) exp(-i 2pi (Gx i + Gy j) / 32)
    # This is exactly the 2D FFT (scaled).
    eps_inv_fft = np.fft.fft2(eps_inv)
    # The FFT output is ordered [0..N-1, 0..N-1].
    # We need to map our G indices (-2..2) to FFT indices.
    # G=0 is index 0. G=1 is index 1. G=-1 is index 31.
    
    def get_eps_inv_coeff(gx, gy):
        idx_x = gx % size
        idx_y = gy % size
        return eps_inv_fft[idx_y, idx_x] / (size * size)
    
    # Precompute matrix A for a given k
    # A[G, G'] = (k-G) . (k-G') * eps_inv(G-G')
    # Note: (k-G).dot(k-G') is a scalar for 2D TM.
    # Actually, the operator is (k-G) x (k-G') for TE? No, for TM it's scalar product.
    # For TM: (k-G) . (k-G')
    
    # k-points to sample
    # Irreducible BZ for square lattice: 0 <= kx <= ky, kx+ky <= 1 (if a=1)
    # High symmetry points: Gamma(0,0), M(0.5,0), K(0.5,0.5), X(0,0.5)
    # Let's sample:
    k_points = [
        (0.0, 0.0),   # Gamma
        (0.5, 0.0),   # X
        (0.5, 0.5),   # M
        (0.25, 0.0),  # Mid Gamma-X
        (0.5, 0.25),  # Mid X-M
        (0.25, 0.25)  # Interior
    ]
    
    min_omega_5 = float('inf')
    max_omega_4 = 0.0
    
    for kx, ky in k_points:
        k = np.array([kx, ky])
        
        # Build matrix A
        N_pw = len(G_vectors)
        A = np.zeros((N_pw, N_pw), dtype=complex)
        
        for i, Gi in enumerate(G_vectors):
            for j, Gj in enumerate(G_vectors):
                # k - Gi
                k_minus_Gi = k - Gi
                k_minus_Gj = k - Gj
                dot_product = np.dot(k_minus_Gi, k_minus_Gj)
                
                # G_diff = Gi - Gj
                G_diff = Gi - Gj
                eps_inv_coeff = get_eps_inv_coeff(G_diff[0], G_diff[1])
                
                A[i, j] = dot_product * eps_inv_coeff
        
        # Solve eigenvalue problem: A c = omega^2 c
        # A is Hermitian
        try:
            eigvals, eigvecs = np.linalg.eigh(A)
            # eigvals are omega^2
            # Filter out small negative values due to numerical noise
            eigvals = np.real(eigvals)
            eigvals = eigvals[eigvals > 1e-6]
            omegas = np.sqrt(eigvals)
            
            # We need bands 4 and 5.
            # Ensure we have at least 5 bands
            if len(omegas) >= 5:
                omega_4 = omegas[3]
                omega_5 = omegas[4]
                
                # Update global min/max
                # The gap is defined as min(omega_5) - max(omega_4)
                # So we track the max of the 4th band and min of the 5th band
                if omega_4 > max_omega_4:
                    max_omega_4 = omega_4
                if omega_5 < min_omega_5:
                    min_omega_5 = omega_5
        except Exception:
            continue
    
    if min_omega_5 == float('inf') or max_omega_4 == 0:
        return 0.0
        
    gap = min_omega_5 - max_omega_4
    midgap = (min_omega_5 + max_omega_4) / 2.0
    
    if midgap == 0:
        return 0.0
        
    ratio = gap / midgap
    # The task asks for 2 * gap / midgap
    return 2.0 * ratio

def design(problem):
    size = problem["pixels"]
    start_time = time.time()
    deadline = start_time + 110.0 # 10s margin
    
    # Initial coefficients
    # c0: threshold, c1..cN: shape weights
    # Let's start with a circle-like shape
    # B1 = x^2+y^2. If c1=1, c0=0.5, we get a circle of radius sqrt(0.5) ~ 0.7 (too big)
    # We want a rod of radius ~0.2-0.3.
    # x^2+y^2 < r^2. 
    # If field = c1*(x^2+y^2) - c0 > 0 is dielectric?
    # If c1=1, c0=0.1, then x^2+y^2 > 0.1 is dielectric. That's the background.
    # We want dielectric in the center.
    # So field = c0 - c1*(x^2+y^2) > 0.
    # Let's adjust signs in _generate_cell or here.
    # In _generate_cell: field = sum c_i B_i - c0.
    # If B1 = x^2+y^2, and we want dielectric where x^2+y^2 < r^2,
    # we need field > 0 inside.
    # So we need - (x^2+y^2) + c0 > 0.
    # So c1 should be negative.
    
    best_coeffs = [0.1, -1.0, 0.0, 0.0, 0.0, 0.0, 0.0] 
    best_cell = _generate_cell(best_coeffs, size)
    best_score = _fast_gap_eval(best_cell, size)
    
    # Optimization loop
    # Number of coefficients to optimize
    n_coeffs = len(best_coeffs)
    
    # Random perturbations
    # We will use a simple hill climbing with random steps
    step_size = 0.05
    
    while time.time() < deadline:
        # Perturb
        new_coeffs = best_coeffs + np.random.normal(0, step_size, n_coeffs)
        
        # Ensure c0 is reasonable (threshold)
        # Keep c0 in [0, 1]
        new_coeffs[0] = np.clip(new_coeffs[0], 0.0, 1.0)
        
        new_cell = _generate_cell(new_coeffs, size)
        
        # Check if the cell is valid (not empty or full)
        if np.sum(new_cell) == 0 or np.sum(new_cell) == size*size:
            continue
            
        new_score = _fast_gap_eval(new_cell, size)
        
        if new_score > best_score:
            best_score = new_score
            best_coeffs = new_coeffs
            best_cell = new_cell
            # Reduce step size slightly on improvement
            step_size *= 0.95
        else:
            # Occasionally increase step size to escape local minima
            if np.random.random() < 0.05:
                step_size *= 1.1
                step_size = min(step_size, 0.2)
    
    return best_cell

# Ensure the function signature matches
def design(problem):
    return _design_impl(problem)

def _design_impl(problem):
    size = problem["pixels"]
    start_time = time.time()
    deadline = start_time + 110.0
    
    # Initial coefficients
    best_coeffs = [0.1, -1.0, 0.0, 0.0, 0.0, 0.0, 0.0] 
    best_cell = _generate_cell(best_coeffs, size)
    best_score = _fast_gap_eval(best_cell, size)
    
    n_coeffs = len(best_coeffs)
    step_size = 0.05
    
    while time.time() < deadline:
        new_coeffs = best_coeffs + np.random.normal(0, step_size, n_coeffs)
        new_coeffs[0] = np.clip(new_coeffs[0], 0.0, 1.0)
        
        new_cell = _generate_cell(new_coeffs, size)
        
        if np.sum(new_cell) == 0 or np.sum(new_cell) == size*size:
            continue
            
        new_score = _fast_gap_eval(new_cell, size)
        
        if new_score > best_score:
            best_score = new_score
            best_coeffs = new_coeffs
            best_cell = new_cell
            step_size *= 0.95
        else:
            if np.random.random() < 0.05:
                step_size *= 1.1
                step_size = min(step_size, 0.2)
                
    return best_cell
