import numpy as np
import time
import random

# Constants for the solver
N_PX = 32
N_BASIS = 15  # 15x15 basis vectors -> 225 planes. Fast and accurate enough.
EPS_HIGH = 12.0
EPS_LOW = 1.0

# Precompute reciprocal lattice vectors
# k = (mx + ny) * b1 + (my - nx) * b2 ? No, standard square lattice:
# G = (nx, ny) in units of 2pi/a.
# We center them around 0.
half_basis = N_BASIS // 2
# Generate indices from -half_basis to half_basis
idx_range = np.arange(-half_basis, half_basis + 1)
# Create meshgrid of reciprocal lattice indices
GX, GY = np.meshgrid(idx_range, idx_range, indexing='ij')
GX_flat = GX.flatten()
GY_flat = GY.flatten()
N_PLS = len(GX_flat)

# TM Matrix Base Part: (G + k)^2 * delta_G,0
# The TM eigenvalue equation is sum_{G'} [ (G+k)^2 delta_{G,G'} + (1/eps)_{G-G'} ] c_{G'} = (omega/k)^2 c_G
# Actually, the standard form is:
# sum_{G'} [ (G+k)^2 delta_{G,G'} + (4pi^2/a^2) (1/eps)_{G-G'} ] c_{G'} = (omega/(k a))^2 c_G
# Let's normalize k in units of 2pi/a. So G are integers.
# Term 1: (Gx + kx)^2 + (Gy + ky)^2
# Term 2: Fourier coeff of 1/eps.

def compute_eps_inv_fourier(cell):
    """
    Compute Fourier coefficients of 1/eps for the 32x32 cell.
    cell is 32x32 list of 0s and 1s.
    Returns numpy array of shape (N_PLS,) of complex coefficients for G in basis.
    """
    eps = np.zeros((N_PX, N_PX))
    for i in range(N_PX):
        for j in range(N_PX):
            if cell[i][j] == 1:
                eps[i, j] = EPS_HIGH
            else:
                eps[i, j] = EPS_LOW
            
    eps_inv = 1.0 / eps
    
    # Compute Fourier coefficients for G in our basis
    # c_G = sum_{i,j} eps_inv[i,j] * exp(-i * 2pi * (Gx*i + Gy*j)/N_PX) / N_PX^2
    # Using FFT is faster, but N_PLS is small (225) and N_PX is 32.
    # Direct sum: 225 * 32 * 32 ~ 230k ops. Very fast.
    coeffs = np.zeros(N_PLS, dtype=np.complex128)
    
    # Precompute phase factors? 
    # For each G, sum over pixels.
    # i, j range 0..31.
    i_coords = np.arange(N_PX)
    j_coords = np.arange(N_PX)
    I, J = np.meshgrid(i_coords, j_coords, indexing='ij')
    
    for k in range(N_PLS):
        gx = GX_flat[k]
        gy = GY_flat[k]
        # Phase: -2pi * (gx*I + gy*J) / N_PX
        phase = -2.0 * np.pi * (gx * I + gy * J) / N_PX
        weights = eps_inv * np.exp(1j * phase)
        coeffs[k] = np.sum(weights) / (N_PX * N_PX)
        
    return coeffs

def build_tm_matrix(kx, ky, coeffs):
    """
    Build the TM matrix for wave vector (kx, ky) in units of 2pi/a.
    kx, ky are real numbers.
    coeffs are the Fourier coefficients of 1/eps.
    """
    M = np.zeros((N_PLS, N_PLS), dtype=np.complex128)
    
    # 1/eps matrix in Fourier space
    # (1/eps)_{G, G'} = (1/eps)_{G-G'}
    # We have coeffs for G in basis. We need the difference G - G'.
    # This is a circulant-like structure but truncated.
    # Instead of building the full matrix via convolution, we can use the fact that
    # M_{ij} = ((G_i + k)^2) delta_{ij} + c_{G_i - G_j}
    
    # Precompute (G_i + k)^2
    Gkx = GX_flat + kx
    Gky = GY_flat + ky
    kinetic_terms = Gkx**2 + Gky**2
    
    # Fill diagonal
    for i in range(N_PLS):
        M[i, i] = kinetic_terms[i]
        
    # Fill off-diagonals with coeffs
    # We need c_{G_i - G_j}
    # We can compute the full convolution or just sum directly.
    # Since N_PLS is 225, O(N^2) is 50k ops. Acceptable.
    for i in range(N_PLS):
        for j in range(N_PLS):
            if i == j:
                continue
            dgx = GX_flat[i] - GX_flat[j]
            dgy = GY_flat[i] - GY_flat[j]
            # Find if (dgx, dgy) is in our basis.
            # If it is, add coeffs[idx]. If not, assume 0 (truncation error).
            # To speed this up, create a dictionary of coeffs indexed by (gx, gy)
            # But doing this every time build_tm_matrix is called is wasteful.
            # Let's precompute a lookup dict for coeffs once.
            
    # This O(N^2) loop inside the function is slow if called many times.
    # Optimization: Precompute the sparse structure of non-zero Fourier coeffs?
    # The coeffs are non-zero for all G in the basis.
    # Let's use the dictionary approach inside the design loop, but precompute the keys.
    
    # Better approach: Use the property that the matrix is formed by kinetic + potential.
    # Potential part is a "convolution" in Fourier space.
    # Let's stick to the direct loop but optimize it by precomputing the diff indices.
    
    # Actually, for 225x225, a pure Python loop is ~0.01s? No, 50k iterations in Python is ~5ms.
    # With 30 k-points and many iterations, this adds up.
    # Let's try to vectorize the potential part construction?
    # M_pot[i,j] = coeffs[ idx_of(G_i - G_j) ]
    
    # Let's return a helper that builds M efficiently.
    # We will pass a pre-computed lookup for coeffs.
    
    # Re-doing the loop with a lookup dict provided externally is better.
    pass

# Let's restructure to precompute the index mapping for differences
# Create a map: (dx, dy) -> index in coeffs array, or -1 if not in basis
diff_to_idx = {}
for i in range(N_PLS):
    for j in range(N_PLS):
        dx = GX_flat[i] - GX_flat[j]
        dy = GY_flat[i] - GY_flat[j]
        # Check if (dx, dy) is in our basis set
        # We can just check if it's in the set of (GX_flat, GY_flat)
        # But doing this for all pairs is expensive.
        
# Simpler: Just compute the potential matrix P where P[i,j] = c_{Gi-Gj}
# We can compute the full Fourier series of 1/eps on a larger grid? No.
# Let's just use the loop but optimize the inner part.
# Precompute the list of (i, j, coeff_index) for all non-zero interactions?
# All interactions in the basis are non-zero.
# There are 225*225 = 50625 interactions.
# We can precompute the list of (i, j, coeff_val) once for a given coeffs array?
# The coeffs change every iteration. So we can't precompute the values, but we can precompute the indices.

# Precompute indices for each (i,j) pair
# i, j in 0..N_PLS-1
# We need to find k such that G_k = G_i - G_j
# Create a set of valid G vectors for fast lookup
valid_Gs = set(zip(GX_flat, GY_flat))
# Map G vector to index
G_to_idx = {}
for idx, (gx, gy) in enumerate(zip(GX_flat, GY_flat)):
    G_to_idx[(gx, gy)] = idx

# Precompute the list of (i, j, k) where k is the index of G_i - G_j if it exists
interaction_indices = []
for i in range(N_PLS):
    for j in range(N_PLS):
        if i == j:
            continue
        dx = GX_flat[i] - GX_flat[j]
        dy = GY_flat[i] - GY_flat[j]
        key = (dx, dy)
        if key in G_to_idx:
            k = G_to_idx[key]
            interaction_indices.append((i, j, k))

# Now building the matrix is just filling M[i,j] = coeffs[k] for these triplets.

def get_tm_bands(kx, ky, coeffs):
    M = np.zeros((N_PLS, N_PLS), dtype=np.complex128)
    
    # Diagonal
    Gkx = GX_flat + kx
    Gky = GY_flat + ky
    diag = Gkx**2 + Gky**2
    np.fill_diagonal(M, diag)
    
    # Off-diagonal from interaction_indices
    for i, j, k in interaction_indices:
        M[i, j] = coeffs[k]
        
    # Solve eigenvalue problem
    # M is Hermitian?
    # Kinetic part is real and symmetric.
    # Potential part: c_{Gi-Gj}. Since 1/eps is real, c_{-G} = conj(c_G).
    # c_{Gj-Gi} = c_{-(Gi-Gj)} = conj(c_{Gi-Gj}).
    # So M[j,i] = coeffs[idx(Gj-Gi)] = conj(coeffs[idx(Gi-Gj)]) = conj(M[i,j]).
    # Yes, M is Hermitian. We can use eigh.
    
    try:
        eigvals = np.linalg.eigvalsh(M)
        # eigvals are in units of (2pi/a)^2. 
        # omega = k * sqrt(eigval). 
        # For gap ratio, k cancels out if we compare at same k, but k varies.
        # Score is 2(min w5 - max w4) / (min w5 + max w4).
        # w = k * sqrt(eig).
        # So we return k * sqrt(eigvals)
        freqs = np.sqrt(np.maximum(eigvals, 0)) * np.sqrt(kx**2 + ky**2)
        return freqs
    except np.linalg.LinAlgError:
        return np.zeros(5) # Fallback

def compute_score(cell):
    coeffs = compute_eps_inv_fourier(cell)
    
    # Sample k-points in irreducible BZ
    # Gamma (0,0), X (0.5, 0), M (0.5, 0.5)
    # Also some points on the lines to capture band maxima/minima better.
    k_points = [
        (0.0, 0.0),
        (0.5, 0.0),
        (0.5, 0.5),
        (0.25, 0.0),
        (0.5, 0.25),
        (0.25, 0.25)
    ]
    
    min_w5 = float('inf')
    max_w4 = -float('inf')
    
    for kx, ky in k_points:
        if kx == 0 and ky == 0:
            # At Gamma, k=0, all bands are 0. 
            # Actually, the frequency is 0. 
            # The band structure usually refers to the dispersion.
            # At Gamma, the eigenvalues are the frequencies squared.
            # But for the gap, we look at the bands in the BZ.
            # Usually, the "band" is defined as the set of frequencies for a given n.
            # At Gamma, n=1..4 are 0? No, they are distinct modes.
            # Wait, for TM, at Gamma (k=0), the vector equation decouples into scalar Laplacians.
            # The eigenvalues are k^2 * lambda. 
            # At k=0, freq=0. 
            # So all bands meet at 0 at Gamma? 
            # Yes, for photonic crystals, all bands start at 0 at Gamma.
            # So max_w4 at Gamma is 0. min_w5 at Gamma is 0.
            # The gap opens away from Gamma.
            # So we should exclude Gamma from the min_w5/max_w4 calculation?
            # Or does the band definition imply the envelope of the dispersion?
            # "gap between the 4th and 5th TM bands"
            # Usually, Band 4 is the 4th lowest frequency for each k.
            # At Gamma, f4=0, f5=0. 
            # The gap is the minimum of (f5(k) - f4(k)) over k.
            # If f4 and f5 cross at Gamma, the gap is 0.
            # However, the 4th and 5th bands do not cross at Gamma in a typical PC?
            # Actually, at Gamma, the modes are degenerate due to symmetry.
            # The 1st, 2nd, 3rd, 4th modes might have non-zero frequencies if we consider the lattice periodicity?
            # No, for a lattice with period a, k=0 is the center.
            # The eigenvalues of the Laplacian on the unit cell with periodic BC are non-zero.
            # Wait. The PWE method solves for the dielectric mode.
            # At k=0, the plane wave expansion reduces to finding the eigenmodes of the dielectric function in the unit cell.
            # These eigenmodes have non-zero frequencies.
            # So Gamma is a valid point to check.
            
            freqs = get_tm_bands(0.0, 0.0, coeffs)
            # The eigenvalues returned by get_tm_bands are k*sqrt(eig).
            # My get_tm_bands implementation multiplies by sqrt(kx^2+ky^2).
            # This is WRONG for the eigenvalue definition in PWE.
            # The eigenvalue lambda from the matrix M is (omega / k)^2 * (2pi/a)^2?
            # Let's check the standard PWE formulation.
            # (G+k)^2 delta + 1/eps = (omega/k)^2
            # So omega = k * sqrt( (G+k)^2 delta + ... )
            # At k=0, this formula breaks because of the 1/k.
            # However, the eigenvalues of the matrix M are proportional to (omega/k)^2.
            # So omega^2 = k^2 * lambda_M.
            # So omega = |k| * sqrt(lambda_M).
            # At k=0, this gives 0. 
            # But physically, the frequencies at Gamma are NOT zero.
            # The issue is that at k=0, the "k" in the denominator is the free-space wavenumber?
            # No, in PWE, we solve for the dielectric modes.
            # The correct relation is omega^2 = c^2 * lambda_M.
            # The matrix M contains (G+k)^2.
            # At k=0, M contains G^2.
            # The eigenvalues of M are the (omega/c)^2.
            # So omega = c * sqrt(lambda_M).
            # The factor of k should NOT be in the final frequency calculation!
            # The (G+k)^2 term handles the momentum.
            # So my `get_tm_bands` returning `freqs = np.sqrt(eigvals) * np.sqrt(kx**2 + ky**2)` is INCORRECT.
            # It should be `freqs = np.sqrt(eigvals)`.
            # Let's verify.
            # For air, eps=1. M = (G+k)^2. Eigenvalues are (G+k)^2.
            # omega^2 = (G+k)^2. omega = |G+k|.
            # This is the dispersion of light in a vacuum. Correct.
            # So the frequency is just sqrt(eigval).
            
            # I need to fix get_tm_bands.
            
        # Corrected get_tm_bands logic:
        # eigvals are (omega)^2 (in units of (2pi/a)^2 if G are integers).
        # So omega = sqrt(eigval).
        
        pass

# Let's rewrite get_tm_bands correctly
def get_tm_bands_correct(kx, ky, coeffs):
    M = np.zeros((N_PLS, N_PLS), dtype=np.complex128)
    
    Gkx = GX_flat + kx
    Gky = GY_flat + ky
    diag = Gkx**2 + Gky**2
    np.fill_diagonal(M, diag)
    
    for i, j, k in interaction_indices:
        M[i, j] = coeffs[k]
        
    try:
        eigvals = np.linalg.eigvalsh(M)
        freqs = np.sqrt(np.maximum(eigvals, 0))
        return freqs
    except np.linalg.LinAlgError:
        return np.zeros(5)

def compute_score_correct(cell):
    coeffs = compute_eps_inv_fourier(cell)
    
    # Sample k-points
    # Irreducible BZ of square lattice: 0 <= ky <= kx <= 0.5
    # Key points: Gamma(0,0), X(0.5,0), M(0.5,0.5)
    # And points in between to find max of band 4 and min of band 5.
    k_points = []
    # Gamma
    k_points.append((0.0, 0.0))
    # X
    k_points.append((0.5, 0.0))
    # M
    k_points.append((0.5, 0.5))
    # Midpoints
    k_points.append((0.25, 0.0))
    k_points.append((0.5, 0.25))
    k_points.append((0.25, 0.25))
    # A few more for better resolution
    k_points.append((0.4, 0.1))
    k_points.append((0.3, 0.2))
    
    min_w5 = float('inf')
    max_w4 = -float('inf')
    
    for kx, ky in k_points:
        freqs = get_tm_bands_correct(kx, ky, coeffs)
        # Sort frequencies
        freqs = np.sort(freqs)
        # Band 4 is index 3, Band 5 is index 4 (0-indexed)
        if len(freqs) >= 5:
            w4 = freqs[3]
            w5 = freqs[4]
            if w4 > max_w4:
                max_w4 = w4
            if w5 < min_w5:
                min_w5 = w5
                
    if min_w5 == float('inf'):
        return -1.0
        
    gap = min_w5 - max_w4
    midgap = (min_w5 + max_w4) / 2.0
    if midgap == 0:
        return -1.0
    return 2.0 * gap / midgap

def design(problem):
    start_time = time.time()
    time_limit = 110.0  # Safety margin
    
    size = problem["pixels"] # 32
    
    # Initialize with a cross shape or random
    # Start with a symmetric cross
    quad = np.zeros((16, 16), dtype=int)
    # Draw a cross in the center of the 16x16 quad
    # Center of 16x16 is (7.5, 7.5)
    # Let's try a ring or a cross
    # Seed: Centered rod radius 0.2 * 32 = 6.4 pixels.
    # In 16x16, radius 3.2.
    for i in range(16):
        for j in range(16):
            dx = i - 7.5
            dy = j - 7.5
            if dx*dx + dy*dy < 3.2**2:
                quad[i, j] = 1
            
    def expand_quad(q):
        # Mirror to 32x32
        # Top-left: q
        # Top-right: q[:, ::-1]
        # Bottom-left: q[::-1, :]
        # Bottom-right: q[::-1, ::-1]
        top = np.hstack((q, q[:, ::-1]))
        bottom = top[::-1, :]
        return np.vstack((top, bottom))
        
    current_cell = expand_quad(quad)
    current_score = compute_score_correct(current_cell)
    best_cell = current_cell.copy()
    best_score = current_score
    
    T = 1.0
    T_min = 0.01
    cooling = 0.99
    
    while time.time() - start_time < time_limit:
        # Pick a random pixel in the quad to flip
        ri = random.randint(0, 15)
        rj = random.randint(0, 15)
        
        quad[ri, rj] = 1 - quad[ri, rj]
        new_cell = expand_quad(quad)
        new_score = compute_score_correct(new_cell)
        
        delta = new_score - current_score
        if delta > 0 or random.random() < np.exp(delta / T):
            current_cell = new_cell
            current_score = new_score
            if current_score > best_score:
                best_score = current_score
                best_cell = current_cell.copy()
        else:
            quad[ri, rj] = 1 - quad[ri, rj] # Revert
            
        T *= cooling
        if T < T_min:
            T = T_min
            
    # Convert best_cell back to list of lists
    return best_cell.tolist()
