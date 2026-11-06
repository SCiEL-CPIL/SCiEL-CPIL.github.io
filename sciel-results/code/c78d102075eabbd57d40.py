"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import math
import time
import numpy as np

def design(problem):
    """
    Design a 32x32 photonic crystal unit cell with D4 symmetry to maximize
    the gap between TM bands 4 and 5.
    
    Strategy:
    1. Use a 2x2 array of circular dielectric rods (air background) as the base geometry,
       which is known to open larger TM band gaps than a single central rod.
    2. Use a fast approximate plane-wave method to estimate the gap between bands 4 and 5.
    3. Perform a local search over rod radius to find the optimal size.
    4. Return the design with the largest estimated gap.
    """
    size = problem.get("pixels", 32)
    eps_low = problem.get("eps_low", 1)
    eps_high = problem.get("eps_high", 12)
    
    start_time = time.time()
    deadline = start_time + 100  # Safety margin
    
    def generate_cell(radius_frac):
        """
        Generate a 32x32 cell with a 2x2 array of central circular rods.
        Rod centers at (0.25, 0.25), (0.25, 0.75), (0.75, 0.25), (0.75, 0.75) in normalized coords.
        """
        cell = [[0] * size for _ in range(size)]
        r_px = radius_frac * size
        centers = [
            (0.25 * size, 0.25 * size),
            (0.25 * size, 0.75 * size),
            (0.75 * size, 0.25 * size),
            (0.75 * size, 0.75 * size)
        ]
        for i in range(size):
            for j in range(size):
                x = j + 0.5
                y = i + 0.5
                for cx, cy in centers:
                    dx = x - cx
                    dy = y - cy
                    if dx * dx + dy * dy < r_px * r_px:
                        cell[i][j] = 1
                        break
        return cell
    
    def compute_fourier_coefficients(cell, max_order=8):
        """
        Compute Fourier coefficients of 1/epsilon for G vectors with |m|,|n| <= max_order.
        """
        eps = np.array([[eps_high if cell[i][j] == 1 else eps_low for j in range(size)] for i in range(size)])
        inv_eps = 1.0 / eps
        
        # Use FFT for efficiency
        # We need coefficients for G = (2*pi*m, 2*pi*n) where m,n in [-max_order, max_order]
        # The discrete Fourier series coefficient is:
        # f_G = (1/N^2) * sum_{i,j} f[i,j] * exp(-i * 2*pi * (m*i + n*j) / N)
        
        # Compute the 2D FFT
        fft_vals = np.fft.fft2(inv_eps)
        
        coeffs = {}
        for m in range(-max_order, max_order + 1):
            for n in range(-max_order, max_order + 1):
                # Map to FFT indices
                m_idx = m % size
                n_idx = n % size
                # Normalize by N^2
                coeffs[(m, n)] = fft_vals[m_idx, n_idx] / (size * size)
        
        return coeffs
    
    def approximate_gap_ratio(cell, max_order=8, k_points=None):
        """
        Approximate the gap ratio between bands 4 and 5.
        Uses a small basis set of plane waves.
        """
        if k_points is None:
            # Use high-symmetry points and some interior points
            k_points = [
                (0, 0),
                (0.5, 0),
                (0.5, 0.5),
                (0.25, 0.25),
                (0.5, 0.25),
                (0.25, 0),
                (0, 0.25),
                (0.35, 0.35),
                (0.45, 0.45),
                (0.5, 0.35),
            ]
        
        coeffs = compute_fourier_coefficients(cell, max_order)
        
        # Define the set of G vectors to use in the basis
        # We need enough G vectors to resolve the bands up to index 5.
        # Let's use m,n in -3..3 for a 7x7 = 49 state basis, which should be sufficient.
        g_vectors = [(m, n) for m in range(-3, 4) for n in range(-3, 4)]
        
        min_omega_5 = float('inf')
        max_omega_4 = 0.0
        
        for kx, ky in k_points:
            # Build matrix
            N = len(g_vectors)
            H = np.zeros((N, N), dtype=complex)
            
            for i, Gp in enumerate(g_vectors):
                for j, G in enumerate(g_vectors):
                    # G_diff = Gp - G
                    m_diff = Gp[0] - G[0]
                    n_diff = Gp[1] - G[1]
                    
                    # Check if (m_diff, n_diff) is in coeffs
                    if (m_diff, n_diff) in coeffs:
                        eps_inv_G = coeffs[(m_diff, n_diff)]
                    else:
                        eps_inv_G = 0.0
                    
                    # k + Gp
                    kx_total = kx + Gp[0]
                    ky_total = ky + Gp[1]
                    k_sq = kx_total**2 + ky_total**2
                    
                    H[i, j] = k_sq * eps_inv_G
            
            # Solve eigenvalue problem
            # H is Hermitian
            eigvals = np.linalg.eigvalsh(H)
            eigvals = np.sort(eigvals.real)
            
            # We are interested in bands 4 and 5 (indices 3 and 4 in 0-based)
            if len(eigvals) >= 5:
                omega_4 = eigvals[3]
                omega_5 = eigvals[4]
                
                if omega_5 < min_omega_5:
                    min_omega_5 = omega_5
                if omega_4 > max_omega_4:
                    max_omega_4 = omega_4
        
        if min_omega_5 == float('inf') or max_omega_4 == 0:
            return -1.0
        
        gap = min_omega_5 - max_omega_4
        midgap = (min_omega_5 + max_omega_4) / 2.0
        
        if midgap <= 0:
            return -1.0
        
        return gap / midgap
    
    # Candidate radii
    # The 2x2 rod geometry typically has optimal radii around 0.12-0.18
    radii = [0.10, 0.12, 0.14, 0.16, 0.18, 0.20, 0.22]
    
    best_score = -1.0
    best_cell = generate_cell(0.14)
    
    for r in radii:
        if time.time() - start_time > deadline:
            break
        cell = generate_cell(r)
        score = approximate_gap_ratio(cell)
        if score > best_score:
            best_score = score
            best_cell = cell
    
    # Enforce D4 symmetry strictly (just in case)
    final_cell = best_cell
    for i in range(size):
        for j in range(size):
            vals = [
                final_cell[i][j],
                final_cell[size-1-i][j],
                final_cell[i][size-1-j],
                final_cell[size-1-i][size-1-j],
                final_cell[j][i],
                final_cell[size-1-j][size-1-i],
                final_cell[j][size-1-i],
                final_cell[size-1-j][i],
            ]
            avg = sum(vals) / len(vals)
            final_cell[i][j] = 1 if avg >= 0.5 else 0
    
    return final_cell

# EVOLVE-BLOCK-END
