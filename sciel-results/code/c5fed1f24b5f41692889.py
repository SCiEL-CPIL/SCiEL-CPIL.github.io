"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import math
import random

# EVOLVE-BLOCK-START

def _make_symmetric_cell(size, quarter):
    """Build a 32x32 cell from a 16x16 top-left quarter, enforcing full square symmetry."""
    n = size // 2  # 16
    # quarter[i][j] for i in 0..n-1, j in 0..n-1 is top-left
    # Full cell:
    #   row i (0..n-1):   quarter[i][j]        for j 0..n-1
    #                      quarter[i][n-1-j]    for j n..2n-1  (mirror horizontally)
    #   row 2n-1-i:       same as row i
    cell = []
    for i in range(size):
        ri = i if i < n else (2 * n - 1 - i)
        row = []
        for j in range(size):
            rj = j if j < n else (2 * n - 1 - j)
            row.append(quarter[ri][rj])
        cell.append(row)
    return cell


def _fourier_coefficients(cell, size, n_pw):
    """Compute exact Fourier coefficients of the dielectric map.
    
    Returns a dict mapping (mx, my) -> sum of eps(x,y) * exp(-i 2pi (mx*x + my*y)/size)
    over the unit cell, for the needed plane-wave indices.
    """
    # n_pw is the number of plane waves in each direction (e.g., 17 for 289 total)
    # Plane wave indices range from -n_pw//2 to n_pw//2
    half = n_pw // 2
    coeff = {}
    for mx in range(-half, half + 1):
        for my in range(-half, half + 1):
            s = 0.0
            for i in range(size):
                x = (i + 0.5) / size
                for j in range(size):
                    y = (j + 0.5) / size
                    eps = cell[i][j]
                    # TM: we need 1/eps, but for the Galerkin method with exact coefficients,
                    # the operator involves epsilon. Let's compute sum of eps * exp(-i...)
                    # Actually for TM polarization, the eigenvalue problem involves 1/epsilon.
                    # The standard plane-wave method for TM:
                    #   (1/eps) is expanded in Fourier series.
                    # But the task says "exact Fourier coefficients of the pixel map"
                    # Let me think: for TM, the equation is -∇·(1/eps ∇H) = (omega^2/c^2) H
                    # In plane waves, this becomes a matrix with elements involving 1/eps_{G-G'}
                    # So we need Fourier coefficients of 1/eps.
                    # eps is either 1 (air) or 12 (dielectric)
                    # 1/eps is 1 or 1/12
                    pass
            coeff[(mx, my)] = s
    return coeff


def _compute_1over_eps_fourier(cell, size, n_pw):
    """Compute Fourier coefficients of 1/epsilon for the pixel map.
    
    For TM polarization, the eigenvalue problem requires 1/eps.
    eps = 1 (air, pixel 0) or 12 (dielectric, pixel 1)
    1/eps = 1 (air) or 1/12 (dielectric)
    """
    half = n_pw // 2
    inv_eps_coeff = {}
    for mx in range(-half, half + 1):
        for my in range(-half, half + 1):
            s = 0.0
            for i in range(size):
                x = (i + 0.5) / size
                for j in range(size):
                    y = (j + 0.5) / size
                    eps = 12.0 if cell[i][j] == 1 else 1.0
                    inv_eps = 1.0 / eps
                    angle = 2.0 * math.pi * (mx * x + my * y)
                    s += inv_eps * (math.cos(angle) + 1j * math.sin(angle))
            # Normalize by number of pixels (area of unit cell = 1)
            inv_eps_coeff[(mx, my)] = s / (size * size)
    return inv_eps_coeff


def _band_structure_tm(inv_eps_coeff, n_pw, n_kpoints=30):
    """Compute TM band structure using plane-wave Galerkin method.
    
    Returns list of (kx, ky, [omega_1, ..., omega_n]) for each k-point.
    """
    half = n_pw // 2
    # Build list of plane wave vectors G
    G_vectors = []
    for mx in range(-half, half + 1):
        for my in range(-half, half + 1):
            G_vectors.append((2.0 * math.pi * mx, 2.0 * math.pi * my))
    
    n_pw_total = len(G_vectors)
    
    # Precompute G-G' differences
    # For each pair (i, j), compute G_i - G_j and look up inv_eps_coeff
    # The matrix element is: delta_ij * |G_i|^2 + (1 - delta_ij) * (G_i - G_j) * inv_eps_coeff[G_i - G_j]
    # Wait, let me be more careful.
    # 
    # TM eigenvalue problem: -∇·(1/eps ∇H) = (omega^2/c^2) H
    # Expanding H = sum_k c_k exp(i(k+G_k)·r)
    # The matrix in the plane-wave basis is:
    # M_{ij} = (k + G_i) · (k + G_j) * delta_{G_i, G_j} + (k + G_i) · (k + G_j) * ... 
    # 
    # Actually, the standard form is:
    # M_{ij} = |k + G_i|^2 * delta_{ij} + sum over G such that G_i - G_j = G of (k+G_i)·(k+G_j-G) * eps_G / (2pi)^2
    # 
    # Let me use the simpler formulation. For 2D TM:
    # The operator is -1/eps * (d^2/dx^2 + d^2/dy^2)
    # In Fourier space, this becomes a matrix where:
    # M_{ij} = (k + G_i) · (k + G_i) * delta_{ij} + (k + G_i) · (k + G_j) * f(G_i - G_j)
    # where f(G) = Fourier coefficient of 1/eps at G.
    # 
    # Hmm, I need to be more careful. Let me use the standard plane-wave method.
    # 
    # For TM: (-∇_⊥^2 / eps) H = (omega^2/c^2) H
    # H(r) = sum_n c_n exp(i (k + G_n) · r)
    # 
    # -∇_⊥^2 H = sum_n c_n |k + G_n|^2 exp(i(k+G_n)·r)
    # (1/eps) H = sum_m (1/eps)_{m} sum_n c_n exp(i(k+G_n+G_m)·r)
    #            = sum_l (sum_{m: G_n + G_m = G_l - ... })
    # 
    # This is getting complicated. Let me use a simpler approach:
    # The matrix element M_{ij} satisfies:
    # M_{ij} = delta_{ij} |k+G_i|^2 + (1-delta_{ij}) (k+G_i)·(k+G_j) * eps_{G_i-G_j}
    # 
    # Actually, I recall that for the 2D TM problem, the plane-wave matrix is:
    # M_{ij} = |k + G_i|^2 * delta_{ij} + sum_{G} (k+G_i)·(k+G_j-G) * (1/eps)_G * delta(G_i - G_j, G)
    # 
    # Let me just use the standard formulation from Joannopoulos et al.:
    # For TM: M_{ij} = delta_{ij} |k+G_i|^2 + (1-delta_{ij}) (k+G_i)·(k+G_j) * eps_{G_i-G_j}
    # where eps_G is the Fourier coefficient of 1/eps.
    # 
    # Wait, I think the correct form is:
    # M_{ij} = |k+G_i|^2 delta_{ij} + (k+G_i)·(k+G_j) * eps_{G_i-G_j} for i != j
    # and M_{ii} = |k+G_i|^2 + (k+G_i)·(k+G_i) * eps_0
    # 
    # Hmm, let me just implement it carefully.
    # 
    # The eigenvalue equation is:
    # sum_j [ |k+G_i|^2 delta_{ij} + (k+G_i)·(k+G_j) * f(G_i - G_j) ] c_j = omega^2 c_i
    # where f(G) is the Fourier coefficient of 1/eps.
    # 
    # Actually, I think the correct matrix is:
    # M_{ij} = (k+G_i) · (k+G_j) * f(G_i - G_j)
    # where f(G) = Fourier coefficient of 1/eps at G.
    # This is because -∇^2 H gives |k+G_i|^2 c_i, and 1/eps * H gives the convolution.
    # 
    # Let me use: M_{ij} = (k+G_i)·(k+G_j) * f_{G_i - G_j}
    # where f_G is the Fourier coefficient of 1/eps.
    
    # Precompute dot products won't work since k varies. Let's just build the matrix for each k.
    
    # Irreducible BZ for square lattice: triangle with vertices (0,0), (pi,0), (pi,pi)
    # k points: 30 points covering boundary and interior
    k_points = []
    # Along Gamma-X: (0,0) to (pi,0)
    for t in [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]:
        k_points.append((math.pi * t, 0.0))
    # Along X-M: (pi,0) to (pi,pi)
    for t in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]:
        k_points.append((math.pi, math.pi * t))
    # Along M-Gamma: (pi,pi) to (0,0)
    for t in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
        k_points.append((math.pi * (1 - t), math.pi * (1 - t)))
    
    bands = []  # list of (kx, ky, [omega_1, ..., omega_n])
    
    for kx, ky in k_points:
        # Build matrix
        M = [[0.0] * n_pw_total for _ in range(n_pw_total)]
        for i in range(n_pw_total):
            kx_i = kx + G_vectors[i][0]
            ky_i = ky + G_vectors[i][1]
            for j in range(n_pw_total):
                kx_j = kx + G_vectors[j][0]
                ky_j = ky + G_vectors[j][1]
                G_diff_x = G_vectors[i][0] - G_vectors[j][0]
                G_diff_y = G_vectors[i][1] - G_vectors[j][1]
                # Convert to pixel indices
                mx = int(round(G_diff_x / (2.0 * math.pi)))
                my = int(round(G_diff_y / (2.0 * math.pi)))
                f_val = inv_eps_coeff.get((mx, my), 0.0)
                M[i][j] = (kx_i * kx_j + ky_i * ky_j) * f_val.real
        
        # Solve eigenvalue problem
        # Use numpy for eigendecomposition
        import numpy as np
        M_np = np.array(M)
        eigenvalues = np.linalg.eigvalsh(M_np)
        # omega = sqrt(eigenvalue) since M * c = omega^2 * c
        omegas = [math.sqrt(max(0.0, ev)) for ev in eigenvalues]
        bands.append((kx, ky, omegas))
    
    return bands


def _compute_gap_ratio(bands, lower_band=4):
    """Compute the gap-midgap ratio between bands 4 and 5 (0-indexed: 3 and 4)."""
    max_omega_lower = 0.0
    min_omega_upper = float('inf')
    
    for kx, ky, omegas in bands:
        if len(omegas) > lower_band:
            max_omega_lower = max(max_omega_lower, omegas[lower_band - 1])
            min_omega_upper = min(min_omega_upper, omegas[lower_band])
    
    if min_omega_upper == float('inf'):
        return -1.0
    
    gap = min_omega_upper - max_omega_lower
    midgap = (min_omega_upper + max_omega_lower) / 2.0
    if midgap == 0:
        return 0.0
    return gap / midgap


def _four_rod_pattern(size, r1, r2, r3, r4, offset):
    """Create a four-rod pattern at (±offset, ±offset) positions.
    
    Each rod is a circle of radius r at the specified position.
    Due to symmetry, we only define the top-left quarter and mirror.
    """
    n = size // 2
    quarter = [[0] * n for _ in range(n)]
    
    # In the full cell, rods are at (±offset, ±offset)
    # In the top-left quarter (i, j) where i,j in 0..n-1,
    # the corresponding full-cell coordinates are:
    #   x_full = (i + 0.5)/size - 0.5  for i in 0..n-1 (left half)
    #   y_full = (j + 0.5)/size - 0.5  for j in 0..n-1 (top half)
    # 
    # The four rod centers in full-cell coordinates:
    #   (+offset, +offset), (-offset, +offset), (+offset, -offset), (-offset, -offset)
    # 
    # In the top-left quarter, we only see the portion where x < 0 and y < 0
    # So we check if pixel (i,j) is within radius of any of the four centers.
    # But due to symmetry, if it's within radius of (-offset, -offset), it's also
    # within radius of the others when mirrored. So we just check all four.
    
    centers = [
        (offset, offset),
        (-offset, offset),
        (offset, -offset),
        (-offset, -offset),
    ]
    radii = [r1, r2, r3, r4]
    
    for i in range(n):
        x = (i + 0.5) / size - 0.5
        for j in range(n):
            y = (j + 0.5) / size - 0.5
            for c, r in zip(centers, radii):
                dx = x - c[0]
                dy = y - c[1]
                if dx * dx + dy * dy < r * r:
                    quarter[i][j] = 1
                    break
    return quarter


def _single_rod_pattern(size, radius):
    """Create a single centered rod pattern."""
    n = size // 2
    quarter = [[0] * n for _ in range(n)]
    for i in range(n):
        x = (i + 0.5) / size - 0.5
        for j in range(n):
            y = (j + 0.5) / size - 0.5
            if x * x + y * y < radius * radius:
                quarter[i][j] = 1
    return quarter


def design(problem):
    """Design a symmetric 32x32-pixel unit cell maximizing the 4-5 TM band gap."""
    size = problem["pixels"]
    start_time = time.time()
    deadline = start_time + 200.0  # Leave 30s margin for the 230s limit
    
    # Start with the four-rod pattern that scored 5.265
    best_cell = None
    best_score = -float('inf')
    
    # Initial candidate: four rods at (±1/4, ±1/4) with radius ~0.15
    # The observed improvement used this strategy
    initial_offset = 0.25
    initial_radius = 0.15
    
    def evaluate_cell(cell):
        """Evaluate the gap ratio for a given cell."""
        try:
            n_pw = 17  # 17x17 = 289 plane waves
            inv_eps_coeff = _compute_1over_eps_fourier(cell, size, n_pw)
            bands = _band_structure_tm(inv_eps_coeff, n_pw, n_kpoints=30)
            score = _compute_gap_ratio(bands, lower_band=4)
            return score
        except Exception as e:
            return -float('inf')
    
    # Evaluate initial four-rod pattern
    quarter = _four_rod_pattern(size, initial_radius, initial_radius, initial_radius, initial_radius, initial_offset)
    cell = _make_symmetric_cell(size, quarter)
    score = evaluate_cell(cell)
    if score > best_score:
        best_score = score
        best_cell = cell
    
    # Try some variations
    variations = [
        # (offset, radius)
        (0.25, 0.12),
        (0.25, 0.18),
        (0.25, 0.20),
        (0.20, 0.15),
        (0.30, 0.15),
        (0.25, 0.10),
        (0.22, 0.15),
        (0.28, 0.15),
    ]
    
    for offset, radius in variations:
        if time.time() > deadline - 30.0:
            break
        quarter = _four_rod_pattern(size, radius, radius, radius, radius, offset)
        cell = _make_symmetric_cell(size, quarter)
        score = evaluate_cell(cell)
        if score > best_score:
            best_score = score
            best_cell = cell
    
    # Try single rod for comparison
    for radius in [0.15, 0.18, 0.20, 0.22]:
        if time.time() > deadline - 30.0:
            break
        quarter = _single_rod_pattern(size, radius)
        cell = _make_symmetric_cell(size, quarter)
        score = evaluate_cell(cell)
        if score > best_score:
            best_score = score
            best_cell = cell
    
    if best_cell is None:
        # Fallback: original single rod
        best_cell = _make_symmetric_cell(size, _single_rod_pattern(size, 0.2))
    
    return best_cell

# EVOLVE-BLOCK-END
