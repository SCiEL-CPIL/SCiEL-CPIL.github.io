"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.linalg import eigh

def _compute_tm_bands(cell, n_pw=17, n_bands=6, k_points=12):
    """Compute TM bands using plane-wave expansion.
    
    For TM: the eigenvalue problem is (K - K0)^2 * epsilon^{-1} * A = omega^2 * A
    where K is the wavevector grid.
    """
    size = len(cell)
    eps = np.array(cell, dtype=float)
    eps[eps == 0] = 1.0
    eps[eps == 1] = 12.0
    
    # Fourier coefficients of 1/eps
    inv_eps = 1.0 / eps
    # Compute Fourier coefficients
    G_max = n_pw
    G_vals = []
    for gx in range(-G_max, G_max + 1):
        for gy in range(-G_max, G_max + 1):
            if gx * gx + gy * gy <= G_max * G_max:
                G_vals.append((gx, gy))
    n_G = len(G_vals)
    
    # Compute Fourier coefficients of inv_eps
    inv_eps_2d = inv_eps.astype(float)
    fourier_inv_eps = np.zeros((n_G, n_G), dtype=complex)
    for i, (gx_i, gy_i) in enumerate(G_vals):
        for j, (gx_j, gy_j) in enumerate(G_vals):
            dgx = gx_i - gx_j
            dgy = gy_i - gy_j
            if dgx == 0 and dgy == 0:
                fourier_inv_eps[i, j] = np.mean(inv_eps_2d)
            else:
                # Fourier coefficient
                phase = np.exp(-1j * 2 * np.pi * (dgx * np.arange(size)[:, None] / size + 
                                                   dgy * np.arange(size)[None, :] / size))
                fourier_inv_eps[i, j] = np.sum(inv_eps_2d * phase) / (size * size)
    
    # Irreducible BZ points for square lattice
    # Irreducible wedge: kx >= ky >= 0, kx + ky <= 1 (in units of 2*pi/a)
    k_points_list = []
    # Boundary points
    k_points_list.append((0.0, 0.0))      # Gamma
    k_points_list.append((0.5, 0.0))      # X
    k_points_list.append((0.5, 0.5))      # M
    k_points_list.append((0.25, 0.0))
    k_points_list.append((0.5, 0.25))
    k_points_list.append((0.25, 0.25))
    k_points_list.append((0.375, 0.0))
    k_points_list.append((0.375, 0.375))
    k_points_list.append((0.125, 0.125))
    k_points_list.append((0.5, 0.125))
    k_points_list.append((0.125, 0.0))
    k_points_list.append((0.4375, 0.0625))
    
    max_omega_4 = 0.0
    min_omega_5 = float('inf')
    
    for (kx, ky) in k_points_list:
        Kx = kx + np.array([g[0] for g in G_vals])
        Ky = ky + np.array([g[1] for g in G_vals])
        
        # Build the matrix: (K - K')^2 * inv_eps_{K-K'}
        # For TM: A(K) satisfies sum_K' (K-K')^2 * inv_eps(K-K') * A(K') = omega^2 * A(K)
        # This is not symmetric, so we need to handle it carefully
        # Actually, for TM the operator is (k+G)^2 * eps^{-1}(G-G') which is not Hermitian
        # We need to use a different formulation
        
        # Standard TM formulation:
        # (k+G)^2 / eps(G-G') * A(G') = omega^2 * A(G)
        # This gives a non-Hermitian matrix. Let's use the standard approach.
        
        # Build matrix M where M[i,j] = (K_i)^2 * inv_eps[G_i - G_j]
        # But we need to find the index mapping
        G_dict = {}
        for idx, (gx, gy) in enumerate(G_vals):
            G_dict[(gx, gy)] = idx
        
        M = np.zeros((n_G, n_G), dtype=complex)
        for i in range(n_G):
            Ki2 = Kx[i] * Kx[i] + Ky[i] * Ky[i]
            for j in range(n_G):
                dgx = G_vals[i][0] - G_vals[j][0]
                dgy = G_vals[i][1] - G_vals[j][1]
                if (dgx, dgy) in G_dict:
                    M[i, j] = Ki2 * fourier_inv_eps[G_dict[(dgx, dgy)], j]
                # Note: fourier_inv_eps is indexed by (i,j) where i is the output G, j is input
                # Actually let me re-derive this properly
        
        # Let me redo the matrix construction more carefully
        # The TM eigenvalue equation in plane waves:
        # sum_{G'} [(k+G)^2] * (1/eps)_{G-G'} * A_{G'} = omega^2 * A_G
        # where (1/eps)_{G-G'} is the Fourier coefficient of 1/eps at (G-G')
        
        # So M[G, G'] = (k+G)^2 * (1/eps)_{G-G'}
        # This matrix is not symmetric in general
        
        M = np.zeros((n_G, n_G), dtype=complex)
        for i in range(n_G):
            Ki2 = Kx[i] * Kx[i] + Ky[i] * Ky[i]
            for j in range(n_G):
                dgx = G_vals[i][0] - G_vals[j][0]
                dgy = G_vals[i][1] - G_vals[j][1]
                # Find the index for (dgx, dgy) in our G list
                # But (dgx,dgy) might not be in our list. We need the Fourier coefficient
                # at that difference. Since we computed fourier_inv_eps for all pairs in G_vals,
                # we can look it up if (dgx,dgy) is in G_dict
                if (dgx, dgy) in G_dict:
                    idx_dg = G_dict[(dgx, dgy)]
                    # fourier_inv_eps[idx_dg, j] gives the Fourier coeff at (G_vals[idx_dg] - G_vals[j])
                    # But we want (1/eps)_{G_i - G_j} which is fourier_inv_eps[i, j]
                    # Wait, let me re-examine.
                    # fourier_inv_eps[i, j] = FT of inv_eps at (G_vals[i] - G_vals[j])
                    # So M[i, j] = Kx[i]^2 + Ky[i]^2 times fourier_inv_eps[i, j]
                    M[i, j] = Ki2 * fourier_inv_eps[i, j]
                else:
                    # If the difference is not in our G list, the Fourier coefficient is 0
                    # (for a finite grid, this should be handled by the sum)
                    M[i, j] = 0.0
        
        # Solve the generalized eigenvalue problem. Since M is not Hermitian,
        # we use eigvals
        try:
            eigvals = np.linalg.eigvals(M)
            real_eigvals = np.sort(np.real(eigvals))
            # Take the smallest n_bands positive eigenvalues
            positive = real_eigvals[real_eigvals > 0]
            if len(positive) >= n_bands:
                omega4 = positive[3]  # 4th band (0-indexed: 3)
                omega5 = positive[4]  # 5th band (0-indexed: 4)
                max_omega_4 = max(max_omega_4, omega4)
                min_omega_5 = min(min_omega_5, omega5)
        except:
            pass
    
    if min_omega_5 == float('inf') or max_omega_4 == 0:
        return -1.0
    
    gap = min_omega_5 - max_omega_4
    midgap = (min_omega_5 + max_omega_4) / 2.0
    if midgap == 0:
        return -1.0
    return gap / midgap

def _make_cell_circle(radius_frac, center=(16, 16), size=32):
    """Create a cell with a central dielectric circle."""
    cell = np.zeros((size, size), dtype=int)
    cy, cx = center
    for i in range(size):
        for j in range(size):
            dx = (j + 0.5) / size - 0.5
            dy = (i + 0.5) / size - 0.5
            if dx * dx + dy * dy < radius_frac * radius_frac:
                cell[i, j] = 1
    return cell

def _make_cell_ring(r_inner, r_outer, size=32):
    """Create a cell with a dielectric ring."""
    cell = np.zeros((size, size), dtype=int)
    for i in range(size):
        for j in range(size):
            dx = (j + 0.5) / size - 0.5
            dy = (i + 0.5) / size - 0.5
            r2 = dx * dx + dy * dy
            if r_inner * r_inner < r2 < r_outer * r_outer:
                cell[i, j] = 1
    return cell

def _make_cell_cross(width_frac, length_frac, size=32):
    """Create a cross-shaped dielectric pattern."""
    cell = np.zeros((size, size), dtype=int)
    half_w = width_frac * size / 2
    half_l = length_frac * size / 2
    for i in range(size):
        for j in range(size):
            dx = abs(j + 0.5 - size / 2)
            dy = abs(i + 0.5 - size / 2)
            if (dx < half_w and dy < half_l) or (dy < half_w and dx < half_l):
                cell[i, j] = 1
    return cell

def _make_cell_square(size_frac, size=32):
    """Create a central dielectric square."""
    cell = np.zeros((size, size), dtype=int)
    half = size_frac * size / 2
    start = int(size / 2 - half)
    end = int(size / 2 + half)
    cell[start:end, start:end] = 1
    return cell

def _make_cell_hole_circle(radius_frac, size=32):
    """Dielectric background with a circular air hole."""
    cell = np.ones((size, size), dtype=int)
    for i in range(size):
        for j in range(size):
            dx = (j + 0.5) / size - 0.5
            dy = (i + 0.5) / size - 0.5
            if dx * dx + dy * dy < radius_frac * radius_frac:
                cell[i, j] = 0
    return cell

def _check_symmetry(cell):
    """Check if cell has required symmetries."""
    cell = np.array(cell)
    if not np.array_equal(cell, cell[::-1, :]):
        return False
    if not np.array_equal(cell, cell[:, ::-1]):
        return False
    if not np.array_equal(cell, cell.T):
        return False
    return True

def design(problem):
    """Design a symmetric 32x32 photonic crystal unit cell for maximum TM band gap."""
    size = problem["pixels"]
    t_start = time.time()
    deadline = t_start + 110.0  # safety margin before 120s limit
    
    best_cell = None
    best_score = -1.0
    
    # Generate candidate geometries
    candidates = []
    
    # 1. Various circle radii (dielectric rod in air)
    for r in [0.15, 0.2, 0.25, 0.3, 0.35, 0.4]:
        candidates.append(_make_cell_circle(r, size=size))
    
    # 2. Dielectric background with air holes
    for r in [0.15, 0.2, 0.25, 0.3, 0.35]:
        candidates.append(_make_cell_hole_circle(r, size=size))
    
    # 3. Ring patterns
    for ri in [0.15, 0.2, 0.25]:
        for ro in [0.3, 0.35, 0.4, 0.45]:
            if ro > ri:
                candidates.append(_make_cell_ring(ri, ro, size=size))
    
    # 4. Cross patterns
    for w in [0.1, 0.15, 0.2]:
        for l in [0.3, 0.4, 0.45]:
            candidates.append(_make_cell_cross(w, l, size=size))
    
    # 5. Square patterns
    for s in [0.3, 0.4, 0.5, 0.6]:
        candidates.append(_make_cell_square(s, size=size))
    
    # Evaluate each candidate
    for cell in candidates:
        if time.time() > deadline - 5:
            break
        
        cell = np.array(cell, dtype=int)
        if not _check_symmetry(cell):
            continue
        
        try:
            score = _compute_tm_bands(cell, n_pw=10, n_bands=6, k_points=8)
            if score > best_score:
                best_score = score
                best_cell = cell.copy()
        except:
            pass
    
    # If no good candidate found, use the baseline
    if best_cell is None:
        best_cell = _make_cell_circle(0.2, size=size)
    
    # Convert to nested list
    return best_cell.tolist()
# EVOLVE-BLOCK-END
