import numpy as np
import time
import math

# EVOLVE-BLOCK-START
def design(problem):
    """
    Design a 32x32 unit cell for a square lattice photonic crystal.
    Uses a fast plane-wave expansion (PWE) solver to optimize the geometry
    for the TM band gap between bands 4 and 5.
    """
    size = problem["pixels"]
    eps_low = problem.get("eps_low", 1.0)
    eps_high = problem.get("eps_high", 12.0)
    
    # Deadline guard
    start_time = time.time()
    deadline = start_time + 200.0  # 200s safety margin
    
    # Fast PWE Solver Parameters
    # Use a smaller cutoff for speed during search
    pw_cutoff = 5  # Number of G-vector shells
    n_bands = 5    # Need bands 4 and 5
    
    # Define candidate geometries
    # Each candidate is a function that returns a 32x32 binary grid
    candidates = []
    
    # 1. Circular Dielectric Rods (varying radius)
    for r in [0.15, 0.2, 0.25, 0.3, 0.35]:
        candidates.append(('rod_circ', r))
        
    # 2. Square Dielectric Rods (varying half-side)
    for s in [0.15, 0.2, 0.25, 0.3, 0.35]:
        candidates.append(('rod_sq', s))
        
    # 3. Circular Air Holes (Dielectric Background)
    for r in [0.15, 0.2, 0.25, 0.3, 0.35]:
        candidates.append(('hole_circ', r))
        
    # 4. Square Air Holes
    for s in [0.15, 0.2, 0.25, 0.3, 0.35]:
        candidates.append(('hole_sq', s))
        
    # 5. Diamond Air Holes (45 deg rotated square)
    for s in [0.2, 0.3, 0.4]:
        candidates.append(('hole_diamond', s))

    best_score = -1.0
    best_cell = None
    
    # Pre-compute G-vectors for the fast solver
    G_vectors = []
    for nx in range(-pw_cutoff, pw_cutoff + 1):
        for ny in range(-pw_cutoff, pw_cutoff + 1):
            if nx*nx + ny*ny <= pw_cutoff*pw_cutoff:
                G_vectors.append((nx, ny))
    G_vectors = np.array(G_vectors)
    n_pw = len(G_vectors)
    
    # K-points in irreducible BZ (approximate)
    # (0,0), (0.5,0), (0.5,0.5), and some interior points
    k_points = [
        (0.0, 0.0),
        (0.5, 0.0),
        (0.5, 0.5),
        (0.25, 0.0),
        (0.25, 0.25),
        (0.4, 0.1),
        (0.4, 0.4),
        (0.1, 0.1)
    ]
    
    for i, (ctype, param) in enumerate(candidates):
        if time.time() > deadline - 10:
            break
            
        cell = _generate_cell(ctype, param, size, eps_low, eps_high)
        
        # Calculate Fourier coefficients of 1/epsilon for TM (or epsilon for TE)
        # We are using the TE-like matrix: H_mn = (k+Gm).(k+Gn) / eps_Gm-Gn
        # So we need 1/epsilon Fourier coeffs? No, the matrix uses 1/eps.
        # Wait, the matrix is H_mn = (k+Gm).(k+Gn) * (1/eps)_{Gm-Gn}
        # So we need the Fourier coefficients of 1/epsilon.
        inv_eps = _get_inv_eps_fourier(cell, G_vectors, eps_low, eps_high)
        
        score = _fast_score(k_points, G_vectors, inv_eps, n_bands)
        
        if score > best_score:
            best_score = score
            best_cell = cell
            
    if best_cell is None:
        # Fallback to seed
        best_cell = _generate_cell('rod_circ', 0.2, size, eps_low, eps_high)
        
    return best_cell

def _generate_cell(ctype, param, size, eps_low, eps_high):
    """Generates a symmetric 32x32 cell."""
    cell = np.zeros((size, size), dtype=int)
    half = size // 2
    
    for i in range(half):
        for j in range(half):
            x = (i + 0.5) / size - 0.5
            y = (j + 0.5) / size - 0.5
            
            is_dielectric = False
            if ctype == 'rod_circ':
                if x*x + y*y < param*param:
                    is_dielectric = True
            elif ctype == 'rod_sq':
                if abs(x) < param and abs(y) < param:
                    is_dielectric = True
            elif ctype == 'hole_circ':
                # Background is dielectric, hole is air
                if x*x + y*y < param*param:
                    is_dielectric = False
                else:
                    is_dielectric = True
            elif ctype == 'hole_sq':
                if abs(x) < param and abs(y) < param:
                    is_dielectric = False
                else:
                    is_dielectric = True
            elif ctype == 'hole_diamond':
                # |x| + |y| < param
                if abs(x) + abs(y) < param:
                    is_dielectric = False
                else:
                    is_dielectric = True
            
            val = 1 if is_dielectric else 0
            
            # Apply symmetry
            cell[i, j] = val
            cell[i, size - 1 - j] = val
            cell[size - 1 - i, j] = val
            cell[size - 1 - i, size - 1 - j] = val
            
    return cell.tolist()

def _get_inv_eps_fourier(cell, G_vectors, eps_low, eps_high):
    """
    Computes Fourier coefficients of 1/epsilon for the given cell.
    cell is a list of lists of 0/1.
    """
    size = len(cell)
    eps_grid = np.zeros((size, size))
    for i in range(size):
        for j in range(size):
            if cell[i][j] == 1:
                eps_grid[i, j] = eps_high
            else:
                eps_grid[i, j] = eps_low
                
    inv_eps_grid = 1.0 / eps_grid
    
    # Compute Fourier coefficients
    # f_G = (1/N^2) sum_{x,y} f(x,y) exp(-i G . r)
    # Since the grid is discrete, we sum over pixels.
    # G is in lattice units. r = (i/N, j/N).
    # exp(-i 2 pi (nx * i/N + ny * j/N))
    
    coeffs = np.zeros(len(G_vectors), dtype=complex)
    for idx, (nx, ny) in enumerate(G_vectors):
        s = 0.0
        for i in range(size):
            for j in range(size):
                angle = -2.0 * math.pi * (nx * i / size + ny * j / size)
                s += inv_eps_grid[i, j] * np.exp(1j * angle)
        coeffs[idx] = s / (size * size)
        
    return coeffs

def _fast_score(k_points, G_vectors, inv_eps_coeffs, n_bands):
    """
    Computes the approximate gap ratio using the TE-like PWE matrix.
    H_mn = (k+G_m) . (k+G_n) * inv_eps_coeffs[G_m - G_n]
    """
    n_pw = len(G_vectors)
    # Precompute index map for G_m - G_n
    # G_vectors are (nx, ny)
    g_map = {}
    for idx, g in enumerate(G_vectors):
        g_map[tuple(g)] = idx
        
    max_omega_4 = 0.0
    min_omega_5 = float('inf')
    
    for kx, ky in k_points:
        # Build matrix H
        H = np.zeros((n_pw, n_pw), dtype=complex)
        for m in range(n_pw):
            km = (kx + G_vectors[m][0], ky + G_vectors[m][1])
            for n in range(n_pw):
                kn = (kx + G_vectors[n][0], ky + G_vectors[n][1])
                # Dot product
                dot = km[0]*kn[0] + km[1]*kn[1]
                
                # G_m - G_n
                dgx = G_vectors[m][0] - G_vectors[n][0]
                dgy = G_vectors[m][1] - G_vectors[n][1]
                
                # Get coefficient for 1/eps
                coeff = 0.0
                # The Fourier coeff index corresponds to G_m - G_n
                # Note: Our coeffs are for exp(-i G r), so coeff[G] is <exp(i G r)>?
                # f(x) = sum G f_G exp(i G r)
                # 1/eps(x) = sum G c_G exp(i G r)
                # The matrix element involves <exp(i G_m r) 1/eps exp(i G_n r)>
                # = sum G c_G <exp(i (G_m + G_n + G) r)>
                # Non-zero if G_m + G_n + G = 0 => G = -G_m - G_n = G_n - G_m
                # So we need c_{G_n - G_m}
                key = (dgy, dgx) # Wait, G_m - G_n is (dgx, dgy). We need G_n - G_m which is (-dgx, -dgy).
                # Let's stick to G_m - G_n.
                # The term is c_{G_n - G_m}.
                key = (-dgx, -dgy)
                if key in g_map:
                    coeff = inv_eps_coeffs[g_map[key]]
                else:
                    # If G_n - G_m is outside the cutoff, coeff is 0 (approx)
                    coeff = 0.0
                    
                H[m, n] = dot * coeff
                
        # Solve eigenvalue problem
        try:
            # H is Hermitian
            eigvals = np.linalg.eigvalsh(H)
            # Filter small positive eigenvalues
            eigvals = eigvals[eigvals > 1e-6]
            
            if len(eigvals) >= n_bands:
                # omega^2 = eigvals
                # omega = sqrt(eigvals)
                w4 = np.sqrt(eigvals[3])
                w5 = np.sqrt(eigvals[4])
                
                if w4 > max_omega_4:
                    max_omega_4 = w4
                if w5 < min_omega_5:
                    min_omega_5 = w5
        except np.linalg.LinAlgError:
            continue
            
    if min_omega_5 == float('inf') or max_omega_4 == 0:
        return -1.0
        
    gap = min_omega_5 - max_omega_4
    midgap = (min_omega_5 + max_omega_4) / 2.0
    ratio = 2.0 * gap / (min_omega_5 + max_omega_4)
    
    return ratio

# EVOLVE-BLOCK-END
