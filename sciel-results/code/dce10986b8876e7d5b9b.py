import numpy as np
import time
import math
from scipy.linalg import eig

def design(problem):
    start_time = time.time()
    deadline = start_time + 210  # Safety margin before 230s wall-clock limit
    
    size = problem["pixels"] # 32
    eps_low = problem["eps_low"] # 1
    eps_high = problem["eps_high"] # 12
    
    # Precompute G vectors for local solver
    # Using 11x11 G vectors for speed and reasonable accuracy
    G_range = 5
    G_vecs = []
    for nx in range(-G_range, G_range + 1):
        for ny in range(-G_range, G_range + 1):
            G_vecs.append((2 * np.pi * nx, 2 * np.pi * ny, nx, ny))
    N_G = len(G_vecs)
    
    # Precompute k-points in IBZ
    # IBZ: 0 <= kx <= ky <= pi
    # We sample a grid and filter
    k_points = []
    # Include high symmetry points
    k_points.append((0, 0))
    k_points.append((np.pi, 0))
    k_points.append((np.pi, np.pi))
    k_points.append((np.pi/2, np.pi/2))
    
    # Grid points
    num_k = 6
    for i in range(num_k):
        for j in range(num_k):
            kx = i / (num_k - 1) * np.pi
            ky = j / (num_k - 1) * np.pi
            if kx <= ky:
                k_points.append((kx, ky))
    
    # Remove duplicates approximately
    unique_k = []
    for kp in k_points:
        is_dup = False
        for ukp in unique_k:
            if np.allclose(kp, ukp, atol=1e-5):
                is_dup = True
                break
        if not is_dup:
            unique_k.append(kp)
    k_points = unique_k

    def get_fourier_coeffs(cell):
        """
        Compute Fourier coefficients of epsilon(r) for the given G vectors.
        cell is a 32x32 list of lists of 0/1.
        """
        # Convert to numpy array for faster access
        cell_arr = np.array(cell, dtype=float)
        eps_vals = eps_low + (eps_high - eps_low) * cell_arr
        
        # We need eps_G = integral eps(r) exp(-i G.r) dr
        # For pixel map, this is sum over pixels.
        # To speed up, we can precompute the sinc terms for each G component.
        # But since cell changes, we just sum.
        # Optimization: Use matrix multiplication?
        # eps_G = sum_{i,j} eps_{ij} * exp(-i Gx (i+0.5)/size) * (2 sin(Gx/2size)/Gx) * exp(-i Gy (j+0.5)/size) * (2 sin(Gy/2size)/Gy)
        # Let X_G = exp(-i Gx (i+0.5)/size) * sinc_x_G
        # Let Y_G = exp(-i Gy (j+0.5)/size) * sinc_y_G
        # Then eps_G = sum_i sum_j eps_{ij} X_G[i] Y_G[j]
        # This is essentially a 2D DFT-like operation but with sinc weights.
        
        coeffs = np.zeros(N_G, dtype=complex)
        
        # Precompute weights for each G vector
        # weight_x[nx][i] = exp(-i 2pi nx (i+0.5)/size) * (2 sin(2pi nx / 2size) / (2pi nx)) if nx!=0 else 1/size
        # Actually, the integral of exp(-i G x) from i/size to (i+1)/size is:
        # [exp(-i G (i+1)/size) - exp(-i G i/size)] / (-i G)
        # = exp(-i G (i+0.5)/size) * 2 sin(G/2size) / G
        # If G=0, it is 1/size.
        
        # We can compute this on the fly for each G, or precompute for all nx, ny.
        # Since N_G is small (121), and size is 32, it's fast.
        
        for idx, (Gx, Gy, nx, ny) in enumerate(G_vecs):
            # Compute weights for x and y
            if nx == 0:
                wx = np.ones(size) / size
            else:
                # i from 0 to size-1
                centers = (np.arange(size) + 0.5) / size
                # exp(-i Gx centers) * 2 sin(Gx/2size) / Gx
                factor = 2 * np.sin(Gx / (2 * size)) / Gx
                wx = np.exp(-1j * Gx * centers) * factor
                
            if ny == 0:
                wy = np.ones(size) / size
            else:
                centers = (np.arange(size) + 0.5) / size
                factor = 2 * np.sin(Gy / (2 * size)) / Gy
                wy = np.exp(-1j * Gy * centers) * factor
            
            # eps_G = sum_{i,j} eps_{ij} wx[i] wy[j]
            # This is (wx^T * eps * wy) if eps is matrix.
            # Using numpy: np.sum(eps_vals * np.outer(wx, wy))
            coeffs[idx] = np.sum(eps_vals * np.outer(wx, wy))
            
        return coeffs

    def compute_gap(cell):
        """
        Compute the gap between band 4 and 5.
        Returns the gap-midgap ratio.
        """
        coeffs = get_fourier_coeffs(cell)
        
        max_band4 = 0.0
        min_band5 = float('inf')
        
        for kx, ky in k_points:
            # Build D matrix (diagonal)
            D_vals = np.zeros(N_G)
            for idx, (Gx, Gy, nx, ny) in enumerate(G_vecs):
                kx_g = kx + Gx
                ky_g = ky + Gy
                D_vals[idx] = kx_g * kx_g + ky_g * ky_g
            
            # Build H matrix (Hermitian)
            # H_{G, G'} = eps_{G - G'}
            # We have coeffs for G in range -5..5.
            # G - G' can be in range -10..10.
            # We only have coeffs for -5..5.
            # So if |dx| > 5 or |dy| > 5, coeff is 0.
            
            H = np.zeros((N_G, N_G), dtype=complex)
            
            # Map (nx, ny) to index
            idx_map = {}
            for idx, (Gx, Gy, nx, ny) in enumerate(G_vecs):
                idx_map[(nx, ny)] = idx
            
            for i in range(N_G):
                nx_i, ny_i = G_vecs[i][2], G_vecs[i][3]
                for j in range(N_G):
                    nx_j, ny_j = G_vecs[j][2], G_vecs[j][3]
                    dx = nx_i - nx_j
                    dy = ny_i - ny_j
                    key = (dx, dy)
                    if key in idx_map:
                        H[i, j] = coeffs[idx_map[key]]
                    else:
                        H[i, j] = 0.0
            
            # Solve H c = omega^2 D c
            # D^{-1/2} H D^{-1/2} c' = omega^2 c'
            # D_inv_sqrt = 1 / sqrt(D_vals)
            # Avoid division by zero (should not happen for k in IBZ and G != -k)
            # But if k+G = 0, D=0. This happens if k = -G.
            # In IBZ, k >= 0. G can be negative.
            # If k=0, G=0 => D=0.
            # We handle this by adding a small epsilon to D or skipping.
            # Actually, for k=0, G=0, the equation is 0*c_0 = omega^2 * 0 * c_0?
            # No, D is |k+G|^2. For k=0, G=0, D=0.
            # The eigenvalue problem is singular.
            # Standard trick: set D=1 for the zero mode? Or just skip k=0?
            # At k=0, the lowest band is constant, omega=0?
            # For TM, omega^2 = c^2 / (1/eps * |k|^2)?
            # Let's just add a tiny epsilon to D_vals to avoid division by zero.
            D_vals_safe = D_vals + 1e-12
            
            D_inv_sqrt = 1.0 / np.sqrt(D_vals_safe)
            H_trans = (H * D_inv_sqrt[:, None]) * D_inv_sqrt[None, :]
            
            # H_trans should be Hermitian
            # H_trans = (H_trans + H_trans.conj().T) / 2
            
            try:
                # Use eigvalsh for Hermitian matrices
                omega2_vals = np.linalg.eigvalsh(H_trans)
                omega_vals = np.sqrt(np.real(omega2_vals))
                omega_vals = np.sort(omega_vals)
                
                # We need bands 4 and 5 (indices 3 and 4)
                # Ensure we have enough bands
                if len(omega_vals) >= 5:
                    b4 = omega_vals[3]
                    b5 = omega_vals[4]
                    max_band4 = max(max_band4, b4)
                    min_band5 = min(min_band5, b5)
            except Exception:
                continue
        
        if min_band5 == float('inf') or max_band4 == 0:
            return -1.0
            
        gap = min_band5 - max_band4
        midgap = (min_band5 + max_band4) / 2.0
        
        if midgap == 0:
            return -1.0
            
        return gap / midgap

    def make_cell_from_shape(shape_type, param1, param2=None):
        """
        Generate a symmetric cell.
        1 = dielectric, 0 = air.
        """
        cell = [[0]*size for _ in range(size)]
        center = (size - 1) / 2.0
        
        for i in range(size):
            for j in range(size):
                x = (i - center) / (size/2.0)
                y = (j - center) / (size/2.0)
                # x, y in [-1, 1]
                
                is_dielectric = False
                
                if shape_type == 'hole_circle':
                    # Dielectric everywhere EXCEPT circle in center
                    r = param1
                    if x*x + y*y > r*r:
                        is_dielectric = True
                elif shape_type == 'hole_square':
                    # Dielectric everywhere EXCEPT square in center
                    side = param1
                    if abs(x) > side/2 or abs(y) > side/2:
                        is_dielectric = True
                elif shape_type == 'hole_super_ellipse':
                    r = param1
                    n = param2
                    # Super-ellipse: |x/r|^n + |y/r|^n < 1 is inside
                    if abs(x/r)**n + abs(y/r)**n >= 1.0:
                        is_dielectric = True
                elif shape_type == 'post_circle':
                    r = param1
                    if x*x + y*y < r*r:
                        is_dielectric = True
                elif shape_type == 'post_square':
                    side = param1
                    if abs(x) < side/2 and abs(y) < side/2:
                        is_dielectric = True

                cell[i][j] = 1 if is_dielectric else 0
        
        # Enforce symmetry strictly
        for i in range(size):
            for j in range(size):
                # Symmetries: (i,j), (size-1-i, j), (i, size-1-j), (size-1-i, size-1-j)
                # Also transpose: (j,i) etc.
                # Simplest: average all symmetric pixels.
                # For 4-fold, there are 4 orbits.
                # Orbit of (i,j): {(i,j), (31-i,j), (i,31-j), (31-i,31-j)}
                # And transposed: {(j,i), (31-j,i), (j,31-i), (31-j,31-i)}
                # If we generate symmetrically, these should be equal.
                # To be safe, we can just ensure the array is symmetric.
                pass
        
        # Explicit symmetry enforcement
        for i in range(size):
            for j in range(size):
                val = cell[i][j]
                cell[31-i][j] = val
                cell[i][31-j] = val
                cell[31-i][31-j] = val
                # Transpose symmetry
                cell[j][i] = val
                cell[j][31-i] = val
                cell[31-j][i] = val
                cell[31-j][31-i] = val
                
        return cell

    # Candidate seeds
    # Hole structures are usually better for TM gaps in square lattice
    candidates = [
        ('hole_circle', 0.3, None),
        ('hole_circle', 0.4, None),
        ('hole_circle', 0.5, None),
        ('hole_square', 0.4, None),
        ('hole_square', 0.5, None),
        ('hole_square', 0.6, None),
        ('hole_super_ellipse', 0.4, 4.0),
        ('hole_super_ellipse', 0.5, 4.0),
        ('post_circle', 0.4, None),
        ('post_square', 0.4, None),
    ]

    best_score = -1.0
    best_cell = None

    # Evaluate candidates
    for shape, p1, p2 in candidates:
        if time.time() > deadline - 30: # Leave time for optimization
            break
            
        cell = make_cell_from_shape(shape, p1, p2)
        score = compute_gap(cell)
        
        if score > best_score:
            best_score = score
            best_cell = cell

    if best_cell is None:
        best_cell = make_cell_from_shape('hole_circle', 0.4, None)
        best_score = compute_gap(best_cell)

    # Local Optimization (Hill Climbing)
    # Flip pixels that improve the score
    # Respect symmetry: flipping (i,j) flips all symmetric pixels.
    
    # Identify unique orbits
    visited = set()
    orbits = []
    for i in range(size):
        for j in range(size):
            if (i, j) in visited:
                continue
            
            # Generate orbit
            orbit = [(i, j), (31-i, j), (i, 31-j), (31-i, 31-j),
                     (j, i), (j, 31-i), (31-j, i), (31-j, 31-i)]
            # Remove duplicates
            unique_orbit = list(set(orbit))
            # Mark as visited
            for uo in unique_orbit:
                visited.add(uo)
            orbits.append(unique_orbit)

    # Sort orbits? Maybe by frequency? Or just iterate.
    # To speed up, we can try flipping one orbit at a time.
    # There are 32*32/8 = 128 orbits.
    
    improved = True
    max_iterations = 50
    
    while improved and time.time() < deadline - 10:
        improved = False
        for orbit in orbits:
            if time.time() > deadline - 10:
                break
            
            # Try flipping this orbit
            # Save current state
            old_vals = [best_cell[i][j] for i, j in orbit]
            
            # Flip
            for i, j in orbit:
                best_cell[i][j] = 1 - best_cell[i][j]
            
            new_score = compute_gap(best_cell)
            
            if new_score > best_score:
                best_score = new_score
                improved = True
            else:
                # Revert
                for (i, j), val in zip(orbit, old_vals):
                    best_cell[i][j] = val

    return best_cell

