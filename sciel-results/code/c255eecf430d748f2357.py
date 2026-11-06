"""Design a 32x32 photonic crystal unit cell to maximize TM band 4-5 gap."""
import numpy as np
import time
import math

def design(problem):
    """
    Design a symmetric 32x32 unit cell for a 2D square-lattice photonic crystal
    with epsilon=12 dielectric in air, maximizing the gap between TM bands 4 and 5.
    """
    size = problem["pixels"]  # 32
    eps_low = problem["eps_low"]   # 1
    eps_high = problem["eps_high"] # 12
    
    # Deadline guard
    t_start = time.time()
    DEADLINE = 210.0  # safety margin before 230s kill
    
    def time_ok():
        return time.time() - t_start < DEADLINE
    
    # Precompute pixel coordinates
    coords = np.array([(i + 0.5) / size - 0.5 for i in range(size)])
    XX, YY = np.meshgrid(coords, coords, indexing='ij')
    
    def enforce_symmetry(cell):
        """Enforce D4 symmetry by averaging over the symmetry group."""
        sym_cells = []
        sym_cells.append(cell)
        sym_cells.append(cell[::-1, :])
        sym_cells.append(cell[:, ::-1])
        sym_cells.append(cell[::-1, ::-1])
        sym_cells.append(cell.T)
        sym_cells.append(cell.T[::-1, :])
        sym_cells.append(cell.T[:, ::-1])
        sym_cells.append(cell.T[::-1, ::-1])
        avg = np.mean(sym_cells, axis=0)
        return (avg >= 0.5).astype(int)
    
    def make_cell_from_radii(radii_4x4):
        """
        Create a 32x32 cell from a 4x4 grid of radii for 8x8 sub-cells.
        Each sub-cell contains a circular dielectric disk centered at the sub-cell center.
        radii_4x4: 4x4 array of radii (in lattice units, 0 to 0.125 max for 8x8 sub-cell)
        The 4x4 superlattice is mapped to the 32x32 pixel grid.
        """
        cell = np.zeros((size, size), dtype=int)
        sub_size = 8  # each sub-cell is 8x8 pixels
        
        for si in range(4):
            for sj in range(4):
                # Center of this sub-cell in the 32x32 grid
                ci = si * sub_size + sub_size // 2  # center pixel row
                cj = sj * sub_size + sub_size // 2  # center pixel col
                
                # Radius in pixels
                r_pixels = radii_4x4[si, sj] * size  # convert from lattice units to pixels
                
                if r_pixels < 0.5:
                    continue
                
                # Define the region for this sub-cell
                i_start = si * sub_size
                i_end = (si + 1) * sub_size
                j_start = sj * sub_size
                j_end = (sj + 1) * sub_size
                
                # Create local coordinates for this sub-cell
                for di in range(sub_size):
                    for dj in range(sub_size):
                        pi = i_start + di
                        pj = j_start + dj
                        # Distance from sub-cell center (in pixel units)
                        dx = (pi + 0.5) - (ci + 0.5)
                        dy = (pj + 0.5) - (cj + 0.5)
                        dist = math.sqrt(dx*dx + dy*dy)
                        if dist <= r_pixels:
                            cell[pi, pj] = 1
        
        return enforce_symmetry(cell)
    
    def calculate_gap(cell, n_pw=7, n_kpoints=10):
        """Calculate the band 4-5 gap ratio using a plane-wave method."""
        eps = eps_low + (eps_high - eps_low) * cell.astype(float)
        inv_eps = 1.0 / eps
        
        N = size
        half = n_pw // 2
        
        # G-vectors
        G_list = []
        for gx in range(-half, half + 1):
            for gy in range(-half, half + 1):
                G_list.append((gx, gy))
        
        nG = len(G_list)
        
        # Precompute inv_eps Fourier coefficients using FFT for speed
        # Use the fact that we can compute all needed coefficients at once
        inv_eps_2d = inv_eps.astype(float)
        
        # Compute Fourier coefficients using FFT
        # The FFT gives us coefficients for G = (0, ..., N-1), but we need negative G too
        # For real-valued functions, F(-G) = conj(F(G))
        
        # Use np.fft.fft2 to compute all coefficients
        fft_2d = np.fft.fft2(inv_eps_2d)
        # The FFT output is ordered as [0, 1, ..., N/2-1, -N/2, ..., -1]
        # We need to map our G-vectors to the FFT indices
        
        def get_fft_coeff(gx, gy):
            """Get the Fourier coefficient for G=(gx,gy) from the FFT."""
            # FFT index for gx
            if gx >= 0:
                idx_x = gx % N
            else:
                idx_x = gx % N  # Python modulo handles negative correctly
            if gy >= 0:
                idx_y = gy % N
            else:
                idx_y = gy % N
            return fft_2d[idx_x, idx_y] / (N * N)
        
        inv_eps_F = {}
        for gx, gy in G_list:
            inv_eps_F[(gx, gy)] = get_fft_coeff(gx, gy)
        
        # Key k-points in the irreducible BZ
        k_points_list = [
            (0.0, 0.0),      # Gamma
            (0.5, 0.0),      # X
            (0.5, 0.5),      # M
            (0.25, 0.0),
            (0.5, 0.25),
            (0.25, 0.25),
            (0.35, 0.15),
            (0.45, 0.2),
            (0.3, 0.3),
            (0.4, 0.1),
        ]
        
        max_omega_4 = 0.0
        min_omega_5 = float('inf')
        
        for kx, ky in k_points_list:
            # Build eigenvalue matrix
            M = np.zeros((nG, nG), dtype=complex)
            
            for ig, (gx, gy) in enumerate(G_list):
                Gkx = gx - kx
                Gky = gy - ky
                for igp, (gxp, gyp) in enumerate(G_list):
                    Gpkx = gxp - kx
                    Gpky = gyp - ky
                    dGx = gx - gxp
                    dGy = gy - gyp
                    if (dGx, dGy) in inv_eps_F:
                        M[ig, igp] = (Gkx * Gpkx + Gky * Gpky) * inv_eps_F[(dGx, dGy)]
            
            # Solve eigenvalue problem
            eigvals = np.linalg.eigvalsh(M)
            eigvals = np.sort(eigvals.real)
            
            if len(eigvals) >= 5:
                omega_4 = np.sqrt(max(0, eigvals[3]))
                omega_5 = np.sqrt(max(0, eigvals[4]))
                max_omega_4 = max(max_omega_4, omega_4)
                min_omega_5 = min(min_omega_5, omega_5)
        
        if min_omega_5 == float('inf'):
            return -1.0
        
        gap_ratio = 2.0 * (min_omega_5 - max_omega_4) / (min_omega_5 + max_omega_4)
        return gap_ratio
    
    # Phase 1: Grid search over 4x4 radius patterns
    # Each radius is in [0, 0.12] (max radius for 8x8 sub-cell is 4 pixels = 4/32 = 0.125)
    # Due to symmetry, we only need to optimize 10 unique radii (out of 16)
    # The 4x4 grid has D4 symmetry, so unique positions are:
    # (0,0), (0,1), (0,2), (0,3), (1,1), (1,2), (1,3), (2,2), (2,3), (3,3)
    # But wait, we need the full 4x4 to be symmetric under the square group
    # For the 4x4 superlattice to be D4 symmetric:
    # r[si][sj] == r[3-si][sj] == r[si][3-sj] == r[3-si][3-sj] == r[sj][si] == r[sj][3-si] == r[3-sj][si] == r[3-sj][3-si]
    
    # Let's parameterize by the independent radii
    # Independent positions in 4x4 D4 orbit:
    # Orbit 1: (0,0) - center of cell corner
    # Orbit 2: (0,1) and (0,2) - these are related by mirror, so r[0][1] == r[0][2]
    # Actually, for 4x4 with indices 0,1,2,3:
    # Mirror about center: (i,j) -> (3-i, 3-j)
    # Transpose: (i,j) -> (j,i)
    # So orbits:
    # (0,0) <-> (3,3), (0,3) <-> (3,0)
    # (1,1) <-> (2,2), (1,2) <-> (2,1)
    # (0,1) <-> (3,2), (0,2) <-> (3,1)
    # (1,0) <-> (2,3), (1,3) <-> (2,0)
    # Plus transpose relations
    
    # Let me just define the 10 independent parameters and build the full 4x4
    
    def build_symmetric_4x4(r00, r01, r02, r03, r11, r12, r13, r22, r23, r33):
        """Build a D4-symmetric 4x4 array from 10 independent parameters."""
        r = np.zeros((4, 4))
        r[0, 0] = r00
        r[3, 3] = r00
        r[0, 3] = r03
        r[3, 0] = r03
        r[0, 1] = r01
        r[3, 2] = r01
        r[0, 2] = r02
        r[3, 1] = r02
        r[1, 0] = r01  # transpose of (0,1)
        r[2, 3] = r01
        r[1, 3] = r02  # transpose of (0,2)
        r[2, 0] = r02
        r[1, 1] = r11
        r[2, 2] = r11
        r[1, 2] = r12
        r[2, 1] = r12
        r[1, 3] = r13  # Wait, this conflicts with above
        # Let me redo this more carefully
        
        # The D4 group for 4x4 grid (indices 0-3):
        # Identity: (i,j)
        # 90° rot: (i,j) -> (j, 3-i)
        # 180° rot: (i,j) -> (3-i, 3-j)
        # 270° rot: (i,j) -> (3-j, i)
        # Mirror x: (i,j) -> (i, 3-j)
        # Mirror y: (i,j) -> (3-i, j)
        # Anti-diag: (i,j) -> (3-j, 3-i)
        # Main diag: (i,j) -> (j, i)
        
        # Let's just assign values to one representative from each orbit
        # and fill in the rest
        
        r = np.zeros((4, 4))
        
        # Orbit of (0,0): {(0,0), (3,3), (0,3), (3,0)}
        r[0, 0] = r00
        r[3, 3] = r00
        r[0, 3] = r03
        r[3, 0] = r03
        
        # Orbit of (1,1): {(1,1), (2,2), (1,2), (2,1)}
        r[1, 1] = r11
        r[2, 2] = r11
        r[1, 2] = r12
        r[2, 1] = r12
        
        # Orbit of (0,1): {(0,1), (3,2), (1,0), (2,3)}
        r[0, 1] = r01
        r[3, 2] = r01
        r[1, 0] = r01
        r[2, 3] = r01
        
        # Orbit of (0,2): {(0,2), (3,1), (2,0), (1,3)}
        r[0, 2] = r02
        r[3, 1] = r02
        r[2, 0] = r02
        r[1, 3] = r02
        
        # Orbit of (1,3): {(1,3), (2,0), (3,1), (0,2)} - already covered
        
        # Check: are all 16 positions covered?
        # (0,0), (0,1), (0,2), (0,3) - row 0 done
        # (1,0), (1,1), (1,2), (1,3) - row 1 done
        # (2,0), (2,1), (2,2), (2,3) - row 2 done
        # (3,0), (3,1), (3,2), (3,3) - row 3 done
        # Yes, all covered.
        
        return r
    
    # Phase 1: Coarse grid search
    best_score = -float('inf')
    best_radii = None
    best_cell = None
    
    # Grid values for each radius (in lattice units)
    # Max radius for 8x8 sub-cell: 4 pixels = 4/32 = 0.125
    # Use values from 0.0 to 0.12
    r_vals = [0.0, 0.03, 0.06, 0.09, 0.12]
    
    # For the coarse search, use a reduced grid to keep it fast
    # Try a few representative patterns
    
    # Pattern 1: Uniform radius
    # Pattern 2: Larger in center, smaller at corners
    # Pattern 3: Larger at corners, smaller in center
    # Pattern 4: Checkerboard-like
    
    candidate_patterns = []
    
    # Uniform
    for r in r_vals:
        candidate_patterns.append([r]*10)
    
    # Center-heavy: larger radii in inner positions
    for r_center in [0.08, 0.10, 0.12]:
        for r_outer in [0.02, 0.04, 0.06]:
            candidate_patterns.append([
                r_outer,  # r00
                r_center,  # r01
                r_center,  # r02
                r_outer,  # r03
                r_center,  # r11
                r_center,  # r12
                r_center,  # r13
                r_center,  # r22
                r_center,  # r23
                r_outer   # r33
            ])
    
    # Corner-heavy
    for r_corner in [0.08, 0.10, 0.12]:
        for r_inner in [0.02, 0.04, 0.06]:
            candidate_patterns.append([
                r_corner,  # r00
                r_inner,   # r01
                r_inner,   # r02
                r_corner,  # r03
                r_inner,   # r11
                r_inner,   # r12
                r_inner,   # r13
                r_inner,   # r22
                r_inner,   # r23
                r_corner   # r33
            ])
    
    # Alternating pattern
    for r1 in [0.06, 0.09, 0.12]:
        for r2 in [0.02, 0.04, 0.06]:
            candidate_patterns.append([
                r1, r2, r1, r2,
                r2, r1, r2, r1,
                r1, r2, r1, r2,
                r2, r1, r2, r1
            ])
    
    # Evaluate candidates
    for pattern in candidate_patterns:
        if not time_ok():
            break
        
        r00, r01, r02, r03, r11, r12, r13, r22, r23, r33 = pattern
        r_4x4 = build_symmetric_4x4(r00, r01, r02, r03, r11, r12, r13, r22, r23, r33)
        cell = make_cell_from_radii(r_4x4)
        score = calculate_gap(cell)
        
        if score > best_score:
            best_score = score
            best_radii = pattern.copy()
            best_cell = cell.copy()
    
    # Phase 2: Local optimization around the best candidate
    if time_ok() and best_radii is not None:
        # Try perturbing each parameter
        for idx in range(10):
            if not time_ok():
                break
            
            base = best_radii.copy()
            current_val = base[idx]
            
            # Try nearby values
            for delta in [-0.02, -0.01, 0.01, 0.02]:
                if not time_ok():
                    break
                
                new_val = current_val + delta
                if new_val < 0 or new_val > 0.13:
                    continue
                
                new_radii = base.copy()
                new_radii[idx] = new_val
                
                r00, r01, r02, r03, r11, r12, r13, r22, r23, r33 = new_radii
                r_4x4 = build_symmetric_4x4(r00, r01, r02, r03, r11, r12, r13, r22, r23, r33)
                cell = make_cell_from_radii(r_4x4)
                score = calculate_gap(cell)
                
                if score > best_score:
                    best_score = score
                    best_radii = new_radii.copy()
                    best_cell = cell.copy()
    
    # Phase 3: Try some additional diverse patterns if time remains
    if time_ok():
        additional_patterns = []
        
        # Gradient-like patterns
        for a in [0.04, 0.06, 0.08]:
            for b in [0.06, 0.08, 0.10]:
                for c in [0.08, 0.10, 0.12]:
                    additional_patterns.append([a, b, c, a, b, c, b, c, b, a])
        
        # Ring-like: larger in middle ring
        for r_in in [0.02, 0.04]:
            for r_mid in [0.10, 0.12]:
                for r_out in [0.04, 0.06]:
                    additional_patterns.append([r_out, r_mid, r_mid, r_out, r_mid, r_in, r_mid, r_in, r_mid, r_out])
        
        for pattern in additional_patterns:
            if not time_ok():
                break
            
            r00, r01, r02, r03, r11, r12, r13, r22, r23, r33 = pattern
            r_4x4 = build_symmetric_4x4(r00, r01, r02, r03, r11, r12, r13, r22, r23, r33)
            cell = make_cell_from_radii(r_4x4)
            score = calculate_gap(cell)
            
            if score > best_score:
                best_score = score
                best_radii = pattern.copy()
                best_cell = cell.copy()
    
    # Fallback: if no good result, use a simple centered rod
    if best_cell is None:
        # Centered circular rod with radius 0.2
        r2 = XX**2 + YY**2
        cell = (r2 <= 0.2**2).astype(int)
        best_cell = enforce_symmetry(cell)
    
    # Convert to nested list
    return best_cell.tolist()
