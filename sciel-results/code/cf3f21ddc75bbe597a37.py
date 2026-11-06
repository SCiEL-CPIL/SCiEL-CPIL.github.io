"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import math
import time
import numpy as np
import random

def design(problem):
    """
    Design a 32x32 unit cell using a multi-stage optimization:
    1. Coarse geometry optimization (4 rods) with 5x5 PW surrogate.
    2. Fine pixel boundary optimization with 9x5 PW surrogate.
    3. Local pixel flip polishing with 9x5 PW surrogate.
    """
    start_time = time.time()
    deadline = start_time + 210.0  # Safety margin before 230s limit
    
    size = problem.get("pixels", 32)
    eps_high = problem.get("eps_high", 12.0)
    eps_low = problem.get("eps_low", 1.0)
    
    # --- High-Fidelity Surrogate Band Structure ---
    def compute_surrogate_gap(cell, pw_size=5):
        """
        Compute an approximate gap ratio using a plane-wave basis.
        pw_size: N x N grid of plane waves (N=5 or 9).
        """
        cell_np = np.array(cell, dtype=np.float64)
        eps_map = eps_low + (eps_high - eps_low) * cell_np
        inv_eps_map = 1.0 / eps_map
        
        # Compute FFT of 1/epsilon
        fft_inv_eps = np.fft.fft2(inv_eps_map) / (size * size)
        
        # Define plane wave grid
        # We want G values from -pw_size//2 to pw_size//2
        # If pw_size is odd, e.g., 5 -> -2, -1, 0, 1, 2
        # If pw_size is even, e.g., 4 -> -2, -1, 0, 1 (asymmetric)
        # Let's stick to odd sizes for symmetry around 0.
        if pw_size % 2 == 0:
            pw_size += 1
            
        half = pw_size // 2
        m_vals = np.arange(-half, half + 1)
        n_vals = np.arange(-half, half + 1)
        
        # Extract Fourier coefficients
        c_G = {}
        for m in m_vals:
            for n in n_vals:
                idx_m = m % size
                idx_n = n % size
                c_G[(m, n)] = fft_inv_eps[idx_m, idx_n]
        
        # K-points: Irreducible BZ of square lattice
        # Gamma (0,0), X (0.5, 0), M (0.5, 0.5)
        # And a few interior points to capture band folding
        k_points = [
            (0.0, 0.0),   # Gamma
            (0.5, 0.0),   # X
            (0.5, 0.5),   # M
            (0.25, 0.0),  # Gamma-X mid
            (0.25, 0.25)  # Gamma-M mid
        ]
        
        # Precompute G vectors and indices
        idx_to_mn = []
        for m in m_vals:
            for n in n_vals:
                idx_to_mn.append((m, n))
        N = len(idx_to_mn)
        
        min_omega_5 = float('inf')
        max_omega_4 = 0.0
        
        for kx, ky in k_points:
            A = np.zeros((N, N))
            B = np.zeros((N, N), dtype=complex)
            
            for i in range(N):
                m_i, n_i = idx_to_mn[i]
                Gx_i = 2*math.pi*m_i
                Gy_i = 2*math.pi*n_i
                kx_G = kx + Gx_i
                ky_G = ky + Gy_i
                A[i, i] = kx_G**2 + ky_G**2
                
                for j in range(N):
                    m_j, n_j = idx_to_mn[j]
                    G_diff_m = m_i - m_j
                    G_diff_n = n_i - n_j
                    if (G_diff_m, G_diff_n) in c_G:
                        B[i, j] = c_G[(G_diff_m, G_diff_n)]
                    else:
                        B[i, j] = 0.0
            
            # Solve generalized eigenvalue problem: A v = omega^2 B v
            # Equivalent to B^{-1} A v = omega^2 v
            # Since B is Hermitian positive definite (for physical eps), we can use eig
            try:
                # Use np.linalg.eig for simplicity, filter real positive parts
                eigvals, _ = np.linalg.eig(np.linalg.inv(B) @ A)
                real_eigvals = np.real(eigvals)
                # Filter out numerical noise and non-physical modes
                real_eigvals = real_eigvals[real_eigvals > 1e-6]
                if len(real_eigvals) < 5:
                    continue
                real_eigvals.sort()
                
                omega_4 = np.sqrt(real_eigvals[3])
                omega_5 = np.sqrt(real_eigvals[4])
                max_omega_4 = max(max_omega_4, omega_4)
                min_omega_5 = min(min_omega_5, omega_5)
            except:
                return -1.0
            
        if max_omega_4 == 0.0 or min_omega_5 == float('inf'):
            return -1.0
            
        gap = min_omega_5 - max_omega_4
        midgap = (min_omega_5 + max_omega_4) / 2.0
        if midgap == 0:
            return -1.0
        return 2.0 * gap / midgap

    # --- Stage 1: Coarse Geometry Optimization ---
    # Initialize with 4 rods
    # Parameters: center_offset (0.25), radius
    # We will sweep radius and offset slightly
    
    best_cell = None
    best_score = -1.0
    
    # Define a function to create a cell from parameters
    def create_cell_4rods(offset, radius):
        cell = [[0 for _ in range(size)] for _ in range(size)]
        centers = [
            (-offset, -offset),
            ( offset, -offset),
            (-offset,  offset),
            ( offset,  offset)
        ]
        for i in range(size):
            y = (i + 0.5) / size - 0.5
            for j in range(size):
                x = (j + 0.5) / size - 0.5
                inside = False
                for cx, cy in centers:
                    dist_sq = (x - cx)**2 + (y - cy)**2
                    if dist_sq < radius**2:
                        inside = True
                        break
                cell[i][j] = 1 if inside else 0
        return cell

    # Sweep offsets and radii
    offsets = [0.22, 0.25, 0.28]
    radii = [0.10, 0.12, 0.14, 0.16, 0.18]
    
    for off in offsets:
        for rad in radii:
            if time.time() > deadline - 60.0:
                break
            cell = create_cell_4rods(off, rad)
            score = compute_surrogate_gap(cell, pw_size=5)
            if score > best_score:
                best_score = score
                best_cell = cell
    
    if best_cell is None:
        # Fallback
        best_cell = create_cell_4rods(0.25, 0.15)
        best_score = compute_surrogate_gap(best_cell, pw_size=5)

    # --- Stage 2: Fine Pixel Boundary Optimization ---
    # We will flip pixels in the "boundary" of the rods to refine the shape.
    # Identify pixels that are on the boundary of the current shape.
    # A pixel is on the boundary if it is 1 and has a 0 neighbor, or 0 and has a 1 neighbor.
    # We will try flipping these pixels (in symmetric orbits) to improve the score.
    
    cell = [row[:] for row in best_cell]
    
    # Generate orbits again
    def get_orbits(cell):
        visited = set()
        orbits = []
        for i in range(size):
            for j in range(size):
                if (i, j) in visited:
                    continue
                orbit = set()
                points = [
                    (i, j),
                    (size - 1 - i, j),
                    (i, size - 1 - j),
                    (size - 1 - i, size - 1 - j),
                    (j, i),
                    (size - 1 - j, i),
                    (j, size - 1 - i),
                    (size - 1 - j, size - 1 - i)
                ]
                for p in points:
                    orbit.add(p)
                    visited.add(p)
                orbits.append(orbit)
        return orbits

    orbits = get_orbits(cell)
    random.seed(42)
    random.shuffle(orbits)
    
    # Filter orbits to only those on the boundary
    def is_boundary_orbit(orbit, cell):
        # Check if any pixel in orbit is a boundary pixel
        for i, j in orbit:
            val = cell[i][j]
            # Check 4-neighbors
            for di, dj in [(-1,0), (1,0), (0,-1), (0,1)]:
                ni, nj = i+di, j+dj
                if 0 <= ni < size and 0 <= nj < size:
                    if cell[ni][nj] != val:
                        return True
        return False

    boundary_orbits = [o for o in orbits if is_boundary_orbit(o, cell)]
    random.shuffle(boundary_orbits)
    
    improved = True
    while improved and time.time() < deadline - 30.0:
        improved = False
        for orbit in boundary_orbits:
            if time.time() > deadline - 30.0:
                break
            
            # Check if all pixels in orbit have the same value
            first_i, first_j = next(iter(orbit))
            val = cell[first_i][first_j]
            all_same = all(cell[i][j] == val for i, j in orbit)
            
            if not all_same:
                continue
            
            # Flip
            new_val = 1 - val
            for i, j in orbit:
                cell[i][j] = new_val
            
            score = compute_surrogate_gap(cell, pw_size=9)
            
            if score > best_score:
                best_score = score
                best_cell = [row[:] for row in cell]
                improved = True
            else:
                for i, j in orbit:
                    cell[i][j] = val

    # --- Stage 3: Local Pixel Flip Polishing ---
    # Try all orbits again with the higher fidelity surrogate
    # But only a subset to save time? Or all?
    # Let's try all orbits one more time, but break early if no improvement in a full pass
    
    cell = [row[:] for row in best_cell]
    orbits = get_orbits(cell)
    random.shuffle(orbits)
    
    improved = True
    while improved and time.time() < deadline - 10.0:
        improved = False
        for orbit in orbits:
            if time.time() > deadline - 10.0:
                break
            
            first_i, first_j = next(iter(orbit))
            val = cell[first_i][first_j]
            all_same = all(cell[i][j] == val for i, j in orbit)
            
            if not all_same:
                continue
            
            new_val = 1 - val
            for i, j in orbit:
                cell[i][j] = new_val
            
            score = compute_surrogate_gap(cell, pw_size=9)
            
            if score > best_score:
                best_score = score
                best_cell = [row[:] for row in cell]
                improved = True
            else:
                for i, j in orbit:
                    cell[i][j] = val

    return best_cell

# EVOLVE-BLOCK-END
