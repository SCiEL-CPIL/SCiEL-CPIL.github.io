"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time

# EVOLVE-BLOCK-START
def design(problem):
    """Four-rod dielectric lattice optimized for TM 4-5 band gap via Fourier-harmonic parameterization."""
    t_start = time.time()
    DEADLINE = 200.0  # safety margin before 230s kill
    
    size = problem["pixels"]
    eps_high = problem["eps_high"]
    eps_low = problem["eps_low"]
    
    # Four-rod pattern: dielectric disks at (±d, ±d) in normalized coords
    # Parameters: rod_radius (fraction of cell), offset (fraction of cell)
    # Symmetry: invariant under 90° rotation, reflection about x, y, and diagonals
    
    def make_cell(rod_radius, offset):
        """Create 32x32 cell with four dielectric rods at (±offset, ±offset)."""
        cell = np.zeros((size, size), dtype=np.int8)
        for i in range(size):
            for j in range(size):
                x = (i + 0.5) / size - 0.5
                y = (j + 0.5) / size - 0.5
                # Check distance to each of four rod centers
                for dx, dy in [(offset, offset), (offset, -offset), (-offset, offset), (-offset, -offset)]:
                    if (x - dx)**2 + (y - dy)**2 < rod_radius**2:
                        cell[i, j] = 1
                        break
        return cell
    
    def fourier_coefficients(cell, nx, ny):
        """Compute 2D DFT of the dielectric indicator function."""
        f = cell.astype(np.float64)
        # Normalize: f is 0 or 1
        F = np.fft.fft2(f)
        F /= (size * size)
        return F
    
    def surrogate_gap(cell):
        """
        Surrogate gap estimate based on Fourier coefficients.
        The TM 4-5 gap is driven by the (1,1) Fourier component coupling.
        Higher |F(1,1)| and |F(1,0)|/|F(0,1)| tend to open the gap.
        We use a heuristic: gap ~ |F(1,1)| * (1 - mean_f) * mean_f
        (product maximized when fill fraction is moderate)
        """
        f = cell.astype(np.float64)
        fill = np.mean(f)
        if fill < 0.01 or fill > 0.99:
            return -1.0
        F = fourier_coefficients(cell, 1, 1)
        # F is indexed as F[nx, ny] after fft2; for periodic, (1,1) is at index [1,1] or [-1,-1]
        # fft2 gives F[kx, ky] where kx,ky are 0..N-1; negative freqs are at N-1, N-2, etc.
        F11 = abs(F[1, 1])
        F10 = abs(F[1, 0])
        F01 = abs(F[0, 1])
        # The gap is enhanced by strong (1,1) coupling
        # Penalize too high or too low fill
        fill_penalty = 1.0 - abs(fill - 0.35) / 0.35
        if fill_penalty < 0:
            fill_penalty = 0.1
        score = (F11 + 0.3 * F10 + 0.3 * F01) * fill_penalty
        return score
    
    # Grid search over rod_radius and offset
    best_score = -np.inf
    best_cell = None
    best_params = None
    
    # Rod radius: 0.05 to 0.25 (fraction of cell side)
    # Offset: 0.05 to 0.35 (fraction of cell side from center)
    radius_vals = np.linspace(0.05, 0.25, 12)
    offset_vals = np.linspace(0.05, 0.35, 12)
    
    for r in radius_vals:
        if time.time() > DEADLINE:
            break
        for d in offset_vals:
            if time.time() > DEADLINE:
                break
            cell = make_cell(r, d)
            s = surrogate_gap(cell)
            if s > best_score:
                best_score = s
                best_cell = cell
                best_params = (r, d)
    
    # Refine around best with finer grid
    if best_params is not None and time.time() < DEADLINE - 30:
        r0, d0 = best_params
        radius_fine = np.linspace(max(0.03, r0 - 0.08), min(0.30, r0 + 0.08), 15)
        offset_fine = np.linspace(max(0.03, d0 - 0.08), min(0.40, d0 + 0.08), 15)
        
        for r in radius_fine:
            if time.time() > DEADLINE:
                break
            for d in offset_fine:
                if time.time() > DEADLINE:
                    break
                cell = make_cell(r, d)
                s = surrogate_gap(cell)
                if s > best_score:
                    best_score = s
                    best_cell = cell
                    best_params = (r, d)
    
    # Second refinement
    if best_params is not None and time.time() < DEADLINE - 20:
        r0, d0 = best_params
        radius_fine2 = np.linspace(max(0.03, r0 - 0.04), min(0.30, r0 + 0.04), 10)
        offset_fine2 = np.linspace(max(0.03, d0 - 0.04), min(0.40, d0 + 0.04), 10)
        
        for r in radius_fine2:
            if time.time() > DEADLINE:
                break
            for d in offset_fine2:
                if time.time() > DEADLINE:
                    break
                cell = make_cell(r, d)
                s = surrogate_gap(cell)
                if s > best_score:
                    best_score = s
                    best_cell = cell
                    best_params = (r, d)
    
    if best_cell is None:
        # Fallback: single centered rod
        best_cell = np.zeros((size, size), dtype=np.int8)
        for i in range(size):
            for j in range(size):
                x = (i + 0.5) / size - 0.5
                y = (j + 0.5) / size - 0.5
                if x*x + y*y < 0.2**2:
                    best_cell[i, j] = 1
    
    # Convert to nested list
    result = best_cell.tolist()
    
    # Verify symmetry
    # Row-reverse
    for i in range(size):
        for j in range(size):
            if result[i][j] != result[size-1-i][j]:
                result[i][j] = result[size-1-i][j]
    # Col-reverse
    for i in range(size):
        for j in range(size):
            if result[i][j] != result[i][size-1-j]:
                result[i][j] = result[i][size-1-j]
    # Transpose
    for i in range(size):
        for j in range(size):
            if result[i][j] != result[j][i]:
                # Force symmetry: use min to ensure consistency
                val = min(result[i][j], result[j][i])
                result[i][j] = val
                result[j][i] = val
    
    return result
# EVOLVE-BLOCK-END
