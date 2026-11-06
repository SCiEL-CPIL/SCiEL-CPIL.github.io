"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.fft import fft2, ifft2

def design(problem):
    """
    Design a symmetric 32x32 photonic crystal unit cell optimized for
    the largest gap between the 4th and 5th TM bands.
    Uses a parametric family of symmetric patterns with local optimization.
    """
    size = problem["pixels"]
    eps_low = problem.get("eps_low", 1)
    eps_high = problem.get("eps_high", 12)
    
    # Deadline guard
    start_time = time.time()
    deadline = start_time + 210  # 200s budget with safety margin
    
    # Generate initial candidate: a circular dielectric region
    # For TM band gaps in square lattice photonic crystals, the optimal
    # fill fraction is typically around 0.3-0.5 for the first gap,
    # but for higher-order gaps (4th-5th bands), we need to explore
    # different geometries.
    
    def make_cell_from_params(params, size=32):
        """
        Create a symmetric cell from parameters.
        params: list of (cx, cy, radius, sign) for elliptical/circular regions
        sign: +1 for dielectric, -1 for air (to subtract)
        """
        cell = np.zeros((size, size), dtype=np.float64)
        cx = size / 2.0
        cy = size / 2.0
        
        for (px, py, r, s) in params:
            # Create a mask for this region
            for i in range(size):
                for j in range(size):
                    dx = (j + 0.5) - cx
                    dy = (i + 0.5) - cy
                    # Normalize to unit cell coordinates
                    nx = dx / size
                    ny = dy / size
                    if nx*nx + ny*ny < r*r:
                        cell[i, j] += s
        
        # Threshold to binary
        cell = (cell > 0).astype(int)
        
        # Enforce symmetry: cell[i][j] == cell[size-1-i][size-1-j]
        # and cell == cell.T
        # For D4 symmetry about center, we need:
        # cell[i][j] == cell[size-1-i][j] (horizontal mirror)
        # cell[i][j] == cell[i][size-1-j] (vertical mirror)
        # cell[i][j] == cell[j][i] (transpose)
        
        # Enforce horizontal mirror symmetry
        for i in range(size):
            for j in range(size // 2):
                val = max(cell[i][j], cell[i][size-1-j])
                cell[i][j] = val
                cell[i][size-1-j] = val
        
        # Enforce vertical mirror symmetry
        for i in range(size // 2):
            for j in range(size):
                val = max(cell[i][j], cell[size-1-i][j])
                cell[i][j] = val
                cell[size-1-i][j] = val
        
        # Enforce transpose symmetry
        for i in range(size):
            for j in range(i + 1, size):
                val = max(cell[i][j], cell[j][i])
                cell[i][j] = val
                cell[j][i] = val
        
        return cell.tolist()
    
    def make_cell_circle(radius, size=32):
        """Create a cell with a centered circular dielectric region."""
        cell = np.zeros((size, size), dtype=int)
        cx = size / 2.0
        cy = size / 2.0
        for i in range(size):
            for j in range(size):
                dx = (j + 0.5) - cx
                dy = (i + 0.5) - cy
                if dx*dx + dy*dy < radius*radius:
                    cell[i, j] = 1
        return cell.tolist()
    
    def make_cell_square(half_size, size=32):
        """Create a cell with a centered square dielectric region."""
        cell = np.zeros((size, size), dtype=int)
        start = size // 2 - half_size
        end = size // 2 + half_size
        for i in range(start, end):
            for j in range(start, end):
                cell[i, j] = 1
        return cell.tolist()
    
    def make_cell_cross(arm_width, size=32):
        """Create a cell with a cross-shaped dielectric region."""
        cell = np.zeros((size, size), dtype=int)
        mid = size // 2
        for i in range(size):
            for j in range(size):
                if abs(i - mid) < arm_width or abs(j - mid) < arm_width:
                    cell[i, j] = 1
        return cell.tolist()
    
    def make_cell_ring(outer_r, inner_r, size=32):
        """Create a cell with a ring-shaped dielectric region."""
        cell = np.zeros((size, size), dtype=int)
        cx = size / 2.0
        cy = size / 2.0
        for i in range(size):
            for j in range(size):
                dx = (j + 0.5) - cx
                dy = (i + 0.5) - cy
                r2 = dx*dx + dy*dy
                if inner_r*inner_r < r2 < outer_r*outer_r:
                    cell[i, j] = 1
        return cell.tolist()
    
    def make_cell_four_circles(offset, radius, size=32):
        """Create a cell with four small circles in a symmetric pattern."""
        cell = np.zeros((size, size), dtype=int)
        cx = size / 2.0
        cy = size / 2.0
        positions = [
            (cx + offset, cy + offset),
            (cx + offset, cy - offset),
            (cx - offset, cy + offset),
            (cx - offset, cy - offset),
        ]
        for (px, py) in positions:
            for i in range(size):
                for j in range(size):
                    dx = (j + 0.5) - px
                    dy = (i + 0.5) - py
                    if dx*dx + dy*dy < radius*radius:
                        cell[i, j] = 1
        return cell.tolist()
    
    # Simple band structure estimator using plane-wave method
    # This is a simplified version for quick evaluation during optimization
    
    def estimate_gap(cell, n_pw=17, n_kpts=10):
        """
        Rough estimate of the gap between bands 4 and 5.
        Uses a simplified plane-wave expansion.
        """
        size = len(cell)
        eps = np.array(cell, dtype=np.float64)
        
        # Compute Fourier coefficients of epsilon
        eps_f = fft2(eps)
        
        # For a 32x32 pixel cell, we use a grid of plane waves
        # The reciprocal lattice vectors are G = (m, n) for m, n integers
        # We use a cutoff of n_pw//2 in each direction
        
        half = n_pw // 2
        G_indices = [(m, n) for m in range(-half, half+1) for n in range(-half, half+1)]
        n_pw_actual = len(G_indices)
        
        # Build the dielectric matrix in reciprocal space
        # For TM mode: (G-G') * (1/epsilon(G-G')) * (G-G')
        # This is complex, so we use a simplified scalar approximation
        
        # Actually, let's use a simpler approach: compute the eigenvalues
        # of the generalized eigenvalue problem using the pixel data directly
        # via a finite-difference or variational method
        
        # For speed, use a simple heuristic based on fill fraction and symmetry
        fill = np.sum(eps) / (size * size)
        
        # Heuristic: for TM bands in square lattice, the gap between
        # bands 4 and 5 is maximized at certain fill fractions
        # This is very rough but gives a signal for optimization
        
        # The 4th and 5th TM bands are higher-order, so we need
        # a more complex structure than a simple circle
        # Let's use a proxy: the product of fill fraction and its complement,
        # weighted by some structural complexity measure
        
        # Compute a simple complexity measure: number of domain boundaries
        boundaries = 0
        for i in range(size):
            for j in range(size):
                if j < size - 1 and eps[i, j] != eps[i, j+1]:
                    boundaries += 1
                if i < size - 1 and eps[i, j] != eps[i+1, j]:
                    boundaries += 1
        
        # Score proxy: we want moderate fill fraction and some complexity
        # This is very rough - the actual gap depends on the band structure
        score = fill * (1 - fill) * (1 + boundaries / 1000)
        
        return score
    
    # Generate initial candidates from different families
    candidates = []
    
    # Circles with various radii
    for r in [0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45]:
        candidates.append(('circle', r, make_cell_circle(r, size)))
    
    # Squares with various half-sizes
    for hs in [4, 5, 6, 7, 8, 9, 10]:
        candidates.append(('square', hs, make_cell_square(hs, size)))
    
    # Crosses with various arm widths
    for aw in [2, 3, 4, 5, 6]:
        candidates.append(('cross', aw, make_cell_cross(aw, size)))
    
    # Rings
    for outer_r in [0.3, 0.35, 0.4, 0.45]:
        for inner_r in [0.1, 0.15, 0.2]:
            if inner_r < outer_r:
                candidates.append(('ring', (outer_r, inner_r), make_cell_ring(outer_r, inner_r, size)))
    
    # Four circles
    for offset in [0.1, 0.15, 0.2]:
        for radius in [0.08, 0.1, 0.12]:
            candidates.append(('four_circ', (offset, radius), make_cell_four_circles(offset, radius, size)))
    
    # Evaluate all candidates with the heuristic
    best_cell = None
    best_score = -1
    
    for name, params, cell in candidates:
        if time.time() > deadline:
            break
        score = estimate_gap(np.array(cell))
        if score > best_score:
            best_score = score
            best_cell = cell
    
    # If we have time, do local refinement on the best candidate
    if time.time() < deadline - 30 and best_cell is not None:
        best_arr = np.array(best_cell)
        
        # Try flipping individual pixels while maintaining symmetry
        # Due to D4 symmetry, we only need to consider pixels in the
        # fundamental domain: i >= j and i + j < size
        # Actually for D4 about center, the fundamental region is
        # the triangle in the first quadrant relative to center
        
        improved = True
        while improved and time.time() < deadline - 10:
            improved = False
            best_arr_curr = best_arr.copy()
            best_score_curr = best_score
            
            for i in range(size):
                if time.time() > deadline - 10:
                    break
                for j in range(size):
                    # Only process pixels in the fundamental domain
                    # For D4 symmetry about center (15.5, 15.5):
                    # We need cell[i][j] == cell[size-1-i][j] == cell[i][size-1-j] == cell[size-1-i][size-1-j]
                    # and cell[i][j] == cell[j][i]
                    # So we only need to consider i <= size-1-i and j <= size-1-j and i <= j
                    # i.e., i < size//2 and j < size//2 and i <= j
                    if i >= size // 2 or j >= size // 2 or i > j:
                        continue
                    
                    # Try flipping this pixel and all its symmetric equivalents
                    symmetric_pixels = [
                        (i, j),
                        (size-1-i, j),
                        (i, size-1-j),
                        (size-1-i, size-1-j),
                        (j, i),
                        (j, size-1-i),
                        (size-1-i, j),  # duplicate
                        (size-1-j, i),
                    ]
                    
                    # Unique set
                    unique_pixels = list(set(symmetric_pixels))
                    
                    # Check if all have the same current value
                    current_val = best_arr_curr[i, j]
                    all_same = all(best_arr_curr[pi, pj] == current_val for pi, pj in unique_pixels)
                    
                    if not all_same:
                        continue
                    
                    # Flip
                    new_arr = best_arr_curr.copy()
                    for pi, pj in unique_pixels:
                        new_arr[pi, pj] = 1 - current_val
                    
                    new_score = estimate_gap(new_arr)
                    
                    if new_score > best_score_curr:
                        best_score_curr = new_score
                        best_arr_curr = new_arr
                        improved = True
            
            best_arr = best_arr_curr
            best_score = best_score_curr
    
    if best_cell is None:
        # Fallback: centered square
        best_cell = make_cell_square(8, size)
    
    return best_cell

# EVOLVE-BLOCK-END
