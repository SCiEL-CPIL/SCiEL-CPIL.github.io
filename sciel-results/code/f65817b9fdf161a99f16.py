"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time

# EVOLVE-BLOCK-START
def design(problem):
    """Design a symmetric 32x32 photonic crystal unit cell for maximum TM band gap."""
    size = problem["pixels"]  # 32
    eps_low = problem.get("eps_low", 1)
    eps_high = problem.get("eps_high", 12)
    
    # Time budget: 120s, leave 30s margin
    start_time = time.time()
    time_budget = 90.0  # seconds
    
    def make_symmetric(cell_8x8):
        """Expand 8x8 quarter-cell to 32x32 full cell with square symmetry."""
        # The 8x8 represents the top-left quadrant
        # Full cell: [Q | flip_h(Q)]
        #            [flip_v(Q) | flip_vh(Q)]
        # where flip_h = horizontal flip, flip_v = vertical flip
        q = np.array(cell_8x8, dtype=int)
        # Top half: left = Q, right = flip horizontally
        top = np.hstack([q, q[:, ::-1]])
        # Bottom half: flip top vertically
        bottom = top[::-1, :]
        full = np.vstack([top, bottom])
        return full.tolist()
    
    def make_symmetric_4x4(cell_4x4):
        """Expand 4x4 to 32x32 by tiling and symmetry."""
        q = np.array(cell_4x4, dtype=int)
        # Tile to 16x16 first, then apply symmetry
        tiled = np.tile(q, (4, 4))
        # Now apply the full symmetry: top-left 16x16 is tiled
        # top = [tiled | flip_h(tiled)]
        top = np.hstack([tiled, tiled[:, ::-1]])
        bottom = top[::-1, :]
        return np.vstack([top, bottom]).tolist()
    
    # Try several candidate patterns
    
    candidates = []
    
    # Candidate 1: Circular rod (seed)
    cell = np.zeros((size, size), dtype=int)
    for i in range(size):
        for j in range(size):
            x = (i + 0.5) / size - 0.5
            y = (j + 0.5) / size - 0.5
            if x*x + y*y < 0.2**2:
                cell[i, j] = 1
    candidates.append(cell.tolist())
    
    # Candidate 2: Square rod (axis-aligned square)
    for half_size in [0.15, 0.2, 0.25, 0.3]:
        cell = np.zeros((size, size), dtype=int)
        h = half_size * size
        for i in range(size):
            for j in range(size):
                x = (i + 0.5) / size - 0.5
                y = (j + 0.5) / size - 0.5
                if abs(x) < h/size and abs(y) < h/size:
                    cell[i, j] = 1
        candidates.append(cell.tolist())
    
    # Candidate 3: Diamond shape
    for radius in [0.15, 0.2, 0.25, 0.3]:
        cell = np.zeros((size, size), dtype=int)
        for i in range(size):
            for j in range(size):
                x = (i + 0.5) / size - 0.5
                y = (j + 0.5) / size - 0.5
                if abs(x) + abs(y) < radius:
                    cell[i, j] = 1
        candidates.append(cell.tolist())
    
    # Candidate 4: Dielectric at corners (4 small squares)
    for corner_size in [0.1, 0.15, 0.2]:
        cell = np.zeros((size, size), dtype=int)
        cs = int(corner_size * size)
        # Four corners
        for i in range(cs):
            for j in range(cs):
                cell[i, j] = 1
                cell[i, size-1-j] = 1
                cell[size-1-i, j] = 1
                cell[size-1-i, size-1-j] = 1
        candidates.append(cell.tolist())
    
    # Candidate 5: Cross/plus shape
    for arm_width in [3, 4, 5, 6]:
        cell = np.zeros((size, size), dtype=int)
        center = size // 2
        # Horizontal bar
        for j in range(size):
            for i in range(center - arm_width//2, center + arm_width//2 + 1):
                if 0 <= i < size:
                    cell[i, j] = 1
        # Vertical bar
        for i in range(size):
            for j in range(center - arm_width//2, center + arm_width//2 + 1):
                if 0 <= j < size:
                    cell[i, j] = 1
        candidates.append(cell.tolist())
    
    # Candidate 6: Ring/annulus shape
    for inner_r, outer_r in [(0.1, 0.2), (0.15, 0.25), (0.1, 0.25), (0.2, 0.3)]:
        cell = np.zeros((size, size), dtype=int)
        for i in range(size):
            for j in range(size):
                x = (i + 0.5) / size - 0.5
                y = (j + 0.5) / size - 0.5
                r = np.sqrt(x*x + y*y)
                if inner_r < r < outer_r:
                    cell[i, j] = 1
        candidates.append(cell.tolist())
    
    # Candidate 7: Air holes in dielectric (inverse)
    for hole_radius in [0.1, 0.15, 0.2]:
        cell = np.ones((size, size), dtype=int)
        for i in range(size):
            for j in range(size):
                x = (i + 0.5) / size - 0.5
                y = (j + 0.5) / size - 0.5
                if x*x + y*y < hole_radius**2:
                    cell[i, j] = 0
        candidates.append(cell.tolist())
    
    # Candidate 8: Four small rods at symmetric positions
    for offset in [0.2, 0.25, 0.3]:
        for rod_r in [0.08, 0.1, 0.12]:
            cell = np.zeros((size, size), dtype=int)
            positions = [(offset, offset), (-offset, offset), (offset, -offset), (-offset, -offset)]
            for px, py in positions:
                for i in range(size):
                    for j in range(size):
                        x = (i + 0.5) / size - 0.5
                        y = (j + 0.5) / size - 0.5
                        if (x - px)**2 + (y - py)**2 < rod_r**2:
                            cell[i, j] = 1
            candidates.append(cell.tolist())
    
    # Candidate 9: Checkerboard-like pattern with symmetry
    for period in [4, 6, 8]:
        cell = np.zeros((size, size), dtype=int)
        for i in range(size):
            for j in range(size):
                # Symmetric checkerboard
                if (i // period + j // period) % 2 == 0:
                    cell[i, j] = 1
        candidates.append(cell.tolist())
    
    # Candidate 10: Concentric squares
    for n_squares in [2, 3, 4]:
        for gap in [0.05, 0.08, 0.1]:
            cell = np.zeros((size, size), dtype=int)
            for i in range(size):
                for j in range(size):
                    x = (i + 0.5) / size - 0.5
                    y = (j + 0.5) / size - 0.5
                    r = max(abs(x), abs(y))  # Chebyshev distance
                    for k in range(n_squares):
                        r_inner = k * gap
                        r_outer = (k + 0.5) * gap
                        if r_inner < r < r_outer:
                            cell[i, j] = 1
                            break
            candidates.append(cell.tolist())
    
    # Now do a local search on the most promising candidates
    # Use a simple heuristic: try to optimize by flipping individual pixels
    # that maintain symmetry
    
    def is_symmetric(cell):
        cell = np.array(cell)
        # Check 180 rotation
        if not np.array_equal(cell, cell[::-1, ::-1]):
            return False
        # Check horizontal mirror
        if not np.array_equal(cell, cell[:, ::-1]):
            return False
        # Check vertical mirror  
        if not np.array_equal(cell, cell[::-1, :]):
            return False
        # Check transpose
        if not np.array_equal(cell, cell.T):
            return False
        return True
    
    # Simple local search: try flipping pixels in symmetric groups
    def local_search(cell, max_iters=200):
        cell = np.array(cell)
        best_score = -1
        best_cell = cell.copy()
        
        # For simplicity, use a proxy score: we can't compute the actual band gap,
        # so we'll just return the initial cell. The evaluator will score it.
        # Instead, let's just return the best candidate based on some heuristic.
        return best_cell.tolist()
    
    # Since we can't compute the actual band gap in the design function,
    # we'll return a carefully chosen candidate.
    # Based on photonic crystal literature, for TM polarization in square lattices,
    # dielectric rods with specific shapes can create large gaps.
    # Let's pick a few top candidates and return the one most likely to work well.
    
    # Check time
    elapsed = time.time() - start_time
    if elapsed > time_budget:
        # Return the first candidate (seed) as fallback
        return candidates[0]
    
    # Heuristic: patterns with moderate fill fraction and smooth boundaries tend to work well
    # Let's pick a diamond shape with radius 0.25 as our best guess
    # Or try the ring shape which can create multiple band gaps
    
    # Actually, let's do a more systematic approach: try all candidates
    # and pick based on a simple structural heuristic
    
    # For now, return a well-chosen candidate
    # The diamond with radius 0.2 is a good compromise
    best_idx = None
    for idx, cand in enumerate(candidates):
        cand_arr = np.array(cand)
        fill = np.mean(cand_arr)
        # Prefer fill fractions around 0.3-0.5 for TM gaps
        if 0.25 < fill < 0.45:
            if best_idx is None:
                best_idx = idx
    
    if best_idx is not None:
        return candidates[best_idx]
    
    # Fallback: return the circular rod
    return candidates[0]
# EVOLVE-BLOCK-END
