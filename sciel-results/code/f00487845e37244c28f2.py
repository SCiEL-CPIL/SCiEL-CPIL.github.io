"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import math

def design(problem):
    """Return a symmetric 32x32 photonic crystal unit cell designed for a TM band gap."""
    size = 32
    t_start = time.time()
    deadline = t_start + 110.0  # safety margin before 120s limit
    
    # For TM polarization in square lattice, dielectric rods in air background
    # with high contrast (eps=12) can produce good band gaps.
    # The 4th-5th TM band gap is typically found in the second Brillouin zone.
    # 
    # Key insight: For a single rod per unit cell (high symmetry), the gap
    # depends critically on the rod radius. For eps=12, optimal radius is
    # around 0.35-0.45 of the lattice constant.
    #
    # We'll use a single central rod with circular cross-section, which has
    # full C4v symmetry.
    
    def _make_cell(radius):
        """Create a 32x32 cell with a single circular rod of given radius."""
        cell = [[0] * size for _ in range(size)]
        for i in range(size):
            x = (i + 0.5) / size - 0.5
            for j in range(size):
                y = (j + 0.5) / size - 0.5
                if x * x + y * y < radius * radius:
                    cell[i][j] = 1
        return cell
    
    def _check_symmetry(cell, size):
        """Check square symmetry."""
        for i in range(size):
            for j in range(size):
                if cell[i][j] != cell[size - 1 - i][size - 1 - j]:
                    return False
                if cell[i][j] != cell[j][i]:
                    return False
                if cell[i][j] != cell[size - 1 - i][j]:
                    return False
                if cell[i][j] != cell[i][size - 1 - j]:
                    return False
        return True
    
    # Strategy: Try multiple candidate geometries and pick the best one
    # based on physical intuition for TM band gaps in square photonic crystals.
    
    # Candidate 1: Single central rod with varying radius
    # For eps=12, the 4-5 TM gap is maximized around r/a ≈ 0.35-0.45
    candidates = []
    
    # Primary candidates: single central rod
    for r in [0.30, 0.33, 0.35, 0.37, 0.40, 0.42, 0.45]:
        cell = _make_cell(r)
        if _check_symmetry(cell, size):
            candidates.append((r, cell))
    
    # Candidate 2: Four rods at quadrant centers (still C4v symmetric)
    for r in [0.12, 0.15, 0.18]:
        offset = 0.25
        cell = [[0] * size for _ in range(size)]
        centers = [(offset, offset), (-offset, offset), (offset, -offset), (-offset, -offset)]
        for i in range(size):
            x = (i + 0.5) / size - 0.5
            for j in range(size):
                y = (j + 0.5) / size - 0.5
                for cx, cy in centers:
                    if (x - cx) ** 2 + (y - cy) ** 2 < r * r:
                        cell[i][j] = 1
                        break
        if _check_symmetry(cell, size):
            candidates.append(('4rod_r{:.2f}'.format(r), cell))
    
    # Candidate 3: Dielectric background with central air hole
    for hole_r in [0.15, 0.20, 0.25, 0.30]:
        cell = [[1] * size for _ in range(size)]
        for i in range(size):
            x = (i + 0.5) / size - 0.5
            for j in range(size):
                y = (j + 0.5) / size - 0.5
                if x * x + y * y < hole_r * hole_r:
                    cell[i][j] = 0
        if _check_symmetry(cell, size):
            candidates.append(('hole_r{:.2f}'.format(hole_r), cell))
    
    # Heuristic scoring based on physics:
    # - Single rod with r ≈ 0.38-0.42 tends to give best 4-5 TM gap for eps=12
    # - Four-rod configurations can also be competitive
    # - Air holes in dielectric background generally give smaller gaps for TM
    
    def heuristic_score(param, cell):
        if isinstance(param, float):
            # Single rod: optimal around 0.38-0.42
            if 0.37 <= param <= 0.43:
                return 1.0 - abs(param - 0.40) * 2.0
            elif 0.35 <= param <= 0.45:
                return 0.7 - abs(param - 0.40) * 1.5
            else:
                return 0.3
        elif isinstance(param, str):
            if param.startswith('4rod'):
                return 0.5
            elif param.startswith('hole'):
                return 0.4
        return 0.3
    
    best_cell = None
    best_heuristic = -1
    
    for param, cell in candidates:
        score = heuristic_score(param, cell)
        if score > best_heuristic:
            best_heuristic = score
            best_cell = cell
    
    # If no valid cell found, create a default
    if best_cell is None:
        best_cell = _make_cell(0.38)
    
    # Final check: ensure the cell meets all requirements
    if not _check_symmetry(best_cell, size):
        # Fallback to known-good symmetric cell
        best_cell = _make_cell(0.38)
    
    return best_cell
# EVOLVE-BLOCK-END
