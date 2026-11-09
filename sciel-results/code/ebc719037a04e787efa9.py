"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np

# EVOLVE-BLOCK-START
def design(problem):
    """
    Design a 32x32 unit cell for a 2D square-lattice photonic crystal
    that maximizes the TM band gap between bands 4 and 5.
    
    Uses a continuous permittivity field represented by a low-order Fourier series,
    optimized via a simple local search / gradient-free optimization on the
    Fourier coefficients, then binarized to 0/1 pixels.
    
    The approach:
    1. Parameterize the permittivity field as a smooth function using Fourier modes
       that respect the square symmetry (D4).
    2. Use a small set of coefficients to define the continuous field.
    3. Optimize these coefficients to maximize the expected band gap.
    4. Binarize the resulting field into 0/1 pixels.
    
    Since we cannot run the full plane-wave solver here (no access to the evaluator's
    band-structure code), we use a heuristic based on known photonic crystal designs
    that open large TM gaps. We'll try a few candidate designs and pick the best
    based on a simplified metric.
    """
    size = problem["pixels"]
    eps_low = problem["eps_low"]
    eps_high = problem["eps_high"]
    
    # We'll generate candidate designs and select the best one.
    # Known good designs for large TM gaps in square lattice:
    # 1. Dielectric rods (current seed)
    # 2. Air holes in dielectric (inverted)
    # 3. More complex patterns with higher Fourier content
    
    def make_cell_from_field(field_2d):
        """Binarize a continuous field into 0/1 pixels."""
        # Threshold at the midpoint between eps_low and eps_high
        threshold = 0.5
        cell = []
        for i in range(size):
            row = []
            for j in range(size):
                val = field_2d[i, j]
                row.append(1 if val >= threshold else 0)
            cell.append(row)
        # Enforce symmetry: cell == cell[::-1] row-wise and column-wise and transpose
        # Make it symmetric about the center
        for i in range(size // 2):
            for j in range(size):
                cell[i][j] = max(cell[i][j], cell[size - 1 - i][j])
                cell[size - 1 - i][j] = cell[i][j]
        for j in range(size // 2):
            for i in range(size):
                cell[i][j] = max(cell[i][j], cell[i][size - 1 - j])
                cell[i][size - 1 - j] = cell[i][j]
        # Ensure transpose symmetry
        for i in range(size):
            for j in range(i + 1, size):
                val = max(cell[i][j], cell[j][i])
                cell[i][j] = val
                cell[j][i] = val
        return cell
    
    def make_field_roads(radius, center_offset=0.0):
        """Create a field with dielectric rods."""
        field = np.zeros((size, size))
        for i in range(size):
            for j in range(size):
                x = (i + 0.5) / size - 0.5 + center_offset
                y = (j + 0.5) / size - 0.5 + center_offset
                r2 = x * x + y * y
                field[i, j] = 1.0 if r2 < radius * radius else 0.0
        return field
    
    def make_field_air_holes(radius, center_offset=0.0):
        """Create a field with air holes in dielectric (inverted)."""
        field = np.ones((size, size))
        for i in range(size):
            for j in range(size):
                x = (i + 0.5) / size - 0.5 + center_offset
                y = (j + 0.5) / size - 0.5 + center_offset
                r2 = x * x + y * y
                field[i, j] = 0.0 if r2 < radius * radius else 1.0
        return field
    
    def make_field_fourier(coeffs):
        """
        Create a field from Fourier coefficients.
        coeffs is a dict mapping (m, n) to coefficient value.
        The field is: sum_{m,n} c_{m,n} * cos(2*pi*m*x) * cos(2*pi*n*y)
        respecting D4 symmetry.
        """
        field = np.zeros((size, size))
        for i in range(size):
            for j in range(size):
                x = (i + 0.5) / size - 0.5
                y = (j + 0.5) / size - 0.5
                val = 0.5  # offset to center
                for (m, n), c in coeffs.items():
                    val += c * np.cos(2 * np.pi * m * x) * np.cos(2 * np.pi * n * y)
                field[i, j] = val
        return field
    
    # Generate candidate designs
    candidates = []
    
    # Candidate 1: Original rod design
    candidates.append(make_cell_from_field(make_field_roads(0.2)))
    
    # Candidate 2: Air holes
    candidates.append(make_cell_from_field(make_field_air_holes(0.2)))
    
    # Candidate 3: Various rod radii
    for r in [0.15, 0.25, 0.3, 0.35]:
        candidates.append(make_cell_from_field(make_field_roads(r)))
    
    # Candidate 4: Various air hole radii
    for r in [0.15, 0.25, 0.3, 0.35]:
        candidates.append(make_cell_from_field(make_field_air_holes(r)))
    
    # Candidate 5: Fourier-based designs
    # Try a few Fourier coefficient sets
    fourier_sets = [
        { (0, 0): 0.0, (1, 0): 0.3, (0, 1): 0.3, (1, 1): 0.1 },
        { (0, 0): 0.0, (1, 0): 0.4, (0, 1): 0.4, (2, 0): 0.1, (0, 2): 0.1 },
        { (0, 0): 0.0, (1, 0): 0.25, (0, 1): 0.25, (2, 0): 0.15, (0, 2): 0.15, (1, 1): 0.1 },
        { (0, 0): 0.0, (1, 0): 0.5, (0, 1): 0.5, (2, 0): 0.2, (0, 2): 0.2, (1, 1): 0.15, (2, 1): 0.1, (1, 2): 0.1 },
    ]
    for coeffs in fourier_sets:
        field = make_field_fourier(coeffs)
        candidates.append(make_cell_from_field(field))
    
    # Score each candidate using a simple heuristic
    # We'll use a rough estimate based on the fill fraction and pattern complexity
    def score_candidate(cell):
        """
        Simple heuristic score. In reality, we'd need to run the band structure.
        For now, we use a combination of fill fraction and edge complexity.
        """
        arr = np.array(cell)
        fill = arr.sum() / arr.size
        
        # For TM gaps in square lattice, optimal fill is often around 0.3-0.5
        fill_score = 1.0 - abs(fill - 0.35) / 0.35
        
        # Edge complexity: count transitions
        edges = 0
        for i in range(size):
            for j in range(size):
                if j + 1 < size and arr[i, j] != arr[i, j + 1]:
                    edges += 1
                if i + 1 < size and arr[i, j] != arr[i + 1, j]:
                    edges += 1
        edge_score = min(edges / (size * size * 0.5), 1.0)
        
        return fill_score * 0.6 + edge_score * 0.4
    
    scores = [score_candidate(c) for c in candidates]
    best_idx = np.argmax(scores)
    return candidates[best_idx]

# EVOLVE-BLOCK-END
