"""Design a D4-symmetric photonic crystal unit cell using cellular automaton dynamics."""
import numpy as np

# EVOLVE-BLOCK-START
def design(problem):
    """
    Generate a 32x32 D4-symmetric binary mask using a 2D cellular automaton
    (Game of Life variant) evolved from a simple symmetric seed, then enforce
    D4 symmetry about the cell center.
    """
    size = problem["pixels"]  # 32
    half = size // 2          # 16
    
    # Initialize a small symmetric seed in the top-left quadrant
    # We'll work with the full grid but enforce symmetry at the end
    # Start with a simple pattern: a cross or small cluster in the center
    grid = np.zeros((size, size), dtype=np.int32)
    
    # Place a small symmetric seed - a 3x3 block at center
    c = size // 2
    # Set a few pixels to create an interesting initial state
    # Use a pattern that will evolve into complex structures
    seed_pattern = [
        [0, 1, 0],
        [1, 1, 1],
        [0, 1, 0],
    ]
    # Place this at the center
    for di in range(-1, 2):
        for dj in range(-1, 2):
            ii = c + di
            jj = c + dj
            if 0 <= ii < size and 0 <= jj < size:
                grid[ii, jj] = seed_pattern[di + 1][dj + 1]
    
    # Also add some additional seed points to make evolution more interesting
    # Add diagonal points
    for d in [2, 3, 4]:
        grid[c - d, c - d] = 1
        grid[c + d, c - d] = 1
        grid[c - d, c + d] = 1
        grid[c + d, c + d] = 1
    
    # Enforce D4 symmetry on the initial grid
    grid = _enforce_d4_symmetry(grid, size)
    
    # Evolve the cellular automaton for several steps
    # Use a modified Game of Life rule that favors dense patterns
    num_steps = 8
    for step in range(num_steps):
        grid = _ca_step(grid, size)
        grid = _enforce_d4_symmetry(grid, size)
    
    # Threshold: convert to binary (already binary, but ensure)
    grid = (grid > 0).astype(np.int32)
    
    # Final D4 symmetry enforcement
    grid = _enforce_d4_symmetry(grid, size)
    
    # Convert to nested list
    cell = grid.tolist()
    
    return cell


def _enforce_d4_symmetry(grid, size):
    """
    Enforce D4 symmetry (square symmetry group) about the cell center.
    The cell must satisfy:
      - cell == cell[::-1] row-wise (vertical flip)
      - cell == cell[::-1] column-wise (horizontal flip)  
      - cell == cell.T (transpose)
    
    This means the cell is symmetric under:
      - Reflection about horizontal center line
      - Reflection about vertical center line
      - Rotation by 90 degrees (which follows from the above two + diagonal)
    
    Actually, for a 32x32 grid, the center is between pixels.
    D4 symmetry about the cell center means:
      - (i, j) == (size-1-i, j)   [horizontal reflection]
      - (i, j) == (i, size-1-j)   [vertical reflection]
      - (i, j) == (j, i)          [diagonal reflection / transpose]
    
    Let's enforce all three conditions.
    """
    g = grid.copy()
    size_n = size
    
    # We need to find a fixed point under all D4 operations.
    # The D4 group has 8 elements: identity, 3 rotations, 4 reflections.
    # For a grid indexed 0..size-1, the center is at (size/2 - 0.5, size/2 - 0.5)
    # 
    # Reflections:
    #   H: (i, j) -> (size-1-i, j)
    #   V: (i, j) -> (i, size-1-j)
    #   D1: (i, j) -> (j, i)  [main diagonal]
    #   D2: (i, j) -> (size-1-j, size-1-i)  [anti-diagonal]
    #
    # To enforce symmetry, we can average (or take majority) over the orbit of each pixel.
    # Since we want binary, let's take the maximum (OR) over the orbit to preserve density,
    # or we could take the mean and threshold. Let's try OR first to keep more structure.
    
    result = np.zeros((size_n, size_n), dtype=np.int32)
    
    for i in range(size_n):
        for j in range(size_n):
            # Get all pixels in the D4 orbit of (i, j)
            orbit = [
                (i, j),
                (size_n - 1 - i, j),       # H
                (i, size_n - 1 - j),       # V
                (j, i),                     # D1
                (size_n - 1 - i, size_n - 1 - j),  # 180 rot = H then V
                (j, size_n - 1 - i),       # D2
                (size_n - 1 - j, i),       # H then D1
                (size_n - 1 - j, size_n - 1 - i),   # V then D1
            ]
            # Take OR (max) over the orbit
            val = 0
            for (oi, oj) in orbit:
                val = val or g[oi, oj]
            result[i, j] = val
    
    return result


def _ca_step(grid, size):
    """
    One step of a modified Game of Life cellular automaton.
    Uses toroidal boundary conditions.
    Rule: A dead cell becomes alive if it has exactly 3 neighbors.
          A live cell stays alive if it has 2 or 3 neighbors.
    This is standard Conway's Game of Life.
    """
    new_grid = np.zeros_like(grid)
    for i in range(size):
        for j in range(size):
            # Count neighbors with toroidal wrapping
            count = 0
            for di in [-1, 0, 1]:
                for dj in [-1, 0, 1]:
                    if di == 0 and dj == 0:
                        continue
                    ni = (i + di) % size
                    nj = (j + dj) % size
                    count += grid[ni, nj]
            
            if grid[i, j] == 1:
                # Live cell survives with 2 or 3 neighbors
                if count == 2 or count == 3:
                    new_grid[i, j] = 1
            else:
                # Dead cell becomes alive with exactly 3 neighbors
                if count == 3:
                    new_grid[i, j] = 1
    
    return new_grid
# EVOLVE-BLOCK-END
