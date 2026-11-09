# EVOLVE-BLOCK-START
import numpy as np

def squares_in_square10() -> tuple:
    """
    Returns a valid packing of 10 squares in the unit square maximizing the sum of side lengths.
    
    Configuration:
    - 2 large squares of side 0.5 at (0.25, 0.25) and (0.75, 0.75).
    - 8 small squares of side 0.25 filling the remaining top-left and bottom-right 0.5x0.5 quadrants.
    
    Total sum of sides = 2 * 0.5 + 8 * 0.25 = 1.0 + 2.0 = 3.0.
    Baseline sum = 2.5.
    Score = 3.0 / 2.5 = 1.2.
    """
    n = 10
    centers = np.zeros((n, 2), dtype=np.float64)
    sides = np.zeros(n, dtype=np.float64)
    
    # Large squares
    centers[0] = [0.25, 0.25]
    sides[0] = 0.5
    
    centers[1] = [0.75, 0.75]
    sides[1] = 0.5
    
    # Small squares in Top-Left quadrant [0, 0.5]x[0.5, 1.0]
    # Quadrant divided into 4 0.25x0.25 cells
    # Cell 1: [0, 0.25]x[0.5, 0.75] -> Center (0.125, 0.625)
    centers[2] = [0.125, 0.625]
    sides[2] = 0.25
    
    # Cell 2: [0.25, 0.5]x[0.5, 0.75] -> Center (0.375, 0.625)
    centers[3] = [0.375, 0.625]
    sides[3] = 0.25
    
    # Cell 3: [0, 0.25]x[0.75, 1.0] -> Center (0.125, 0.875)
    centers[4] = [0.125, 0.875]
    sides[4] = 0.25
    
    # Cell 4: [0.25, 0.5]x[0.75, 1.0] -> Center (0.375, 0.875)
    centers[5] = [0.375, 0.875]
    sides[5] = 0.25
    
    # Small squares in Bottom-Right quadrant [0.5, 1.0]x[0, 0.5]
    # Quadrant divided into 4 0.25x0.25 cells
    # Cell 5: [0.5, 0.75]x[0, 0.25] -> Center (0.625, 0.125)
    centers[6] = [0.625, 0.125]
    sides[6] = 0.25
    
    # Cell 6: [0.75, 1.0]x[0, 0.25] -> Center (0.875, 0.125)
    centers[7] = [0.875, 0.125]
    sides[7] = 0.25
    
    # Cell 7: [0.5, 0.75]x[0.25, 0.5] -> Center (0.625, 0.375)
    centers[8] = [0.625, 0.375]
    sides[8] = 0.25
    
    # Cell 8: [0.75, 1.0]x[0.25, 0.5] -> Center (0.875, 0.375)
    centers[9] = [0.875, 0.375]
    sides[9] = 0.25
    
    return centers, sides

# EVOLVE-BLOCK-END
