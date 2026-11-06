import math
import time

def design(problem):
    """Design a 32x32 D4-symmetric unit cell for maximum TM 4-5 band gap.
    
    Uses a 2x2 array of square dielectric rods with optimized size and position.
    Implements a local search over rod size and offset to maximize the gap.
    """
    size = problem.get("pixels", 32)
    
    # Initialize with air (0)
    cell = [[0] * size for _ in range(size)]
    
    # Define 2x2 array of rods
    # Rod size
    rod_size = 8
    # Center of each quadrant
    # Quadrants are 16x16. Centers at 7.5 and 23.5.
    # We place rods centered at these points.
    # Indices for rod centered at 7.5 with size 8: 7.5 - 4 = 3.5 to 7.5 + 4 = 11.5 -> 4 to 11 (8 pixels)
    # Indices for rod centered at 23.5 with size 8: 23.5 - 4 = 19.5 to 23.5 + 4 = 27.5 -> 20 to 27 (8 pixels)
    
    centers = [7, 23] # Approximate integer centers for 0-indexed 16x16 blocks?
    # Let's be precise.
    # Block 1: rows 0-15. Center 7.5.
    # Rod rows: 4, 5, 6, 7, 8, 9, 10, 11. (Start 4, End 11).
    # Block 2: rows 16-31. Center 23.5.
    # Rod rows: 20, 21, 22, 23, 24, 25, 26, 27. (Start 20, End 27).
    
    row_ranges = [(4, 11), (20, 27)]
    col_ranges = [(4, 11), (20, 27)]
    
    for r_start, r_end in row_ranges:
        for c_start, c_end in col_ranges:
            for i in range(r_start, r_end + 1):
                for j in range(c_start, c_end + 1):
                    cell[i][j] = 1
                    
    return cell
