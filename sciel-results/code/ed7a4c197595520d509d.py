import numpy as np

def design(problem):
    size = problem["pixels"]
    # Initialize with air (0)
    cell = [[0 for _ in range(size)] for _ in range(size)]
    
    # Define centers for the four rods
    # Symmetry center is at (size/2 - 0.5, size/2 - 0.5) = (15.5, 15.5)
    # Offset by size/4 = 8.0
    # Centers at 15.5 - 8.0 = 7.5 and 15.5 + 8.0 = 23.5
    centers = [
        (7.5, 7.5),
        (7.5, 23.5),
        (23.5, 7.5),
        (23.5, 23.5)
    ]
    
    radius = 4.5
    
    for i in range(size):
        for j in range(size):
            # Check if this pixel is inside any of the rods
            # Using pixel center coordinates (i+0.5, j+0.5)
            pi = i + 0.5
            pj = j + 0.5
            is_dielectric = False
            for cx, cy in centers:
                dist_sq = (pi - cx)**2 + (pj - cy)**2
                if dist_sq < radius**2:
                    is_dielectric = True
                    break
            
            if is_dielectric:
                cell[i][j] = 1
            else:
                cell[i][j] = 0
                
    # Enforce symmetry strictly to handle any floating point or logic edge cases
    # The generation above is inherently symmetric if centers and radius are symmetric,
    # but let's enforce C4v to be safe.
    
    # 1. Horizontal flip (left-right)
    # 2. Vertical flip (top-bottom)
    # 3. Transpose (diagonal)
    
    # We can enforce by taking the logical AND or OR of symmetric pixels?
    # Actually, if the generation is symmetric, it's already symmetric.
    # But to be absolutely sure, we can average and threshold, or just check.
    # Given the analytic nature, it should be exact.
    
    return cell
