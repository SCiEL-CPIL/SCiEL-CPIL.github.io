import math

def design(problem):
    size = problem["pixels"]
    cell = []
    # Approach: Central Dielectric Square Pillar
    # This differs from the annulus approach by having a connected center and sharp corners.
    w = 0.45  # Half-width of the square relative to cell size (0 to 0.5 is half cell)
    # Actually, let's define it as the range. 
    # x in [-0.5, 0.5]. Square from -w/2 to w/2? 
    # Let's use a parameter 'half_size' such that the square extends from -half_size to half_size.
    half_size = 0.25 # Square of side 0.5, centered.
    
    for i in range(size):
        row = []
        for j in range(size):
            x = (i + 0.5) / size - 0.5
            y = (j + 0.5) / size - 0.5
            # Check if inside the central square
            if abs(x) < half_size and abs(y) < half_size:
                row.append(1) # Dielectric
            else:
                row.append(0) # Air
        cell.append(row)
    return cell
