import time

def design(problem):
    """
    Design a symmetric 32x32 pixel unit cell for a 2D square-lattice photonic crystal.
    Uses a dielectric square frame (square rod with a central square air hole)
    which is known to open larger TM band gaps than a simple circular rod.
    """
    size = problem["pixels"]
    # Parameters for the square frame
    # Outer half-width (distance from center to edge of dielectric square)
    W = 13.0
    # Inner half-width (distance from center to edge of air hole)
    H = 5.0
    
    cell = []
    center = (size - 1) / 2.0 # 15.5
    
    for i in range(size):
        row = []
        for j in range(size):
            # Calculate distance from center in "square" metric (max norm)
            # We use Euclidean distance for the square boundaries? No, square boundaries are axis-aligned.
            # A square centered at (cx, cy) with half-width W is defined by |x-cx| < W and |y-cy| < W.
            dx = abs(i - center)
            dy = abs(j - center)
            
            is_outer = (dx < W) and (dy < W)
            is_inner = (dx < H) and (dy < H)
            
            # Dielectric if in outer square and NOT in inner square
            if is_outer and not is_inner:
                row.append(1)
            else:
                row.append(0)
        cell.append(row)
        
    return cell
