import math
import random

N = 12

def construct():
    # Set random seed for reproducibility if needed, but not strictly required
    random.seed(42)
    
    best_s = float('inf')
    best_inner = None
    
    # Function to calculate minimal side length S for a set of centers
    def get_min_side(centers):
        # Centers must be inside hexagon of side L = S-1
        # Condition for (x,y) to be in hexagon of side L:
        # |x| <= L
        # |y| <= sqrt(3)*L - (sqrt(3)/2)*|x|
        # => L >= |x|
        # => L >= (2/sqrt(3))*|y| + (1/2)*|x|
        max_l = 0.0
        for x, y in centers:
            l1 = abs(x)
            l2 = (2.0 / math.sqrt(3)) * abs(y) + 0.5 * abs(x)
            max_l = max(max_l, l1, l2)
        return max_l + 1.0

    # Generate lattice points
    # We want to pick 12 points from a triangular lattice
    # such that they are close to the origin to minimize the bounding hexagon
    
    # Try different rotations of the lattice
    for angle_deg in [0, 15, 30, 45, 60, 75, 90]:
        angle = math.radians(angle_deg)
        cos_a = math.cos(angle)
        sin_a = math.sin(angle)
        
        # Generate a grid of lattice points
        # Triangular lattice vectors
        u = (math.sqrt(3), 0)
        v = (math.sqrt(3)/2, 1.5)
        
        points = []
        for i in range(-5, 6):
            for j in range(-5, 6):
                # x = i*u_x + j*v_x
                # y = i*u_y + j*v_y
                x = i * u[0] + j * v[0]
                y = i * u[1] + j * v[1]
                
                # Rotate
                xr = x * cos_a - y * sin_a
                yr = x * sin_a + y * cos_a
                
                points.append((xr, yr))
        
        # Sort points by distance from origin (approximate radial distance)
        # We want the 12 closest points to the origin
        # But they must be distinct and non-overlapping (lattice ensures this)
        
        # Calculate a "radius" for sorting
        def sort_key(p):
            return p[0]**2 + p[1]**2
            
        points.sort(key=sort_key)
        
        # Take the first 12
        centers = points[:12]
        
        s = get_min_side(centers)
        
        if s < best_s:
            best_s = s
            best_inner = [[x, y, 0.0] for x, y in centers]
            
    # If best_inner is None (should not happen), fallback
    if best_inner is None:
        return [[4.0 * k, 0.0, 0.0] for k in range(-(N // 2), N - N // 2)], (0.0, 0.0), 40.0, 0.0

    return best_inner, (0.0, 0.0), best_s, 0.0
