"""New approach: Points on concentric circles to exploit radial symmetry."""
import math

# EVOLVE-BLOCK-START
def construct():
    points = []
    # Generate points on circles with radii 5, 10, 15
    # x^2 + y^2 = R^2
    radii = [5, 10, 15]
    for r in radii:
        r2 = r * r
        for x in range(-r, r + 1):
            y2 = r2 - x * x
            y = int(round(y2 ** 0.5))
            if y * y == y2:
                points.append((x, y))
                if y != 0:
                    points.append((x, -y))
    
    # We have 36 points total (12 per circle).
    # We need exactly 30.
    # To maintain a consistent selection, we sort by radius and angle.
    
    unique_points = []
    seen = set()
    for p in points:
        if p not in seen:
            seen.add(p)
            unique_points.append(p)
            
    # Sort by radius, then angle (atan2)
    def key_func(p):
        return (p[0]**2 + p[1]**2, math.atan2(p[1], p[0]))
        
    unique_points.sort(key=key_func)
    
    # Return the first 30 points
    return unique_points[:30]
# EVOLVE-BLOCK-END
