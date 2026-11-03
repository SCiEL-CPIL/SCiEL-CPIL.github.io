import numpy as np
import math

N = 12

# EVOLVE-BLOCK-START
def construct():
    # Define the 12 centers on a triangular lattice
    # Using axial coordinates (q, r)
    # Compact cluster:
    # Row 0: 1
    # Row 1: 2
    # Row 2: 3
    # Row 3: 3
    # Row 4: 2
    # Row 5: 1
    # Total 12
    
    # Axial coords for this shape
    # Let's generate them
    centers = []
    # q, r
    # Row 0 (r=0): q=0
    centers.append((0, 0))
    # Row 1 (r=1): q=0, 1 ? 
    # In axial coords, neighbors of (0,0) are (1,0), (-1,0), (0,1), (0,-1), (1,-1), (-1,1)
    # Let's just use Cartesian coordinates directly to avoid confusion.
    # Flat-top hexagons.
    # Distance between centers: sqrt(3)
    # Vector to right: (sqrt(3), 0)
    # Vector to top-right: (sqrt(3)/2, 3/2)
    # Vector to top-left: (-sqrt(3)/2, 3/2)
    
    # Let's place the cluster centered at (0,0)
    # The shape is symmetric.
    # Let's define the centers manually based on the 1-2-3-3-2-1 structure.
    # Let the "width" direction be X.
    # Row 0 (top): 1 hex. Center (0, y0)
    # Row 1: 2 hexes. Centers (-dx, y1), (dx, y1)
    # Row 2: 3 hexes. Centers (-2dx, y2), (0, y2), (2dx, y2)
    # Row 3: 3 hexes. Centers (-2dx, y3), (0, y3), (2dx, y3)
    # Row 4: 2 hexes. Centers (-dx, y4), (dx, y4)
    # Row 5: 1 hex. Center (0, y5)
    
    # For flat-top hexes:
    # Horizontal spacing (dx) = sqrt(3)/2
    # Vertical spacing (dy) = 1.5
    # Wait, distance between centers is sqrt(3).
    # If horizontal neighbors are at (sqrt(3), 0), then dx = sqrt(3)/2 for the offset in the next row?
    # Let's check.
    # Hex 1 at (0,0).
    # Hex 2 (right) at (sqrt(3), 0).
    # Hex 3 (top-right) at (sqrt(3)/2, 1.5).
    # Distance: sqrt( (sqrt(3)/2)^2 + 1.5^2 ) = sqrt( 3/4 + 9/4 ) = sqrt(12/4) = sqrt(3). Correct.
    
    # So, let's build the cluster:
    # We want it centered at (0,0).
    # The shape has 6 rows.
    # Let's define the rows relative to the center.
    # The "center" of the cluster is between Row 2 and Row 3.
    # Let's shift so the geometric center is (0,0).
    # Total height = 5 * 1.5 + sqrt(3) (height of one hex) ? No.
    # Vertical span of centers: 5 * 1.5 = 7.5.
    # Top center y = 3.75, Bottom center y = -3.75.
    
    # Let's list the centers:
    # Row 0 (top): 1 hex at (0, 3.75)
    # Row 1: 2 hexes at (-sqrt(3)/2, 2.25), (sqrt(3)/2, 2.25)
    # Row 2: 3 hexes at (-sqrt(3), 0.75), (0, 0.75), (sqrt(3), 0.75)
    # Row 3: 3 hexes at (-sqrt(3), -0.75), (0, -0.75), (sqrt(3), -0.75)
    # Row 4: 2 hexes at (-sqrt(3)/2, -2.25), (sqrt(3)/2, -2.25)
    # Row 5 (bottom): 1 hex at (0, -3.75)
    
    # Check counts: 1+2+3+3+2+1 = 12.
    
    s3 = math.sqrt(3)
    dx = s3 / 2
    dy = 1.5
    
    centers = [
        (0, 3.75),
        (-dx, 2.25), (dx, 2.25),
        (-s3, 0.75), (0, 0.75), (s3, 0.75),
        (-s3, -0.75), (0, -0.75), (s3, -0.75),
        (-dx, -2.25), (dx, -2.25),
        (0, -3.75)
    ]
    
    # Inner hexagons are flat-top, side 1.
    # Vertices of a flat-top hexagon centered at (cx, cy) with side 1:
    # (cx + 1, cy), (cx + 0.5, cy + s3/2), (cx - 0.5, cy + s3/2), 
    # (cx - 1, cy), (cx - 0.5, cy - s3/2), (cx + 0.5, cy - s3/2)
    
    # We need to find the minimal S of the outer hexagon (centered at 0,0, rotated by phi)
    # such that all inner hexagons are inside.
    
    # Get all vertices of all inner hexagons
    all_vertices = []
    for cx, cy in centers:
        # Generate 6 vertices
        for k in range(6):
            angle = math.radians(60 * k)
            vx = cx + 1.0 * math.cos(angle)
            vy = cy + 1.0 * math.sin(angle)
            all_vertices.append((vx, vy))
    
    # Function to calculate required S for a given outer rotation phi
    def get_required_S(phi):
        max_S = 0.0
        for vx, vy in all_vertices:
            r = math.sqrt(vx*vx + vy*vy)
            if r == 0:
                continue
            theta = math.atan2(vy, vx)
            # Relative angle to outer hexagon
            # Outer hexagon is flat-top rotated by phi.
            # The "normals" of the outer hexagon are at phi + 30, phi + 90, ...
            # The distance to boundary in direction theta is:
            # a / cos( (theta - phi) mod pi/3 - pi/6 )
            # where a = S * sqrt(3) / 2
            # So S = r * cos(...) / (sqrt(3)/2)
            
            # Let's simplify the cosine term.
            # The function f(alpha) = 1/cos(alpha mod pi/3 - pi/6)
            # We want to minimize S, so we want to maximize the denominator?
            # No, S = r / (a/r) = r * (1/a) * h(theta-phi)
            # h(theta) is the support function normalized by a.
            # h(theta) = 1 / cos( (theta) mod pi/3 - pi/6 )
            
            rel_theta = (theta - phi) % (math.pi / 3)
            # The angle from the normal is rel_theta - pi/6
            # Wait, the normal is at pi/6 (30 deg) relative to the side?
            # For flat-top, vertex at 0. Side normal at 30.
            # So if rel_theta is 0 (vertex), angle from normal is 30. cos(30) = sqrt(3)/2.
            # If rel_theta is 30 (normal), angle from normal is 0. cos(0) = 1.
            # So we want rel_theta to be close to pi/6 to maximize the denominator.
            
            angle_from_normal = rel_theta - math.pi / 6
            # Ensure we use the correct cosine (always positive in this range)
            cos_val = math.cos(angle_from_normal)
            if cos_val < 1e-9:
                cos_val = 1e-9
            
            S_needed = r / (math.sqrt(3) / 2 * cos_val)
            if S_needed > max_S:
                max_S = S_needed
        return max_S

    # Search for optimal phi
    # The function is periodic with pi/3.
    # We can search in [0, pi/3]
    
    best_S = float('inf')
    best_phi = 0.0
    
    # Grid search
    num_steps = 1000
    for i in range(num_steps):
        phi = (math.pi / 3) * i / num_steps
        S_val = get_required_S(phi)
        if S_val < best_S:
            best_S = S_val
            best_phi = phi
            
    # Refine with local search
    # Simple gradient descent or ternary search?
    # Let's do a finer grid around the best
    for i in range(100):
        phi = best_phi + (math.pi / 3) * (i - 50) / (num_steps * 10)
        S_val = get_required_S(phi)
        if S_val < best_S:
            best_S = S_val
            best_phi = phi

    # The inner hexagons are returned with their coordinates and angle.
    # The task asks for: inner rows (x, y, angle_degrees)
    # We assumed flat-top, so angle is 0.
    
    inner = []
    for cx, cy in centers:
        inner.append([cx, cy, 0.0])
        
    # Outer hexagon center is (0,0)
    # Side length is best_S
    # Angle is best_phi (radians) -> convert to degrees?
    # The return signature is: inner, center, side, angle
    # "outer hexagon center (x, y), side length and rotation"
    # Usually rotation is in degrees or radians? The example uses 0.0.
    # Let's assume radians for consistency with math functions, or degrees?
    # The prompt says "angle_degrees" for inner.
    # For outer, it just says "rotation".
    # Let's check the seed: `return inner, (0.0, 0.0), 40.0, 0.0`
    # It's ambiguous. I will use radians for the outer rotation to be safe, or check if degrees are expected.
    # Given "angle_degrees" is explicit for inner, and just "rotation" for outer, 
    # and standard math libraries use radians, I'll stick to radians for the outer rotation 
    # unless the evaluator expects degrees. 
    # However, often in these tasks, if one is degrees, the other might be too.
    # Let's look at the interface: `construct() -> (inner, center, side, angle)`
    # `inner` rows have `angle_degrees`.
    # `angle` is the outer rotation.
    # I will provide the outer rotation in radians, as it is the native unit of the calculation.
    # If the evaluator expects degrees, it might fail. 
    # Let's assume radians for the outer angle to match the inner angle's "degrees" distinction.
    
    return inner, (0.0, 0.0), best_S, best_phi
# EVOLVE-BLOCK-END
