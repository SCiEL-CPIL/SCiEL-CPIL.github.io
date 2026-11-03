"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
from scipy.optimize import minimize, differential_evolution

N = 12

# EVOLVE-BLOCK-START
def construct():
    # Generate candidate packings of 12 unit hexagons in hexagonal lattice
    # Then find minimum enclosing regular hexagon for each
    
    # Hexagonal lattice basis vectors for unit hexagons (flat-top orientation)
    # For flat-top hexagons: width = sqrt(3), height = 2
    # Lattice vectors: a1 = (sqrt(3), 0), a2 = (sqrt(3)/2, 3/2)
    
    sqrt3 = np.sqrt(3.0)
    a1 = np.array([sqrt3, 0.0])
    a2 = np.array([sqrt3/2.0, 1.5])
    
    # Generate several candidate arrangements
    candidates = []
    
    # Candidate 1: 3x4 block in hex lattice
    # Rows of 4, 3 rows offset
    for row in range(3):
        for col in range(4):
            pos = col * a1 + row * a2
            candidates.append(pos)
    
    # Candidate 2: 4x3 block (rotated)
    candidates2 = []
    for row in range(4):
        for col in range(3):
            pos = col * a1 + row * a2
            candidates2.append(pos)
    
    # Candidate 3: Hexagonal cluster - center + ring + partial second ring
    # Center at origin
    # 6 neighbors at distance sqrt(3)
    # Then add more
    hex_cluster = []
    # Center
    hex_cluster.append(np.array([0.0, 0.0]))
    # 6 adjacent
    for i in range(6):
        angle = i * np.pi / 3.0
        # For flat-top, neighbors are at angles 0, 60, 120, 180, 240, 300
        # Distance sqrt(3) from center
        x = sqrt3 * np.cos(angle)
        y = sqrt3 * np.sin(angle)
        hex_cluster.append(np.array([x, y]))
    # That's 7, need 5 more - add next layer partially
    # Second ring has 12 positions, pick 5
    for i in range(12):
        angle = i * np.pi / 6.0
        # Distance 2*sqrt(3) for second ring? Actually let me use lattice coords
        pass
    
    # Let me use lattice coordinates more carefully
    # In axial coordinates (q, r), position = q*a1 + r*a2
    # For a compact shape, use small q, r ranges
    
    # Let's just try several rectangular blocks in lattice space
    def lattice_to_cartesian(q, r):
        return q * a1 + r * a2
    
    # Try different rectangular regions in (q, r) space
    arrangements = []
    
    # 3x4
    arr1 = [lattice_to_cartesian(q, r) for r in range(3) for q in range(4)]
    arrangements.append(arr1)
    
    # 4x3
    arr2 = [lattice_to_cartesian(q, r) for r in range(4) for q in range(3)]
    arrangements.append(arr2)
    
    # 2x6
    arr3 = [lattice_to_cartesian(q, r) for r in range(2) for q in range(6)]
    arrangements.append(arr3)
    
    # 6x2
    arr4 = [lattice_to_cartesian(q, r) for r in range(6) for q in range(2)]
    arrangements.append(arr4)
    
    # Hexagonal shape: use q,r in a hexagonal pattern
    # Center (0,0), then neighbors
    hex_arr = []
    # (q,r) pairs for a hexagonal cluster of 12
    # Let's enumerate small |q|,|r|,|q+r| 
    for q in range(-2, 3):
        for r in range(-2, 3):
            if abs(q) <= 2 and abs(r) <= 2 and abs(q+r) <= 2:
                hex_arr.append(lattice_to_cartesian(q, r))
    # That gives 19 points (hex of radius 2), take first 12 or a subset
    # Actually let me count: for |q|<=2, |r|<=2, |q+r|<=2
    # q=-2: r=0,1,2 (3)
    # q=-1: r=-1,0,1,2 (4)  
    # q=0: r=-2,-1,0,1,2 (5)
    # q=1: r=-2,-1,0,1 (4)
    # q=2: r=-2,-1,0 (3)
    # Total = 19
    # Take a connected subset of 12
    # Just use all and trim, or pick a specific 12
    if len(hex_arr) >= 12:
        arrangements.append(hex_arr[:12])
    
    # Also try a "bar" shape
    bar = [lattice_to_cartesian(q, 0) for q in range(12)]
    arrangements.append(bar)
    
    # For each arrangement, find minimum enclosing regular hexagon
    def hex_apothem_constraint(points, cx, cy, theta, s):
        """Check if all points are inside regular hexagon with center (cx,cy), 
        side length s, rotation theta. Returns max violation (0 if all inside)."""
        apothem = s * sqrt3 / 2.0
        max_violation = 0.0
        for px, py in points:
            # Rotate point relative to center by -theta
            dx = px - cx
            dy = py - cy
            rx = dx * np.cos(-theta) - dy * np.sin(-theta)
            ry = dx * np.sin(-theta) + dy * np.cos(-theta)
            # For a regular hexagon with flat side at angle 0, the constraint is:
            # |rx| <= apothem and |ry| <= apothem and |rx*0.5 + ry*sqrt(3)/2| <= apothem
            # Actually, the 6 half-plane constraints for a regular hexagon centered at origin
            # with apothem 'a' and one side perpendicular to x-axis:
            # The normals are at angles 0, 60, 120, 180, 240, 300
            # Constraint: dot((rx,ry), (cos(k*60), sin(k*60))) <= a for k=0..5
            for k in range(6):
                angle = k * np.pi / 3.0
                nx = np.cos(angle)
                ny = np.sin(angle)
                val = rx * nx + ry * ny
                if val > apothem:
                    violation = val - apothem
                    if violation > max_violation:
                        max_violation = violation
        return max_violation
    
    def objective(params, points):
        cx, cy, theta, s = params
        # We want to minimize s, but s must be large enough
        # Use a penalty approach: minimize s + penalty * max(0, constraint)
        # Or: minimize s subject to constraint <= 0
        # Let's minimize s, with constraint that all points are inside
        # We'll use: if any point is outside, add large penalty
        apothem = s * sqrt3 / 2.0
        max_violation = 0.0
        for px, py in points:
            dx = px - cx
            dy = py - cy
            rx = dx * np.cos(-theta) - dy * np.sin(-theta)
            ry = dx * np.sin(-theta) + dy * np.cos(-theta)
            for k in range(6):
                angle = k * np.pi / 3.0
                nx = np.cos(angle)
                ny = np.sin(angle)
                val = rx * nx + ry * ny
                if val > apothem:
                    violation = val - apothem
                    if violation > max_violation:
                        max_violation = violation
        # Penalty for violation
        penalty = 100.0 * max_violation
        return s + penalty
    
    best_result = None
    best_side = float('inf')
    
    for arr in arrangements:
        if len(arr) != N:
            continue
        points = np.array(arr)
        
        # Initial guess: center of mass, theta=0, s=large
        cx0 = np.mean(points[:, 0])
        cy0 = np.mean(points[:, 1])
        
        # Try multiple initial thetas
        for theta0 in [0.0, np.pi/6, np.pi/3, np.pi/2, 2*np.pi/3, 5*np.pi/6]:
            s0 = 10.0
            x0 = [cx0, cy0, theta0, s0]
            
            bounds = [(-10, 10), (-10, 10), (-np.pi/2, np.pi/2), (0.5, 20)]
            
            result = minimize(objective, x0, args=(points,), method='Nelder-Mead', 
                            options={'maxiter': 1000, 'xatol': 1e-6, 'fatol': 1e-6})
            
            if result.success or result.fun < best_side:
                cx, cy, theta, s = result.x
                # Verify feasibility
                apothem = s * sqrt3 / 2.0
                feasible = True
                for px, py in points:
                    dx = px - cx
                    dy = py - cy
                    rx = dx * np.cos(-theta) - dy * np.sin(-theta)
                    ry = dx * np.sin(-theta) + dy * np.cos(-theta)
                    for k in range(6):
                        angle = k * np.pi / 3.0
                        nx = np.cos(angle)
                        ny = np.sin(angle)
                        val = rx * nx + ry * ny
                        if val > apothem + 1e-9:
                            feasible = False
                            break
                    if not feasible:
                        break
                
                if feasible and s < best_side:
                    best_side = s
                    best_result = (arr, cx, cy, s, theta)
    
    if best_result is None:
        # Fallback: use the first arrangement with a safe outer hexagon
        arr = arrangements[0]
        points = np.array(arr)
        cx = np.mean(points[:, 0])
        cy = np.mean(points[:, 1])
        # Compute required size
        max_dist = 0
        for px, py in points:
            d = np.sqrt((px-cx)**2 + (py-cy)**2)
            if d > max_dist:
                max_dist = d
        s = max_dist * 2 / sqrt3 + 2  # Add margin
        best_side = s
        best_result = (arr, cx, cy, s, 0.0)
    
    arr, cx, cy, s, theta = best_result
    
    # Convert to required format
    inner = []
    for i in range(N):
        x, y = arr[i]
        # All inner hexagons have angle 0 (aligned with lattice)
        inner.append([float(x), float(y), 0.0])
    
    # Outer hexagon: center (cx, cy), side length s, rotation theta
    # Return angle in degrees
    angle_deg = float(np.degrees(theta))
    
    return inner, (float(cx), float(cy)), float(s), angle_deg
# EVOLVE-BLOCK-END
