<<<<<<< SEARCH
def _generate_candidate(type_id, params=None):
    """
    Generate structured candidates known to have high symmetry.
    """
    if params is None:
        params = {}
        
    points = []
    
    if type_id == 'regular_polygon_concentric':
        # Concentric regular polygons
        # e.g., 1 ring of 15, or 2 rings of 10, 3 rings of 10, etc.
        # We need 30 points total.
        # Strategy: Try rings of size 10, 10, 10 or 15, 15.
        ring_sizes = params.get('ring_sizes', [10, 10, 10])
        radii = params.get('radii', [10, 20, 30])
        
        for r, size in zip(radii, ring_sizes):
            for k in range(size):
                angle = 2 * np.pi * k / size
                x = int(round(r * np.cos(angle)))
                y = int(round(r * np.sin(angle)))
                points.append((x, y))
                
    elif type_id == 'triangular_lattice':
        # Select 30 points from a triangular lattice
        # Triangular lattice points: (i, 0) and (i+0.5, h)
        # Let's use a grid of (i, j) and map to triangular coords
        # Or simpler: generate a large triangular grid and pick the center 30.
        pts = []
        # Generate points in a hexagonal-ish shape
        for x in range(-20, 21):
            for y in range(-20, 21):
                # Check if in triangular lattice?
                # Actually, let's just use a standard grid but shifted
                # or use a specific lattice generator.
                pass
        
        # Simpler: Use a 6x5 grid but with triangular offsets
        # Row 0: y=0, x = -10, -8, -6, -4, -2, 0, 2, 4, 6, 8, 10 (12 pts)
        # Row 1: y=h, x = -9, -7, ... (11 pts)
        # This is getting complex. Let's use a known good lattice:
        # Eisenstein integers lattice points.
        # z = i + j * w where w = (-1 + i*sqrt(3))/2
        # We want integer coordinates in [-30, 30].
        
        lattice_pts = []
        for i in range(-15, 16):
            for j in range(-15, 16):
                # Approximate triangular lattice
                x = i + j // 2
                y = j * 3 // 2 # Scale y to keep aspect ratio reasonable for int grid
                # Better:
                # x = i + j
                # y = j
                # This is just a sheared square grid, which is equivalent to square grid.
                
                # Let's use the standard triangular lattice vectors:
                # v1 = (1, 0)
                # v2 = (0.5, sqrt(3)/2) ~ (0.5, 0.866)
                # To keep integer coords, we can use:
                # P = i * (2, 0) + j * (1, 3)  -- This is a distorted triangle
                # Let's try a subset of the integer grid that forms a nice pattern.
                
                # Actually, the "Regular Polygon" approach is often better for high multiplicity.
                # Let's add a "Hexagonal Lattice" generator.
                pass
        
        # Generate a hexagonal lattice of side length 3 (30 points)
        # Center at (0,0).
        # Rows of length 1, 2, 3, 4, 5, 6, 5, 4, 3, 2, 1 -> Sum = 36 (too many)
        # Rows: 1, 2, 3, 4, 5, 6, 5, 4, 3, 2, 1 is 36.
        # Remove 6 corner points to get 30?
        # Or use a 5x6 grid of triangular lattice?
        
        # Let's use a simpler structured grid:
        # A 5x6 grid where every other row is shifted by 0.5 (scaled to 1).
        # y = 0, 2, 4, 6, 8
        # x = 0, 2, 4, 6, 8, 10
        # Shift row 1, 3 by 1.
        # This creates a "brick wall" or "triangular" feel.
        
        pts = []
        for r in range(5):
            y = (r - 2) * 4 # -8, -4, 0, 4, 8
            for c in range(6):
                x = (c - 2.5) * 4 # -10, -6, -2, 2, 6, 10
                if r % 2 == 1:
                    x += 2 # Shift by 1 unit in x (which is 2 in our scaled space? No, 2/4 = 0.5 shift)
                    # If x step is 4, shift by 2 is half step.
                pts.append((int(x), y))
        points = pts

    elif type_id == 'circle_grid':
        # Points on concentric circles
        # 10 points on r=10, 10 on r=20, 10 on r=30
        for r, count in [(10, 10), (20, 10), (30, 10)]:
            for k in range(count):
                angle = 2 * np.pi * k / count
                x = int(round(r * np.cos(angle)))
                y = int(round(r * np.sin(angle)))
                points.append((x, y))
                
    elif type_id == 'square_grid':
        # Standard 5x6 grid
        pts = []
        for x in range(-10, 11, 4):
            for y in range(-8, 9, 4):
                pts.append((x, y))
        points = pts

    return points
=======
def _generate_candidate(type_id, params=None):
    """
    Generate structured candidates known to have high symmetry.
    """
    if params is None:
        params = {}
        
    points = []
    
    if type_id == 'square_grid':
        # Standard 5x6 grid, centered
        # x: -10, -6, -2, 2, 6, 10
        # y: -8, -4, 0, 4, 8
        xs = [-10, -6, -2, 2, 6, 10]
        ys = [-8, -4, 0, 4, 8]
        for x in xs:
            for y in ys:
                points.append((x, y))
                
    elif type_id == 'hex_grid':
        # Hexagonal lattice subset
        # Generate points (i, j) in a hex grid
        # x = i + j//2, y = j * 3 // 2 ?
        # Let's use a standard hex grid coordinate system
        # Axial coordinates (q, r)
        # x = q + r/2
        # y = r * sqrt(3)/2
        # To make integer, scale by 2:
        # X = 2q + r
        # Y = r * 3 # approximating sqrt(3)*2 ~ 3.46, using 3 is distorted but keeps ints
        # Actually, let's just pick a dense cluster from a "brick" pattern
        pts = []
        # 5 rows
        # Row 0: 6 pts
        # Row 1: 6 pts (shifted)
        # Row 2: 6 pts
        # Row 3: 6 pts (shifted)
        # Row 4: 6 pts
        # Total 30
        for r in range(5):
            y = (r - 2) * 6 # -12, -6, 0, 6, 12
            for c in range(6):
                x = (c - 2.5) * 4 # -10, -6, -2, 2, 6, 10
                if r % 2 == 1:
                    x += 2
                pts.append((int(x), y))
        points = pts

    elif type_id == 'circle_points':
        # Pick 30 points on circles x^2+y^2 <= 900
        # Prioritize points with many symmetries
        all_pts = []
        for x in range(-30, 31):
            for y in range(-30, 31):
                all_pts.append((x, y, x*x + y*y))
        # Sort by distance, then pick?
        # Or just pick points on specific radii
        # r=0: (0,0)
        # r=5: (0,5),(5,0)... 8 pts
        # r=10: (0,10),(10,0)... 8 pts
        # r=15: (0,15),(15,0)... 8 pts
        # r=20: (0,20),(20,0)... 8 pts
        # Total 33. Remove 3.
        pts = []
        # Add (0,0)
        pts.append((0,0))
        # Add 8 pts on r=5
        pts.extend([(5,0), (0,5), (-5,0), (0,-5), (3,4), (4,3), (-3,-4), (-4,-3)])
        # Add 8 pts on r=10
        pts.extend([(10,0), (0,10), (-10,0), (0,-10), (6,8), (8,6), (-6,-8), (-8,-6)])
        # Add 8 pts on r=15
        pts.extend([(15,0), (0,15), (-15,0), (0,-15), (9,12), (12,9), (-9,-12), (-12,-9)])
        # Add 7 pts on r=20 (remove one)
        pts.extend([(20,0), (0,20), (-20,0), (0,-20), (12,16), (16,12), (-12,-16)])
        points = pts

    elif type_id == 'random':
        pts = []
        while len(pts) < 30:
            x = random.randint(-30, 30)
            y = random.randint(-30, 30)
            if (x, y) not in pts:
                pts.append((x, y))
        points = pts

    return points
>>>>>>> REPLACE

