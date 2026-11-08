<<<<<<< SEARCH
    # Try different packings
    best_result = None
    best_score = float('inf')
    
    # First, try hexagonal packings of various sizes
    for r in range(0, 5):
        if time.time() > deadline:
            break
        n_hexes = 1 + 3 * r * (r + 1)
        hexes = generate_hexagonal_packing(r)
        
        if len(hexes) != n_hexes:
            continue
        
        # Compute minimum enclosing hexagon
        cx, cy, side, rot = min_enclosing_hexagon(hexes)
        
        if side < best_score:
            best_score = side
            best_result = (hexes, (cx, cy), side, rot)
    
    # Also try the original linear arrangement for comparison
    for n in range(3, 15):
        if time.time() > deadline:
            break
        # Linear arrangement along x-axis
        hexes = [(sqrt3 * k, 0.0, 0.0) for k in range(-(n // 2), n - n // 2)]
        # Actually, for non-overlapping unit hexagons in a row, spacing should be sqrt(3)
        # But they might overlap at boundaries. Let's use spacing that ensures no overlap.
        # For flat-top hexagons side by side, center-to-center = 2 * apothem = sqrt(3)
        # For pointy-top, center-to-center horizontally = 2 * apothem = sqrt(3)
        
        cx, cy, side, rot = min_enclosing_hexagon(hexes)
        
        if side < best_score:
            best_score = side
            best_result = (hexes, (cx, cy), side, rot)
    
    if best_result is None:
        # Fallback
        return [[0.0, 0.0, 0.0]], (0.0, 0.0), 1.0, 0.0
    
    hexes, center, side, rot = best_result
    inner = [[h[0], h[1], h[2]] for h in hexes]
    return inner, center, side, rot
=======
    # Try different packings
    best_result = None
    best_score = float('inf')
    
    # First, try hexagonal packings of various sizes
    # r=0 is 1 hex. r=1 is 7 hexes.
    for r in range(0, 5):
        if time.time() > deadline:
            break
        hexes = generate_hexagonal_packing(r)
        
        # Compute minimum enclosing hexagon
        cx, cy, side, rot = min_enclosing_hexagon(hexes)
        
        if side < best_score:
            best_score = side
            best_result = (hexes, (cx, cy), side, rot)
    
    # Also try the original linear arrangement for comparison
    for n in range(3, 15):
        if time.time() > deadline:
            break
        # Linear arrangement along x-axis
        # k ranges from -n//2 to n//2 (inclusive) to center it
        start_k = -(n // 2)
        end_k = n - n // 2
        hexes = [(sqrt3 * k, 0.0, 0.0) for k in range(start_k, end_k)]
        
        cx, cy, side, rot = min_enclosing_hexagon(hexes)
        
        if side < best_score:
            best_score = side
            best_result = (hexes, (cx, cy), side, rot)
    
    if best_result is None:
        # Fallback
        return [[0.0, 0.0, 0.0]], (0.0, 0.0), 1.0, 0.0
    
    hexes, center, side, rot = best_result
    inner = [[h[0], h[1], h[2]] for h in hexes]
    return inner, center, side, rot
>>>>>>> REPLACE
