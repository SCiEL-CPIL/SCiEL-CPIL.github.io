import time
import random
import itertools
import numpy as np

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 50  # 50s safety margin for 60s limit
    
    # Strategy: 
    # 1. We want to maximize the multiplicity of a single positive doubled area.
    # 2. The seed program (6x5 grid) has area 1 multiplicity: 
    #    In a 6x5 grid, how many unit triangles? 
    #    Actually, let's analyze the seed.
    #    Seed: x in 0..5, y in 0..4.
    #    Area 1 triangles are formed by (x,y), (x+1,y), (x,y+1) etc.
    #    Number of such triangles in a W x H grid is roughly (W-1)*(H-1)*2 + ...
    #    For 6x5 (30 points), W=6, H=5.
    #    Unit squares: 5*4 = 20. Each square gives 2 unit triangles. Total 40.
    #    Plus other configurations?
    #    Let's calculate the score of the seed to know the baseline.
    
    # Let's compute the score of the seed configuration first.
    seed_points = [(x, y) for x in range(6) for y in range(5)]
    seed_score = _calculate_max_multiplicity(seed_points)
    
    best_score = seed_score
    best_points = seed_points
    
    # We will try to improve upon this.
    # A common strategy for "many equal area triangles" is to use a lattice.
    # If we use a denser grid or a different aspect ratio, we might get more unit triangles.
    # Or, we can target a specific area A > 1 if unit triangles are limited by geometry.
    # However, unit triangles are usually the most abundant in small grids.
    
    # Let's try a few different grid dimensions that fit 30 points.
    # 30 = 1*30, 2*15, 3*10, 5*6, 6*5.
    # The seed is 6x5.
    # Let's try 5x6 (transposed, same score).
    # Let's try 3x10.
    # Let's try 2x15.
    # Let's try 1x30 (all collinear, score 0).
    
    # Let's evaluate 3x10 and 2x15.
    for w, h in [(3, 10), (2, 15), (10, 3), (15, 2)]:
        if time.time() > deadline - 5: break
        pts = [(x, y) for x in range(w) for y in range(h)]
        if len(pts) != 30: continue
        score = _calculate_max_multiplicity(pts)
        if score > best_score:
            best_score = score
            best_points = pts
    
    # Now, let's try to improve the best grid found (likely 6x5 or 5x6) 
    # by perturbing points to increase the count of the dominant area.
    # However, perturbing a grid usually *decreases* the count of small integer areas 
    # because it breaks the lattice structure. 
    # So, a pure grid is likely optimal for small integer areas.
    
    # Is there a configuration that yields MORE than the 6x5 grid for area 1?
    # In a 6x5 grid:
    # Points: 30.
    # Pairs with vector (1,0): 5*5 = 25.
    # Pairs with vector (0,1): 6*4 = 24.
    # Pairs with vector (1,1): 5*4 = 20.
    # Pairs with vector (1,-1): 5*4 = 20.
    # 
    # Triangles of area 1 (doubled area 2) correspond to vectors u, v with det(u,v) = 2.
    # Common vectors: (1,0) and (0,2) -> det 2.
    # (1,0) and (1,2) -> det 2.
    # (1,0) and (2,2) -> det 4 (no).
    # (1,0) and (0,1) -> det 1 (area 0.5, doubled 1).
    # 
    # Let's check the seed score calculation.
    # The seed score is likely dominated by doubled area 1 or 2.
    
    # Let's run a quick local search on the best grid to see if we can squeeze more.
    # We will try to move points slightly to create more of the target area.
    # But moving points off the lattice is risky.
    
    # Alternative: Try a different lattice structure.
    # E.g., a hexagonal lattice approximation? 
    # Or a "compressed" grid where some rows are shifted?
    
    # Let's try a heuristic search:
    # Start with best grid.
    # Identify the most frequent doubled area A*.
    # Try to add more triangles of area A* by moving points.
    # This is complex.
    
    # Given the time constraint and the nature of the problem, the grid is often the local optimum.
    # However, the prompt asks for a DIFFERENT strategy.
    # The current program is a static grid.
    # My strategy: Evaluate multiple grid aspect ratios and select the best one.
    # This is a "search over grid dimensions" strategy.
    
    # Let's also try a "random lattice" approach:
    # Generate random integer points in a box, calculate score.
    # This is unlikely to beat a grid, but it's a different strategy.
    # Given 50s, we can do a few hundred random restarts.
    
    # Actually, the most promising "different" strategy is to optimize the grid aspect ratio 
    # and potentially use a "staggered" grid (like a brick wall pattern) which might 
    # increase the number of unit triangles or specific area triangles.
    
    # Let's try staggered grids.
    # Row 0: x = 0..5
    # Row 1: x = 0.5..5.5 -> not integer.
    # So integer staggered grids are tricky.
    # Maybe shift every other row by 1?
    # Row 0: 0..5
    # Row 1: 1..6
    # Row 2: 0..5
    # Row 3: 1..6
    # Row 4: 0..5
    # This is still a subset of the integer lattice, just a different subset.
    # Let's test this "brick wall" pattern.
    
    w, h = 6, 5
    brick_pts = []
    for y in range(h):
        offset = y % 2
        for x in range(w):
            px = x + offset
            py = y
            if 0 <= px <= 30 and 0 <= py <= 30:
                brick_pts.append((px, py))
    # Ensure 30 distinct points
    if len(brick_pts) == 30 and len(set(brick_pts)) == 30:
        score = _calculate_max_multiplicity(brick_pts)
        if score > best_score:
            best_score = score
            best_points = brick_pts
    
    # Let's try a few random perturbations of the best grid found so far.
    # We will pick a random point and move it to a random nearby integer coordinate.
    # If the score improves, keep it.
    
    current_best = list(best_points)
    current_best_score = best_score
    current_best_area = _get_most_frequent_area(current_best)
    
    attempts = 0
    while time.time() < deadline - 2:
        attempts += 1
        # Perturb a random point
        idx = random.randint(0, 29)
        orig_x, orig_y = current_best[idx]
        new_x = orig_x + random.randint(-1, 1)
        new_y = orig_y + random.randint(-1, 1)
        
        if new_x == orig_x and new_y == orig_y:
            continue
            
        if not (0 <= new_x <= 30 and 0 <= new_y <= 30):
            continue
            
        # Check distinctness
        new_set = set(current_best)
        if (new_x, new_y) in new_set:
            continue
            
        new_set.remove((orig_x, orig_y))
        new_set.add((new_x, new_y))
        
        candidate = list(new_set)
        score = _calculate_max_multiplicity(candidate)
        
        if score > current_best_score:
            current_best_score = score
            current_best = candidate
            current_best_area = _get_most_frequent_area(candidate)
            if current_best_score > best_score:
                best_score = current_best_score
                best_points = current_best
    
    return best_points

def _calculate_max_multiplicity(points):
    """Calculate the maximum multiplicity of any positive doubled area."""
    n = len(points)
    if n < 3:
        return 0
        
    # Precompute all pairwise differences to speed up triangle area calculation?
    # Or just iterate through triples.
    # N=30, triples = 4060. This is fast.
    
    areas = {}
    for i in range(n):
        x1, y1 = points[i]
        for j in range(i + 1, n):
            x2, y2 = points[j]
            for k in range(j + 1, n):
                x3, y3 = points[k]
                # Doubled area = |det([x2-x1, y2-y1], [x3-x1, y3-y1])|
                det = abs((x2 - x1) * (y3 - y1) - (y2 - y1) * (x3 - x1))
                if det > 0:
                    areas[det] = areas.get(det, 0) + 1
                    
    if not areas:
        return 0
        
    return max(areas.values())

def _get_most_frequent_area(points):
    """Return the area value with the highest multiplicity."""
    n = len(points)
    areas = {}
    for i in range(n):
        x1, y1 = points[i]
        for j in range(i + 1, n):
            x2, y2 = points[j]
            for k in range(j + 1, n):
                x3, y3 = points[k]
                det = abs((x2 - x1) * (y3 - y1) - (y2 - y1) * (x3 - x1))
                if det > 0:
                    areas[det] = areas.get(det, 0) + 1
                    
    if not areas:
        return 0
    max_mult = max(areas.values())
    for area, mult in areas.items():
        if mult == max_mult:
            return area
    return 0
# EVOLVE-BLOCK-END
