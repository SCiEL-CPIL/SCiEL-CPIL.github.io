"""Construct 32 unit pennies on the triangular lattice maximizing unit-distance pairs."""
import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 140  # Safety margin before 150s kill

    # Distance squared in axial coordinates (a, b) for omega = exp(i*pi/3):
    # point = a + b*omega, |omega|=1, Re(omega)=1/2
    # |a + b*omega|^2 = a^2 + b^2 + a*b
    # Two points (a1,b1) and (a2,b2): diff = (da, db)
    # dist_sq = da^2 + db^2 + da*db
    # Unit distance: dist_sq = 1
    # Minimum distance >= 1: dist_sq >= 1 (since dist_sq is integer for integer coords)
    # Actually dist_sq can be 0 (same point) or >= 1. We need distinct points so dist_sq >= 1.
    # Wait: dist_sq = da^2 + db^2 + da*db. For integer da, db this is always a non-negative integer.
    # dist_sq = 0 iff da=0, db=0. Otherwise dist_sq >= 1.
    # So "mutual distance at least one" means all pairs are distinct (which we ensure) since
    # the minimum non-zero squared distance is 1.
    
    # The six unit neighbors of (a,b) are:
    # (a+1, b), (a-1, b), (a, b+1), (a, b-1), (a+1, b-1), (a-1, b+1)
    # Check: (1,0): 1+0+0=1 ✓, (0,1): 0+1+0=1 ✓, (1,-1): 1+1-1=1 ✓
    # (-1,0): 1+0+0=1 ✓, (0,-1): 0+1+0=1 ✓, (-1,1): 1+1-1=1 ✓

    def dist_sq(p1, p2):
        da = p1[0] - p2[0]
        db = p1[1] - p2[1]
        return da*da + db*db + da*db

    def count_unit_edges(points):
        """Count pairs at distance exactly 1."""
        n = len(points)
        count = 0
        for i in range(n):
            for j in range(i+1, n):
                if dist_sq(points[i], points[j]) == 1:
                    count += 1
        return count

    def is_valid(points):
        """Check all pairs have distance >= 1 (i.e., distinct points)."""
        n = len(points)
        for i in range(n):
            for j in range(i+1, n):
                if dist_sq(points[i], points[j]) == 0:
                    return False
        return True

    def check_bounds(points):
        for p in points:
            if abs(p[0]) > 64 or abs(p[1]) > 64:
                return False
        return True

    # Strategy: Use a spiral/hexagonal packing and then optimize with local search.
    # Harborth's theorem: optimum is 76 for n=32.
    # A compact hexagonal cluster of 32 points should be close to optimal.
    
    # Generate a hexagonal cluster of 32 points.
    # Hexagonal numbers: 1, 7, 19, 37, ...
    # 32 is between 19 (radius 2) and 37 (radius 3).
    # Start with radius 2 hexagon (19 points) and add 13 more from radius 3 ring.
    
    def hex_points(R):
        """Generate hexagonal cluster of radius R (max(|a|,|b|,|a+b|) <= R)."""
        pts = []
        for a in range(-R, R+1):
            for b in range(-R, R+1):
                if max(abs(a), abs(b), abs(a+b)) <= R:
                    pts.append((a, b))
        return pts

    def hex_ring(R):
        """Generate the ring at distance exactly R from center."""
        all_R = set(hex_points(R))
        all_Rm1 = set(hex_points(R-1)) if R > 0 else set()
        return list(all_R - all_Rm1)

    # Build initial solution: 19 points from R=2, plus 13 from R=3 ring
    base = hex_points(2)  # 19 points
    ring3 = hex_ring(3)   # 18 points in the ring
    
    # We need 13 more points. Choose them to maximize edges.
    # Use greedy: add points that maximize new edges while maintaining validity.
    
    current = list(base)
    remaining = list(ring3)
    
    # Also consider points from a larger area if needed
    # For now, try to pick 13 from ring3 greedily
    random.seed(42)
    random.shuffle(remaining)
    
    for pt in remaining:
        if len(current) >= 32:
            break
        # Check if adding pt keeps all distances >= 1 (it will since it's distinct)
        current.append(pt)
    
    if len(current) < 32:
        # Need more points - expand to larger area
        # Generate a larger set of candidate points
        candidates = hex_points(5)
        candidates = [p for p in candidates if p not in set(current)]
        random.shuffle(candidates)
        for pt in candidates:
            if len(current) >= 32:
                break
            current.append(pt)
    
    # Verify we have exactly 32 distinct points
    current = list(set(current))
    if len(current) != 32:
        # Fallback: use a simple strip
        current = [(a, b) for a in range(16) for b in range(2)]
    
    # Local search to maximize unit-distance edges
    def local_search(points, max_iters=50000):
        """Try to improve by moving points to nearby positions."""
        current = list(points)
        best_score = count_unit_edges(current)
        
        # Precompute neighbors for fast distance checks
        # For each point, try moving it to nearby lattice positions
        moves = [(0,0), (1,0), (-1,0), (0,1), (0,-1), (1,-1), (-1,1),
                 (2,0), (-2,0), (0,2), (0,-2), (2,-2), (-2,2),
                 (1,1), (-1,-1), (2,-1), (-2,1), (1,-2), (-1,2)]
        
        for iteration in range(max_iters):
            if time.time() > deadline - 5:
                break
            
            improved = False
            # Try moving each point
            for i in range(32):
                orig = current[i]
                for da, db in moves:
                    new_pt = (orig[0] + da, orig[1] + db)
                    if new_pt == orig:
                        continue
                    if abs(new_pt[0]) > 64 or abs(new_pt[1]) > 64:
                        continue
                    
                    # Check if new_pt is already used by another point
                    if new_pt in current and new_pt != orig:
                        continue
                    
                    # Tentatively move
                    current[i] = new_pt
                    
                    # Check validity: all pairs distinct
                    valid = True
                    for j in range(32):
                        if j == i:
                            continue
                        if dist_sq(new_pt, current[j]) == 0:
                            valid = False
                            break
                    
                    if not valid:
                        current[i] = orig
                        continue
                    
                    new_score = count_unit_edges(current)
                    if new_score > best_score:
                        best_score = new_score
                        improved = True
                    else:
                        current[i] = orig
            
            if not improved:
                break
        
        return current, best_score
    
    # Run local search
    current, score = local_search(current)
    
    # Try multiple restarts with different initial configurations
    best_solution = current
    best_score = score
    
    for restart in range(20):
        if time.time() > deadline - 10:
            break
        
        # Create a perturbed version
        perturbed = list(best_solution)
        for _ in range(5):
            idx = random.randint(0, 31)
            da = random.choice([-2,-1,0,1,2])
            db = random.choice([-2,-1,0,1,2])
            new_pt = (perturbed[idx][0]+da, perturbed[idx][1]+db)
            if abs(new_pt[0]) <= 64 and abs(new_pt[1]) <= 64:
                if new_pt not in perturbed:
                    perturbed[idx] = new_pt
        
        # Validate
        if len(set(perturbed)) != 32:
            continue
        if not check_bounds(perturbed):
            continue
        
        improved, new_score = local_search(perturbed, max_iters=20000)
        if new_score > best_score:
            best_score = new_score
            best_solution = improved
    
    # Final verification
    if len(set(best_solution)) != 32:
        best_solution = [(a, b) for a in range(16) for b in range(2)]
    if not check_bounds(best_solution):
        best_solution = [(a, b) for a in range(16) for b in range(2)]
    
    return [[a, b] for a, b in best_solution]
