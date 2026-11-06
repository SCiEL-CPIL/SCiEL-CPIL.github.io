import time
import itertools
from fractions import Fraction

def construct():
    start_time = time.time()
    DEADLINE = 140.0  # safety margin before 150s kill
    
    # Start with parabola points
    base = [(x, x * x) for x in range(-10, 10)]
    
    # We'll perturb some points to create collinear triples.
    # Strategy: pick pairs of base points, find a third point on the line through them
    # that is close to another base point, then move that base point to the intersection.
    
    # Actually, better approach: 
    # Keep most points on parabola. For selected indices i, j, k, 
    # move point k so that points i, j, k are collinear.
    # The line through base[i] and base[j] has a specific equation.
    # We place point k on that line, but at a different x than base[k].
    
    # Let's try: for pairs (i,j), the line through base[i], base[j] 
    # intersects the vertical line x = base[k][0] at some y.
    # Move base[k] to that y.
    
    # To get disjoint triples, pick non-overlapping index sets.
    
    # Let's manually design a configuration with several collinear triples.
    
    # Points on parabola: index -10 to 9, so indices 0..19 correspond to x=-10..9
    # base[i] = (i-10, (i-10)^2)
    
    # Let me try a different approach: use known constructions.
    # A (k, n)-set or use the "no 4 collinear" constraint carefully.
    
    # Simpler: randomly perturb and check.
    # But let's be more systematic.
    
    # Design: take parabola points. Select 5 disjoint pairs of indices (a,b),
    # and for each, pick a third index c not used, and move point c to lie on line ab.
    
    # Line through (x1,y1) and (x2,y2): 
    # If x1 != x2: y = y1 + (y2-y1)/(x2-x1) * (x - x1)
    # For point at x=xc: yc = y1 + (y2-y1)/(x2-x1) * (xc - x1)
    
    def line_y(x1, y1, x2, y2, xc):
        if x1 == x2:
            return None  # vertical line
        return y1 + Fraction(y2 - y1, x2 - x1) * (xc - x1)
    
    # Try to find good triples where the resulting y is integer and |y| <= 100
    # and doesn't create accidental 4-collinear.
    
    # Let's just try a fixed construction and verify.
    
    # I'll use a heuristic search: start with parabola, greedily add collinear triples.
    
    import random
    random.seed(42)
    
    best_points = None
    best_score = 21
    
    def check_no_4_collinear(points):
        """Check no 4 points are collinear."""
        n = len(points)
        for i in range(n):
            for j in range(i+1, n):
                for k in range(j+1, n):
                    # Check if any other point lies on line ij
                    x1, y1 = points[i]
                    x2, y2 = points[j]
                    x3, y3 = points[k]
                    # Cross product to check collinearity
                    # (x2-x1)(y3-y1) - (y2-y1)(x3-x1) == 0
                    def collinear(p1, p2, p3):
                        return (p2[0]-p1[0])*(p3[1]-p1[1]) == (p2[1]-p1[1])*(p3[0]-p1[0])
                    if collinear(points[i], points[j], points[k]):
                        # Check if a 4th point is also on this line
                        for m in range(n):
                            if m in (i, j, k):
                                continue
                            if collinear(points[i], points[j], points[m]):
                                return False
        return True
    
    def max_no_3_collinear(points):
        """Find largest subset with no three collinear using branch and bound."""
        n = len(points)
        
        # Precompute collinear triples
        triple_set = set()
        for i in range(n):
            for j in range(i+1, n):
                for k in range(j+1, n):
                    if (points[j][0]-points[i][0])*(points[k][1]-points[i][1]) == \
                       (points[j][1]-points[i][1])*(points[k][0]-points[i][0]):
                        triple_set.add((i, j, k))
        
        # Branch and bound
        best = [0]
        
        def backtrack(remaining, current_size, current_set):
            if not remaining:
                if current_size > best[0]:
                    best[0] = current_size
                return
            if current_size + len(remaining) <= best[0]:
                return
            
            # Pick a point from remaining
            idx = remaining[0]
            rest = remaining[1:]
            
            # Option 1: include idx
            # Check if adding idx creates a triple with any two in current_set
            can_add = True
            for a, b in itertools.combinations(current_set, 2):
                if (a, idx, b) in triple_set or (a, b, idx) in triple_set or (idx, a, b) in triple_set:
                    can_add = False
                    break
            if can_add:
                new_current = current_set + [idx]
                # Filter remaining: remove points that form triple with two in new_current
                new_remaining = []
                for r in rest:
                    ok = True
                    for a, b in itertools.combinations(new_current, 2):
                        t = tuple(sorted((a, b, r)))
                        if t in triple_set:
                            ok = False
                            break
                    if ok:
                        new_remaining.append(r)
                backtrack(new_remaining, current_size + 1, new_current)
            
            # Option 2: exclude idx
            backtrack(rest, current_size, current_set)
        
        backtrack(list(range(n)), 0, [])
        return best[0]
    
    # Generate candidate configurations
    # Start with parabola, create collinear triples by moving points
    
    candidates = []
    
    # Candidate 1: original parabola (score 20)
    candidates.append([(x, x*x) for x in range(-10, 10)])
    
    # Candidate 2-?: perturb to create triples
    # For each set of 5 disjoint pairs, move 5 points to form triples
    
    # Let's try specific constructions
    # Triple 1: move point at x=0 to lie on line through x=-10 and x=9
    # Line through (-10,100) and (9,81): slope = (81-100)/(9-(-10)) = -19/19 = -1
    # y = 100 + (-1)(x - (-10)) = 100 - x - 10 = 90 - x
    # At x=0: y=90. So move (0,0) to (0,90)
    
    # Triple 2: move point at x=1 to lie on line through x=-9 and x=8
    # Line through (-9,81) and (8,64): slope = (64-81)/(8-(-9)) = -17/17 = -1
    # y = 81 + (-1)(x+9) = 81 - x - 9 = 72 - x
    # At x=1: y=71. Move (1,1) to (1,71)
    
    # Triple 3: move point at x=2 to lie on line through x=-8 and x=7
    # Line through (-8,64) and (7,49): slope = (49-64)/(7-(-8)) = -15/15 = -1
    # y = 64 + (-1)(x+8) = 64 - x - 8 = 56 - x
    # At x=2: y=54. Move (2,4) to (2,54)
    
    # Triple 4: move point at x=3 to lie on line through x=-7 and x=6
    # Line through (-7,49) and (6,36): slope = (36-49)/(6-(-7)) = -13/13 = -1
    # y = 49 + (-1)(x+7) = 49 - x - 7 = 42 - x
    # At x=3: y=39. Move (3,9) to (3,39)
    
    # Triple 5: move point at x=4 to lie on line through x=-6 and x=5
    # Line through (-6,36) and (5,25): slope = (25-36)/(5-(-6)) = -11/11 = -1
    # y = 36 + (-1)(x+6) = 36 - x - 6 = 30 - x
    # At x=4: y=26. Move (4,16) to (4,26)
    
    # So the configuration is:
    # For x in -10..-6: (x, x^2)
    # For x in -5..-1: (x, x^2)
    # For x in 0..4: (x, 90-x, 71-x, 54-x, 39-x, 26-x) respectively
    # For x in 5..9: (x, x^2)
    
    # Wait, let me recompute. The moved points are at x=0,1,2,3,4.
    # Their new y values: 90, 71, 54, 39, 26
    
    # Let me verify these are all distinct and |y|<=100:
    # x=0: y=90 ✓
    # x=1: y=71 ✓
    # x=2: y=54 ✓
    # x=3: y=39 ✓
    # x=4: y=26 ✓
    
    # All good. Now check for accidental 4-collinear.
    
    cand2 = []
    for x in range(-10, 10):
        if x in (0, 1, 2, 3, 4):
            y_map = {0: 90, 1: 71, 2: 54, 3: 39, 4: 26}
            cand2.append((x, y_map[x]))
        else:
            cand2.append((x, x*x))
    
    candidates.append(cand2)
    
    # Let's also try more triples to reduce score further
    # Triple 6: move x=5 to lie on line through x=-5 and x=6? 
    # Line through (-5,25) and (6,36): slope = (36-25)/(6-(-5)) = 11/11 = 1
    # y = 25 + 1*(x+5) = x + 30
    # At x=5: y=35. But (5,25) is currently on parabola. Move to (5,35).
    
    # Triple 7: move x=6 to lie on line through x=-4 and x=7
    # Line through (-4,16) and (7,49): slope = (49-16)/(7-(-4)) = 33/11 = 3
    # y = 16 + 3*(x+4) = 16 + 3x + 12 = 3x + 28
    # At x=6: y=18+28=46. Move (6,36) to (6,46)
    
    # But wait, if I move x=5 and x=6, I need to check they don't conflict with existing triples.
    
    # Let me build a larger configuration with 7 triples.
    
    cand3 = []
    y_overrides = {0: 90, 1: 71, 2: 54, 3: 39, 4: 26, 5: 35, 6: 46}
    for x in range(-10, 10):
        if x in y_overrides:
            cand3.append((x, y_overrides[x]))
        else:
            cand3.append((x, x*x))
    
    candidates.append(cand3)
    
    # Try even more: add triples for x=7,8,9
    # Triple 8: move x=7 to lie on line through x=-3 and x=8
    # Line through (-3,9) and (8,64): slope = (64-9)/(8-(-3)) = 55/11 = 5
    # y = 9 + 5*(x+3) = 9 + 5x + 15 = 5x + 24
    # At x=7: y=35+24=59. Move (7,49) to (7,59)
    
    # Triple 9: move x=8 to lie on line through x=-2 and x=9
    # Line through (-2,4) and (9,81): slope = (81-4)/(9-(-2)) = 77/11 = 7
    # y = 4 + 7*(x+2) = 4 + 7x + 14 = 7x + 18
    # At x=8: y=56+18=74. Move (8,64) to (8,74)
    
    # Triple 10: move x=9... but x=9 is the last one. Let's see.
    # Actually we have 20 points, x=-10..9. If we move x=0..9, that's 10 points moved.
    # But each "triple" uses 2 unmoved points and 1 moved point.
    # With 10 triples, we'd use 20 points in triples, but each triple shares the moved point.
    # Actually each triple is disjoint in terms of the moved point, but the 2 anchor points are shared.
    
    # Let's just try with x=0..9 moved (10 points), using anchors from the negative side.
    
    cand4 = []
    y_overrides4 = {0: 90, 1: 71, 2: 54, 3: 39, 4: 26, 5: 35, 6: 46, 7: 59, 8: 74}
    # For x=9, let's find a line. Use x=-1 and x=10? But x=10 is out of range.
    # Use x=-1 and some other. Line through (-1,1) and (9,81): but we're moving (9,81).
    # Let's use x=-1 and x=0... but x=0 is moved. 
    # This is getting complex. Let's just use 9 triples (x=0..8 moved).
    
    for x in range(-10, 10):
        if x in y_overrides4:
            cand4.append((x, y_overrides4[x]))
        else:
            cand4.append((x, x*x))
    
    candidates.append(cand4)
    
    # Evaluate all candidates
    valid_candidates = []
    for cand in candidates:
        if len(cand) != 20:
            continue
        if len(set(cand)) != 20:
            continue
        if any(abs(x) > 100 or abs(y) > 100 for x, y in cand):
            continue
        if not check_no_4_collinear(cand):
            continue
        valid_candidates.append(cand)
    
    if not valid_candidates:
        # Fallback: return parabola
        return [[x, x*x] for x in range(-10, 10)]
    
    # Find the one with minimum score
    best_cand = None
    best_sc = 21
    for cand in valid_candidates:
        if time.time() > DEADLINE:
            break
        sc = max_no_3_collinear(cand)
        if sc < best_sc:
            best_sc = sc
            best_cand = cand
    
    if best_cand is None:
        best_cand = valid_candidates[0]
    
    return [[x, y] for x, y in best_cand]
