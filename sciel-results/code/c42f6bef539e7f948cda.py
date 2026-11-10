"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from math import gcd, isqrt
from itertools import combinations

def construct():
    """Search for integral point sets with no three collinear and no four concyclic."""
    t_start = time.time()
    DEADLINE = t_start + 140  # 10s safety margin before 150s kill

    def time_ok():
        return time.time() < DEADLINE

    def is_square(n):
        if n < 0:
            return False
        r = isqrt(n)
        return r * r == n

    def check_collinear(p1, p2, p3):
        """Check if three points are collinear using exact arithmetic."""
        (x1, y1), (x2, y2), (x3, y3) = p1, p2, p3
        return (x2 - x1) * (y3 - y1) - (y2 - y1) * (x3 - x1) == 0

    def check_concyclic(p1, p2, p3, p4):
        """Check if four points are concyclic using exact arithmetic."""
        (x1, y1), (x2, y2), (x3, y3), (x4, y4) = p1, p2, p3, p4
        a1, b1, c1 = x1, y1, x1*x1 + y1*y1
        a2, b2, c2 = x2, y2, x2*x2 + y2*y2
        a3, b3, c3 = x3, y3, x3*x3 + y3*y3
        a4, b4, c4 = x4, y4, x4*x4 + y4*y4
        
        det = (a1 * (b2 * (c3 - c4) - c2 * (b3 - b4) + (b3 * c4 - c3 * b4))
             - b1 * (a2 * (c3 - c4) - c2 * (a3 - a4) + (a3 * c4 - c3 * a4))
             + c1 * (a2 * (b3 - b4) - b2 * (a3 - a4) + (a3 * b4 - b3 * a4))
             - (a2 * (b3 * c4 - c3 * b4) - b2 * (a3 * c4 - c3 * a4) + c2 * (a3 * b4 - b3 * a4)))
        
        return det == 0

    def validate_incremental(points, new_point):
        """Check if new_point can be added to points (all constraints satisfied)."""
        # Check distance to all existing points
        for p in points:
            dx = new_point[0] - p[0]
            dy = new_point[1] - p[1]
            d2 = dx*dx + dy*dy
            if not is_square(d2):
                return False
        
        # Check collinearity with all pairs
        n = len(points)
        for i in range(n):
            for j in range(i+1, n):
                if check_collinear(points[i], points[j], new_point):
                    return False
        
        # Check concyclicity with all triples
        for i in range(n):
            for j in range(i+1, n):
                for k in range(j+1, n):
                    if check_concyclic(points[i], points[j], points[k], new_point):
                        return False
        
        return True

    def generate_pythagorean_candidates(center, max_radius=300):
        """Generate points at integer distance from center."""
        candidates = set()
        for r2 in range(1, max_radius*max_radius + 1):
            r = isqrt(r2)
            if r*r != r2:
                continue
            for dx in range(-r, r+1):
                dy2 = r2 - dx*dx
                if dy2 < 0:
                    continue
                dy = isqrt(dy2)
                if dy*dy != dy2:
                    continue
                for dy_sign in ([1, -1] if dy != 0 else [0]):
                    for dx_sign in ([1, -1] if dx != 0 else [0]):
                        cand = (center[0] + dx_sign*dx, center[1] + dy_sign*dy)
                        if cand != center:
                            candidates.add(cand)
        return candidates

    def generate_multi_constraint_candidates(points, max_radius=200):
        """Generate candidates that are at integer distance from ALL points."""
        if not points:
            return set()
        
        # Start with candidates from the first point
        cands = generate_pythagorean_candidates(points[0], max_radius)
        
        # Intersect with candidates from other points
        for p in points[1:]:
            other_cands = generate_pythagorean_candidates(p, max_radius)
            cands &= other_cands
        
        # Remove existing points
        point_set = set(points)
        cands -= point_set
        
        return cands

    def find_best_candidate(points, point_set, candidates):
        """Find the best candidate to add from a list."""
        for cand in candidates:
            if cand in point_set:
                continue
            if validate_incremental(points, cand):
                return cand
        return None

    def try_add_incremental(points, max_radius=200):
        """Try to add one point to the set using multi-constraint candidates."""
        # First try multi-constraint candidates (most promising)
        multi_cands = generate_multi_constraint_candidates(points, max_radius)
        if multi_cands:
            cand_list = list(multi_cands)
            random.shuffle(cand_list)
            found = find_best_candidate(points, set(points), cand_list)
            if found:
                return found
        
        # Then try single-constraint candidates
        all_cands = set()
        for p in points:
            all_cands.update(generate_pythagorean_candidates(p, max_radius))
        all_cands -= set(points)
        
        if all_cands:
            cand_list = list(all_cands)
            random.shuffle(cand_list)
            found = find_best_candidate(points, set(points), cand_list)
            if found:
                return found
        
        return None

    def search_with_backtracking(initial_points, target_size, max_depth=5):
        """Search for a larger set using backtracking."""
        if len(initial_points) >= target_size:
            return initial_points
        
        # Try to add a point
        found = try_add_incremental(initial_points, max_radius=200)
        if found:
            new_points = initial_points + [found]
            result = search_with_backtracking(new_points, target_size, max_depth)
            if len(result) > len(initial_points):
                return result
        
        # If we can't add, try removing a point and exploring alternatives
        if max_depth > 0 and len(initial_points) > 3:
            for i in range(len(initial_points) - 1, -1, -1):
                if i == 0:
                    break  # Keep at least one point
                reduced = initial_points[:i] + initial_points[i+1:]
                result = search_with_backtracking(reduced, target_size, max_depth - 1)
                if len(result) > len(initial_points):
                    return result
        
        return initial_points

    # Start with known good 5-point set
    best_points = [(0, 0), (-56, 33), (-16, 63), (0, 33), (16, 63)]
    best_size = len(best_points)
    
    random.seed(42)
    
    # Multiple search strategies
    for strategy in range(10):
        if not time_ok():
            break
        
        # Reset to base set for each strategy
        current_points = list(best_points)
        
        # Try incremental addition with different parameters
        max_radius = 150 + strategy * 20
        for attempt in range(5):
            if not time_ok():
                break
            
            found = try_add_incremental(current_points, max_radius)
            if found:
                current_points.append(found)
                if len(current_points) > best_size:
                    best_points = list(current_points)
                    best_size = len(current_points)
                    break
            else:
                # Try backtracking
                if len(current_points) > 3:
                    # Remove a random point and try again
                    idx = random.randint(1, len(current_points) - 1)
                    removed = current_points.pop(idx)
                    found = try_add_incremental(current_points, max_radius)
                    if found:
                        current_points.append(found)
                        if len(current_points) > best_size:
                            best_points = list(current_points)
                            best_size = len(current_points)
                    else:
                        current_points.append(removed)
        
        if best_size >= 8:
            break
    
    # Final validation
    valid = True
    for i in range(len(best_points)):
        for j in range(i+1, len(best_points)):
            dx = best_points[i][0] - best_points[j][0]
            dy = best_points[i][1] - best_points[j][1]
            if not is_square(dx*dx + dy*dy):
                valid = False
                break
        if not valid:
            break
        for k in range(j+1, len(best_points)):
            if check_collinear(best_points[i], best_points[j], best_points[k]):
                valid = False
                break
        if not valid:
            break
    if valid:
        for i in range(len(best_points)):
            for j in range(i+1, len(best_points)):
                for k in range(j+1, len(best_points)):
                    for l in range(k+1, len(best_points)):
                        if check_concyclic(best_points[i], best_points[j], best_points[k], best_points[l]):
                            valid = False
                            break
                    if not valid:
                        break
                if not valid:
                    break
            if not valid:
                break
        if not valid:
            # Fallback to known good set
            best_points = [(0, 0), (-56, 33), (-16, 63), (0, 33), (16, 63)]
    
    # Format the result
    return {'D': 1, 'points': [[[x, 1], [y, 1]] for x, y in best_points]}
# EVOLVE-BLOCK-END
