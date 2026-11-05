"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from collections import defaultdict

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # safety margin before 150s kill
    
    # Start with the 4x4 grid which gives 20 four-point lines
    points = [[x, y] for x in range(4) for y in range(4)]
    
    def count_four_lines(pts):
        """Count lines containing exactly 4 of the points."""
        n = len(pts)
        # Map from line key to set of point indices
        line_map = defaultdict(set)
        for i in range(n):
            for j in range(i+1, n):
                x1, y1 = pts[i]
                x2, y2 = pts[j]
                dx = x2 - x1
                dy = y2 - y1
                # Normalize direction vector
                g = _gcd(abs(dx), abs(dy))
                if g > 0:
                    dx //= g
                    dy //= g
                else:
                    dx, dy = 0, 1
                # Normalize sign
                if dx < 0 or (dx == 0 and dy < 0):
                    dx = -dx
                    dy = -dy
                # Line key: normalized direction + a canonical point on the line
                # Use the point with minimum x, then minimum y as anchor
                # Actually, use (dx, dy, x1*dx - y1*dy) as a line identifier
                # The line equation: dy*x - dx*y = c where c = dy*x1 - dx*y1
                c = dy * x1 - dx * y1
                key = (dx, dy, c)
                line_map[key].add(i)
                line_map[key].add(j)
        
        count = 0
        for key, idx_set in line_map.items():
            if len(idx_set) == 4:
                count += 1
        return count
    
    def _gcd(a, b):
        while b:
            a, b = b, a % b
        return a
    
    def get_line_keys(pts):
        """Get all line keys and their point sets for incremental updates."""
        n = len(pts)
        line_map = defaultdict(set)
        for i in range(n):
            for j in range(i+1, n):
                x1, y1 = pts[i]
                x2, y2 = pts[j]
                dx = x2 - x1
                dy = y2 - y1
                g = _gcd(abs(dx), abs(dy))
                if g > 0:
                    dx //= g
                    dy //= g
                else:
                    dx, dy = 0, 1
                if dx < 0 or (dx == 0 and dy < 0):
                    dx = -dx
                    dy = -dy
                c = dy * x1 - dx * y1
                key = (dx, dy, c)
                line_map[key].add(i)
                line_map[key].add(j)
        return line_map
    
    def is_valid(pts):
        """Check no five collinear and all distinct, |x|,|y| <= 1000."""
        if len(set(tuple(p) for p in pts)) != len(pts):
            return False
        for p in pts:
            if abs(p[0]) > 1000 or abs(p[1]) > 1000:
                return False
        line_map = get_line_keys(pts)
        for key, idx_set in line_map.items():
            if len(idx_set) >= 5:
                return False
        return True
    
    # Initial evaluation
    current_score = count_four_lines(points)
    best_points = [p[:] for p in points]
    best_score = current_score
    
    # Hill climbing with random restarts
    max_iter = 500000
    
    for restart in range(50):
        if time.time() > deadline:
            break
        
        # For first restart, use the grid; for others, perturb
        if restart == 0:
            pts = [p[:] for p in points]
        else:
            # Start from best found so far with some perturbation
            pts = [p[:] for p in best_points]
            # Randomly move a few points
            for _ in range(random.randint(1, 3)):
                idx = random.randint(0, 15)
                pts[idx] = [random.randint(-10, 10), random.randint(-10, 10)]
        
        if not is_valid(pts):
            continue
        
        score = count_four_lines(pts)
        if score > best_score:
            best_score = score
            best_points = [p[:] for p in pts]
        
        # Hill climbing
        for it in range(max_iter):
            if time.time() > deadline:
                break
            
            # Pick a random point to move
            idx = random.randint(0, 15)
            old_pt = pts[idx][:]
            
            # Generate candidate moves
            moved = False
            for _ in range(20):
                # Move to a nearby random position
                new_pt = [old_pt[0] + random.randint(-3, 3), 
                         old_pt[1] + random.randint(-3, 3)]
                if abs(new_pt[0]) > 1000 or abs(new_pt[1]) > 1000:
                    continue
                if new_pt == old_pt:
                    continue
                # Check distinctness
                if any(p == new_pt for p in pts):
                    continue
                
                pts[idx] = new_pt
                
                if not is_valid(pts):
                    pts[idx] = old_pt
                    continue
                
                new_score = count_four_lines(pts)
                if new_score >= score:
                    score = new_score
                    moved = True
                    if score > best_score:
                        best_score = score
                        best_points = [p[:] for p in pts]
                    break
                else:
                    pts[idx] = old_pt
            
            if not moved:
                # Try a larger jump
                new_pt = [random.randint(-15, 15), random.randint(-15, 15)]
                if abs(new_pt[0]) <= 1000 and abs(new_pt[1]) <= 1000:
                    if not any(p == new_pt for p in pts):
                        pts[idx] = new_pt
                        if is_valid(pts):
                            new_score = count_four_lines(pts)
                            if new_score >= score:
                                score = new_score
                                if score > best_score:
                                    best_score = score
                                    best_points = [p[:] for p in pts]
                            else:
                                pts[idx] = old_pt
        
        # Plateau escape: if stuck, do a random restart from best
        if time.time() > deadline:
            break
    
    # Final validation
    if not is_valid(best_points):
        # Fall back to the grid
        best_points = [[x, y] for x in range(4) for y in range(4)]
    
    return best_points
# EVOLVE-BLOCK-END
