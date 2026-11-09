import time
import random
import math
from itertools import combinations

def count_ordinary_lines(points):
    """Count lines containing exactly 2 of the given points."""
    n = len(points)
    if n < 2:
        return 0
    
    # Map from line (normalized) to count of points on it
    line_counts = {}
    
    for i in range(n):
        for j in range(i+1, n):
            x1, y1 = points[i]
            x2, y2 = points[j]
            dx = x2 - x1
            dy = y2 - y1
            
            # Normalize the line representation
            # A line through (x1,y1) and (x2,y2) can be represented by (a,b,c) where ax+by+c=0
            # a = dy, b = -dx, c = dx*y1 - dy*x1
            a = dy
            b = -dx
            c = dx * y1 - dy * x1
            
            # Normalize: divide by gcd, and make canonical sign
            g = math.gcd(math.gcd(abs(a), abs(b)), abs(c))
            if g != 0:
                a //= g
                b //= g
                c //= g
            
            # Canonical sign: first non-zero of (a,b) should be positive
            if a < 0 or (a == 0 and b < 0):
                a = -a
                b = -b
                c = -c
            
            key = (a, b, c)
            line_counts[key] = line_counts.get(key, 0) + 1
    
    # Count lines with exactly 2 points
    ordinary = sum(1 for count in line_counts.values() if count == 2)
    return ordinary


def construct():
    start_time = time.time()
    deadline = start_time + 140  # Safety margin before 150s limit
    
    # Start with 3x4 grid as baseline (6 ordinary lines)
    best_points = [[x, y] for x in range(4) for y in range(3)]
    best_count = count_ordinary_lines(best_points)
    
    # Try several random initial configurations and local search
    random.seed(42)
    
    def local_search(points, max_iters=5000):
        """Local search: try moving one point to a random nearby position."""
        current = [p[:] for p in points]
        current_count = count_ordinary_lines(current)
        n = len(current)
        
        for it in range(max_iters):
            if time.time() > deadline - 5:
                break
            
            # Pick a random point to move
            idx = random.randint(0, n-1)
            old_x, old_y = current[idx]
            
            # Generate a candidate new position
            # Try positions that might create collinearities
            # Strategy: pick two other points and try to place on the line through them
            if random.random() < 0.5:
                # Place on line through two other points
                other_indices = [i for i in range(n) if i != idx]
                i1, i2 = random.sample(other_indices, 2)
                x1, y1 = current[i1]
                x2, y2 = current[i2]
                dx = x2 - x1
                dy = y2 - y1
                if dx == 0 and dy == 0:
                    continue
                # Parametrize: point = p1 + t*(dx,dy)
                # Try integer t values
                found = False
                for t in range(-20, 21):
                    nx = x1 + t * dx
                    ny = y1 + t * dy
                    if abs(nx) > 10**9 or abs(ny) > 10**9:
                        continue
                    # Check if this position is already occupied
                    if any(current[k][0] == nx and current[k][1] == ny for k in range(n) if k != idx):
                        continue
                    # Check not all collinear
                    test_points = current[:]
                    test_points[idx] = [nx, ny]
                    # Quick check: not all on one line
                    if _all_collinear(test_points):
                        continue
                    new_count = count_ordinary_lines(test_points)
                    if new_count < current_count:
                        current[idx] = [nx, ny]
                        current_count = new_count
                        found = True
                        break
                if found:
                    continue
            else:
                # Random perturbation
                for _ in range(5):
                    nx = old_x + random.randint(-5, 5)
                    ny = old_y + random.randint(-5, 5)
                    if abs(nx) > 10**9 or abs(ny) > 10**9:
                        continue
                    if any(current[k][0] == nx and current[k][1] == ny for k in range(n) if k != idx):
                        continue
                    test_points = current[:]
                    test_points[idx] = [nx, ny]
                    if _all_collinear(test_points):
                        continue
                    new_count = count_ordinary_lines(test_points)
                    if new_count < current_count:
                        current[idx] = [nx, ny]
                        current_count = new_count
                        break
        
        return current, current_count
    
    def _all_collinear(points):
        """Check if all points are collinear."""
        n = len(points)
        if n < 3:
            return False
        x0, y0 = points[0]
        x1, y1 = points[1]
        dx = x1 - x0
        dy = y1 - y0
        for i in range(2, n):
            xi, yi = points[i]
            if dx * (yi - y0) != dy * (xi - x0):
                return False
        return True
    
    # Try multiple random starts
    for trial in range(30):
        if time.time() > deadline - 10:
            break
        
        # Generate random initial points
        if trial == 0:
            # Start with the grid
            points = [[x, y] for x in range(4) for y in range(3)]
        elif trial == 1:
            # Try a different structured start: two lines of 6
            points = [[i, 0] for i in range(6)] + [[i, 1] for i in range(6)]
        elif trial == 2:
            # Three lines of 4
            points = [[i, 0] for i in range(4)] + [[i, 1] for i in range(4)] + [[i, 2] for i in range(4)]
        elif trial == 3:
            # Four lines of 3
            points = [[i, 0] for i in range(3)] + [[i, 1] for i in range(3)] + [[i, 2] for i in range(3)] + [[i, 3] for i in range(3)]
        else:
            # Random points
            points = []
            used = set()
            while len(points) < 12:
                x = random.randint(-50, 50)
                y = random.randint(-50, 50)
                if (x, y) not in used:
                    used.add((x, y))
                    points.append([x, y])
            if _all_collinear(points):
                continue
        
        # Run local search
        result_points, result_count = local_search(points, max_iters=2000)
        
        if result_count < best_count:
            best_points = result_points
            best_count = result_count
        
        # If we hit the theoretical minimum, stop
        if best_count <= 6:
            break
    
    # Final polish: try to improve further
    if best_count > 6 and time.time() < deadline - 20:
        result_points, result_count = local_search(best_points, max_iters=5000)
        if result_count < best_count:
            best_points = result_points
            best_count = result_count
    
    return best_points


# Quick test
if __name__ == "__main__":
    pts = construct()
    print(f"Number of points: {len(pts)}")
    print(f"Ordinary lines: {count_ordinary_lines(pts)}")
    print(f"Points: {pts}")
