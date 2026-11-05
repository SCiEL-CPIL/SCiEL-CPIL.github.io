import time
import random
import itertools

# EVOLVE-BLOCK-START
def _count_lines(points):
    """
    Count the number of lines containing exactly 4 points.
    points: list of [x, y]
    """
    n = len(points)
    if n < 4:
        return 0
    
    # Map point to index
    point_to_idx = {tuple(p): i for i, p in enumerate(points)}
    
    # Count points on each line defined by pair of points
    # Line key: (A, B, C) normalized such that A^2 + B^2 is minimized?
    # Standard line representation: Ax + By + C = 0
    # Normalize: gcd(A, B, C) = 1, and A > 0 or (A == 0 and B > 0) or (A == 0 and B == 0 and C > 0)
    
    line_counts = {}
    
    for i in range(n):
        for j in range(i + 1, n):
            x1, y1 = points[i]
            x2, y2 = points[j]
            
            # Direction vector
            dx = x2 - x1
            dy = y2 - y1
            
            # Normal vector (A, B)
            A = dy
            B = -dx
            
            # C = -(A*x1 + B*y1)
            C = -(A * x1 + B * y1)
            
            # Normalize (A, B, C)
            # Find GCD
            def gcd(a, b):
                while b:
                    a, b = b, a % b
                return abs(a)
            
            g = gcd(gcd(A, B), C)
            if g == 0:
                continue # Should not happen for distinct points
            
            A //= g
            B //= g
            C //= g
            
            # Ensure canonical sign
            if A < 0 or (A == 0 and B < 0) or (A == 0 and B == 0 and C < 0):
                A = -A
                B = -B
                C = -C
            
            key = (A, B, C)
            
            if key not in line_counts:
                line_counts[key] = {i, j}
            else:
                line_counts[key].update({i, j})
                
    count = 0
    for key, pts_in_line in line_counts.items():
        if len(pts_in_line) == 4:
            count += 1
            
    return count

def construct():
    start_time = time.time()
    deadline = start_time + 110.0 # Safety margin
    
    # Initial configuration: 4x4 grid
    # Shift to center to allow movement in all directions
    # Grid: x in [-1, 0, 1, 2]? No, 0..3 is fine, bounds are +/-1000.
    # Let's use a centered grid to be safe for "nearby" moves if we shift.
    # Actually, 0..3 is far from boundaries.
    points = [[x, y] for x in range(4) for y in range(4)]
    
    best_score = _count_lines(points)
    best_points = [p[:] for p in points]
    
    # Local search
    # We try to move one point to a new location
    # To speed up, we can pre-calculate contributions or just recalculate.
    # Recalculating is O(N^2) = 256 ops. Very fast.
    # We can do many iterations.
    
    # Define a neighborhood: move one point by (dx, dy) where dx, dy in [-2, 2]
    # Ensure new point is distinct and within bounds.
    
    no_improvement_count = 0
    max_no_improvement = 500
    
    while time.time() < deadline:
        improved = False
        
        # Shuffle points to avoid bias?
        # Or just iterate through indices.
        for i in range(16):
            if time.time() > deadline:
                break
                
            orig_x, orig_y = points[i]
            
            # Try small moves
            # We want to find a move that increases the score.
            # We can sample a few random moves or try all nearby moves.
            # Nearby moves: dx, dy in [-1, 1] excluding (0,0)
            # Also try some larger jumps to escape local minima?
            # Let's stick to small moves first.
            
            for dx in range(-2, 3):
                for dy in range(-2, 3):
                    if dx == 0 and dy == 0:
                        continue
                    
                    new_x = orig_x + dx
                    new_y = orig_y + dy
                    
                    if new_x < -1000 or new_x > 1000 or new_y < -1000 or new_y > 1000:
                        continue
                    
                    # Check distinctness
                    # If (new_x, new_y) is already in points (except i), skip
                    # Optimization: keep a set of occupied points
                    # For simplicity, check list
                    is_valid = True
                    for k in range(16):
                        if k != i and points[k][0] == new_x and points[k][1] == new_y:
                            is_valid = False
                            break
                    if not is_valid:
                        continue
                    
                    # Try the move
                    points[i] = [new_x, new_y]
                    score = _count_lines(points)
                    
                    if score > best_score:
                        best_score = score
                        best_points = [p[:] for p in points]
                        improved = True
                        no_improvement_count = 0
                    else:
                        # Revert
                        points[i] = [orig_x, orig_y]
                        
        if not improved:
            no_improvement_count += 1
            if no_improvement_count > max_no_improvement:
                # Try a random restart or larger jump
                # Randomly perturb a few points
                if time.time() > deadline - 10.0:
                    break
                
                # Perturb 2 random points significantly
                for _ in range(2):
                    idx = random.randint(0, 15)
                    orig_x, orig_y = points[idx]
                    # Jump to a random nearby spot
                    new_x = orig_x + random.randint(-10, 10)
                    new_y = orig_y + random.randint(-10, 10)
                    if -1000 <= new_x <= 1000 and -1000 <= new_y <= 1000:
                        # Check distinctness
                        valid = True
                        for k in range(16):
                            if k != idx and points[k][0] == new_x and points[k][1] == new_y:
                                valid = False
                                break
                        if valid:
                            points[idx] = [new_x, new_y]
                            score = _count_lines(points)
                            if score > best_score:
                                best_score = score
                                best_points = [p[:] for p in points]
                                no_improvement_count = 0
                # Reset no_improvement count to allow more exploration
                no_improvement_count = 0
        else:
            # If improved, continue local search
            pass
            
    return best_points
# EVOLVE-BLOCK-END
