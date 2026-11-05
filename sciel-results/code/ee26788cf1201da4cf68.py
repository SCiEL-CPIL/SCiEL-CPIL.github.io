import time
import random
import math
from collections import defaultdict

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin
    
    # Initial 4x4 grid
    # Shift to center to allow movement in both directions
    grid = [[x, y] for x in range(4) for y in range(4)]
    # Center it
    grid = [[x - 1.5, y - 1.5] for x, y in grid]
    # Scale to use integer space better? No, keep small for now.
    # Actually, just use integers directly.
    init_grid = [[x, y] for x in range(4) for y in range(4)]
    
    def evaluate(points):
        """
        Returns (score, is_valid).
        score: number of lines with exactly 4 points.
        is_valid: True if no line has 5+ points.
        """
        line_points = defaultdict(set)
        n = len(points)
        for i in range(n):
            x1, y1 = points[i]
            for j in range(i + 1, n):
                x2, y2 = points[j]
                dx = x2 - x1
                dy = y2 - y1
                if dx == 0 and dy == 0:
                    continue # Distinct points guaranteed, but just in case
                
                g = math.gcd(abs(dx), abs(dy))
                dx_n = dx // g
                dy_n = dy // g
                
                # Normalize direction
                if dx_n < 0 or (dx_n == 0 and dy_n < 0):
                    dx_n = -dx_n
                    dy_n = -dy_n
                
                # Line equation: dy_n * x - dx_n * y = c
                c = dy_n * x1 - dx_n * y1
                
                key = (dx_n, dy_n, c)
                line_points[key].add(i)
                line_points[key].add(j)
        
        score = 0
        for key, pts in line_points.items():
            if len(pts) == 4:
                score += 1
            elif len(pts) > 4:
                return 0, False # Invalid: 5+ collinear
        
        return score, True

    best_score, _ = evaluate(init_grid)
    best_points = [p[:] for p in init_grid]
    
    # Random restart search
    # We try to improve upon the grid.
    # The grid has score 10.
    
    # Helper to generate a random point within bounds
    def random_point():
        return [random.randint(-1000, 1000), random.randint(-1000, 1000)]
    
    # Helper to check if points are distinct
    def are_distinct(points):
        seen = set()
        for p in points:
            if tuple(p) in seen:
                return False
            seen.add(tuple(p))
        return True

    # Simulated Annealing / Hill Climbing with Restarts
    current_points = [p[:] for p in init_grid]
    current_score, current_valid = evaluate(current_points)
    if not current_valid:
        current_score = -1000
        current_points = [p[:] for p in init_grid]
        current_score, _ = evaluate(current_points)
        
    # If grid is valid, start from there
    if current_valid:
        best_score = current_score
        best_points = [p[:] for p in current_points]
    else:
        # Fallback to a valid random config if grid fails (it shouldn't)
        while True:
            cand = [random_point() for _ in range(16)]
            if are_distinct(cand):
                s, v = evaluate(cand)
                if v:
                    best_score = s
                    best_points = cand
                    break

    # Search loop
    # We will try local moves.
    # Move types:
    # 1. Move one point to a random location.
    # 2. Swap two points.
    
    # To escape local optima, we can use a "kick" strategy:
    # If no improvement for X steps, randomly perturb several points.
    
    no_improve_count = 0
    max_no_improve = 500
    
    while time.time() < deadline:
        # Choose a move
        if random.random() < 0.8:
            # Move one point
            idx = random.randint(0, 15)
            old_p = current_points[idx][:]
            # Try to move to a nearby location first? Or global?
            # Global is better for exploration, but local for exploitation.
            # Let's mix.
            if random.random() < 0.5:
                # Local move: +/- 10
                nx = old_p[0] + random.randint(-10, 10)
                ny = old_p[1] + random.randint(-10, 10)
                if -1000 <= nx <= 1000 and -1000 <= ny <= 1000:
                    current_points[idx] = [nx, ny]
            else:
                # Global move
                current_points[idx] = random_point()
            
            # Check distinctness
            if not are_distinct(current_points):
                current_points[idx] = old_p
                continue
            
            new_score, new_valid = evaluate(current_points)
            if new_valid:
                if new_score > current_score:
                    current_score = new_score
                    no_improve_count = 0
                    if current_score > best_score:
                        best_score = current_score
                        best_points = [p[:] for p in current_points]
                else:
                    # Accept worse moves with some probability? 
                    # For simplicity, pure hill climbing with restarts is safer.
                    # But let's accept slightly worse to escape?
                    # No, let's stick to strict improvement or kick.
                    no_improve_count += 1
                    # Revert if not improved? 
                    # Actually, if we don't revert, we might wander.
                    # Let's revert if not improved to stay on the "ridge".
                    current_points[idx] = old_p
            else:
                # Invalid, revert
                current_points[idx] = old_p
                no_improve_count += 1
                
        else:
            # Swap two points
            i, j = random.sample(range(16), 2)
            current_points[i], current_points[j] = current_points[j][:], current_points[i][:]
            
            if not are_distinct(current_points):
                current_points[i], current_points[j] = current_points[j][:], current_points[i][:]
                continue
            
            new_score, new_valid = evaluate(current_points)
            if new_valid:
                if new_score > current_score:
                    current_score = new_score
                    no_improve_count = 0
                    if current_score > best_score:
                        best_score = current_score
                        best_points = [p[:] for p in current_points]
                else:
                    no_improve_count += 1
                    current_points[i], current_points[j] = current_points[j][:], current_points[i][:]
            else:
                current_points[i], current_points[j] = current_points[j][:], current_points[i][:]
                no_improve_count += 1
        
        # Kick if stuck
        if no_improve_count > max_no_improve:
            # Perturb 3-5 points randomly
            num_perturb = random.randint(3, 5)
            indices = random.sample(range(16), num_perturb)
            for idx in indices:
                current_points[idx] = random_point()
            
            # Ensure distinct
            if not are_distinct(current_points):
                # Revert to best known if invalid
                current_points = [p[:] for p in best_points]
                current_score, _ = evaluate(current_points)
            else:
                new_score, new_valid = evaluate(current_points)
                if new_valid:
                    current_score = new_score
                    if current_score > best_score:
                        best_score = current_score
                        best_points = [p[:] for p in current_points]
                else:
                    current_points = [p[:] for p in best_points]
                    current_score, _ = evaluate(current_points)
            
            no_improve_count = 0

    return best_points
# EVOLVE-BLOCK-END
