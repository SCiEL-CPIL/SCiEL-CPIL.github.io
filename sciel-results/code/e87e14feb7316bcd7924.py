import time
import random
import math
from collections import defaultdict

def construct():
    start_time = time.time()
    deadline = start_time + 140  # Safety margin (evaluator kills at 150s)
    
    def gcd(a, b):
        while b:
            a, b = b, a % b
        return a

    def line_key(p1, p2):
        x1, y1 = p1
        x2, y2 = p2
        dx = x2 - x1
        dy = y2 - y1
        if dx == 0 and dy == 0:
            return None
        g = gcd(abs(dx), abs(dy))
        dx //= g
        dy //= g
        # Normalize direction to ensure unique key for the line
        if dx < 0 or (dx == 0 and dy < 0):
            dx = -dx
            dy = -dy
        # Equation: dy*x - dx*y = C
        C = dy * x1 - dx * y1
        return (dx, dy, C)

    def in_bounds(p):
        return abs(p[0]) <= 1000 and abs(p[1]) <= 1000

    def random_point():
        return (random.randint(-1000, 1000), random.randint(-1000, 1000))

    def evaluate_full(points):
        """Full evaluation to initialize state. Returns (score, line_counts, line_members)"""
        line_counts = defaultdict(int)
        line_members = defaultdict(set)
        for i in range(16):
            for j in range(i + 1, 16):
                key = line_key(points[i], points[j])
                if key:
                    line_counts[key] += 1
                    line_members[key].add(i)
                    line_members[key].add(j)
        
        score = 0
        for count in line_counts.values():
            if count == 4:
                score += 1
            elif count >= 5:
                return -1, {}, {} # Invalid
        
        return score, line_counts, line_members

    def simulate_annealing(points, max_time=130):
        # Initialize state
        score, line_counts, line_members = evaluate_full(points)
        if score < 0:
            # Try to fix or restart
            return points, 0
        
        best_score = score
        best_points = [list(p) for p in points]
        current_points = [list(p) for p in points]
        
        # Temperature parameters
        T = 50.0
        cooling = 0.9999
        
        # Pre-compute all possible deltas for small moves? 
        # No, just generate on the fly.
        
        iterations = 0
        while time.time() < start_time + max_time:
            iterations += 1
            if iterations % 10000 == 0:
                T *= cooling
            
            idx = random.randint(0, 15)
            old_p = tuple(current_points[idx])
            
            # Generate new point
            # Strategy: 
            # 1. 50%: Small random move
            # 2. 30%: Move along a line defined by two other points
            # 3. 20%: Large random jump
            
            r = random.random()
            if r < 0.5:
                dx = random.randint(-10, 10)
                dy = random.randint(-10, 10)
                new_p = (old_p[0] + dx, old_p[1] + dy)
            elif r < 0.8:
                # Pick two other points to define a line, then pick a point on that line
                j = random.randint(0, 15)
                k = random.randint(0, 15)
                if j == idx or k == idx or j == k:
                    new_p = random_point()
                else:
                    p_j = current_points[j]
                    p_k = current_points[k]
                    # Line through j and k
                    # Find an integer point on this line, not j, not k, not idx
                    # Parametric: P = P_j + t * (P_k - P_j)
                    dx = p_k[0] - p_j[0]
                    dy = p_k[1] - p_j[1]
                    g = gcd(abs(dx), abs(dy))
                    if g == 0:
                        new_p = random_point()
                    else:
                        dx //= g
                        dy //= g
                        # Pick a random t
                        # t can be any integer. 
                        # We want new_p to be distinct from existing points.
                        # Let's try a few random t values
                        found = False
                        for _ in range(10):
                            t = random.randint(-10, 10)
                            if t == 0: continue
                            cand = (p_j[0] + t * dx, p_j[1] + t * dy)
                            if in_bounds(cand) and cand != old_p and cand not in current_points:
                                new_p = cand
                                found = True
                                break
                        if not found:
                            new_p = random_point()
            else:
                new_p = random_point()
            
            # Check validity of new_p
            if not in_bounds(new_p):
                continue
            if new_p in current_points:
                continue
            
            # Incremental update
            # Remove idx from lines involving old_p
            # Add idx to lines involving new_p
            
            # We need to be careful with the line_counts and line_members dicts.
            # We will simulate the move.
            
            # 1. Identify lines affected by removing old_p
            # Lines are defined by pairs. 
            # The lines that contained idx and old_p are the lines (idx, j) for all j != idx.
            # We need to decrement counts for these lines.
            
            # However, a line might contain idx and multiple other points.
            # The key is determined by the line geometry.
            
            # Let's collect the unique line keys that idx is on with old_p
            old_lines = set()
            for j in range(16):
                if j == idx:
                    continue
                key = line_key(old_p, current_points[j])
                if key:
                    old_lines.add(key)
            
            # Let's collect the unique line keys that idx will be on with new_p
            new_lines = set()
            for j in range(16):
                if j == idx:
                    continue
                key = line_key(new_p, current_points[j])
                if key:
                    new_lines.add(key)
            
            # Calculate delta score and check for 5-collinear
            
            # We need to check if any line count becomes 5 after the move.
            # And calculate change in score (number of lines with count 4).
            
            # A line's count changes only if it is in old_lines or new_lines.
            # Actually, a line might be in both (if new_p, old_p, and some j are collinear? No, line_key is unique for the line).
            # If old_p and new_p are on the same line with j, the key is the same.
            
            # Let's track the counts for the specific lines involved.
            # We need a temporary map of count changes.
            
            # Simpler: 
            # For each line in union(old_lines, new_lines):
            #   old_count = line_counts[line]
            #   new_count = old_count
            #   if line in old_lines: new_count -= 1
            #   if line in new_lines: new_count += 1
            #   if new_count >= 5: Invalid move.
            
            # Score change:
            #   if old_count == 4 and new_count != 4: score -= 1
            #   if old_count != 4 and new_count == 4: score += 1
            
            is_valid = True
            delta_score = 0
            
            lines_to_check = old_lines.union(new_lines)
            for key in lines_to_check:
                old_count = line_counts.get(key, 0)
                new_count = old_count
                if key in old_lines:
                    new_count -= 1
                if key in new_lines:
                    new_count += 1
                
                if new_count >= 5:
                    is_valid = False
                    break
                
                if old_count == 4 and new_count != 4:
                    delta_score -= 1
                elif old_count != 4 and new_count == 4:
                    delta_score += 1
            
            if not is_valid:
                continue
            
            new_score = score + delta_score
            
            # Acceptance criterion (Simulated Annealing)
            if new_score > score:
                accept = True
            else:
                # Delta E is score - new_score (positive if worse)
                delta_e = score - new_score
                if delta_e == 0:
                    accept = True # Accept equal to explore
                else:
                    # Probability of accepting worse move
                    p = math.exp(-delta_e / T)
                    accept = (random.random() < p)
            
            if accept:
                # Update state
                # Apply changes to line_counts and line_members
                for key in lines_to_check:
                    old_count = line_counts.get(key, 0)
                    new_count = old_count
                    if key in old_lines:
                        new_count -= 1
                        if new_count > 0:
                            line_members[key].discard(idx)
                        else:
                            del line_counts[key]
                            del line_members[key]
                    if key in new_lines:
                        new_count += 1
                        if key not in line_counts:
                            line_counts[key] = 0
                            line_members[key] = set()
                        line_counts[key] = new_count
                        line_members[key].add(idx)
                    
                    # If key became 0, clean up (handled above)
                    # Note: If key is in both, we effectively moved idx from old to new on the same line?
                    # No, if key is in both, it means old_p and new_p are on the same line with some j.
                    # Then count stays same. 
                    # The logic above handles this: -1 then +1.
                
                current_points[idx] = new_p
                score = new_score
                
                if score > best_score:
                    best_score = score
                    best_points = [list(p) for p in current_points]
                    
            # End of iteration
            
        return best_points, best_score

    # Initial configurations
    best_overall_score = 0
    best_overall_points = None
    
    # 1. 4x4 Grid
    grid_points = [(x, y) for x in range(4) for y in range(4)]
    p, s = simulate_annealing(grid_points)
    if s > best_overall_score:
        best_overall_score = s
        best_overall_points = p
    
    # 2. Random restarts with SA
    for _ in range(10):
        if time.time() > deadline:
            break
        random_points = [random_point() for _ in range(16)]
        # Ensure distinct
        attempts = 0
        while len(set(random_points)) < 16 and attempts < 100:
            random_points[random.randint(0, 15)] = random_point()
            attempts += 1
        if len(set(random_points)) < 16:
            continue
            
        p, s = simulate_annealing(random_points, max_time=10) # Shorter time per restart
        if s > best_overall_score:
            best_overall_score = s
            best_overall_points = p
            
    # 3. Final polish
    if best_overall_points and time.time() < deadline - 5:
        p, s = simulate_annealing(best_overall_points)
        if s > best_overall_score:
            best_overall_score = s
            best_overall_points = p

    if best_overall_points is None:
        return [[x, y] for x in range(4) for y in range(4)]
    
    return [[x, y] for x, y in best_overall_points]
