import time
import random
import math
from itertools import combinations

def construct():
    start_time = time.time()
    time_limit = 100.0  # Leave margin for 120s limit
    
    def get_line_key(p1, p2):
        x1, y1 = p1
        x2, y2 = p2
        A = y1 - y2
        B = x2 - x1
        C = x1 * y2 - x2 * y1
        
        if A == 0 and B == 0:
            return None # Should not happen for distinct points
            
        g = math.gcd(abs(A), math.gcd(abs(B), abs(C)))
        A //= g
        B //= g
        C //= g
        
        # Canonical sign
        if A < 0:
            A = -A
            B = -B
            C = -C
        elif A == 0 and B < 0:
            B = -B
            C = -C
            
        return (A, B, C)

    def evaluate(points):
        # points is a list of 16 tuples (x, y)
        # Returns (score, num_5_plus)
        # We want to maximize score, minimize num_5_plus.
        # Actually, if num_5_plus > 0, it's invalid.
        # Score = count of lines with exactly 4 points.
        
        line_counts = {}
        
        # Generate all unique lines defined by pairs
        # Using a set to avoid duplicate line keys
        unique_lines = set()
        for i in range(16):
            for j in range(i+1, 16):
                key = get_line_key(points[i], points[j])
                if key:
                    unique_lines.add(key)
        
        num_4 = 0
        num_5_plus = 0
        
        for key in unique_lines:
            A, B, C = key
            count = 0
            for p in points:
                if A * p[0] + B * p[1] == C:
                    count += 1
            if count == 4:
                num_4 += 1
            elif count >= 5:
                num_5_plus += 1
        
        return num_4, num_5_plus

    def is_valid(points):
        # Check bounds and distinctness
        for p in points:
            if not (-1000 <= p[0] <= 1000 and -1000 <= p[1] <= 1000):
                return False
        if len(set(points)) != 16:
            return False
        return True

    def evaluate_incremental(points, line_index, idx, old_p, new_p):
        """
        Efficiently evaluate the change in score when moving point idx from old_p to new_p.
        line_index: dict mapping line_key -> set of point indices on that line
        Returns (new_score, new_num_5_plus, new_line_index)
        """
        # Copy line_index
        new_line_index = {k: set(v) for k, v in line_index.items()}
        
        # Remove old_p from all lines it was on
        for i in range(16):
            if i == idx:
                continue
            key = get_line_key(old_p, points[i])
            if key in new_line_index:
                new_line_index[key].discard(idx)
        
        # Add new_p to all lines it's on
        for i in range(16):
            if i == idx:
                continue
            key = get_line_key(new_p, points[i])
            if key:
                if key not in new_line_index:
                    new_line_index[key] = set()
                new_line_index[key].add(idx)
        
        # Also need to handle lines where new_p is the only point or pairs with idx
        # Actually, we need to rebuild lines involving idx
        # Let's just recount all lines for simplicity in the incremental approach
        
        # Actually, let's do a full recount but using the line_index to be faster
        num_4 = 0
        num_5_plus = 0
        
        for key, pts_in_line in new_line_index.items():
            if len(pts_in_line) >= 2:
                count = len(pts_in_line)
                if count == 4:
                    num_4 += 1
                elif count >= 5:
                    num_5_plus += 1
        
        return num_4, num_5_plus, new_line_index

    def build_line_index(points):
        """Build a line index: dict mapping line_key -> set of point indices"""
        line_index = {}
        for i in range(16):
            for j in range(i+1, 16):
                key = get_line_key(points[i], points[j])
                if key:
                    if key not in line_index:
                        line_index[key] = set()
                    line_index[key].add(i)
                    line_index[key].add(j)
        return line_index

    best_points = None
    best_score = -1
    
    # Initial configuration: 4x4 grid (affine plane of order 4)
    initial_points = [(i, j) for i in range(4) for j in range(4)]
    current_points = initial_points[:]
    
    # Evaluate initial
    curr_score, curr_inv = evaluate(current_points)
    if curr_inv == 0:
        best_score = curr_score
        best_points = current_points[:]
    
    # Build initial line index
    line_index = build_line_index(current_points)
    
    # Local Search with incremental evaluation
    no_improve = 0
    max_steps = 10000
    
    for step in range(max_steps):
        if time.time() - start_time > time_limit:
            break
            
        idx = random.randint(0, 15)
        old_x, old_y = current_points[idx]
        
        # Generate candidate moves - try small moves first
        moves = []
        for dx in range(-5, 6):
            for dy in range(-5, 6):
                if dx == 0 and dy == 0:
                    continue
                new_p = (old_x + dx, old_y + dy)
                if -1000 <= new_p[0] <= 1000 and -1000 <= new_p[1] <= 1000:
                    if new_p not in current_points:
                        moves.append(new_p)
        
        random.shuffle(moves)
        
        improved = False
        for new_p in moves:
            # Try the move
            current_points[idx] = new_p
            new_score, new_inv, new_line_index = evaluate_incremental(
                current_points, line_index, idx, (old_x, old_y), new_p
            )
            
            # Acceptance criterion
            accept = False
            
            if curr_inv > 0 and new_inv == 0:
                # Became valid
                accept = True
            elif curr_inv == 0 and new_inv == 0:
                # Both valid
                if new_score > curr_score:
                    accept = True
                elif new_score == curr_score and random.random() < 0.1:
                    # Plateau move
                    accept = True
            elif curr_inv > 0 and new_inv > 0:
                # Both invalid, try to reduce invalidity
                if random.random() < 0.05:
                    accept = True
            elif curr_inv == 0 and new_inv > 0:
                # Became invalid, reject
                accept = False

            if accept:
                improved = True
                curr_score = new_score
                curr_inv = new_inv
                line_index = new_line_index
                
                if curr_inv == 0 and curr_score > best_score:
                    best_score = curr_score
                    best_points = current_points[:]
                break
            else:
                # Revert
                current_points[idx] = (old_x, old_y)
        
        if not improved:
            no_improve += 1
            if no_improve > 100:
                # Restart with perturbation
                no_improve = 0
                for _ in range(3):
                    idx = random.randint(0, 15)
                    new_x = random.randint(-50, 50)
                    new_y = random.randint(-50, 50)
                    current_points[idx] = (new_x, new_y)
                curr_score, curr_inv = evaluate(current_points)
                line_index = build_line_index(current_points)
        else:
            no_improve = 0

    # Final polish: try larger moves around best solution
    if best_points and time.time() - start_time < time_limit - 10:
        current_points = best_points[:]
        line_index = build_line_index(current_points)
        curr_score, curr_inv = evaluate(current_points)
        
        for step in range(2000):
            if time.time() - start_time > time_limit:
                break
                
            idx = random.randint(0, 15)
            old_x, old_y = current_points[idx]
            
            # Try larger moves
            moves = []
            for dx in range(-20, 21):
                for dy in range(-20, 21):
                    if dx == 0 and dy == 0:
                        continue
                    new_p = (old_x + dx, old_y + dy)
                    if -1000 <= new_p[0] <= 1000 and -1000 <= new_p[1] <= 1000:
                        if new_p not in current_points:
                            moves.append(new_p)
            
            random.shuffle(moves)
            
            for new_p in moves:
                current_points[idx] = new_p
                new_score, new_inv, new_line_index = evaluate_incremental(
                    current_points, line_index, idx, (old_x, old_y), new_p
                )
                
                accept = False
                if curr_inv == 0 and new_inv == 0 and new_score > curr_score:
                    accept = True
                elif curr_inv > 0 and new_inv == 0:
                    accept = True
                
                if accept:
                    curr_score = new_score
                    curr_inv = new_inv
                    line_index = new_line_index
                    if curr_inv == 0 and curr_score > best_score:
                        best_score = curr_score
                        best_points = current_points[:]
                    break
                else:
                    current_points[idx] = (old_x, old_y)

    # Convert to list of lists as required
    if best_points is None:
        return [[i, j] for i in range(4) for j in range(4)]
    
    return [list(p) for p in best_points]
