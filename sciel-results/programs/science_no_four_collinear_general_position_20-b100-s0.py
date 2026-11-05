import time
import random
import itertools
from math import gcd

def construct():
    start_time = time.time()
    deadline = start_time + 115.0
    
    def is_collinear(p1, p2, p3):
        return (p2[0] - p1[0]) * (p3[1] - p1[1]) == (p3[0] - p1[0]) * (p2[1] - p1[1])
    
    def line_key(p1, p2):
        """Compute a canonical key for the line through p1 and p2."""
        dx = p2[0] - p1[0]
        dy = p2[1] - p1[1]
        g = gcd(abs(dx), abs(dy))
        if g == 0:
            return (0, 0, p1[0], p1[1])
        dx //= g
        dy //= g
        # Canonical direction: ensure dx > 0, or dx == 0 and dy > 0
        if dx < 0 or (dx == 0 and dy < 0):
            dx = -dx
            dy = -dy
        # Line equation: dy*x - dx*y = c
        c = dy * p1[0] - dx * p1[1]
        return (dx, dy, c)
    
    def count_triples_and_check_four(points):
        """Count collinear triples and check for 4-collinear in O(n^2)."""
        n = len(points)
        line_counts = {}
        for i in range(n):
            for j in range(i + 1, n):
                key = line_key(points[i], points[j])
                line_counts[key] = line_counts.get(key, 0) + 1
        
        triples = 0
        has_four = False
        for key, count in line_counts.items():
            if count >= 4:
                has_four = True
            if count >= 3:
                # C(count, 3) triples
                triples += count * (count - 1) * (count - 2) // 6
        
        return triples, has_four
    
    def generate_initial_points():
        # Start with a 4x5 grid structure which has many collinearities
        pts = []
        for x in range(4):
            for y in range(5):
                pts.append([x * 10, y * 10])
        return pts

    def get_score(points):
        triples, has_four = count_triples_and_check_four(points)
        if has_four:
            return -1000000
        return triples
    
    def find_lines_with_three(points):
        """Find all lines that contain exactly 3 points."""
        n = len(points)
        line_to_points = {}
        for i in range(n):
            for j in range(i + 1, n):
                key = line_key(points[i], points[j])
                if key not in line_to_points:
                    line_to_points[key] = set()
                line_to_points[key].add(i)
                line_to_points[key].add(j)
        
        lines_three = []
        for key, pts in line_to_points.items():
            if len(pts) == 3:
                lines_three.append((key, pts))
        return lines_three
    
    def generate_valid_initial():
        pts = []
        for x in range(4):
            for y in range(5):
                pts.append([x * 10, y * 10])
        
        current_pts = pts
        valid = False
        attempts = 0
        while not valid and attempts < 500:
            _, has_four = count_triples_and_check_four(current_pts)
            if not has_four:
                valid = True
            else:
                # Find a line with 4+ points and move one of them
                line_to_points = {}
                for i in range(20):
                    for j in range(i + 1, 20):
                        key = line_key(current_pts[i], current_pts[j])
                        if key not in line_to_points:
                            line_to_points[key] = set()
                        line_to_points[key].add(i)
                        line_to_points[key].add(j)
                
                moved = False
                for key, pts in line_to_points.items():
                    if len(pts) >= 4:
                        idx = random.choice(list(pts))
                        old_p = current_pts[idx]
                        for _ in range(50):
                            dx = random.randint(-3, 3)
                            dy = random.randint(-3, 3)
                            if dx == 0 and dy == 0:
                                continue
                            new_p = [old_p[0] + dx, old_p[1] + dy]
                            if -100 <= new_p[0] <= 100 and -100 <= new_p[1] <= 100:
                                if new_p not in current_pts:
                                    current_pts[idx] = new_p
                                    moved = True
                                    break
                        if moved:
                            break
                attempts += 1
        return current_pts
    
    def directed_moves(points):
        """Generate directed moves that try to create new collinear triples."""
        n = len(points)
        moves = []
        
        # For each pair of points, try to find a third point that would be collinear
        for i in range(n):
            for j in range(i + 1, n):
                p1 = points[i]
                p2 = points[j]
                dx = p2[0] - p1[0]
                dy = p2[1] - p1[1]
                g = gcd(abs(dx), abs(dy))
                if g == 0:
                    continue
                dx_n = dx // g
                dy_n = dy // g
                
                # Try extending the line in both directions
                for t in range(1, 4):
                    for sign in [1, -1]:
                        new_p = [p2[0] + sign * t * dx_n, p2[1] + sign * t * dy_n]
                        if -100 <= new_p[0] <= 100 and -100 <= new_p[1] <= 100:
                            if new_p not in points:
                                moves.append(new_p)
                        new_p = [p1[0] - sign * t * dx_n, p1[1] - sign * t * dy_n]
                        if -100 <= new_p[0] <= 100 and -100 <= new_p[1] <= 100:
                            if new_p not in points:
                                moves.append(new_p)
        
        # Also add some random moves for diversity
        for _ in range(10):
            new_p = [random.randint(-100, 100), random.randint(-100, 100)]
            if new_p not in points:
                moves.append(new_p)
        
        return moves
    
    def local_search(points, time_limit):
        current_pts = [p[:] for p in points]
        current_score = get_score(current_pts)
        best_pts = [p[:] for p in current_pts]
        best_score = current_score
        
        no_improve_count = 0
        
        while time.time() < time_limit:
            improved = False
            all_moves = directed_moves(current_pts)
            
            # Shuffle moves for diversity
            random.shuffle(all_moves)
            
            # Try each move on a random point
            for new_p in all_moves:
                if time.time() > time_limit:
                    break
                
                # Try moving each point to new_p
                best_move = None
                best_move_score = current_score
                
                for i in range(20):
                    if current_pts[i] == new_p:
                        continue
                    test_pts = current_pts[:]
                    test_pts[i] = new_p
                    new_score = get_score(test_pts)
                    if new_score > best_move_score:
                        best_move_score = new_score
                        best_move = i
                
                if best_move is not None:
                    current_pts[best_move] = new_p
                    current_score = best_move_score
                    improved = True
                    no_improve_count = 0
                    if current_score > best_score:
                        best_score = current_score
                        best_pts = [p[:] for p in current_pts]
                    break
            
            if not improved:
                no_improve_count += 1
                if no_improve_count >= 3:
                    # Random jump to escape local optimum
                    idx = random.randint(0, 19)
                    new_p = [random.randint(-100, 100), random.randint(-100, 100)]
                    if new_p not in current_pts:
                        test_pts = current_pts[:]
                        test_pts[idx] = new_p
                        new_score = get_score(test_pts)
                        if new_score > -1000000:
                            current_pts = test_pts
                            current_score = new_score
                            no_improve_count = 0
                else:
                    break
        
        return best_pts
    
    # Run multiple restarts
    best_overall = None
    best_overall_score = -1
    
    # Initial point
    init_pts = generate_valid_initial()
    
    # Run local search with multiple restarts
    t_end = deadline
    restart_count = 0
    while time.time() < t_end - 5:
        search_time = min(8, t_end - 5 - time.time())
        if search_time < 1:
            break
        current_best = local_search(init_pts, time.time() + search_time)
        score = get_score(current_best)
        if score > best_overall_score:
            best_overall_score = score
            best_overall = current_best
        
        restart_count += 1
        
        # Generate a new initial point for next restart
        if time.time() < t_end - 10:
            init_pts = generate_valid_initial()
        else:
            break
            
    return best_overall
