import time
import random
from math import gcd
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 115.0
    
    def line_key(p1, p2):
        dx = p2[0] - p1[0]
        dy = p2[1] - p1[1]
        g = gcd(abs(dx), abs(dy))
        if g == 0:
            return (0, 0, p1[0], p1[1])
        dx //= g
        dy //= g
        if dx < 0 or (dx == 0 and dy < 0):
            dx = -dx
            dy = -dy
        c = dy * p1[0] - dx * p1[1]
        return (dx, dy, c)
    
    def compute_line_counts(points):
        n = len(points)
        line_counts = {}
        for i in range(n):
            for j in range(i + 1, n):
                key = line_key(points[i], points[j])
                line_counts[key] = line_counts.get(key, 0) + 1
        return line_counts
    
    def count_triples_and_check_four(points):
        line_counts = compute_line_counts(points)
        triples = 0
        has_four = False
        for key, count in line_counts.items():
            if count >= 4:
                has_four = True
            if count >= 3:
                triples += count * (count - 1) * (count - 2) // 6
        return triples, has_four
    
    def get_score(points):
        triples, has_four = count_triples_and_check_four(points)
        if has_four:
            return -10**9
        return triples
    
    def is_valid(points):
        if len(points) != 20:
            return False
        seen = set()
        for p in points:
            if not (-100 <= p[0] <= 100 and -100 <= p[1] <= 100):
                return False
            if tuple(p) in seen:
                return False
            seen.add(tuple(p))
        _, has_four = count_triples_and_check_four(points)
        return not has_four
    
    # Finite field inspired construction: use points from a structured grid
    # that maximizes collinear triples
    def construct_from_structure(seed):
        rng = random.Random(seed)
        
        # Strategy: Create clusters of collinear points
        # Each cluster of k points on a line contributes C(k,3) triples
        # We want to maximize total triples subject to no 4 on any line
        
        # Build from scratch using greedy line packing
        points = []
        used = set()
        
        # Place points to create many 3-point lines
        # Use a structured approach: pick lines, place 3 points on each
        # Ensure lines don't create 4-collinear sets
        
        # Method: Start with a base grid, then perturb
        # Try different base structures
        
        structures = []
        
        # Structure 1: Two interleaved 3x3 grids + 2 extra
        for scale in [2, 3, 4, 5]:
            for offset in range(-5, 6):
                pts = []
                # First 9 points: 3x3 grid
                for i in range(3):
                    for j in range(3):
                        x = i * scale + offset
                        y = j * scale
                        pts.append([x, y])
                # Second 9 points: shifted 3x3 grid
                for i in range(3):
                    for j in range(3):
                        x = i * scale + offset + scale // 2
                        y = j * scale + scale // 2
                        pts.append([x, y])
                # 2 extra points
                pts.append([offset + scale, scale * 2 + offset])
                pts.append([offset + 2 * scale, scale + offset])
                
                # Check bounds and validity
                valid = True
                for p in pts:
                    if not (-100 <= p[0] <= 100 and -100 <= p[1] <= 100):
                        valid = False
                        break
                    if tuple(p) in used:
                        valid = False
                        break
                if valid and len(set(tuple(p) for p in pts)) == 20:
                    structures.append(pts)
        
        # Structure 2: Line-based construction
        # Pick several lines, place 3 points on each
        for attempt in range(20):
            pts = []
            lines_used = []
            
            # Pick 7 lines (each will have 3 points = 21 slots, but some overlap)
            # Actually, we need exactly 20 points
            # Let's pick lines that share endpoints
            
            # Use a star pattern: multiple lines through a common region
            center_x = rng.randint(-30, 30)
            center_y = rng.randint(-30, 30)
            
            # Create 6-7 lines through/near center, place 3 points on each
            angles = []
            for _ in range(7):
                angle = rng.uniform(0, 3.14159)
                angles.append(angle)
            angles.sort()
            
            for k, angle in enumerate(angles):
                dx = int(20 * __import__('math').cos(angle))
                dy = int(20 * __import__('math').sin(angle))
                if dx == 0 and dy == 0:
                    continue
                g = gcd(abs(dx), abs(dy))
                if g > 0:
                    dx //= g
                    dy //= g
                
                # Place 3 points on this line
                for t in [-1, 0, 1]:
                    px = center_x + t * dx * 3
                    py = center_y + t * dy * 3
                    if -100 <= px <= 100 and -100 <= py <= 100:
                        pts.append([px, py])
            
            # Remove duplicates, keep first 20
            seen = set()
            unique_pts = []
            for p in pts:
                if tuple(p) not in seen:
                    seen.add(tuple(p))
                    unique_pts.append(p)
            
            if len(unique_pts) >= 20:
                pts = unique_pts[:20]
                # Verify no 4 collinear
                if is_valid(pts):
                    structures.append(pts)
        
        # Structure 3: Perturbed grids
        for _ in range(30):
            base_dx = rng.randint(3, 8)
            base_dy = rng.randint(3, 8)
            ox = rng.randint(-50, 50)
            oy = rng.randint(-50, 50)
            
            pts = []
            # 4x5 grid with perturbations
            for i in range(4):
                for j in range(5):
                    x = ox + i * base_dx + rng.randint(-1, 1)
                    y = oy + j * base_dy + rng.randint(-1, 1)
                    pts.append([x, y])
            
            # Deduplicate
            seen = set()
            unique_pts = []
            for p in pts:
                if tuple(p) not in seen and -100 <= p[0] <= 100 and -100 <= p[1] <= 100:
                    seen.add(tuple(p))
                    unique_pts.append(p)
            
            if len(unique_pts) == 20 and is_valid(unique_pts):
                structures.append(unique_pts)
        
        return structures
    
    def local_search(points, time_limit, seed=0):
        current_pts = [p[:] for p in points]
        current_score = get_score(current_pts)
        best_pts = [p[:] for p in current_pts]
        best_score = current_score
        
        rng = random.Random(seed)
        
        def generate_moves(pts):
            moves = []
            n = len(pts)
            
            # Directed moves along existing lines
            for i in range(n):
                for j in range(i + 1, n):
                    p1, p2 = pts[i], pts[j]
                    dx = p2[0] - p1[0]
                    dy = p2[1] - p1[1]
                    g = gcd(abs(dx), abs(dy))
                    if g == 0:
                        continue
                    dx_n = dx // g
                    dy_n = dy // g
                    
                    for t in range(1, 8):
                        for sign in [1, -1]:
                            new_p = [p2[0] + sign * t * dx_n, p2[1] + sign * t * dy_n]
                            if -100 <= new_p[0] <= 100 and -100 <= new_p[1] <= 100:
                                if new_p not in pts:
                                    moves.append((new_p, i))
                                    moves.append((new_p, j))
            
            # Random moves
            for _ in range(50):
                new_p = [rng.randint(-100, 100), rng.randint(-100, 100)]
                idx = rng.randint(0, 19)
                if new_p not in pts:
                    moves.append((new_p, idx))
            
            return moves
        
        no_improve_count = 0
        
        while time.time() < time_limit:
            moves = generate_moves(current_pts)
            
            # Shuffle and evaluate
            rng.shuffle(moves)
            
            best_move_score = current_score
            best_move = None
            
            # Evaluate up to 200 moves
            for new_p, idx in moves[:200]:
                if new_p in current_pts:
                    continue
                
                test_pts = current_pts[:]
                test_pts[idx] = new_p
                
                new_score = get_score(test_pts)
                if new_score > best_move_score:
                    best_move_score = new_score
                    best_move = (new_p, idx)
            
            if best_move is not None:
                current_pts[best_move[1]] = best_move[0]
                current_score = best_move_score
                no_improve_count = 0
                
                if current_score > best_score:
                    best_score = current_score
                    best_pts = [p[:] for p in current_pts]
            else:
                no_improve_count += 1
                
                # Restart with perturbation
                if no_improve_count >= 10:
                    idx = rng.randint(0, 19)
                    new_p = [rng.randint(-100, 100), rng.randint(-100, 100)]
                    if new_p not in current_pts:
                        test_pts = current_pts[:]
                        test_pts[idx] = new_p
                        if get_score(test_pts) > -10**9:
                            current_pts = test_pts
                            current_score = get_score(current_pts)
                    no_improve_count = 0
        
        return best_pts
    
    # Main search
    best_overall = None
    best_overall_score = -1
    
    # Generate initial structures
    all_structures = []
    for seed in range(50):
        if time.time() > deadline - 10:
            break
        structures = construct_from_structure(seed)
        all_structures.extend(structures)
    
    # Evaluate all structures
    for pts in all_structures:
        if time.time() > deadline - 5:
            break
        score = get_score(pts)
        if score > best_overall_score:
            best_overall_score = score
            best_overall = pts
    
    # If no valid structure found, create one
    if best_overall is None:
        # Fallback: simple grid
        pts = []
        for i in range(4):
            for j in range(5):
                pts.append([i * 10 + 1, j * 10 + 1])
        best_overall = pts
        best_overall_score = get_score(pts)
    
    # Local search refinement
    if time.time() < deadline - 5:
        search_time = min(3.0, deadline - 5 - time.time())
        if search_time > 0.5:
            refined = local_search(best_overall, time.time() + search_time, seed=42)
            refined_score = get_score(refined)
            if refined_score > best_overall_score:
                best_overall_score = refined_score
                best_overall = refined
    
    # Final polish
    if time.time() < deadline - 2:
        search_time = min(2.0, deadline - 2 - time.time())
        if search_time > 0.3:
            refined = local_search(best_overall, time.time() + search_time, seed=123)
            refined_score = get_score(refined)
            if refined_score > best_overall_score:
                best_overall_score = refined_score
                best_overall = refined
    
    return best_overall
