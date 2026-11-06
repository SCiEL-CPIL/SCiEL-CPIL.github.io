import math
import random
import time
from collections import defaultdict

def line_hash(p1, p2):
    x1, y1 = p1
    x2, y2 = p2
    if x1 == x2:
        return ('v', x1)
    if y1 == y2:
        return ('h', y1)
    dy = y2 - y1
    dx = x2 - x1
    g = math.gcd(abs(dx), abs(dy))
    dx_n = dx // g
    dy_n = dy // g
    if dx_n < 0:
        dx_n = -dx_n
        dy_n = -dy_n
    elif dx_n == 0 and dy_n < 0:
        dy_n = -dy_n
    a = dy_n
    b = -dx_n
    c = dx_n * y1 - dy_n * x1
    g = math.gcd(abs(a), abs(b), abs(c))
    if g > 0:
        a //= g
        b //= g
        c //= g
    if a < 0 or (a == 0 and b < 0) or (a == 0 and b == 0 and c < 0):
        a = -a
        b = -b
        c = -c
    return (a, b, c)

def evaluate(points):
    n = len(points)
    if n != 16:
        return -1
    if len(set(map(tuple, points))) != 16:
        return -1
    line_counts = defaultdict(int)
    for i in range(n):
        for j in range(i + 1, n):
            key = line_hash(points[i], points[j])
            line_counts[key] += 1
    
    score = 0
    for key, count in line_counts.items():
        # count is the number of PAIRS.
        # If a line has k points, it has k*(k-1)/2 pairs.
        # We need to recover k from count.
        # k^2 - k - 2*count = 0
        # k = (1 + sqrt(1 + 8*count)) / 2
        disc = 1 + 8 * count
        sqrt_disc = int(math.isqrt(disc))
        if sqrt_disc * sqrt_disc != disc:
            return -1
        k = (1 + sqrt_disc) // 2
        if k >= 5:
            return -1
        if k == 4:
            score += 1
    return score

def get_line_points_extended(p1, p2, bounds=1000):
    """
    Returns a list of (x, y) integer points on the line defined by p1, p2
    that are within bounds.
    """
    x1, y1 = p1
    x2, y2 = p2
    dx = x2 - x1
    dy = y2 - y1
    g = math.gcd(abs(dx), abs(dy))
    if g == 0:
        return [p1]
    step_x = dx // g
    step_y = dy // g
    
    points = []
    # Search backwards from p1
    k = 0
    while True:
        px = x1 - k * step_x
        py = y1 - k * step_y
        if abs(px) > bounds or abs(py) > bounds:
            break
        points.append((px, py))
        k += 1
        if k > 2000: break
        
    # Search forwards from p1
    k = 1
    while True:
        px = x1 + k * step_x
        py = y1 + k * step_y
        if abs(px) > bounds or abs(py) > bounds:
            break
        points.append((px, py))
        k += 1
        if k > 2000: break
        
    return points

def construct():
    start_time = time.time()
    deadline = start_time + 115.0 # Safety margin
    
    # Initial candidate: 4x4 grid
    initial_points = [[x, y] for x in range(4) for y in range(4)]
    
    best_score = evaluate(initial_points)
    best_points = [p[:] for p in initial_points]
    
    random.seed(42)
    
    def get_lines_and_points(points):
        """
        Returns:
        lines: dict mapping line_key to list of point indices on that line
        """
        lines = defaultdict(list)
        for i in range(16):
            for j in range(i + 1, 16):
                key = line_hash(points[i], points[j])
                lines[key].append((i, j))
        
        # Convert pairs to sets of indices
        line_points = {}
        for key, pairs in lines.items():
            pts = set()
            for i, j in pairs:
                pts.add(i)
                pts.add(j)
            line_points[key] = pts
        
        return line_points

    def local_search_line_completion(points, max_iters=10000):
        current_points = [p[:] for p in points]
        current_score = evaluate(current_points)
        if current_score < 0:
            return current_points, current_score
        
        best_local = [p[:] for p in current_points]
        best_local_score = current_score
        
        temp = 1.0
        cooling = 0.999
        no_improve = 0
        
        for _ in range(max_iters):
            if time.time() > deadline:
                break
                
            # Get current line structure
            line_points = get_lines_and_points(current_points)
            
            # Find 3-point lines
            three_point_lines = []
            for key, pts in line_points.items():
                if len(pts) == 3:
                    three_point_lines.append((key, pts))
            
            move_made = False
            
            if three_point_lines:
                # Pick a random 3-point line to try to complete
                key, pts = random.choice(three_point_lines)
                # Get the actual coordinates of the 3 points
                p_coords = [current_points[i] for i in pts]
                
                # Determine the line direction and step
                # Use the first two points to get the step
                p1, p2 = p_coords[0], p_coords[1]
                dx = p2[0] - p1[0]
                dy = p2[1] - p1[1]
                g = math.gcd(abs(dx), abs(dy))
                if g == 0:
                    continue
                step_x = dx // g
                step_y = dy // g
                
                # The 3 points are on the line.
                # We want to find the "next" integer point in the sequence.
                # The points are p1 + k * step.
                # We can just check the points adjacent to the existing ones.
                # Since they are 3 points, the 4th point is likely adjacent to one of them.
                
                # Let's find the min and max k for the existing points relative to p1.
                # Actually, simpler: just generate candidate points around the existing ones.
                candidates = set()
                for p in p_coords:
                    # Check p + step and p - step
                    c1 = (p[0] + step_x, p[1] + step_y)
                    c2 = (p[0] - step_x, p[1] - step_y)
                    if -1000 <= c1[0] <= 1000 and -1000 <= c1[1] <= 1000:
                        candidates.add(c1)
                    if -1000 <= c2[0] <= 1000 and -1000 <= c2[1] <= 1000:
                        candidates.add(c2)
                
                # Filter out points that are already in current_points
                current_set = set(map(tuple, current_points))
                valid_candidates = [c for c in candidates if c not in current_set]
                
                if valid_candidates:
                    target = random.choice(valid_candidates)
                    
                    # Choose a source point to move
                    # Prefer points that are NOT on this line
                    source_indices = [i for i in range(16) if i not in pts]
                    if not source_indices:
                        source_indices = list(range(16))
                    
                    src_idx = random.choice(source_indices)
                    old_pt = current_points[src_idx][:]
                    
                    # Perform move
                    current_points[src_idx] = [target[0], target[1]]
                    
                    new_score = evaluate(current_points)
                    
                    if new_score == -1:
                        current_points[src_idx] = old_pt
                        no_improve += 1
                        continue
                    
                    delta = new_score - current_score
                    
                    if delta > 0:
                        current_score = new_score
                        no_improve = 0
                        if current_score > best_local_score:
                            best_local_score = current_score
                            best_local = [p[:] for p in current_points]
                    else:
                        if delta == 0:
                            current_score = new_score
                            no_improve += 1
                        else:
                            prob = math.exp(delta / temp)
                            if random.random() < prob:
                                current_score = new_score
                            else:
                                current_points[src_idx] = old_pt
                            no_improve += 1
                    
                    move_made = True
            
            if not move_made:
                # Fallback: Random move
                idx = random.randint(0, 15)
                old_pt = current_points[idx][:]
                
                # Random small move
                step = random.randint(1, 5)
                new_x = old_pt[0] + random.randint(-step, step)
                new_y = old_pt[1] + random.randint(-step, step)
                
                if not (-1000 <= new_x <= 1000 and -1000 <= new_y <= 1000):
                    continue
                    
                new_pt = [new_x, new_y]
                
                is_distinct = True
                for i, p in enumerate(current_points):
                    if i != idx and p == new_pt:
                        is_distinct = False
                        break
                if not is_distinct:
                    continue
                    
                current_points[idx] = new_pt
                new_score = evaluate(current_points)
                
                if new_score == -1:
                    current_points[idx] = old_pt
                    no_improve += 1
                    continue
                    
                delta = new_score - current_score
                
                if delta > 0:
                    current_score = new_score
                    no_improve = 0
                    if current_score > best_local_score:
                        best_local_score = current_score
                        best_local = [p[:] for p in current_points]
                else:
                    if delta == 0:
                        current_score = new_score
                        no_improve += 1
                    else:
                        prob = math.exp(delta / temp)
                        if random.random() < prob:
                            current_score = new_score
                        else:
                            current_points[idx] = old_pt
                        no_improve += 1
            
            temp *= cooling
            if temp < 0.001:
                temp = 0.001
                
            if no_improve > 5000:
                break
                
        return best_local, best_local_score

    # Run local search
    pts, score = local_search_line_completion(initial_points, max_iters=20000)
    if score > best_score:
        best_score = score
        best_points = [p[:] for p in pts]
        
    # Try a few random restarts with perturbed grids
    for _ in range(10):
        if time.time() > deadline:
            break
        pts = [p[:] for p in initial_points]
        for i in range(16):
            if random.random() < 0.5:
                pts[i] = [pts[i][0] + random.randint(-2, 2), pts[i][1] + random.randint(-2, 2)]
        
        s = evaluate(pts)
        if s < 0:
            continue
            
        pts, score = local_search_line_completion(pts, max_iters=10000)
        if score > best_score:
            best_score = score
            best_points = [p[:] for p in pts]

    return best_points
