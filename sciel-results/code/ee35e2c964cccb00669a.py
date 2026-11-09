import random
import time
from itertools import combinations

def _are_collinear(p1, p2, p3):
    return (p1[0]*(p2[1]-p3[1]) + p2[0]*(p3[1]-p1[1]) + p3[0]*(p1[1]-p2[1])) == 0

def _has_four_collinear(points):
    n = len(points)
    for i, j, k, l in combinations(range(n), 4):
        # Check if all 4 are on the same line
        # It's sufficient to check if p1,p2,p3 are collinear AND p1,p2,p4 are collinear
        # because if p1,p2,p3 and p1,p2,p4 are collinear, then all 4 are on line p1-p2
        if _are_collinear(points[i], points[j], points[k]) and _are_collinear(points[i], points[j], points[l]):
            return True
    return False

def _max_no_three_collinear(points):
    n = len(points)
    if n == 0:
        return 0
    
    # Precompute conflicts: 
    # conflicts[i] is a list of bitmasks representing pairs {j, k} 
    # such that points[i], points[j], points[k] are collinear.
    conflicts = [[] for _ in range(n)]
    
    for i in range(n):
        for j in range(i+1, n):
            for k in range(j+1, n):
                if _are_collinear(points[i], points[j], points[k]):
                    # Add pair (j,k) to conflicts[i]
                    # Add pair (i,k) to conflicts[j]
                    # Add pair (i,j) to conflicts[k]
                    conflicts[i].append(1 << j | 1 << k)
                    conflicts[j].append(1 << i | 1 << k)
                    conflicts[k].append(1 << i | 1 << j)
    
    best = [0]
    
    def can_add(p, mask):
        # Check if adding point p to set represented by mask creates a collinear triple
        # We need to check if there exists a pair {q, r} in mask such that {q, r} is in conflicts[p]
        for pair_mask in conflicts[p]:
            if (pair_mask & mask) == pair_mask:
                return False
        return True

    def bnb(mask, remaining):
        # mask: bitmask of currently selected points
        # remaining: bitmask of points not yet considered
        if remaining == 0:
            if bin(mask).count('1') > best[0]:
                best[0] = bin(mask).count('1')
            return
            
        # Upper bound
        # max possible size = current size + number of remaining points
        if bin(mask).count('1') + bin(remaining).count('1') <= best[0]:
            return
            
        # Heuristic: pick the point in remaining that has the most conflicts with mask?
        # Or just pick the lowest set bit for determinism and speed
        # Let's pick the point that is "most constrained" by the current mask
        # i.e., the point that would be invalid to add? No, we branch on Add/Not Add.
        # A good heuristic is to pick a point that is involved in many conflicts.
        
        # For simplicity and speed, let's just iterate through bits in 'remaining'
        # We can optimize by picking the point that minimizes the branching factor?
        
        # Let's stick to a simple heuristic: pick the first available point
        # Or better, pick the point that is 'closest' to forming a conflict?
        
        # To improve speed, let's try picking the point with the most conflicts in total?
        # Precompute degree?
        
        # Let's just use the lowest index for now.
        p_bit = remaining & -remaining
        p = p_bit.bit_length() - 1
        
        # Option 1: Add p
        if can_add(p, mask):
            bnb(mask | p_bit, remaining ^ p_bit)
            
        # Option 2: Don't add p
        bnb(mask, remaining ^ p_bit)
        
    bnb(0, (1 << n) - 1)
    return best[0]

def _generate_initial_points():
    """
    Generate a set of 20 points that is likely to have many collinear triples.
    A perturbed grid is a good start.
    """
    points = []
    # Create a 5x4 grid base
    base_x = [-20, -10, 0, 10, 20]
    base_y = [-15, -5, 5, 15]
    
    candidates = []
    for x in base_x:
        for y in base_y:
            candidates.append((x, y))
            
    # We need 20 points. The grid has 20 points.
    # But a perfect grid has many 4-collinear points (rows and cols).
    # We need to perturb them to break 4-collinearities while keeping 3-collinearities.
    
    pts = []
    for x, y in candidates:
        # Perturb slightly
        dx = random.randint(-2, 2)
        dy = random.randint(-2, 2)
        pts.append([x + dx, y + dy])
        
    # Ensure distinctness and bounds
    valid_pts = []
    for p in pts:
        if -100 <= p[0] <= 100 and -100 <= p[1] <= 100:
            if p not in valid_pts:
                valid_pts.append(p)
            else:
                # If duplicate, nudge it
                while True:
                    np = [p[0] + random.randint(-5, 5), p[1] + random.randint(-5, 5)]
                    if -100 <= np[0] <= 100 and -100 <= np[1] <= 100 and np not in valid_pts:
                        valid_pts.append(np)
                        break
                        
    # If we still don't have 20 (due to duplicates), add random ones
    while len(valid_pts) < 20:
        x = random.randint(-100, 100)
        y = random.randint(-100, 100)
        if [x, y] not in valid_pts:
            valid_pts.append([x, y])
            
    return valid_pts[:20]

def _local_search(points, deadline):
    current_points = points[:]
    current_score = _max_no_three_collinear(current_points)
    best_points = current_points[:]
    best_score = current_score
    
    while time.time() < deadline:
        # Pick a random point to move
        idx = random.randint(0, 19)
        old_p = current_points[idx]
        
        # Generate new position
        # Try to move it to a nearby position first? Or random?
        # Random within bounds is simpler.
        new_p = [random.randint(-100, 100), random.randint(-100, 100)]
        
        if new_p == old_p:
            continue
            
        # Check distinctness
        if new_p in current_points:
            continue
            
        new_points = current_points[:]
        new_points[idx] = new_p
        
        # Check validity: no 4 collinear
        if _has_four_collinear(new_points):
            continue
            
        # Evaluate new score
        new_score = _max_no_three_collinear(new_points)
        
        if new_score < current_score:
            current_score = new_score
            current_points = new_points
            if current_score < best_score:
                best_score = current_score
                best_points = current_points[:]
                
    return best_points, best_score

def construct():
    start_time = time.time()
    deadline = start_time + 115  # Safety margin
    
    best_points = None
    best_score = float('inf')
    
    # Try a few initial configurations
    for _ in range(3):
        if time.time() > deadline - 30:
            break
            
        init_points = _generate_initial_points()
        
        # Ensure initial points are valid (no 4 collinear)
        # If not, perturb until valid or timeout
        attempts = 0
        while _has_four_collinear(init_points) and attempts < 100 and time.time() < deadline - 30:
            # Perturb a random point
            idx = random.randint(0, 19)
            init_points[idx] = [random.randint(-100, 100), random.randint(-100, 100)]
            attempts += 1
            
        if _has_four_collinear(init_points):
            continue
            
        pts, score = _local_search(init_points, deadline)
        
        if score < best_score:
            best_score = score
            best_points = pts
            
    if best_points is None:
        # Fallback: return a valid set (might not be optimal)
        # Generate random valid set
        while True:
            pts = []
            for _ in range(20):
                while True:
                    p = [random.randint(-100, 100), random.randint(-100, 100)]
                    if p not in pts:
                        break
                pts.append(p)
            if not _has_four_collinear(pts):
                return pts
                
    return best_points
