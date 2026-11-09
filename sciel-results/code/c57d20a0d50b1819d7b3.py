import math
import time

def _hexdist(q1, r1, q2, r2):
    """Calculate hex distance between two axial coordinates."""
    return max(abs(q1 - q2), abs(r1 - r2), abs((q1 + r1) - (q2 + r2)))

def _count_edges(points):
    """Count the number of unit-distance pairs in the set of points."""
    n = len(points)
    count = 0
    for i in range(n):
        q1, r1 = points[i]
        for j in range(i + 1, n):
            q2, r2 = points[j]
            if _hexdist(q1, r1, q2, r2) == 1:
                count += 1
    return count

def _generate_optimal_32():
    """
    Generate the known optimal configuration of 32 points.
    This is a hexagon of radius 2 (19 points) plus 13 points from radius 3.
    """
    points = set()
    
    # Add hexagon of radius 2
    for q in range(-2, 3):
        for r in range(-2, 3):
            if _hexdist(0, 0, q, r) <= 2:
                points.add((q, r))
                
    # Add 13 points from radius 3
    # The ring of radius 3 has 18 points.
    # We pick a contiguous segment of 13 to maximize internal edges.
    # The points on ring 3 are (q,r) such that hexdist(0,0,q,r)==3.
    ring3 = []
    for q in range(-3, 4):
        for r in range(-3, 4):
            if _hexdist(0, 0, q, r) == 3:
                # Sort by angle to form a contiguous sequence
                x = q + 0.5 * r
                y = math.sqrt(3) / 2 * r
                angle = math.atan2(y, x)
                ring3.append((angle, q, r))
                
    ring3.sort()
    
    # Take the first 13
    for i in range(13):
        _, q, r = ring3[i]
        points.add((q, r))
        
    return list(points)

def _local_search(points, max_iters=1000):
    """
    Local search to maximize edges.
    """
    current_points = [tuple(p) for p in points]
    current_score = _count_edges(current_points)
    
    # Directions for moves
    directions = [(1,0), (0,1), (-1,1), (-1,0), (0,-1), (1,-1)]
    
    for _ in range(max_iters):
        improved = False
        for i in range(len(current_points)):
            q, r = current_points[i]
            # Try moving to adjacent empty spots
            for dq, dr in directions:
                new_q, new_r = q + dq, r + dr
                # Check bounds
                if abs(new_q) > 64 or abs(new_r) > 64:
                    continue
                # Check if occupied
                if (new_q, new_r) in current_points:
                    continue
                
                # Calculate delta score
                # Remove point i, add new point
                # This is expensive, so we approximate or just re-count if promising
                # For 32 points, re-counting is fast enough (32*31/2 = 496 checks)
                
                # Temporarily update
                old_point = current_points[i]
                current_points[i] = (new_q, new_r)
                
                new_score = _count_edges(current_points)
                
                if new_score > current_score:
                    current_score = new_score
                    improved = True
                else:
                    current_points[i] = old_point
                    
                if improved:
                    break
            if improved:
                break
        if not improved:
            break
            
    return current_points, current_score

def construct():
    start_time = time.time()
    deadline = start_time + 10  # 10 seconds safety margin
    
    # 1. Generate the known optimal candidate
    candidate = _generate_optimal_32()
    
    # Ensure exactly 32 distinct points
    if len(candidate) != 32:
        # Fallback to spiral generation if construction fails (shouldn't happen)
        candidates = []
        for q in range(-10, 11):
            for r in range(-10, 11):
                dist = max(abs(q), abs(r), abs(q + r))
                x = q + 0.5 * r
                y = math.sqrt(3) / 2 * r
                angle = math.atan2(y, x)
                candidates.append((dist, angle, q, r))
        candidates.sort(key=lambda x: (x[0], x[1]))
        candidate = [c[2:] for c in candidates[:32]]
        
    # 2. Verify the score
    score = _count_edges(candidate)
    
    # 3. If score is optimal (76), return immediately
    if score == 76:
        return [[q, r] for q, r in candidate]
        
    # 4. If not optimal, try local search
    if time.time() < deadline:
        optimized_points, opt_score = _local_search(candidate)
        if opt_score > score:
            candidate = optimized_points
            score = opt_score
            
    # 5. Return the best found
    return [[q, r] for q, r in candidate]
