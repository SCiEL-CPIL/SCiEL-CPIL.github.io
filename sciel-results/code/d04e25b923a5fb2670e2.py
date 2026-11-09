"""No-isosceles set: maximize points in [0,64)^2 with no 3 forming isosceles triangle."""
import time
import random
from collections import defaultdict

N = 64

def construct():
    start = time.time()
    deadline = start + 280  # 50s safety margin for 330s limit
    
    # Precompute all points
    all_points = [(x, y) for x in range(N) for y in range(N)]
    
    # For efficiency, represent points as integers
    # Point (x,y) -> x * N + y
    
    def dist2(p, q):
        dx = p[0] - q[0]
        dy = p[1] - q[1]
        return dx * dx + dy * dy
    
    def is_isosceles_free(points_set, new_point):
        """Check if adding new_point to points_set creates any isosceles triangle."""
        pts = list(points_set)
        for i, p in enumerate(pts):
            d_np = dist2(new_point, p)
            for j in range(i + 1, len(pts)):
                q = pts[j]
                d_nq = dist2(new_point, q)
                d_pq = dist2(p, q)
                # Check if any two of the three distances are equal
                if d_np == d_nq or d_np == d_pq or d_nq == d_pq:
                    return False
        return True
    
    # Greedy construction with randomized ordering
    best_set = set()
    
    # Try multiple random orderings
    for trial in range(200):
        if time.time() > deadline - 30:
            break
        
        random.seed(trial * 7919 + 13)
        order = all_points[:]
        random.shuffle(order)
        
        current_set = set()
        for p in order:
            if is_isosceles_free(current_set, p):
                current_set.add(p)
        
        if len(current_set) > len(best_set):
            best_set = current_set
            if time.time() > deadline - 60:
                break
    
    # Local search: try to add more points, or swap points
    if time.time() < deadline - 30:
        improved = True
        while improved and time.time() < deadline - 30:
            improved = False
            # Try adding each unselected point
            selected = set(best_set)
            for p in all_points:
                if p not in selected:
                    if is_isosceles_free(selected, p):
                        selected.add(p)
                        improved = True
                        break
            if len(selected) > len(best_set):
                best_set = selected
                improved = True
        
        # Try swaps: remove one, try to add two
        if time.time() < deadline - 20:
            improved = True
            while improved and time.time() < deadline - 20:
                improved = False
                selected = set(best_set)
                unselected = [p for p in all_points if p not in selected]
                
                for rem in list(selected):
                    if time.time() > deadline - 25:
                        break
                    candidate = selected - {rem}
                    added_count = 0
                    for add in unselected:
                        if is_isosceles_free(candidate, add):
                            candidate.add(add)
                            added_count += 1
                            if added_count >= 2:
                                break
                    if len(candidate) > len(best_set):
                        best_set = candidate
                        improved = True
                        break
    
    return list(best_set)
