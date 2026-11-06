"""No-isosceles set in [0, N)^2 grid."""
import time
import random
import numpy as np

N = 64
TIME_LIMIT = 280.0

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + TIME_LIMIT
    
    best_points = None
    best_len = 0
    
    # Generate all candidate points
    all_points = [(x, y) for x in range(N) for y in range(N)]
    
    for restart in range(200):
        if time.time() > deadline - 10:
            break
        
        random.seed(restart * 1000 + 42)
        random.shuffle(all_points)
        
        # Greedy construction
        points = []
        point_set = set()
        
        for p in all_points:
            if time.time() > deadline - 5:
                break
            if _can_add(p, points):
                points.append(p)
                point_set.add(p)
        
        # Local search: try to improve by swapping
        improved = True
        while improved and time.time() < deadline - 5:
            improved = False
            # Try removing each point and adding a better one
            for i in range(len(points)):
                if time.time() > deadline - 5:
                    break
                # Try to find a point that can replace points[i]
                removed = points[i]
                remaining = [p for j, p in enumerate(points) if j != i]
                # Try adding random candidates
                for _ in range(50):
                    if time.time() > deadline - 5:
                        break
                    candidate = random.choice(all_points)
                    if candidate == removed:
                        continue
                    if candidate in remaining:
                        continue
                    if _can_add(candidate, remaining):
                        # Accept the swap
                        points[i] = candidate
                        improved = True
                        break
        
        if len(points) > best_len:
            best_len = len(points)
            best_points = list(points)
    
    if best_points is None:
        # Fallback: simple greedy
        points = []
        for p in all_points:
            if _can_add(p, points):
                points.append(p)
        best_points = points
    
    return best_points


def _can_add(new_pt, existing_pts):
    """Check if adding new_pt to existing_pts creates no isosceles triangle."""
    nx, ny = new_pt
    for i in range(len(existing_pts)):
        ax, ay = existing_pts[i]
        d1 = (nx - ax) ** 2 + (ny - ay) ** 2
        for j in range(i + 1, len(existing_pts)):
            bx, by = existing_pts[j]
            d2 = (ax - bx) ** 2 + (ay - by) ** 2
            d3 = (nx - bx) ** 2 + (ny - by) ** 2
            if d1 == d2 or d1 == d3 or d2 == d3:
                return False
    return True
# EVOLVE-BLOCK-END
