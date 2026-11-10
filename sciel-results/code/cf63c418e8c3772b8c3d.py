"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import math

N = 64

# EVOLVE-BLOCK-START
def construct():
    deadline = time.time() + 320  # Safety margin before 330s kill
    start_time = time.time()
    
    # Precompute all squared distances between all pairs of grid points
    # For a point p, we need to check if adding it creates an isosceles triangle
    # with any two existing points.
    
    # A triangle with vertices A, B, C is isosceles if:
    #   dist(A,B) == dist(A,C) or dist(A,B) == dist(B,C) or dist(A,C) == dist(B,C)
    # We use squared distances to avoid floating point issues.
    
    def sq_dist(p1, p2):
        dx = p1[0] - p2[0]
        dy = p1[1] - p2[1]
        return dx * dx + dy * dy
    
    def can_add(point, existing):
        """Check if adding 'point' to 'existing' set creates any isosceles triangle."""
        n = len(existing)
        for i in range(n):
            d1 = sq_dist(point, existing[i])
            for j in range(i + 1, n):
                d2 = sq_dist(point, existing[j])
                d3 = sq_dist(existing[i], existing[j])
                if d1 == d2 or d1 == d3 or d2 == d3:
                    return False
        return True
    
    # Greedy approach with randomization and multiple restarts
    best_set = []
    
    def greedy_random():
        """Greedy construction with random ordering."""
        all_points = [(x, y) for x in range(N) for y in range(N)]
        random.shuffle(all_points)
        
        result = []
        for p in all_points:
            if can_add(p, result):
                result.append(p)
        
        return result
    
    # First, try a few random greedy passes
    for _ in range(20):
        if time.time() > deadline:
            break
        candidate = greedy_random()
        if len(candidate) > len(best_set):
            best_set = candidate
    
    # Local improvement: try to add points that were rejected, or swap
    # We'll try a more sophisticated approach: iterative improvement
    for iteration in range(100):
        if time.time() > deadline:
            break
        
        current = list(best_set)
        # Try to add one more point
        all_points = [(x, y) for x in range(N) for y in range(N)]
        random.shuffle(all_points)
        
        improved = False
        for p in all_points:
            if p in current:
                continue
            if can_add(p, current):
                current.append(p)
                improved = True
                break
        
        if improved:
            if len(current) > len(best_set):
                best_set = current
        else:
            # Try local search: remove a random point, then try to add two
            if len(current) > 2:
                idx = random.randint(0, len(current) - 1)
                removed = current.pop(idx)
                # Try to add back something else
                for p in all_points:
                    if p in current or p == removed:
                        continue
                    if can_add(p, current):
                        current.append(p)
                        # Try to add one more
                        for p2 in all_points:
                            if p2 in current or p2 == removed:
                                continue
                            if can_add(p2, current):
                                current.append(p2)
                                break
                        break
                if len(current) > len(best_set):
                    best_set = current
                else:
                    # Revert
                    if removed not in current:
                        # Try to put removed back
                        pass
    
    # Final: try a deterministic greedy with sorted order as backup
    if time.time() < deadline - 5:
        # Try sorted by distance from center
        center = (N / 2 - 0.5, N / 2 - 0.5)
        all_points = [(x, y) for x in range(N) for y in range(N)]
        all_points.sort(key=lambda p: (p[0] - center[0])**2 + (p[1] - center[1])**2)
        
        result = []
        for p in all_points:
            if can_add(p, result):
                result.append(p)
        
        if len(result) > len(best_set):
            best_set = result
    
    return best_set
# EVOLVE-BLOCK-END
