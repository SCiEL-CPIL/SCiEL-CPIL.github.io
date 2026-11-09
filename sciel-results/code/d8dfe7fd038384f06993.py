"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import math

N = 64
TIME_LIMIT = 280  # Safety margin below 330s

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    best_pts = []
    best_len = 0
    
    # Precompute all grid points
    all_points = [(x, y) for x in range(N) for y in range(N)]
    
    def dist2(a, b):
        return (a[0] - b[0])**2 + (a[1] - b[1])**2
    
    def is_valid_addition(p, pts):
        """Check if adding p to pts creates any isosceles triangle."""
        k = len(pts)
        if k == 0:
            return True
        if k == 1:
            return True
        # For each pair of existing points, check if p forms an isosceles triangle
        for i in range(k):
            for j in range(i + 1, k):
                d_ij = dist2(pts[i], pts[j])
                d_ip = dist2(pts[i], p)
                d_jp = dist2(pts[j], p)
                # Check all three ways to form isosceles
                if d_ij == d_ip or d_ij == d_jp or d_ip == d_jp:
                    return False
        return True
    
    def greedy_construct(pts, time_limit):
        """Continue building pts greedily until time runs out or no more can be added."""
        remaining = [p for p in all_points if p not in pts]
        rng = random.Random(42)
        
        while remaining:
            if time.time() - start_time > time_limit:
                break
            
            # Try to find a point that can be added
            added = False
            for _ in range(min(len(remaining), 100)):
                idx = rng.randint(0, len(remaining) - 1)
                p = remaining[idx]
                if is_valid_addition(p, pts):
                    pts.append(p)
                    remaining.pop(idx)
                    added = True
                    break
            
            if not added:
                break
        
        return pts
    
    # Run multiple attempts with different random seeds
    for seed in range(1000):
        if time.time() - start_time > TIME_LIMIT:
            break
        
        rng = random.Random(seed)
        pts = []
        
        # Start with a random point
        if all_points:
            pts.append(rng.choice(all_points))
        
        # Greedily add points
        pts = greedy_construct(pts, TIME_LIMIT)
        
        if len(pts) > best_len:
            best_len = len(pts)
            best_pts = pts[:]
    
    return best_pts
# EVOLVE-BLOCK-END
