"""No-isosceles set: find max points in [0,N)^2 with no three forming an isosceles triangle."""
import time
import random
import math

N = 64
TIME_LIMIT = 280  # Safety margin below 330s wall-clock limit

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    best_pts = []
    best_len = 0
    rng = random.Random(12345)
    
    all_points = [(x, y) for x in range(N) for y in range(N)]
    
    def dist2(a, b):
        dx = a[0] - b[0]
        dy = a[1] - b[1]
        return dx * dx + dy * dy
    
    def is_isosceles(p, q, r):
        """Check if p, q, r form an isosceles triangle (including degenerate)."""
        d_pq = dist2(p, q)
        d_pr = dist2(p, r)
        d_qr = dist2(q, r)
        # Isosceles if any two sides are equal
        if d_pq == d_pr or d_pq == d_qr or d_pr == d_qr:
            return True
        return False
    
    def can_add(p, pts):
        """Check if point p can be added to pts without creating an isosceles triangle."""
        for i in range(len(pts)):
            for j in range(i + 1, len(pts)):
                if is_isosceles(p, pts[i], pts[j]):
                    return False
        return True
    
    def greedy_construct(seed_rng):
        """Greedy construction with randomized ordering."""
        pts = []
        candidates = list(range(len(all_points)))
        seed_rng.shuffle(candidates)
        
        for idx in candidates:
            p = all_points[idx]
            if can_add(p, pts):
                pts.append(p)
            # Time check
            if time.time() - start_time > TIME_LIMIT - 5:
                break
        
        return pts
    
    def local_improvement(pts, seed_rng, max_iters=500):
        """Try to add more points via local search."""
        pts = list(pts)
        remaining = set(range(len(all_points)))
        for p in pts:
            remaining.discard(all_points.index(p))
        
        # Build a list of remaining points
        rem_list = [all_points[i] for i in remaining]
        seed_rng.shuffle(rem_list)
        
        improved = True
        iters = 0
        while improved and iters < max_iters:
            improved = False
            iters += 1
            for p in rem_list:
                if can_add(p, pts):
                    pts.append(p)
                    rem_list.remove(p)
                    improved = True
                    break
            if time.time() - start_time > TIME_LIMIT - 5:
                break
        
        return pts
    
    # Multiple randomized greedy restarts
    num_restarts = 20
    for r in range(num_restarts):
        if time.time() - start_time > TIME_LIMIT - 10:
            break
        
        restart_rng = random.Random(1000 + r * 7919)
        pts = greedy_construct(restart_rng)
        
        if len(pts) > best_len:
            best_pts = pts
            best_len = len(pts)
        
        # Local improvement on the best so far
        if len(pts) >= best_len:
            improved = local_improvement(pts, restart_rng)
            if len(improved) > best_len:
                best_pts = improved
                best_len = len(best_pts)
        
        # Check time
        if time.time() - start_time > TIME_LIMIT - 15:
            break
    
    # Final local improvement pass on best
    if time.time() - start_time < TIME_LIMIT - 10 and best_len > 0:
        final_rng = random.Random(99999)
        improved = local_improvement(best_pts, final_rng, max_iters=1000)
        if len(improved) > best_len:
            best_pts = improved
            best_len = len(best_pts)
    
    return [(x, y) for x, y in best_pts]
# EVOLVE-BLOCK-END
