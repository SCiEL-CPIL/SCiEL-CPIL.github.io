"""No-isosceles set: construct maximum points in [0,N)^2 with no isosceles triangle."""
import time
import random
import math

N = 64
TIME_LIMIT = 290  # Safety margin below 330s

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    
    def elapsed():
        return time.time() - start_time
    
    def dist2(a, b):
        dx = a[0] - b[0]
        dy = a[1] - b[1]
        return dx * dx + dy * dy
    
    def check_no_isosceles(pts):
        """Check if a set of points has no three forming an isosceles triangle."""
        n = len(pts)
        for i in range(n):
            for j in range(i + 1, n):
                d2 = dist2(pts[i], pts[j])
                for k in range(n):
                    if k == i or k == j:
                        continue
                    if d2 == dist2(pts[i], pts[k]) or d2 == dist2(pts[j], pts[k]):
                        # Check if it's actually a triangle (not collinear with same distance)
                        # An isosceles triangle requires the two equal sides to meet at a vertex
                        # If d2 == dist2(pts[i], pts[k]), then pts[i] is the apex with equal legs to pts[j] and pts[k]
                        # This is valid as long as pts[j] != pts[k] (which they aren't since distinct points)
                        # But we need to make sure it's not degenerate (all three collinear with apex in middle)
                        # Actually, three distinct points always form a triangle unless collinear
                        # If collinear and distances are equal, it's still "isosceles" in the degenerate sense
                        # The problem says "forming an isosceles triangle" - typically this means non-degenerate
                        # Let's check: if points are collinear and two distances from apex are equal,
                        # the third point would be on one side, so it's not a triangle
                        # Let's verify non-collinearity
                        ax, ay = pts[i]
                        bx, by = pts[j]
                        cx, cy = pts[k]
                        cross = (bx - ax) * (cy - ay) - (by - ay) * (cx - ax)
                        if cross != 0:
                            return False
        return True
    
    def is_valid_addition(p, pts):
        """Check if adding point p to pts keeps the set isosceles-free."""
        for i, a in enumerate(pts):
            for j in range(i + 1, len(pts)):
                b = pts[j]
                d_pa = dist2(p, a)
                d_pb = dist2(p, b)
                d_ab = dist2(a, b)
                # Check if any two of d_pa, d_pb, d_ab are equal
                # and the three points are not collinear
                if d_pa == d_pb or d_pa == d_ab or d_pb == d_ab:
                    # Check non-collinearity
                    px, py = p
                    ax, ay = a
                    bx, by = b
                    cross = (ax - px) * (by - py) - (ay - py) * (bx - px)
                    if cross != 0:
                        return False
        return True
    
    def greedy_construct(order, max_points=None):
        """Greedy: add points in given order if valid."""
        pts = []
        for p in order:
            if is_valid_addition(p, pts):
                pts.append(p)
                if max_points and len(pts) >= max_points:
                    break
        return pts
    
    # All grid points
    all_points = [(x, y) for x in range(N) for y in range(N)]
    
    # Try multiple random orders and greedy
    best_pts = []
    best_len = 0
    
    # First, try a structured approach: random permutations
    rng = random.Random(42)
    
    for iteration in range(2000):
        if elapsed() > TIME_LIMIT - 10:
            break
        
        # Create a random order with some structure
        # Mix of random and semi-structured
        if iteration % 3 == 0:
            # Pure random
            order = all_points[:]
            rng.shuffle(order)
        elif iteration % 3 == 1:
            # Row-by-row with random row order
            rows = list(range(N))
            rng.shuffle(rows)
            order = []
            for r in rows:
                row_pts = [(x, r) for x in range(N)]
                rng.shuffle(row_pts)
                order.extend(row_pts)
        else:
            # Diagonal-based random
            order = all_points[:]
            rng.shuffle(order)
        
        pts = greedy_construct(order)
        
        if len(pts) > best_len:
            best_len = len(pts)
            best_pts = pts[:]
    
    # Try local improvement: for each point in best set, try swapping with a random point
    if best_pts:
        improvement_rounds = 0
        while improvement_rounds < 50:
            if elapsed() > TIME_LIMIT - 10:
                break
            
            improved = False
            pts_copy = best_pts[:]
            removed = None
            
            # Try removing each point and adding a better one
            for idx in range(len(pts_copy)):
                if elapsed() > TIME_LIMIT - 10:
                    break
                
                removed = pts_copy.pop(idx)
                
                # Try to add a random point
                for _ in range(20):
                    if elapsed() > TIME_LIMIT - 10:
                        break
                    candidate = rng.choice(all_points)
                    if candidate not in pts_copy:
                        if is_valid_addition(candidate, pts_copy):
                            pts_copy.append(candidate)
                            if len(pts_copy) > best_len:
                                best_len = len(pts_copy)
                                best_pts = pts_copy[:]
                                improved = True
                            break
                
                if not improved:
                    pts_copy.append(removed)
            
            improvement_rounds += 1
            if not improved:
                break
    
    # Final verification
    if best_pts and not check_no_isosceles(best_pts):
        # Fallback to a simple valid set
        # Points on a line with Sidon-like spacing
        xs, used = [], set()
        x = 0
        while x < N:
            diffs = {abs(x - y) for y in xs}
            if not (diffs & used):
                used |= diffs
                xs.append(x)
            x += 1
        best_pts = [(x, 0) for x in xs]
    
    return best_pts
# EVOLVE-BLOCK-END
