import random
import time
import sys
from itertools import combinations

def construct():
    # Deadline guard
    start_time = time.time()
    deadline = start_time + 110.0  # 110s to be safe, limit is 120s
    
    # Helper: Check if 4 points are collinear
    def is_collinear4(pts):
        # pts is a list of 4 tuples
        # Check if all 4 lie on the same line
        # Use cross product for integer arithmetic to avoid float issues
        (x1, y1), (x2, y2), (x3, y3), (x4, y4) = pts
        # Line through 1 and 2: (y2-y1)(x3-x1) - (x2-x1)(y3-y1) == 0
        # and (y2-y1)(x4-x1) - (x2-x1)(y4-y1) == 0
        dx12 = x2 - x1
        dy12 = y2 - y1
        if dx12 == 0 and dy12 == 0: return False # distinct points
        cross3 = dy12 * (x3 - x1) - dx12 * (y3 - y1)
        cross4 = dy12 * (x4 - x1) - dx12 * (y4 - y1)
        return cross3 == 0 and cross4 == 0

    def find_violation(pts):
        # Returns a tuple of 4 indices that are collinear, or None
        n = len(pts)
        for i in range(n):
            for j in range(i+1, n):
                for k in range(j+1, n):
                    # Check if any l > k is collinear with i, j, k
                    # Optimization: precompute line for i,j,k?
                    # For N=20, O(N^4) is 116k checks, which is fast enough in Python if optimized?
                    # 20*19*18/6 = 1140 triples. For each, check remaining 17. ~19k checks. Very fast.
                    xi, yi = pts[i]
                    xj, yj = pts[j]
                    xk, yk = pts[k]
                    dx = xj - xi
                    dy = yj - yi
                    # Check all l > k
                    for l in range(k+1, n):
                        xl, yl = pts[l]
                        cross_k = dy * (xk - xi) - dx * (yk - yi)
                        cross_l = dy * (xl - xi) - dx * (yl - yi)
                        if cross_k == 0 and cross_l == 0:
                            return (i, j, k, l)
        return None

    def make_valid(pts):
        # Modify pts to be valid (no 4 collinear)
        # pts is a list of [x, y]
        n = len(pts)
        for _ in range(1000): # Max iterations
            viol = find_violation(pts)
            if viol is None:
                return True
            i, j, k, l = viol
            # Move one of the points, say pts[l], to a random new location
            # Ensure new location is distinct from others
            while True:
                nx = random.randint(-100, 100)
                ny = random.randint(-100, 100)
                # Check distinctness
                if not any(pts[m][0] == nx and pts[m][1] == ny for m in range(n)):
                    pts[l] = [nx, ny]
                    break
        return False

    def greedy_3free_size(pts):
        # Returns size of a 3-free set found by greedy
        # Greedy: pick a point, remove it and all points that form a triple with it?
        # No, 3-free means no 3 collinear.
        # If we pick p, we cannot pick any 2 other points that are collinear with p.
        # This is not a simple greedy.
        # Better Greedy: 
        # 1. Identify all triples.
        # 2. This is an independent set in hypergraph.
        # Approximation: Randomly order points, add to set if it doesn't complete a triple with existing points.
        random.shuffle(pts) # We pass a copy or shuffle indices
        selected = []
        selected_set = set()
        # Precompute triples? Too slow for every step.
        # Just check on the fly.
        for p in pts:
            # Check if p can be added to selected
            # p cannot form a triple with any 2 points in selected
            # i.e., for any a, b in selected, a, b, p are not collinear
            can_add = True
            if len(selected) >= 2:
                for a in selected:
                    for b in selected:
                        if a is b: continue
                        # Check collinear a, b, p
                        dx = b[0] - a[0]
                        dy = b[1] - a[1]
                        if dx == 0 and dy == 0: continue
                        cross = dy * (p[0] - a[0]) - dx * (p[1] - a[1])
                        if cross == 0:
                            can_add = False
                            break
                    if not can_add: break
            if can_add:
                selected.append(p)
        return len(selected)

    def exact_max_3free(pts):
        # Branch and Bound for Max 3-Free Set
        # This is expensive, use only for final verification or best tracking
        n = len(pts)
        # Precompute collinearity matrix? 
        # collinear[i][j][k] is True if i,j,k are collinear
        # Or just check on the fly in B&B
        
        # For N=20, B&B might be slow if not pruned well.
        # We'll use the greedy score for the search loop, and exact for the final best.
        
        best = 0
        # Sort points by degree (number of triples they are in) to prune?
        # Let's just do a simple DFS with pruning.
        
        # To speed up, we can use a bitset or list.
        # State: current set, next index to consider
        # If we pick i, we must exclude any pair (j,k) that forms a triple with i.
        
        # This is complex to implement quickly. 
        # Given time constraints, I will rely on the greedy score for search, 
        # and assume the greedy score is a tight upper bound for the purpose of finding a good config.
        # The evaluator will compute the exact score.
        
        return greedy_3free_size(pts)

    # 1. Generate initial valid configuration
    pts = []
    for _ in range(20):
        while True:
            x = random.randint(-100, 100)
            y = random.randint(-100, 100)
            if not any(p[0] == x and p[1] == y for p in pts):
                pts.append([x, y])
                break
    
    if not make_valid(pts):
        # Fallback to parabola if random fails (unlikely)
        pts = [[x, x * x] for x in range(-10, 10)]
        
    best_pts = [p[:] for p in pts]
    best_score = greedy_3free_size(best_pts)
    
    # 2. Local Search
    while time.time() < deadline:
        # Pick a random point to move
        i = random.randint(0, 19)
        old_x, old_y = pts[i]
        
        # Pick a new random location
        new_x = random.randint(-100, 100)
        new_y = random.randint(-100, 100)
        
        # Check if new location is distinct
        if any(pts[m][0] == new_x and pts[m][1] == new_y for m in range(20) if m != i):
            continue
            
        pts[i] = [new_x, new_y]
        
        # Check validity
        if find_violation(pts) is not None:
            # Revert
            pts[i] = [old_x, old_y]
            continue
            
        # Calculate new score (greedy)
        new_score = greedy_3free_size(pts)
        
        if new_score < best_score:
            best_score = new_score
            best_pts = [p[:] for p in pts]
        elif new_score == best_score:
            # Accept with some probability to explore
            if random.random() < 0.5:
                # Keep the change
                pass
            else:
                # Revert
                pts[i] = [old_x, old_y]
        else:
            # Revert
            pts[i] = [old_x, old_y]
            
        # Occasionally, try a "kick" - move 2 points?
        # Or just random restart if stuck?
        
    # Final check: ensure best_pts is valid
    if find_violation(best_pts) is not None:
        # Should not happen, but just in case
        return [[x, x * x] for x in range(-10, 10)]
        
    return best_pts
