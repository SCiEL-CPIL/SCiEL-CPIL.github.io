"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import math

N = 64

# EVOLVE-BLOCK-START
def construct():
    deadline = time.time() + 290  # safety margin before 330s kill
    
    def is_valid_addition(points_set, px, py, dist_cache):
        """Check if adding (px,py) keeps all distances from each existing point distinct."""
        for (qx, qy) in points_set:
            d = (px - qx) ** 2 + (py - qy) ** 2
            # Check if this distance already exists from (qx,qy) to another point
            if d in dist_cache.get((qx, qy), set()):
                return False
        # Also check: from (px,py) to all existing points, distances must be distinct
        seen_dists = set()
        for (qx, qy) in points_set:
            d = (px - qx) ** 2 + (py - qy) ** 2
            if d in seen_dists:
                return False
            seen_dists.add(d)
        return True
    
    def greedy_construct(random_seed, order_mode):
        random.seed(random_seed)
        points = []
        points_set = set()
        # dist_cache[(qx,qy)] = set of squared distances from (qx,qy) to other selected points
        dist_cache = {}
        
        # Generate candidate points
        if order_mode == 0:
            # All grid points, shuffled
            candidates = [(x, y) for x in range(N) for y in range(N)]
            random.shuffle(candidates)
        elif order_mode == 1:
            # Sparse grid with step 2
            candidates = [(x, y) for x in range(0, N, 2) for y in range(0, N, 2)]
            random.shuffle(candidates)
        elif order_mode == 2:
            # Sparse grid with step 3
            candidates = [(x, y) for x in range(0, N, 3) for y in range(0, N, 3)]
            random.shuffle(candidates)
        elif order_mode == 3:
            # All points, sorted by some hash
            candidates = [(x, y) for x in range(N) for y in range(N)]
            random.shuffle(candidates)
        else:
            # Mixed: some dense, some sparse
            candidates = []
            for x in range(N):
                for y in range(N):
                    if (x + y) % 2 == 0:
                        candidates.append((x, y))
            random.shuffle(candidates)
        
        for (px, py) in candidates:
            if time.time() > deadline:
                break
            if is_valid_addition(points_set, px, py, dist_cache):
                points.append((px, py))
                points_set.add((px, py))
                # Update dist_cache
                dist_cache[(px, py)] = set()
                for (qx, qy) in points_set:
                    if (qx, qy) == (px, py):
                        continue
                    d = (px - qx) ** 2 + (py - qy) ** 2
                    dist_cache[(px, py)].add(d)
                    if (qx, qy) not in dist_cache:
                        dist_cache[(qx, qy)] = set()
                    dist_cache[(qx, qy)].add(d)
        
        return points
    
    best = []
    
    # Try multiple random seeds and order modes
    for seed in range(200):
        if time.time() > deadline:
            break
        for mode in range(5):
            if time.time() > deadline:
                break
            pts = greedy_construct(seed * 100 + mode, mode)
            if len(pts) > len(best):
                best = pts
    
    # If we haven't found anything good, fall back to the line-based Sidon approach
    if len(best) < 5:
        xs, used = [], set()
        x = 0
        while x < N:
            diffs = {abs(x - y) for y in xs}
            if not (diffs & used):
                used |= diffs
                xs.append(x)
            x += 1
        best = [(x, 0) for x in xs]
    
    return best
# EVOLVE-BLOCK-END
