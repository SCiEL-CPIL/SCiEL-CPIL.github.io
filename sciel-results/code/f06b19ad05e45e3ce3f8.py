import random
import time

N = 64

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 285.0  # safety margin before 330s kill
    
    # Precompute all grid points
    all_points = [(x, y) for x in range(N) for y in range(N)]
    point_index = {p: i for i, p in enumerate(all_points)}
    num_points = len(all_points)
    
    def sq_dist(p1, p2):
        dx = p1[0] - p2[0]
        dy = p1[1] - p2[1]
        return dx * dx + dy * dy
    
    def is_isosceles(a, b, c):
        """Check if triangle with vertices a, b, c is isosceles (including degenerate)."""
        d_ab = sq_dist(a, b)
        d_bc = sq_dist(b, c)
        d_ca = sq_dist(c, a)
        if d_ab == d_bc or d_ab == d_ca or d_bc == d_ca:
            return True
        return False
    
    def can_add(points, p):
        """Check if adding point p to existing points creates an isosceles triangle."""
        n = len(points)
        for i in range(n):
            a = points[i]
            da = sq_dist(p, a)
            for j in range(i + 1, n):
                b = points[j]
                db = sq_dist(p, b)
                # Check if da == db or da == dist(a,b) or db == dist(a,b)
                # But we need dist(a,b) too... let's compute it
                # Actually we need all three distances
                dc = sq_dist(a, b)
                if da == db or da == dc or db == dc:
                    return False
        return True
    
    def greedy_random(seed=0):
        """Greedy algorithm with randomization."""
        random.seed(seed)
        points = []
        remaining = all_points[:]
        random.shuffle(remaining)
        
        for p in remaining:
            if time.time() > deadline:
                break
            if can_add(points, p):
                points.append(p)
        
        return points
    
    def local_search(points, max_adds=3000, max_swaps=300, max_replaces=100):
        """Local search: try adding, swapping, and replacing points."""
        points = list(points)
        point_set = set(points)
        
        def is_valid_set(pts):
            """Check if the set of points has no isosceles triangles."""
            n = len(pts)
            for i in range(n):
                for j in range(i + 1, n):
                    for k in range(j + 1, n):
                        if is_isosceles(pts[i], pts[j], pts[k]):
                            return False
            return True
        
        # Phase 1: Try adding points
        added_count = 0
        candidates = [p for p in all_points if p not in point_set]
        random.shuffle(candidates)
        for p in candidates:
            if added_count >= max_adds:
                break
            if time.time() > deadline:
                break
            if can_add(points, p):
                points.append(p)
                point_set.add(p)
                added_count += 1
        
        # Phase 2: Try swapping one point for another
        swapped_count = 0
        attempts = 0
        while swapped_count < max_swaps and attempts < 5000:
            if time.time() > deadline:
                break
            attempts += 1
            # Pick a random existing point to potentially remove
            idx = random.randint(0, len(points) - 1)
            remove_p = points[idx]
            # Try adding a random new point
            add_p = random.choice(all_points)
            if add_p in point_set:
                continue
            # Check if removing remove_p and adding add_p keeps validity
            new_points = points[:idx] + points[idx+1:] + [add_p]
            # Check validity of new set
            valid = True
            n = len(new_points)
            # Only need to check triangles involving add_p or that were affected by removal
            # For efficiency, check all triangles with add_p
            for i in range(n):
                if new_points[i] == add_p:
                    continue
                for j in range(i + 1, n):
                    if new_points[j] == add_p:
                        continue
                    if new_points[i] == remove_p or new_points[j] == remove_p:
                        # This triangle existed before, so it was valid
                        continue
                    if is_isosceles(new_points[i], new_points[j], add_p):
                        valid = False
                        break
                if not valid:
                    break
            if valid:
                # Also need to check that no new isosceles was created among remaining points
                # But removing a point can't create new isosceles, so we're fine
                points = new_points
                point_set.discard(remove_p)
                point_set.add(add_p)
                swapped_count += 1
        
        # Phase 3: Try replacing one point with another (similar to swap but more targeted)
        replaced_count = 0
        attempts = 0
        while replaced_count < max_replaces and attempts < 3000:
            if time.time() > deadline:
                break
            attempts += 1
            idx = random.randint(0, len(points) - 1)
            remove_p = points[idx]
            # Try all points not in set
            for add_p in all_points:
                if add_p in point_set:
                    continue
                # Quick check: can add_p be added if remove_p were not there?
                # Check triangles involving add_p with all other points except remove_p
                valid = True
                for other in points:
                    if other == remove_p:
                        continue
                    # Check if add_p and other form isosceles with any third point
                    # This is complex; let's just do a full check for small sets
                    pass
                # Full validity check
                new_points = points[:idx] + points[idx+1:] + [add_p]
                n = len(new_points)
                for i in range(n):
                    for j in range(i + 1, n):
                        for k in range(j + 1, n):
                            if is_isosceles(new_points[i], new_points[j], new_points[k]):
                                valid = False
                                break
                        if not valid:
                            break
                    if not valid:
                        break
                if valid:
                    points = new_points
                    point_set.discard(remove_p)
                    point_set.add(add_p)
                    replaced_count += 1
                    break
        
        return points
    
    # First, try the simple line-based approach from the seed
    def line_sidon():
        xs, used = [], set()
        x = 0
        while x < N:
            diffs = {abs(x - y) for y in xs}
            if not (diffs & used):
                used |= diffs
                xs.append(x)
            x += 1
        return [(x, 0) for x in xs]
    
    best = line_sidon()
    best_len = len(best)
    
    # Try deterministic greedy
    if time.time() < deadline:
        det_result = greedy_random(seed=12345)
        if len(det_result) > best_len:
            best = det_result
            best_len = len(det_result)
    
    # Try multiple random seeds with local search
    seed = 0
    while time.time() < deadline - 10.0:
        result = greedy_random(seed)
        # Apply local search
        result = local_search(result, max_adds=3000, max_swaps=300, max_replaces=100)
        if len(result) > best_len:
            best = result
            best_len = len(result)
        seed += 1
    
    return best

# EVOLVE-BLOCK-END
