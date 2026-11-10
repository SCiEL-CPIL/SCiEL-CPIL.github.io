"""Combined: multi-start two-parabola init + efficient bitmask B&B + greedy local search."""
import time
import random
from itertools import combinations

_DEADLINE = 140.0  # safety margin before 150s kill

def _elapsed(start):
    return time.time() - start

def _cross(p1, p2, p3):
    return (p2[0] - p1[0]) * (p3[1] - p1[1]) - (p2[1] - p1[1]) * (p3[0] - p1[0])

def _check_no_four_collinear(points):
    n = len(points)
    for i in range(n):
        for j in range(i + 1, n):
            x1, y1 = points[i]
            x2, y2 = points[j]
            dx = x2 - x1
            dy = y2 - y1
            count = 2
            for k in range(n):
                if k == i or k == j:
                    continue
                x3, y3 = points[k]
                if dx * (y3 - y1) == dy * (x3 - x1):
                    count += 1
            if count >= 4:
                return False
    return True

def _build_triple_masks(points):
    """Build list of triple bitmasks (each triple is a 3-bit mask over point indices)."""
    n = len(points)
    # Group points by line: for each pair, find the line, collect all points on it
    # Use normalized line representation
    from math import gcd
    
    lines = {}  # key -> list of point indices on that line
    for i in range(n):
        for j in range(i + 1, n):
            x1, y1 = points[i]
            x2, y2 = points[j]
            dx = x2 - x1
            dy = y2 - y1
            # Normalize direction
            if dx == 0:
                key = (0, 1, x1)  # vertical line x = x1
            elif dy == 0:
                key = (1, 0, y1)  # horizontal line y = y1
            else:
                g = gcd(abs(dx), abs(dy))
                dx_n = dx // g
                dy_n = dy // g
                if dx_n < 0:
                    dx_n = -dx_n
                    dy_n = -dy_n
                # Line: y - y1 = (dy_n/dx_n)(x - x1) => dy_n*x - dx_n*y = dy_n*x1 - dx_n*y1
                c = dy_n * x1 - dx_n * y1
                key = (dx_n, dy_n, c)
            
            if key not in lines:
                lines[key] = []
            lines[key].append(i)
            if j not in lines[key]:
                lines[key].append(j)
    
    # Deduplicate and build triples
    seen_triples = set()
    triple_masks = []
    for pts_on_line in lines.values():
        # Deduplicate
        unique_pts = list(dict.fromkeys(pts_on_line))
        m = len(unique_pts)
        if m >= 3:
            for combo in combinations(range(m), 3):
                triple = tuple(sorted([unique_pts[combo[0]], unique_pts[combo[1]], unique_pts[combo[2]]]))
                if triple not in seen_triples:
                    seen_triples.add(triple)
                    mask = (1 << triple[0]) | (1 << triple[1]) | (1 << triple[2])
                    triple_masks.append(mask)
    
    return triple_masks

def _max_independent_set_size(triple_masks, n, timeout=5.0, start_time=None):
    """
    Exact branch and bound: maximize kept points such that no triple is fully kept.
    Equivalent to: minimize removals to hit all triples.
    """
    if start_time is None:
        start_time = time.time()
    
    if not triple_masks:
        return n
    
    # Sort triples: put those with fewer "free" points first for better pruning
    # Actually, just use them as-is for simplicity
    
    best_found = [0]
    
    # Precompute: for each point, which triples it's in
    point_in_triples = [[] for _ in range(n)]
    for t_idx, mask in enumerate(triple_masks):
        for p in range(n):
            if mask & (1 << p):
                point_in_triples[p].append(t_idx)
    
    # Degree: number of triples each point participates in
    degrees = [len(point_in_triples[p]) for p in range(n)]
    
    def _search(available_mask, remaining_triples, removed_count):
        nonlocal best_found
        if time.time() - start_time > timeout:
            return
        
        # Prune: upper bound on kept = n - removed_count
        if n - removed_count <= best_found[0]:
            return
        
        if not remaining_triples:
            kept = available_mask.bit_count()
            if kept > best_found[0]:
                best_found[0] = kept
            return
        
        # Pick a triple: prefer one with the fewest available candidates (most constrained)
        best_t_idx = -1
        best_candidates = None
        best_cand_count = n + 1
        
        for t_idx in remaining_triples:
            t_mask = triple_masks[t_idx]
            candidates = []
            for p in range(n):
                if t_mask & (1 << p) and available_mask & (1 << p):
                    candidates.append(p)
            if len(candidates) < best_cand_count:
                best_cand_count = len(candidates)
                best_candidates = candidates
                best_t_idx = t_idx
            if best_cand_count <= 1:
                break
        
        if best_cand_count == 0:
            # This triple is already hit
            _search(available_mask, remaining_triples[1:], removed_count)
            return
        
        if best_cand_count == 1:
            # Forced: must remove this point
            p = best_candidates[0]
            new_available = available_mask & ~(1 << p)
            new_remaining = [idx for idx in remaining_triples if not (triple_masks[idx] & (1 << p))]
            _search(new_available, new_remaining, removed_count + 1)
            return
        
        # Branch: try removing each candidate
        # Order by degree (remove high-degree points first for better pruning)
        best_candidates.sort(key=lambda p: -degrees[p])
        
        for p in best_candidates:
            new_available = available_mask & ~(1 << p)
            new_remaining = [idx for idx in remaining_triples if not (triple_masks[idx] & (1 << p))]
            _search(new_available, new_remaining, removed_count + 1)
            if time.time() - start_time > timeout:
                return
    
    all_mask = (1 << n) - 1
    _search(all_mask, list(range(len(triple_masks))), 0)
    return best_found[0]

def _score(points, start_time=None, timeout=8.0):
    if start_time is None:
        start_time = time.time()
    triple_masks = _build_triple_masks(points)
    return _max_independent_set_size(triple_masks, len(points), timeout=timeout, start_time=start_time)

def _two_parabola(shift=1, x_range=None):
    """Generate 20 points from two shifted parabolas."""
    if x_range is None:
        x_range = list(range(-5, 5))
    points = []
    for x in x_range:
        points.append([x, x * x])
    for x in x_range:
        points.append([x, x * x + shift])
    
    # Check distinctness
    seen = set()
    unique = []
    for p in points:
        key = tuple(p)
        if key not in seen:
            seen.add(key)
            unique.append(p)
    
    if len(unique) != 20:
        return None
    if not all(abs(p[0]) <= 100 and abs(p[1]) <= 100 for p in unique):
        return None
    return unique

def _random_config(seed=None):
    """Generate a random valid configuration."""
    rng = random.Random(seed)
    points = []
    seen = set()
    attempts = 0
    while len(points) < 20 and attempts < 50000:
        x = rng.randint(-100, 100)
        y = rng.randint(-100, 100)
        key = (x, y)
        if key not in seen:
            seen.add(key)
            points.append([x, y])
        attempts += 1
    
    if len(points) != 20:
        return None
    return points

def _perturb(points, rng):
    """Perturb one or two points slightly."""
    new_points = [list(p) for p in points]
    num_perturb = rng.choice([1, 1, 2])
    for _ in range(num_perturb):
        idx = rng.randint(0, 19)
        dx = rng.randint(-4, 4)
        dy = rng.randint(-4, 4)
        nx = new_points[idx][0] + dx
        ny = new_points[idx][1] + dy
        if -100 <= nx <= 100 and -100 <= ny <= 100:
            new_points[idx] = [nx, ny]
    
    seen = set()
    for p in new_points:
        key = tuple(p)
        if key in seen:
            return None
        seen.add(key)
    return new_points

def _local_search(initial_points, start_time, deadline, rng):
    """Greedy hill-climbing local search."""
    current = [list(p) for p in initial_points]
    current_score = _score(current, start_time=start_time, timeout=8.0)
    best_points = [list(p) for p in current]
    best_score = current_score
    
    stall = 0
    while _elapsed(start_time) < deadline:
        perturbed = _perturb(current, rng)
        if perturbed is None:
            stall += 1
            if stall > 50:
                break
            continue
        
        if not _check_no_four_collinear(perturbed):
            stall += 1
            if stall > 50:
                break
            continue
        
        s = _score(perturbed, start_time=start_time, timeout=8.0)
        if s < current_score:
            current = perturbed
            current_score = s
            best_points = [list(p) for p in current]
            best_score = s
            stall = 0
        else:
            stall += 1
            if stall > 100:
                break
    
    return best_points, best_score

def construct():
    start_time = time.time()
    deadline = start_time + _DEADLINE
    
    best_points = None
    best_score = 999
    rng = random.Random(42)
    
    # Multi-start: try several initial configurations
    initial_configs = []
    
    # Two-parabola configurations with various shifts
    for shift in [1, 2, 3, -1, -2, 5, 7, -3]:
        config = _two_parabola(shift)
        if config is not None and _check_no_four_collinear(config):
            initial_configs.append(config)
    
    # Also try with different x ranges
    for shift in [1, 2, -1]:
        for xr in [list(range(-6, 4)), list(range(-4, 6)), list(range(-7, 3))]:
            config = _two_parabola(shift, xr)
            if config is not None and _check_no_four_collinear(config):
                initial_configs.append(config)
    
    # Add a couple of random configs
    for seed in [1, 2]:
        config = _random_config(seed)
        if config is not None and _check_no_four_collinear(config):
            initial_configs.append(config)
    
    # Fallback: single parabola
    if not initial_configs:
        initial_configs.append([[x, x * x] for x in range(-10, 10)])
    
    # Try each initial config with local search
    for i, config in enumerate(initial_configs):
        if _elapsed(start_time) > deadline - 10:
            break
        
        result_points, result_score = _local_search(config, start_time, deadline - 5, rng)
        
        if result_score < best_score:
            best_score = result_score
            best_points = [list(p) for p in result_points]
        
        # If we've found a very good score, we can stop early
        if best_score <= 10:
            break
    
    # Final validity check
    if best_points is None:
        best_points = [[x, x * x] for x in range(-10, 10)]
    
    if len(set(tuple(p) for p in best_points)) != 20:
        best_points = [[x, x * x] for x in range(-10, 10)]
    
    if not _check_no_four_collinear(best_points):
        best_points = [[x, x * x] for x in range(-10, 10)]
    
    for p in best_points:
        if abs(p[0]) > 100 or abs(p[1]) > 100:
            best_points = [[x, x * x] for x in range(-10, 10)]
            break
    
    return best_points
