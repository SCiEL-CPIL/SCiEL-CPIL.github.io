import time
import random
import sys
from math import gcd

def _check_no_four_collinear(points):
    """Check if no four points are collinear. Returns True if valid."""
    n = len(points)
    lines = {}
    for i in range(n):
        for j in range(i + 1, n):
            x1, y1 = points[i]
            x2, y2 = points[j]
            dx = x2 - x1
            dy = y2 - y1
            A = dy
            B = -dx
            C = dx * y1 - dy * x1
            if A == 0 and B == 0:
                continue
            g = gcd(gcd(abs(A), abs(B)), abs(C))
            if g != 0:
                A //= g
                B //= g
                C //= g
            if A < 0 or (A == 0 and B < 0):
                A = -A
                B = -B
                C = -C
            key = (A, B, C)
            if key not in lines:
                lines[key] = 2
            else:
                lines[key] += 1
                if lines[key] >= 4:
                    return False
    return True


def _max_independent_set(points):
    """Exact branch and bound for largest subset with no three collinear."""
    n = len(points)
    if n == 0:
        return 0

    # Precompute: for each pair (i,j), which other points k are collinear with them
    collinear_with = [[0] * n for _ in range(n)]
    for i in range(n):
        for j in range(i + 1, n):
            x1, y1 = points[i]
            x2, y2 = points[j]
            dx = x2 - x1
            dy = y2 - y1
            mask = 0
            for k in range(n):
                if k == i or k == j:
                    continue
                x3, y3 = points[k]
                if dx * (y3 - y1) == dy * (x3 - x1):
                    mask |= (1 << k)
            collinear_with[i][j] = mask
            collinear_with[j][i] = mask

    best = 0

    def backtrack(current_mask, candidates_mask):
        nonlocal best
        cand_count = candidates_mask.bit_count()
        cur_count = current_mask.bit_count()
        if cur_count + cand_count <= best:
            return
        if cand_count == 0:
            if cur_count > best:
                best = cur_count
            return

        # Pick point with highest conflict degree among candidates for better pruning
        p_idx = (candidates_mask & -candidates_mask).bit_length() - 1
        rest_mask = candidates_mask & ~(1 << p_idx)

        # Branch 1: Include p_idx
        blocked_mask = 0
        cm = current_mask
        while cm:
            r_idx = (cm & -cm).bit_length() - 1
            blocked_mask |= collinear_with[p_idx][r_idx]
            cm &= cm - 1

        new_candidates = rest_mask & ~blocked_mask
        backtrack(current_mask | (1 << p_idx), new_candidates)

        # Branch 2: Exclude p_idx
        backtrack(current_mask, rest_mask)

    backtrack(0, (1 << n) - 1)
    return best


def _build_affine_plane_config():
    """
    Build a configuration from the affine plane AG(2,5) minus 5 points.
    AG(2,5) has 25 points, 31 lines, each line has 5 points.
    We map the 25 points to integer coordinates, then remove 5 to get 20.
    The remaining 20 points will have many 3-collinear triples (from the
    original 5-point lines, after removing 2 points from each line we get 3).
    
    We use a simple bijection: map (a,b) in GF(5)^2 to integer point (a, b) 
    scaled to fit in [-100,100]. Then we need to break 4-collinear lines by
    removing points strategically.
    
    Actually, a cleaner approach: use the standard integer representation.
    AG(2,5) points are (i,j) for i,j in 0..4. Lines are sets of 5 points.
    If we keep all 25, horizontal lines have 5 collinear -> invalid.
    We need to remove points to break all 4+ collinear sets.
    
    Better: use a different mapping. Map (i,j) -> (i * 7 + j * 3, i * 3 + j * 7)
    This is a linear map from GF(5)^2 to Z^2. Lines in GF(5) map to lines in Z^2
    (since the map is injective and linear). But we still have 5-point lines.
    
    Strategy: Start with 20 points chosen from AG(2,5) such that no 4 are collinear.
    We can greedily select 20 of the 25 points avoiding 4-collinear sets.
    """
    # All 25 points of AG(2,5) in integer coords (scaled)
    scale = 15
    all_pts = []
    for i in range(5):
        for j in range(5):
            all_pts.append([i * scale, j * scale])
    
    # Greedily select 20 points with no 4 collinear
    selected = []
    random.seed(123)
    indices = list(range(25))
    random.shuffle(indices)
    
    for idx in indices:
        candidate = all_pts[idx]
        test = selected + [candidate]
        if _check_no_four_collinear(test):
            selected.append(candidate)
        if len(selected) == 20:
            break
    
    if len(selected) < 20:
        # Fallback: try different shuffles
        for seed in range(100):
            random.seed(seed)
            indices = list(range(25))
            random.shuffle(indices)
            selected = []
            for idx in indices:
                candidate = all_pts[idx]
                test = selected + [candidate]
                if _check_no_four_collinear(test):
                    selected.append(candidate)
                if len(selected) == 20:
                    break
            if len(selected) == 20:
                break
    
    return selected


def _count_triples_fast(points):
    """Count total number of collinear triples."""
    n = len(points)
    count = 0
    for i in range(n):
        for j in range(i + 1, n):
            x1, y1 = points[i]
            x2, y2 = points[j]
            dx = x2 - x1
            dy = y2 - y1
            for k in range(j + 1, n):
                x3, y3 = points[k]
                if dx * (y3 - y1) == dy * (x3 - x1):
                    count += 1
    return count


def _delta_triples(points, idx, new_point):
    """
    Compute the change in triple count when moving point idx to new_point.
    Returns (delta, is_valid) where delta = new_triples - old_triples.
    is_valid checks no 4-collinear after the move.
    """
    n = len(points)
    old_p = points[idx]
    
    # Count triples involving idx with old position
    old_count = 0
    for i in range(n):
        if i == idx:
            continue
        for j in range(i + 1, n):
            if j == idx:
                continue
            # Check if old_p, points[i], points[j] are collinear
            x1, y1 = old_p
            x2, y2 = points[i]
            x3, y3 = points[j]
            if (x2 - x1) * (y3 - y1) == (y2 - y1) * (x3 - x1):
                old_count += 1
    
    # Count triples involving idx with new position
    new_count = 0
    for i in range(n):
        if i == idx:
            continue
        for j in range(i + 1, n):
            if j == idx:
                continue
            x1, y1 = new_point
            x2, y2 = points[i]
            x3, y3 = points[j]
            if (x2 - x1) * (y3 - y1) == (y2 - y1) * (x3 - x1):
                new_count += 1
    
    delta = new_count - old_count
    
    # Check validity: no 4 collinear
    points[idx] = new_point
    valid = _check_no_four_collinear(points)
    points[idx] = old_p
    
    return delta, valid


def _simulated_annealing(points, time_limit):
    """
    Simulated annealing to maximize triple count (minimize independence number).
    Uses fast delta evaluation.
    """
    current_pts = [p[:] for p in points]
    current_triples = _count_triples_fast(current_pts)
    
    if not _check_no_four_collinear(current_pts):
        return None, float('inf')
    
    best_pts = [p[:] for p in current_pts]
    best_triples = current_triples
    
    start = time.time()
    T = 5.0  # Initial temperature
    T_min = 0.01
    cooling = 0.9995
    
    while time.time() - start < time_limit:
        # Pick a random point to move
        idx = random.randint(0, 19)
        ox, oy = current_pts[idx]
        
        # Try a random new position near the current one
        radius = random.choice([3, 5, 8, 12])
        nx = ox + random.randint(-radius, radius)
        ny = oy + random.randint(-radius, radius)
        
        if not (-100 <= nx <= 100 and -100 <= ny <= 100):
            continue
        new_p = [nx, ny]
        if new_p == current_pts[idx]:
            continue
        # Check duplicate
        dup = False
        for i in range(20):
            if i != idx and current_pts[i] == new_p:
                dup = True
                break
        if dup:
            continue
        
        delta, valid = _delta_triples(current_pts, idx, new_p)
        
        if not valid:
            continue
        
        if delta > 0:
            # Improvement: always accept
            current_pts[idx] = new_p
            current_triples += delta
            if current_triples > best_triples:
                best_triples = current_triples
                best_pts = [p[:] for p in current_pts]
        else:
            # Worsening: accept with probability exp(delta/T)
            if random.random() < min(1.0, 2.718281828 ** (delta / max(T, 0.001))):
                current_pts[idx] = new_p
                current_triples += delta
        
        T *= cooling
        if T < T_min:
            T = T_min
    
    return best_pts, best_triples


def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin
    
    best_points = None
    best_score = float('inf')
    
    # Generate initial configurations
    configs = []
    
    # Config 1: Affine plane based
    ap_config = _build_affine_plane_config()
    if ap_config is not None and len(ap_config) == 20:
        configs.append(ap_config)
    
    # Config 2: Another affine plane variant with different scale/offset
    scale2 = 12
    all_pts2 = []
    for i in range(5):
        for j in range(5):
            all_pts2.append([i * scale2 + 2, j * scale2 - 1])
    selected2 = []
    random.seed(777)
    indices2 = list(range(25))
    random.shuffle(indices2)
    for idx in indices2:
        candidate = all_pts2[idx]
        test = selected2 + [candidate]
        if _check_no_four_collinear(test):
            selected2.append(candidate)
        if len(selected2) == 20:
            break
    if len(selected2) == 20:
        configs.append(selected2)
    
    # Config 3: Random with high triple density via line clustering
    random.seed(42)
    for seed in range(3):
        random.seed(seed)
        pts = []
        while len(pts) < 20:
            x = random.randint(-50, 50)
            y = random.randint(-50, 50)
            p = [x, y]
            if p not in pts:
                pts.append(p)
        if _check_no_four_collinear(pts):
            configs.append(pts)
    
    if not configs:
        # Fallback random
        random.seed(0)
        pts = []
        while len(pts) < 20:
            x = random.randint(-100, 100)
            y = random.randint(-100, 100)
            p = [x, y]
            if p not in pts and _check_no_four_collinear(pts + [p]):
                pts.append(p)
        configs.append(pts)
    
    # Run simulated annealing on each config
    for idx, cfg in enumerate(configs):
        remaining = deadline - time.time()
        if remaining < 10:
            break
        
        time_limit = min(remaining / max(1, len(configs) - idx), 30.0)
        
        res_pts, res_triples = _simulated_annealing(cfg, time_limit)
        
        if res_pts is not None and _check_no_four_collinear(res_pts):
            # Compute actual score (independence number)
            score = _max_independent_set(res_pts)
            if score < best_score:
                best_score = score
                best_points = res_pts
                print(f"Config {idx}: Score {score}, triples {res_triples}", file=sys.stderr)
    
    # If no valid config found, return a simple valid one
    if best_points is None:
        random.seed(0)
        pts = []
        while len(pts) < 20:
            x = random.randint(-100, 100)
            y = random.randint(-100, 100)
            p = [x, y]
            if p not in pts:
                if _check_no_four_collinear(pts + [p]):
                    pts.append(p)
        best_points = pts
    
    return best_points


if __name__ == "__main__":
    pts = construct()
    print(f"Score: {_max_independent_set(pts)}")
    print(f"Valid: {_check_no_four_collinear(pts)}")
