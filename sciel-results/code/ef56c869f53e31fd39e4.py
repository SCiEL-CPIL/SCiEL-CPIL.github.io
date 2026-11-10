import time
import random
from math import gcd
from functools import lru_cache

_START_TIME = None
_DEADLINE = 145.0

def _elapsed():
    global _START_TIME
    if _START_TIME is None:
        _START_TIME = time.time()
        return 0.0
    return time.time() - _START_TIME

def _check_no_four_collinear(points):
    """Returns True if no 4 points are collinear."""
    n = len(points)
    for i in range(n):
        for j in range(i + 1, n):
            x1, y1 = points[i]
            x2, y2 = points[j]
            dx = x2 - x1
            dy = y2 - y1
            if dx == 0 and dy == 0:
                continue
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

def _build_triple_data(points):
    """
    Build all collinear triples and precompute for fast B&B.
    Returns:
      - triples: list of (i, j, k) tuples (0-indexed)
      - triple_mask_for_pair: dict mapping (min(i,j), max(i,j)) -> bitmask of k values
    """
    n = len(points)
    triples = []
    triple_set = set()
    triple_mask_for_pair = {}
    
    for i in range(n):
        for j in range(i + 1, n):
            x1, y1 = points[i]
            x2, y2 = points[j]
            dx = x2 - x1
            dy = y2 - y1
            if dx == 0 and dy == 0:
                continue
            line_pts = []
            for k in range(n):
                if k == i or k == j:
                    continue
                x3, y3 = points[k]
                if dx * (y3 - y1) == dy * (x3 - x1):
                    line_pts.append(k)
            m = len(line_pts)
            if m >= 1:
                for k in line_pts:
                    t = (min(i, j, k), min(max(i, j, k), max(i, j, k)), max(i, j, k))
                    if t not in triple_set:
                        triple_set.add(t)
                        triples.append(t)
                    # For the pair (i,j), k forms a triple
                    pair_key = (min(i, j), max(i, j))
                    if pair_key not in triple_mask_for_pair:
                        triple_mask_for_pair[pair_key] = 0
                    triple_mask_for_pair[pair_key] |= (1 << k)
    
    return triples, triple_mask_for_pair

def _max_subset_no_three_collinear(points):
    """
    Exact branch and bound to find largest subset with no three collinear.
    Uses bitmask-based conflict checking for speed.
    """
    n = len(points)
    triples, triple_mask_for_pair = _build_triple_data(points)
    
    if not triples:
        return n
    
    # Precompute: for each point i, list of (j, k) pairs such that {i, j, k} is a triple
    # and the mask of j's that conflict with i (i.e., there exists k with {i,j,k} triple)
    conflict_mask = [0] * n
    for (a, b, c) in triples:
        conflict_mask[a] |= (1 << b) | (1 << c)
        conflict_mask[b] |= (1 << a) | (1 << c)
        conflict_mask[c] |= (1 << a) | (1 << b)
    
    # Order points by decreasing conflict degree for better pruning
    degrees = [bin(conflict_mask[i]).count('1') for i in range(n)]
    order = sorted(range(n), key=lambda i: -degrees[i])
    
    # Remap: pos[i] = original index at position i
    # orig_to_pos[orig] = position
    pos = order
    orig_to_pos = [0] * n
    for p, o in enumerate(order):
        orig_to_pos[o] = p
    
    # Rebuild conflict in sorted order
    sorted_conflict = [0] * n
    for i in range(n):
        for j in range(n):
            if conflict_mask[i] & (1 << j):
                sorted_conflict[orig_to_pos[i]] |= (1 << orig_to_pos[j])
    
    # For fast checking: when we have selected set S (as bitmask in sorted order),
    # can we add point p (in sorted order)?
    # We need: no triple {p, a, b} with a, b both in S.
    # Precompute for each sorted point p, the list of (a, b) pairs (in sorted order)
    # such that {orig(p), orig(a), orig(b)} is a triple.
    
    triple_pairs_for_point = [[] for _ in range(n)]
    for (a, b, c) in triples:
        sa, sb, sc = orig_to_pos[a], orig_to_pos[b], orig_to_pos[c]
        # For point sa: pairs (sb, sc)
        triple_pairs_for_point[sa].append((sb, sc))
        triple_pairs_for_point[sb].append((sa, sc))
        triple_pairs_for_point[sc].append((sa, sb))
    
    best = [0]
    deadline = _DEADLINE - 5
    
    # Use iterative deepening or just plain B&B with good ordering
    selected_mask = 0
    
    def can_add(p, sel_mask):
        """Check if point p (sorted order) can be added to selected set sel_mask."""
        for (a, b) in triple_pairs_for_point[p]:
            if (sel_mask >> a) & 1 and (sel_mask >> b) & 1:
                return False
        return True
    
    def upper_bound(p, sel_mask):
        """Upper bound on additional points we can select from positions p..n-1."""
        # Simple: n - p
        # Better: count how many of the remaining points don't conflict with sel_mask
        # but that's expensive. Use n - p for now.
        return n - p
    
    def backtrack(p, count, sel_mask):
        nonlocal best
        if _elapsed() > deadline:
            return
        if p == n:
            if count > best[0]:
                best[0] = count
            return
        
        # Upper bound
        if count + (n - p) <= best[0]:
            return
        
        # Try to add point p
        if can_add(p, sel_mask):
            backtrack(p + 1, count + 1, sel_mask | (1 << p))
        
        # Try to skip point p
        backtrack(p + 1, count, sel_mask)
    
    backtrack(0, 0, 0)
    return best[0]

def _structured_config():
    """
    Generate a structured configuration with high triple density.
    Uses a grid-based approach with careful perturbation to maximize
    collinear triples while avoiding 4-collinear.
    """
    # Start with a 5x4 grid (20 points)
    points = []
    used = set()
    for x in range(-4, 6):
        for y in range(-4, 4):
            if len(points) >= 20:
                break
            key = (x, y)
            if key not in used:
                used.add(key)
                points.append([x, y])
        if len(points) >= 20:
            break
    
    # Perturb to remove 4-collinear while trying to preserve triples
    for _ in range(200):
        if _check_no_four_collinear(points):
            break
        # Find a 4-collinear set and move one point
        found = False
        for i in range(20):
            for j in range(i + 1, 20):
                x1, y1 = points[i]
                x2, y2 = points[j]
                dx, dy = x2 - x1, y2 - y1
                if dx == 0 and dy == 0:
                    continue
                line_pts = []
                for k in range(20):
                    x3, y3 = points[k]
                    if dx * (y3 - y1) == dy * (x3 - x1):
                        line_pts.append(k)
                if len(line_pts) >= 4:
                    # Move the point that is "most central" in the collinear set
                    idx = line_pts[len(line_pts) // 2]
                    old_x, old_y = points[idx]
                    for attempt in range(30):
                        new_x = old_x + random.randint(-3, 3)
                        new_y = old_y + random.randint(-3, 3)
                        if -100 <= new_x <= 100 and -100 <= new_y <= 100:
                            other_pts = [(p[0], p[1]) for k, p in enumerate(points) if k != idx]
                            if (new_x, new_y) not in other_pts:
                                points[idx] = [new_x, new_y]
                                found = True
                                break
                    if found:
                        break
            if found:
                break
    
    return points

def _random_valid_config():
    """Generate a random valid configuration with bias toward grid-like structures."""
    while True:
        if _elapsed() > _DEADLINE - 20:
            return None
        points = []
        seen = set()
        # Bias toward smaller coordinate range for more potential collinearity
        max_coord = random.choice([8, 10, 12, 15])
        while len(points) < 20:
            x = random.randint(-max_coord, max_coord)
            y = random.randint(-max_coord, max_coord)
            key = (x, y)
            if key not in seen:
                seen.add(key)
                points.append([x, y])
        if _check_no_four_collinear(points):
            return points

def _count_triples(points):
    """Count collinear triples."""
    n = len(points)
    count = 0
    seen_lines = set()
    for i in range(n):
        for j in range(i + 1, n):
            x1, y1 = points[i]
            x2, y2 = points[j]
            dx = x2 - x1
            dy = y2 - y1
            if dx == 0 and dy == 0:
                continue
            # Normalize line
            if dx == 0:
                a, b, c = 1, 0, -x1
            elif dy == 0:
                a, b, c = 0, 1, -y1
            else:
                g = gcd(abs(dx), abs(dy))
                dx //= g
                dy //= g
                if dx < 0 or (dx == 0 and dy < 0):
                    dx, dy = -dx, -dy
                a, b = dy, -dx
                c = -(a * x1 + b * y1)
                g2 = gcd(gcd(abs(a), abs(b)), abs(c))
                if g2 > 0:
                    a //= g2
                    b //= g2
                    c //= g2
            line_key = (a, b, c)
            if line_key in seen_lines:
                continue
            seen_lines.add(line_key)
            line_pts = []
            for k in range(n):
                x3, y3 = points[k]
                if a * x3 + b * y3 + c == 0:
                    line_pts.append(k)
            m = len(line_pts)
            if m >= 3:
                count += m * (m - 1) * (m - 2) // 6
    return count

def _improve_triples(points, iterations=200):
    """Try to increase the number of triples while maintaining validity."""
    current_triples = _count_triples(points)
    for _ in range(iterations):
        if _elapsed() > _DEADLINE - 15:
            break
        idx = random.randint(0, 19)
        old_x, old_y = points[idx]
        
        improved = False
        # Try multiple candidate positions
        for attempt in range(10):
            # Bias toward nearby positions for smoother transitions
            if attempt < 5:
                new_x = old_x + random.randint(-3, 3)
                new_y = old_y + random.randint(-3, 3)
            else:
                new_x = random.randint(-15, 15)
                new_y = random.randint(-15, 15)
            
            if new_x == old_x and new_y == old_y:
                continue
            if not (-100 <= new_x <= 100 and -100 <= new_y <= 100):
                continue
            if (new_x, new_y) in [(p[0], p[1]) for i, p in enumerate(points) if i != idx]:
                continue
            
            new_points = [list(p) for p in points]
            new_points[idx] = [new_x, new_y]
            
            if not _check_no_four_collinear(new_points):
                continue
            
            new_triples = _count_triples(new_points)
            if new_triples > current_triples:
                points = new_points
                current_triples = new_triples
                improved = True
                break
    
    return points

def construct():
    global _START_TIME
    _START_TIME = time.time()
    
    best_points = None
    best_score = 21
    
    candidates = []
    
    # Generate structured config first (likely has good properties)
    if _elapsed() < _DEADLINE - 40:
        config = _structured_config()
        if _check_no_four_collinear(config):
            candidates.append(config)
    
    # Generate initial random candidates
    for _ in range(5):
        if _elapsed() > _DEADLINE - 30:
            break
        config = _random_valid_config()
        if config:
            candidates.append(config)
    
    # Evaluate and improve each candidate
    for config in candidates:
        if _elapsed() > _DEADLINE - 10:
            break
        
        # Try to improve triple count
        improved = _improve_triples(config, iterations=150)
        
        if _elapsed() < _DEADLINE - 5:
            s = _max_subset_no_three_collinear(improved)
            if s < best_score:
                best_score = s
                best_points = [list(p) for p in improved]
    
    # If we have a best, try more improvements with fresh random restarts
    if best_points and _elapsed() < _DEADLINE - 15:
        for _ in range(5):
            if _elapsed() > _DEADLINE - 5:
                break
            improved = _improve_triples(best_points, iterations=100)
            s = _max_subset_no_three_collinear(improved)
            if s < best_score:
                best_score = s
                best_points = [list(p) for p in improved]
    
    if best_points is None:
        # Fallback: return any valid config
        config = _random_valid_config()
        if config is None:
            # Last resort: simple grid
            config = [[x, y] for x in range(5) for y in range(4)]
        best_points = config
    
    return best_points
