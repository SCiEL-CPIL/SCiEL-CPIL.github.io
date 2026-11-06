import time
import random
import math

def _neighbors(p):
    a, b = p
    return ((a + 1, b), (a - 1, b), (a, b + 1),
            (a, b - 1), (a + 1, b - 1), (a - 1, b + 1))

def _dist_sq(p, q):
    """Squared distance between two lattice points a+b*omega."""
    da = p[0] - q[0]
    db = p[1] - q[1]
    # |a + b*omega|^2 = a^2 + ab + b^2 for omega = exp(i*pi/3)
    return da * da + da * db + db * db

def _edge_count(Sset):
    """Count pairs at distance exactly 1."""
    cnt = 0
    for p in Sset:
        for q in _neighbors(p):
            if q in Sset and p < q:
                cnt += 1
    return cnt

def _valid(Sset):
    """Check validity: 32 distinct points, |a|,|b| <= 64, min distance >= 1."""
    if len(Sset) != 32:
        return False
    if len(set(Sset)) != 32:
        return False
    for a, b in Sset:
        if abs(a) > 64 or abs(b) > 64:
            return False
    pts = list(Sset)
    for i in range(len(pts)):
        for j in range(i + 1, len(pts)):
            if _dist_sq(pts[i], pts[j]) < 1:
                return False
    return True

def _spiral_config(n=32):
    """
    Generate the first n points of a triangular lattice spiral starting from (0,0).
    This creates a compact, hexagonal-like shape.
    """
    candidates = []
    for a in range(-10, 11):
        for b in range(-10, 11):
            d_sq = _dist_sq((a,b), (0,0))
            x = a + 0.5 * b
            y = math.sqrt(3)/2 * b
            angle = math.atan2(y, x)
            candidates.append((d_sq, angle, (a,b)))
            
    candidates.sort(key=lambda x: (x[0], x[1]))
    
    points = []
    seen = set()
    for d_sq, angle, p in candidates:
        if p in seen:
            continue
        valid = True
        for q in points:
            if _dist_sq(p, q) < 1:
                valid = False
                break
        if valid:
            points.append(p)
            seen.add(p)
            if len(points) == n:
                break
    return points

def _simulated_annealing(initial, max_iters=5000, deadline=None):
    """
    Simulated Annealing to maximize edge count.
    """
    rng = random.Random(999)
    current = list(initial)
    current_edges = _edge_count(set(current))
    best = list(current)
    best_edges = current_edges
    
    T = 1.0
    cooling = 0.99
    
    for i in range(max_iters):
        if deadline is not None and time.time() > deadline:
            break
            
        idx = rng.randrange(len(current))
        old_pos = current[idx]
        
        min_a = min(p[0] for p in current) - 2
        max_a = max(p[0] for p in current) + 2
        min_b = min(p[1] for p in current) - 2
        max_b = max(p[1] for p in current) + 2
        
        candidates = []
        for nb in _neighbors(old_pos):
            candidates.append(nb)
        for _ in range(5):
            candidates.append((rng.randint(min_a, max_a), rng.randint(min_b, max_b)))
            
        rng.shuffle(candidates)
        
        moved = False
        for new_pos in candidates:
            if new_pos == old_pos:
                continue
            if new_pos in set(current):
                continue
            
            valid = True
            for other_idx, other_pos in enumerate(current):
                if other_idx == idx:
                    continue
                if _dist_sq(new_pos, other_pos) < 1:
                    valid = False
                    break
            
            if not valid:
                continue
            
            current[idx] = new_pos
            new_edges = _edge_count(set(current))
            delta = new_edges - current_edges
            
            if delta > 0 or rng.random() < math.exp(delta / T):
                current_edges = new_edges
                if new_edges > best_edges:
                    best_edges = new_edges
                    best = list(current)
                moved = True
                break
            else:
                current[idx] = old_pos
        
        T *= cooling
        if T < 0.01:
            T = 0.01

    return best

def _greedy_construct(seed):
    """
    Greedily construct a set of 32 points.
    """
    rng = random.Random(seed)
    points = set()
    
    candidates = []
    for a in range(-8, 9):
        for b in range(-8, 9):
            candidates.append((a, b))
            
    rng.shuffle(candidates)
    
    for _ in range(32):
        best_edge_gain = -1
        best_pos = None
        
        start_idx = rng.randrange(len(candidates))
        
        for i in range(len(candidates)):
            idx = (start_idx + i) % len(candidates)
            p = candidates[idx]
            if p in points:
                continue
            
            valid = True
            for q in points:
                if _dist_sq(p, q) < 1:
                    valid = False
                    break
            if not valid:
                continue
            
            gain = 0
            for nb in _neighbors(p):
                if nb in points:
                    gain += 1
            
            if gain > best_edge_gain:
                best_edge_gain = gain
                best_pos = p
        
        if best_pos is not None:
            points.add(best_pos)
        else:
            break
            
    return list(points)

def _local_search(Sset, max_iters=500, deadline=None):
    """
    Local search: try moving points to improve edge count.
    """
    current_edges = _edge_count(Sset)
    best_edges = current_edges
    best_set = set(Sset)
    
    candidates = []
    for a in range(-8, 9):
        for b in range(-8, 9):
            candidates.append((a, b))
            
    rng = random.Random(12345)
    
    for it in range(max_iters):
        if deadline is not None and time.time() > deadline:
            break
        
        idx = rng.randrange(len(Sset))
        old_pos = Sset[idx]
        
        move_candidates = list(_neighbors(old_pos))
        rng.shuffle(move_candidates)
        for _ in range(3):
            move_candidates.append(rng.choice(candidates))
            
        improved = False
        for new_pos in move_candidates:
            if new_pos == old_pos:
                continue
            if new_pos in Sset:
                continue
            
            valid = True
            for other in Sset:
                if other == old_pos:
                    continue
                if _dist_sq(new_pos, other) < 1:
                    valid = False
                    break
            if not valid:
                continue
            
            Sset[idx] = new_pos
            new_edges = _edge_count(Sset)
            
            if new_edges > current_edges:
                current_edges = new_edges
                improved = True
                if new_edges > best_edges:
                    best_edges = new_edges
                    best_set = set(Sset)
                break
            else:
                Sset[idx] = old_pos
        
    return list(best_set)

def construct():
    start_time = time.time()
    deadline = start_time + 140
    
    best_result = None
    best_edges = -1
    
    # Strategy 1: Spiral + Simulated Annealing
    spiral = _spiral_config(32)
    if len(spiral) == 32:
        result = _simulated_annealing(spiral, max_iters=5000, deadline=deadline - 10)
        edges = _edge_count(set(result))
        if edges > best_edges:
            best_edges = edges
            best_result = result
    
    # Strategy 2: Greedy + Local Search
    for i in range(10):
        if time.time() > deadline - 5:
            break
        seed = i * 1000 + 42
        initial = _greedy_construct(seed)
        if len(initial) != 32:
            continue
        result = _local_search(initial, max_iters=500, deadline=deadline - 2)
        edges = _edge_count(set(result))
        if edges > best_edges:
            best_edges = edges
            best_result = result
            
    if best_result is None or len(best_result) != 32:
        best_result = [(a, b) for a in range(8) for b in range(4)]
        
    if not _valid(best_result):
        best_result = [(a, b) for a in range(8) for b in range(4)]
        
    return [[a, b] for a, b in best_result]
