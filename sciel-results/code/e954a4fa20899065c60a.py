"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import math

N = 16
DEADLINE = 320.0  # seconds, safety margin before 330s kill

# EVOLVE-BLOCK-START

def _dist2(a, b):
    """Squared Euclidean distance between two grid points."""
    dx = a[0] - b[0]
    dy = a[1] - b[1]
    return dx * dx + dy * dy


def _is_isosceles(p, chosen):
    """Check if adding point p to chosen set creates any isosceles triangle.
    
    For each pair (a, b) in chosen, check if {p, a, b} forms an isosceles triangle.
    An isosceles triangle has at least two equal side lengths.
    """
    for i in range(len(chosen)):
        a = chosen[i]
        da = _dist2(p, a)
        for j in range(i + 1, len(chosen)):
            b = chosen[j]
            db = _dist2(p, b)
            ab = _dist2(a, b)
            # Check if any two sides are equal
            if da == db or da == ab or db == ab:
                return True
    return False


def _is_isosceles_fast(p, chosen, dist_to_chosen):
    """Faster check using precomputed distances from p to each chosen point."""
    n = len(chosen)
    for i in range(n):
        da = dist_to_chosen[i]
        for j in range(i + 1, n):
            db = dist_to_chosen[j]
            ab = _dist2(chosen[i], chosen[j])
            if da == db or da == ab or db == ab:
                return True
    return False


def _precompute_pairwise_dists(chosen):
    """Precompute pairwise squared distances among chosen points."""
    n = len(chosen)
    dists = [[0] * n for _ in range(n)]
    for i in range(n):
        for j in range(i + 1, n):
            d = _dist2(chosen[i], chosen[j])
            dists[i][j] = d
            dists[j][i] = d
    return dists


def _check_can_add(p, chosen, pair_dists):
    """Check if p can be added to chosen without creating isosceles triangles."""
    n = len(chosen)
    # Compute distances from p to each chosen point
    dists_to_p = [_dist2(p, c) for c in chosen]
    for i in range(n):
        da = dists_to_p[i]
        for j in range(i + 1, n):
            db = dists_to_p[j]
            ab = pair_dists[i][j]
            if da == db or da == ab or db == ab:
                return False
    return True


def _greedy_construct(order=None):
    """Greedy construction: iterate through points in given order, add if possible."""
    if order is None:
        order = [(x, y) for x in range(N) for y in range(N)]
    
    chosen = []
    pair_dists = []
    
    for p in order:
        if not _check_can_add(p, chosen, pair_dists):
            continue
        # Add p
        n = len(chosen)
        new_dists = [_dist2(p, c) for c in chosen]
        # Update pair_dists
        pair_dists.append([0] * (n + 1))
        for i in range(n):
            pair_dists[i][n] = new_dists[i]
            pair_dists[n][i] = new_dists[i]
        chosen.append(p)
    
    return chosen, pair_dists


def _greedy_with_backtracking(order, time_limit=5.0):
    """Greedy with limited backtracking to try alternative choices."""
    start_time = time.time()
    best = None
    best_size = -1
    
    # First pass: simple greedy
    chosen, pair_dists = _greedy_construct(order)
    best = chosen[:]
    best_size = len(chosen)
    
    # Try some local improvements: remove one point, try to add different points
    for _ in range(50):  # limited iterations
        if time.time() - start_time > time_limit:
            break
        if not best:
            break
        
        # Try removing a random point and rebuilding
        idx = random.randint(0, len(best) - 1)
        new_chosen = best[:idx] + best[idx+1:]
        new_pair_dists = []
        n = len(new_chosen)
        new_pair_dists = [[0] * n for _ in range(n)]
        for i in range(n):
            for j in range(i + 1, n):
                d = _dist2(new_chosen[i], new_chosen[j])
                new_pair_dists[i][j] = d
                new_pair_dists[j][i] = d
        
        # Try to add points from remaining candidates
        remaining = [p for p in order if p not in new_chosen]
        random.shuffle(remaining)
        
        improved = False
        for p in remaining:
            if time.time() - start_time > time_limit:
                break
            if _check_can_add(p, new_chosen, new_pair_dists):
                n2 = len(new_chosen)
                new_dists = [_dist2(p, c) for c in new_chosen]
                new_pair_dists.append([0] * (n2 + 1))
                for i in range(n2):
                    new_pair_dists[i][n2] = new_dists[i]
                    new_pair_dists[n2][i] = new_dists[i]
                new_chosen.append(p)
                improved = True
        
        if len(new_chosen) > best_size:
            best = new_chosen[:]
            best_size = len(new_chosen)
    
    return best


def _random_order(seed=None):
    """Generate a random ordering of grid points."""
    if seed is not None:
        random.seed(seed)
    points = [(x, y) for x in range(N) for y in range(N)]
    random.shuffle(points)
    return points


def _diagonal_order():
    """Order by distance from corner, then random within same distance."""
    points = [(x, y) for x in range(N) for y in range(N)]
    points.sort(key=lambda p: (p[0] + p[1], p[0]))
    return points


def _spiral_order():
    """Spiral ordering from center."""
    cx, cy = N // 2, N // 2
    points = []
    visited = set()
    max_dist = max(cx, cy, N - 1 - cx, N - 1 - cy)
    for d in range(max_dist + 1):
        for dx in range(-d, d + 1):
            for dy in range(-d, d + 1):
                if max(abs(dx), abs(dy)) != d:
                    continue
                x, y = cx + dx, cy + dy
                if 0 <= x < N and 0 <= y < N and (x, y) not in visited:
                    points.append((x, y))
                    visited.add((x, y))
    return points


def construct():
    start_time = time.time()
    
    # Try multiple strategies and keep the best (smallest) result
    best_result = None
    best_size = float('inf')
    
    # Strategy 1: Row-major greedy
    if time.time() - start_time < DEADLINE - 30:
        result, _ = _greedy_construct()
        if len(result) < best_size:
            best_size = len(result)
            best_result = result[:]
    
    # Strategy 2: Diagonal order
    if time.time() - start_time < DEADLINE - 30:
        order = _diagonal_order()
        result, _ = _greedy_construct(order)
        if len(result) < best_size:
            best_size = len(result)
            best_result = result[:]
    
    # Strategy 3: Spiral order
    if time.time() - start_time < DEADLINE - 30:
        order = _spiral_order()
        result, _ = _greedy_construct(order)
        if len(result) < best_size:
            best_size = len(result)
            best_result = result[:]
    
    # Strategy 4: Random orders with greedy
    num_random_trials = 20
    for trial in range(num_random_trials):
        if time.time() - start_time > DEADLINE - 15:
            break
        order = _random_order(seed=trial * 12345 + 67890)
        result, _ = _greedy_construct(order)
        if len(result) < best_size:
            best_size = len(result)
            best_result = result[:]
    
    # Strategy 5: Backtracking improvement on best result
    if time.time() - start_time < DEADLINE - 10:
        order = _random_order(seed=42)
        improved = _greedy_with_backtracking(order, time_limit=DEADLINE - start_time - 5)
        if improved and len(improved) < best_size:
            best_size = len(improved)
            best_result = improved[:]
    
    # Ensure we have a valid result
    if best_result is None:
        best_result, _ = _greedy_construct()
    
    # Verify maximality: no further point can be added
    # (The greedy construction ensures maximality by definition)
    
    return best_result

# EVOLVE-BLOCK-END
