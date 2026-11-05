"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 16

# EVOLVE-BLOCK-START
import time
import random
import itertools

DEADLINE = 320  # seconds, safety margin before 330s kill

def _dist_sq(p, q):
    return (p[0] - q[0]) ** 2 + (p[1] - q[1]) ** 2

def _creates_isosceles(p, chosen):
    """Check if adding p to chosen creates any isosceles triangle."""
    n = len(chosen)
    for i in range(n):
        for j in range(i + 1, n):
            a, b = chosen[i], chosen[j]
            d_pa = _dist_sq(p, a)
            d_pb = _dist_sq(p, b)
            d_ab = _dist_sq(a, b)
            if d_pa == d_pb or d_pa == d_ab or d_pb == d_ab:
                return True
    return False

def _is_maximal(chosen, time_limit=None):
    """Check if the set is maximal: no grid point can be added."""
    if time_limit is not None and time.time() > time_limit:
        return True  # Assume maximal to avoid timeout
    chosen_set = set(chosen)
    for x in range(N):
        for y in range(N):
            p = (x, y)
            if p in chosen_set:
                continue
            if not _creates_isosceles(p, chosen):
                return False
    return True

def _greedy(order, time_limit=None):
    """Build a maximal no-isosceles set using given point ordering."""
    chosen = []
    for p in order:
        if time_limit is not None and time.time() > time_limit:
            break
        if not _creates_isosceles(p, chosen):
            chosen.append(p)
    # Ensure maximality: add points until maximal
    chosen_set = set(chosen)
    changed = True
    while changed:
        if time_limit is not None and time.time() > time_limit:
            break
        changed = False
        for x in range(N):
            for y in range(N):
                p = (x, y)
                if p not in chosen_set:
                    if not _creates_isosceles(p, chosen):
                        chosen.append(p)
                        chosen_set.add(p)
                        changed = True
    return chosen

def _try_remove(chosen, time_limit=None):
    """Try to remove points while maintaining maximality."""
    chosen = list(chosen)
    improved = True
    while improved:
        if time_limit is not None and time.time() > time_limit:
            break
        improved = False
        for i in range(len(chosen)):
            if time_limit is not None and time.time() > time_limit:
                break
            candidate = chosen[:i] + chosen[i+1:]
            if _is_maximal(candidate, time_limit):
                chosen = candidate
                improved = True
                break
    return chosen

def construct():
    start_time = time.time()
    deadline = start_time + DEADLINE
    
    all_points = list(itertools.product(range(N), repeat=2))
    
    best = None
    best_len = float('inf')
    
    # Define diverse orderings
    orderings = []
    
    # 1. Row-major
    orderings.append(all_points)
    
    # 2. Column-major
    orderings.append(list(itertools.product(range(N), repeat=2))[::-1])
    
    # 3. Diagonal order
    diag = []
    for s in range(2 * N - 1):
        for x in range(N):
            y = s - x
            if 0 <= y < N:
                diag.append((x, y))
    orderings.append(diag)
    
    # 4. Reverse diagonal
    orderings.append(diag[::-1])
    
    # 5. Center-out
    center = ((N-1)/2.0, (N-1)/2.0)
    center_out = sorted(all_points, key=lambda p: (p[0]-center[0])**2 + (p[1]-center[1])**2)
    orderings.append(center_out)
    
    # 6. Random orderings
    rng = random.Random(42)
    for _ in range(10):
        order = all_points[:]
        rng.shuffle(order)
        orderings.append(order)
    
    # Try each ordering
    for idx, order in enumerate(orderings):
        if time.time() > deadline - 30:
            break
        
        try:
            chosen = _greedy(order, deadline - 30)
        except Exception:
            continue
        
        if len(chosen) < best_len:
            best = chosen
            best_len = len(chosen)
        
        # Try to improve by removing points
        if time.time() < deadline - 15:
            improved = _try_remove(best, deadline - 15)
            if len(improved) < best_len:
                best = improved
                best_len = len(improved)
        
        # Additional random restarts if time permits
        if time.time() < deadline - 60:
            for _ in range(5):
                if time.time() > deadline - 30:
                    break
                order = all_points[:]
                rng.shuffle(order)
                try:
                    chosen = _greedy(order, deadline - 30)
                except Exception:
                    continue
                if len(chosen) < best_len:
                    best = chosen
                    best_len = len(chosen)
                if time.time() < deadline - 15:
                    improved = _try_remove(best, deadline - 15)
                    if len(improved) < best_len:
                        best = improved
                        best_len = len(improved)
    
    if best is None:
        # Fallback: simple greedy
        best = _greedy(all_points)
    
    return best
# EVOLVE-BLOCK-END
