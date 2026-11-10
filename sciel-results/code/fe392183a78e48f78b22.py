import itertools
import random
import time

N = 16
POINTS = [(x, y) for x in range(N) for y in range(N)]
NUM_POINTS = N * N

# Precompute squared distances
# dist[i][j] is squared distance between POINTS[i] and POINTS[j]
dist = [[0] * NUM_POINTS for _ in range(NUM_POINTS)]
for i in range(NUM_POINTS):
    xi, yi = POINTS[i]
    for j in range(i + 1, NUM_POINTS):
        xj, yj = POINTS[j]
        d = (xi - xj) ** 2 + (yi - yj) ** 2
        dist[i][j] = d
        dist[j][i] = d

def can_add(k, S):
    """
    Check if point k can be added to set S (list of indices) without forming an isosceles triangle.
    S is a list of indices.
    """
    for a in S:
        da = dist[k][a]
        for b in S:
            if b == a:
                continue
            db = dist[k][b]
            dab = dist[a][b]
            # Check if triangle (k, a, b) is isosceles
            if da == db or da == dab or db == dab:
                return False
    return True

def is_maximal(S):
    """
    Check if set S is maximal.
    S is a set of indices.
    """
    S_set = set(S)
    for k in range(NUM_POINTS):
        if k in S_set:
            continue
        # Check if k can be added
        if can_add(k, list(S_set)):
            return False
    return True

def greedy_maximal(order):
    """
    Construct a maximal set using the given order of point indices.
    """
    S = []
    for k in order:
        if can_add(k, S):
            S.append(k)
    return S

def shrink(S):
    """
    Try to remove points from S while maintaining maximality.
    S is a list of indices.
    """
    S = set(S)
    improved = True
    while improved:
        improved = False
        # Try to remove each point
        for p in list(S):
            S_trial = S - {p}
            if is_maximal(S_trial):
                S = S_trial
                improved = True
                break
    return list(S)

def construct():
    start_time = time.time()
    deadline = start_time + 280  # 20s safety margin
    
    best_set = None
    best_size = NUM_POINTS + 1
    
    # Use a fixed seed for reproducibility in local testing, but random for general robustness?
    # The task implies a single evaluation run. We can use random.
    rng = random.Random(42)
    
    # Initial greedy with row-major to get a baseline
    base_order = list(range(NUM_POINTS))
    S_base = greedy_maximal(base_order)
    S_base_shrunk = shrink(S_base)
    best_set = S_base_shrunk
    best_size = len(best_set)
    
    # Randomized greedy attempts
    while time.time() < deadline:
        order = list(range(NUM_POINTS))
        rng.shuffle(order)
        
        S = greedy_maximal(order)
        S_shrunk = shrink(S)
        
        if len(S_shrunk) < best_size:
            best_set = S_shrunk
            best_size = len(best_set)
            # If we get very small, we might want to stop early? 
            # But let's keep searching until deadline.
            
    return [POINTS[i] for i in best_set]
