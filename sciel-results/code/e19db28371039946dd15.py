import time
import random

N = 16
POINTS = [(x, y) for x in range(N) for y in range(N)]
NUM_POINTS = len(POINTS)
POINT_INDEX = {p: i for i, p in enumerate(POINTS)}

# Precompute squared distances
dist2 = [[0] * NUM_POINTS for _ in range(NUM_POINTS)]
for i in range(NUM_POINTS):
    x1, y1 = POINTS[i]
    for j in range(i + 1, NUM_POINTS):
        x2, y2 = POINTS[j]
        d = (x1 - x2) ** 2 + (y1 - y2) ** 2
        dist2[i][j] = d
        dist2[j][i] = d

# Precompute forbidden pairs for each point
# forbidden_pairs[p] is a list of (i, j) such that {i, j, p} is isosceles
forbidden_pairs = [[] for _ in range(NUM_POINTS)]

for p in range(NUM_POINTS):
    # We need to find all pairs (i, j) with i < j, i != p, j != p
    # such that {i, j, p} is isosceles
    # Iterate i
    for i in range(NUM_POINTS):
        if i == p:
            continue
        for j in range(i + 1, NUM_POINTS):
            if j == p:
                continue
            d_ip = dist2[i][p]
            d_jp = dist2[j][p]
            d_ij = dist2[i][j]
            if d_ip == d_jp or d_ip == d_ij or d_jp == d_ij:
                forbidden_pairs[p].append((i, j))

def can_add(p_idx, S_set):
    """Check if point p_idx can be added to set S (represented as a set of indices)."""
    for i, j in forbidden_pairs[p_idx]:
        if i in S_set and j in S_set:
            return False
    return True

def is_maximal(S):
    """Check if set S (list of indices) is maximal."""
    S_set = set(S)
    for p in range(NUM_POINTS):
        if p not in S_set:
            if can_add(p, S_set):
                return False
    return True

def random_greedy(seed):
    """Construct a maximal no-isosceles set using randomized greedy."""
    random.seed(seed)
    S = []
    S_set = set()
    candidates = list(range(NUM_POINTS))
    random.shuffle(candidates)
    
    for p in candidates:
        if can_add(p, S_set):
            S.append(p)
            S_set.add(p)
            
    return S

def minimize_maximal(S):
    """Try to minimize a maximal set by removing points."""
    S = list(S)
    changed = True
    while changed:
        changed = False
        S_set = set(S)
        # Try removing each point
        for i in range(len(S)):
            p = S[i]
            # Remove p temporarily
            S_set.remove(p)
            if is_maximal(S_set):
                # Accept removal
                S.pop(i)
                changed = True
                break # Restart loop as set changed
            else:
                # Restore p
                S_set.add(p)
    return S

def construct():
    start_time = time.time()
    deadline = start_time + 290.0 # Safety margin
    
    best_set = None
    best_size = NUM_POINTS + 1
    
    seed = 42
    while time.time() < deadline:
        # Generate a random maximal set
        S = random_greedy(seed)
        # Minimize it
        S_min = minimize_maximal(S)
        
        if len(S_min) < best_size:
            best_size = len(S_min)
            best_set = S_min
            # Optional: print or log improvement
            
        seed += 1
        
        # If we have time, we can try more complex local searches
        # But simple restarts with fast checks might be best

    if best_set is None:
        # Fallback
        S = []
        S_set = set()
        for p in range(NUM_POINTS):
            if can_add(p, S_set):
                S.append(p)
                S_set.add(p)
        return [POINTS[i] for i in S]
        
    return [POINTS[i] for i in best_set]
