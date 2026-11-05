"""Search for a Hamilton cycle with minimal max edge-arrival index."""
import time
import random

# Build the edge order from the seed program's data
_EDGE_ORDER = [[3, 11], [9, 18], [6, 7], [8, 18], [5, 15], [19, 22], [0, 23], [13, 19], [1, 3], [21, 23], [0, 2], [15, 23], [4, 12], [16, 21], [3, 8], [10, 23], [15, 22], [9, 13], [10, 13], [18, 19], [3, 7], [9, 12], [0, 17], [12, 17], [3, 14], [6, 23], [0, 12], [2, 5], [8, 20], [11, 19], [21, 22], [10, 16], [4, 11], [0, 19], [13, 21], [15, 16], [9, 11], [5, 8], [5, 12], [7, 8], [9, 14], [2, 23], [17, 20], [1, 23], [4, 7], [14, 22], [18, 22], [10, 15], [11, 22], [16, 17], [6, 16], [1, 5], [1, 20], [6, 10], [0, 21], [0, 10], [15, 21], [12, 23], [7, 19], [8, 23], [8, 21], [4, 14], [0, 22], [3, 22], [13, 18], [20, 23], [17, 19], [14, 17], [3, 17], [0, 4], [1, 8], [2, 22], [2, 20], [8, 22], [1, 16], [3, 19], [14, 20], [7, 18], [4, 15], [10, 12], [7, 13], [18, 20], [2, 19], [1, 19], [13, 17], [1, 2], [9, 21], [12, 15], [8, 16], [19, 20], [1, 11], [5, 17], [13, 14], [11, 17], [13, 16], [7, 17], [2, 15], [16, 22], [2, 7], [9, 16], [7, 23], [9, 23], [8, 9], [5, 23], [7, 20], [1, 17], [0, 7], [9, 22], [2, 17], [3, 18], [17, 22], [0, 5], [9, 15], [0, 8], [1, 9], [6, 11], [5, 7], [0, 15], [12, 19], [7, 21], [15, 18], [11, 15], [7, 10], [3, 10], [0, 13], [12, 13], [6, 20], [9, 10], [2, 18], [1, 14], [4, 22], [14, 15], [3, 9], [16, 20], [16, 23], [10, 20], [4, 9], [6, 18], [0, 14], [0, 6], [17, 23], [5, 10], [20, 22], [10, 19], [18, 21], [16, 19], [11, 21], [7, 12], [1, 10], [6, 8], [0, 11], [9, 19], [0, 9], [4, 6], [3, 16], [3, 13], [1, 6], [6, 21], [2, 14], [12, 18], [17, 21], [12, 16], [14, 21], [13, 20], [11, 20], [5, 16], [3, 23], [1, 18], [7, 9], [6, 12], [8, 19], [11, 12], [10, 11], [7, 22], [8, 11], [20, 21], [2, 12], [8, 17], [8, 10], [4, 23], [9, 17], [14, 18], [8, 13], [9, 20], [1, 15], [5, 9], [3, 6], [2, 11], [3, 15], [3, 20], [4, 19], [4, 21], [6, 15], [10, 22], [7, 14], [2, 6], [13, 23], [0, 18], [15, 19], [4, 5], [19, 23], [12, 21], [5, 14], [4, 20], [5, 21], [8, 14], [12, 14], [12, 22], [0, 3], [7, 11], [5, 13], [13, 15], [7, 16], [1, 13], [4, 8], [14, 23], [15, 17], [6, 9], [2, 8], [1, 7], [4, 10], [2, 21], [7, 15], [13, 22], [2, 4], [16, 18], [6, 17], [10, 21], [5, 22], [2, 9], [1, 12], [0, 1], [12, 20], [4, 16], [22, 23], [19, 21], [5, 11], [10, 18], [5, 20], [1, 4], [17, 18], [18, 23], [14, 16], [11, 16], [8, 12], [2, 13], [3, 5], [5, 18], [6, 22], [6, 14], [4, 13], [2, 3], [6, 19], [0, 20], [1, 21], [5, 6], [10, 17], [11, 14], [15, 20], [11, 23], [10, 14], [8, 15], [11, 13], [3, 21], [14, 19], [11, 18], [3, 4], [4, 17], [3, 12], [0, 16], [2, 10], [2, 16], [1, 22], [4, 18], [5, 19], [6, 13]]

N = 24

# Precompute: for each edge (u,v) with u<v, its 1-based arrival index
_edge_index = {}
for i, (u, v) in enumerate(_EDGE_ORDER):
    if u > v:
        u, v = v, u
    _edge_index[(u, v)] = i + 1

def get_edge_idx(u, v):
    if u > v:
        u, v = v, u
    return _edge_index[(u, v)]

def cycle_max_index(cycle):
    """Compute the max 1-based arrival index over all edges of the cycle."""
    n = len(cycle)
    mx = 0
    for i in range(n):
        u = cycle[i]
        v = cycle[(i + 1) % n]
        idx = get_edge_idx(u, v)
        if idx > mx:
            mx = idx
    return mx

def two_opt(cycle, max_iter=200000):
    """Local search: 2-opt moves that reduce the max edge index."""
    n = len(cycle)
    best = list(cycle)
    best_score = cycle_max_index(best)
    
    for _ in range(max_iter):
        improved = False
        for i in range(n):
            if improved:
                break
            for j in range(i + 2, n):
                if i == 0 and j == n - 1:
                    continue
                # Try reversing segment [i..j]
                new_cycle = best[:i] + best[i:j+1][::-1] + best[j+1:]
                ns = cycle_max_index(new_cycle)
                if ns < best_score:
                    best = new_cycle
                    best_score = ns
                    improved = True
        if not improved:
            break
    return best, best_score

def random_greedy_cycle(rng):
    """Build a cycle greedily: at each step pick the neighbor with smallest edge index."""
    n = N
    used = [False] * n
    cycle = [0]
    used[0] = True
    for _ in range(n - 1):
        cur = cycle[-1]
        candidates = []
        for v in range(n):
            if not used[v]:
                idx = get_edge_idx(cur, v)
                candidates.append((idx, v))
        candidates.sort()
        # Pick from top few with some randomness
        k = min(len(candidates), 3)
        chosen = rng.choice(candidates[:k])
        cycle.append(chosen[1])
        used[chosen[1]] = True
    return cycle

def random_cycle(rng):
    """Random permutation as cycle."""
    c = list(range(N))
    rng.shuffle(c)
    return c

def construct():
    start = time.time()
    deadline = start + 80.0  # safety margin before 90s kill
    
    # Initial solution: identity
    best_cycle = list(range(N))
    best_score = cycle_max_index(best_cycle)
    
    # Try greedy
    rng = random.Random(42)
    for attempt in range(50):
        c = random_greedy_cycle(rng)
        c, s = two_opt(c, max_iter=50000)
        if s < best_score:
            best_cycle = c
            best_score = s
        if time.time() > deadline:
            break
    
    # Try random starts with 2-opt
    for attempt in range(200):
        if time.time() > deadline:
            break
        c = random_cycle(rng)
        c, s = two_opt(c, max_iter=30000)
        if s < best_score:
            best_cycle = c
            best_score = s
    
    # Further polish the best
    best_cycle, best_score = two_opt(best_cycle, max_iter=200000)
    
    # 3-opt style: try swapping two non-adjacent pairs
    # Actually let's do a more targeted local search: for each edge in the cycle,
    # try replacing it with a cheaper edge that still forms a valid cycle.
    # This is complex; instead do more 2-opt passes with different orderings.
    for _ in range(10):
        if time.time() > deadline:
            break
        best_cycle, best_score = two_opt(best_cycle, max_iter=100000)
    
    return list(best_cycle)
