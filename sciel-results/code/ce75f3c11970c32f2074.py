import time
import random
import math

N = 64
NUM_POINTS = N * N
MAX_D2 = (N-1)**2 + (N-1)**2  # 7840 for N=64
BIT_COUNT = 312  # 312 * 256 = 80064 > 7841, so indices 0..7840 fit

# Precompute all coordinates
COORDS = [(i // N, i % N) for i in range(NUM_POINTS)]

# Precompute squared distances between all pairs
# dist2[i] is a list of length NUM_POINTS, dist2[i][j] = (xi-xj)^2 + (yi-yj)^2
# This is a 4096x4096 matrix = ~16M ints, about 128MB. Acceptable.
# Actually, let's use a more memory-efficient approach:
# dist2[i][j] can be computed on the fly, but precomputing helps speed.
# Let's use numpy if available for speed, but the task says stdlib + numpy + scipy.

import numpy as np

# Precompute all coordinates as numpy arrays
xs = np.array([i // N for i in range(NUM_POINTS)])
ys = np.array([i % N for i in range(NUM_POINTS)])

# Precompute dist2 as a 2D array: dist2[i, j] = squared distance
# Memory: 4096 * 4096 * 8 bytes (int64) = ~128 MB. OK.
dx_all = xs[:, None] - xs[None, :]
dy_all = ys[:, None] - ys[None, :]
dist2_mat = (dx_all * dx_all + dy_all * dy_all).astype(np.int32)
# dist2_mat[i, j] is the squared distance from point i to point j

def construct():
    start_time = time.time()
    deadline = start_time + 280
    
    # S: list of point indices in the current set
    # in_set: boolean array of length NUM_POINTS
    # For each point in S, we track which squared distances are "used"
    # i.e., for point i in S, used_d2[i] is a set of d2 values that appear
    # as the distance from i to some other point in S.
    # Also, for adding a new point p, we need to check:
    #   1. For each a in S: d2(p, a) not in used_d2[a] (no isosceles with apex a)
    #   2. All d2(p, a) for a in S are distinct (no isosceles with apex p)
    #      i.e., no two points in S are at the same distance from p.
    
    # We'll maintain:
    # S: list of indices
    # in_set: bytearray or bool array
    # used_d2: list of sets, one per point index (only for points in S we care, but we can have for all)
    #   used_d2[i] = set of d2 values from i to other points in S
    
    S = []
    in_set = [False] * NUM_POINTS
    used_d2 = [set() for _ in range(NUM_POINTS)]
    
    def can_add(p):
        if in_set[p]:
            return False
        # Check distinctness of distances from p to all points in S
        d2s_seen = set()
        for a in S:
            d2 = int(dist2_mat[p, a])
            if d2 in d2s_seen:
                return False
            d2s_seen.add(d2)
            # Check if this distance is already used from a's perspective
            if d2 in used_d2[a]:
                return False
        return True
    
    def add_point(p):
        S.append(p)
        in_set[p] = True
        used_d2[p] = set()
        for a in S[:-1]:
            d2 = int(dist2_mat[p, a])
            used_d2[a].add(d2)
            used_d2[p].add(d2)
    
    def remove_point(p):
        idx = S.index(p)
        S.pop(idx)
        in_set[p] = False
        for a in S:
            d2 = int(dist2_mat[p, a])
            used_d2[a].discard(d2)
        used_d2[p] = set()
    
    # ---- Phase 1: Greedy with multiple random orderings ----
    best_S = []
    best_size = 0
    
    def greedy_one():
        S_local = []
        in_set_local = [False] * NUM_POINTS
        used_d2_local = [set() for _ in range(NUM_POINTS)]
        
        def can_add_local(p):
            if in_set_local[p]:
                return False
            d2s_seen = set()
            for a in S_local:
                d2 = int(dist2_mat[p, a])
                if d2 in d2s_seen:
                    return False
                d2s_seen.add(d2)
                if d2 in used_d2_local[a]:
                    return False
            return True
        
        def add_local(p):
            S_local.append(p)
            in_set_local[p] = True
            used_d2_local[p] = set()
            for a in S_local[:-1]:
                d2 = int(dist2_mat[p, a])
                used_d2_local[a].add(d2)
                used_d2_local[p].add(d2)
        
        order = list(range(NUM_POINTS))
        random.shuffle(order)
        for p in order:
            if can_add_local(p):
                add_local(p)
        
        return S_local
    
    # Run several greedy passes
    for _ in range(30):
        if time.time() > deadline:
            break
        S_g = greedy_one()
        if len(S_g) > best_size:
            best_size = len(S_g)
            best_S = S_g[:]
    
    # Initialize current state from best_S
    for p in best_S:
        add_point(p)
    
    # ---- Phase 2: Local search (add/remove/swap) ----
    # Try to improve by adding points, removing and swapping, etc.
    
    def try_add_random():
        for _ in range(50):
            p = random.randint(0, NUM_POINTS - 1)
            if not in_set[p] and can_add(p):
                add_point(p)
                return True
        return False
    
    def try_remove_and_add():
        # Remove one point, then try to add another
        if not S:
            return False
        q = random.choice(S)
        remove_point(q)
        # Try to add a random point
        for _ in range(100):
            p = random.randint(0, NUM_POINTS - 1)
            if not in_set[p] and can_add(p):
                add_point(p)
                return True
        # If we couldn't add, put q back
        add_point(q)
        return False
    
    def try_swap():
        if not S:
            return False
        q = random.choice(S)
        remove_point(q)
        added = False
        for _ in range(100):
            p = random.randint(0, NUM_POINTS - 1)
            if not in_set[p] and can_add(p):
                add_point(p)
                added = True
                break
        if not added:
            add_point(q)
        return added
    
    stagnation = 0
    max_stagnation = 50
    
    while time.time() < deadline:
        moved = False
        r = random.random()
        if r < 0.5:
            moved = try_add_random()
        elif r < 0.75:
            moved = try_remove_and_add()
        else:
            moved = try_swap()
        
        if moved:
            stagnation = 0
            if len(S) > best_size:
                best_size = len(S)
                best_S = S[:]
        else:
            stagnation += 1
            if stagnation > max_stagnation:
                # Perturbation: remove a random subset and rebuild
                num_remove = max(1, len(S) // 10)
                for _ in range(num_remove):
                    if S:
                        q = random.choice(S)
                        remove_point(q)
                stagnation = 0
    
    return [COORDS[i] for i in best_S]
