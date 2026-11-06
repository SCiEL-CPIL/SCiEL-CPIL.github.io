"""Rewritten program for minimum maximal no-isosceles set."""
N = 16

# EVOLVE-BLOCK-START
import time
import random
import numpy as np
import itertools

# Precompute all grid points and squared distances
POINTS = list(itertools.product(range(N), repeat=2))
V = len(POINTS)  # 256

# DIST2[i,j] = squared Euclidean distance between POINTS[i] and POINTS[j]
DIST2 = np.zeros((V, V), dtype=np.int64)
for i in range(V):
    xi, yi = POINTS[i]
    for j in range(i + 1, V):
        xj, yj = POINTS[j]
        d = (xi - xj) ** 2 + (yi - yj) ** 2
        DIST2[i, j] = d
        DIST2[j, i] = d

# For each point u, precompute list of (v, d) where d = dist2(u,v), sorted by d
# This helps in checking isosceles conditions efficiently
# Actually, let's precompute for each u, the set of distances to all other points
# And for each u, group points by distance

# For fast isosceles checking:
# A set S is no-isosceles iff for all distinct a,b,c in S:
#   dist2(a,b) != dist2(a,c) and dist2(a,b) != dist2(b,c) and dist2(a,c) != dist2(b,c)
# Equivalent: for each a in S, all dist2(a,s) for s in S\{a} are distinct,
#             and for each pair a,b in S, dist2(a,b) is not equal to dist2(a,c) for any c in S\{a,b}
#             and not equal to dist2(b,c) for any c in S\{a,b}

# Let's use a different approach: for each point u, precompute the multiset of distances to all other points
# Actually, let's think about what makes a point "blocked" by a set S.

# A point u is blocked by S if there exist a, b in S such that:
#   dist2(u,a) == dist2(u,b)  [isosceles with u as apex]
#   dist2(u,a) == dist2(a,b)  [isosceles with a as apex, u and b as base]
#   dist2(u,b) == dist2(a,b)  [isosceles with b as apex, u and a as base]

# For the "no-isosceles" constraint on S itself:
# When we add point p to set S, we need:
# 1. For all a in S: dist2(p,a) must be distinct (no two points in S at same distance from p)
# 2. For all a in S: dist2(p,a) must not equal dist2(a,b) for any b in S, b != a
#    (this means p,a,b don't form isosceles with a as apex)
#    Wait, let me re-check: triangle (p,a,b) is isosceles if any two sides equal.
#    Sides: dist2(p,a), dist2(p,b), dist2(a,b)
#    So we need:
#    - dist2(p,a) != dist2(p,b) for all a,b in S, a!=b  [condition 1 above]
#    - dist2(p,a) != dist2(a,b) for all a,b in S, a!=b  [p as one of the equal sides with a]
#    - dist2(p,b) != dist2(a,b) for all a,b in S, a!=b  [p as one of the equal sides with b]
#    
#    But condition 2 and 3 are symmetric. Let me re-examine.
#    For triangle (p, a, b):
#    Side PA = dist2(p,a), Side PB = dist2(p,b), Side AB = dist2(a,b)
#    Isosceles if PA==PB or PA==AB or PB==AB.
#    
#    So when adding p to S, for every pair (a,b) in S:
#    - dist2(p,a) != dist2(p,b)  [this must hold for ALL pairs, so all dist2(p,s) for s in S must be distinct]
#    - dist2(p,a) != dist2(a,b)
#    - dist2(p,b) != dist2(a,b)

# Let's precompute for each point p, and each distance d, the list of points at distance d from p.
# This is a dictionary: dist_map[p][d] = list of points at squared distance d from p

dist_map = {}
for p in range(V):
    dm = {}
    for q in range(V):
        if p == q:
            continue
        d = DIST2[p, q]
        if d not in dm:
            dm[d] = []
        dm[d].append(q)
    dist_map[p] = dm

# Precompute for each pair (a,b), the set of distances that are "forbidden" for a third point p
# Actually, let's just implement the check directly.

def _can_add(p, S):
    """Check if point p can be added to set S without creating an isosceles triangle."""
    if len(S) == 0:
        return True
    if len(S) == 1:
        return True
    
    # Condition 1: All distances from p to points in S must be distinct
    d_seen = set()
    for a in S:
        d = DIST2[p, a]
        if d in d_seen:
            return False
        d_seen.add(d)
    
    # Condition 2: For each pair (a,b) in S, dist2(p,a) != dist2(a,b) and dist2(p,b) != dist2(a,b)
    S_list = list(S)
    n = len(S_list)
    for i in range(n):
        a = S_list[i]
        d_pa = DIST2[p, a]
        for j in range(i + 1, n):
            b = S_list[j]
            d_ab = DIST2[a, b]
            if d_pa == d_ab:
                return False
            d_pb = DIST2[p, b]
            if d_pb == d_ab:
                return False
    
    return True

def _is_blocked(u, S):
    """Check if point u is blocked by set S (u cannot be added to S without creating isosceles)."""
    if len(S) < 2:
        return False
    S_list = list(S)
    n = len(S_list)
    for i in range(n):
        a = S_list[i]
        d_ua = DIST2[u, a]
        for j in range(i + 1, n):
            b = S_list[j]
            d_ub = DIST2[u, b]
            d_ab = DIST2[a, b]
            if d_ua == d_ub or d_ua == d_ab or d_ub == d_ab:
                return True
    return False

def _count_unblocked(S):
    """Count points not in S that are not blocked by S."""
    S_set = set(S)
    count = 0
    for u in range(V):
        if u in S_set:
            continue
        if not _is_blocked(u, S):
            count += 1
    return count

def _greedy_construct(seed_rng):
    """Construct a maximal no-isosceles set using greedy approach with random tie-breaking."""
    S = []
    S_set = set()
    remaining = list(range(V))
    seed_rng.shuffle(remaining)
    
    for p in remaining:
        if p in S_set:
            continue
        if _can_add(p, S):
            S.append(p)
            S_set.add(p)
    
    # Now minimize: try to remove points one by one
    changed = True
    while changed:
        changed = False
        for i in range(len(S)):
            p = S[i]
            new_S = S[:i] + S[i+1:]
            # Check if new_S is still maximal
            # i.e., is p blocked by new_S?
            if _is_blocked(p, new_S):
                # p is redundant, remove it
                S = new_S
                changed = True
                break
    
    return S

def construct():
    start_time = time.time()
    deadline = start_time + 290.0  # Safety margin
    
    best_size = V
    best_set = []
    
    # Multiple randomized restarts
    for restart in range(500):
        if time.time() > deadline:
            break
        
        seed = random.randint(0, 2**31 - 1)
        rng = random.Random(seed)
        
        S = _greedy_construct(rng)
        
        if len(S) < best_size:
            best_size = len(S)
            best_set = S[:]
            if best_size <= 8:
                break
    
    # Additional local search on best_set
    if time.time() < deadline - 10:
        S = best_set[:]
        improved = True
        while improved and time.time() < deadline - 5:
            improved = False
            for i in range(len(S)):
                for j in range(len(S)):
                    if i == j:
                        continue
                    # Try swapping S[i] and S[j]? No, let's try removing S[i] and adding S[j]
                    # Actually, let's try: remove S[i], then see if we can add some other point
                    # This is complex. Let's stick with simple removal.
                    pass
        
        # Simpler: try to remove any point
        changed = True
        while changed and time.time() < deadline - 5:
            changed = False
            for i in range(len(S)):
                p = S[i]
                new_S = S[:i] + S[i+1:]
                if _is_blocked(p, new_S):
                    S = new_S
                    changed = True
                    break
            if not changed:
                # Try random removal + re-addition
                if len(S) > 1:
                    i = random.randint(0, len(S) - 1)
                    p = S[i]
                    new_S = S[:i] + S[i+1:]
                    if _is_blocked(p, new_S):
                        S = new_S
                        changed = True
        
        if len(S) < best_size:
            best_size = len(S)
            best_set = S[:]
    
    return [POINTS[i] for i in best_set]

# EVOLVE-BLOCK-END
