import time
import random
import numpy as np

N = 16
TOTAL_POINTS = N * N
DEADLINE = 325.0

# Precomputation
# dist2[i][j] is squared distance between point i and point j
# conflicts[i][j] is a list of points k such that {i,j,k} is isosceles
# Note: i, j, k must be distinct.

def _precompute():
    # Map index to (x, y)
    coords = [(x, y) for y in range(N) for x in range(N)]
    
    # Precompute squared distances
    dist2 = np.zeros((TOTAL_POINTS, TOTAL_POINTS), dtype=np.int32)
    for i in range(TOTAL_POINTS):
        x1, y1 = coords[i]
        for j in range(i + 1, TOTAL_POINTS):
            x2, y2 = coords[j]
            d = (x1 - x2) ** 2 + (y1 - y2) ** 2
            dist2[i][j] = d
            dist2[j][i] = d
            
    # Precompute conflicts
    # conflicts[i][j] will be a list of k
    # Using list of lists for flexibility, or a set? List is fine.
    conflicts = [[None for _ in range(TOTAL_POINTS)] for _ in range(TOTAL_POINTS)]
    
    for i in range(TOTAL_POINTS):
        for j in range(i + 1, TOTAL_POINTS):
            d_ij = dist2[i][j]
            conflict_list = []
            for k in range(TOTAL_POINTS):
                if k == i or k == j:
                    continue
                d_ik = dist2[i][k]
                d_jk = dist2[j][k]
                # Check isosceles conditions
                if d_ij == d_ik or d_ij == d_jk or d_ik == d_jk:
                    conflict_list.append(k)
            conflicts[i][j] = conflict_list
            conflicts[j][i] = conflict_list # Symmetric
            
    return coords, dist2, conflicts

def construct():
    t0 = time.time()
    coords, dist2, conflicts = _precompute()
    
    def elapsed():
        return time.time() - t0

    def is_valid_set(S):
        # Check if S is a valid no-isosceles set
        S_list = list(S)
        for i in range(len(S_list)):
            for j in range(i + 1, len(S_list)):
                # Check if any k in S_list (k != i, j) forms isosceles
                # Actually, just check if any point in S_list is in conflicts[i][j]
                # But conflicts[i][j] contains ALL grid points.
                # We need to check intersection with S_list.
                # For speed, we can just check pairwise in the small set S.
                pass 
        # The greedy construction ensures validity, so we might not need this 
        # unless we do random swaps that might break validity. 
        # In our perturbation strategy, we remove points (keeping validity) 
        # and add points only if they don't conflict with existing ones (maintaining validity).
        # So validity is maintained by construction.
        return True

    def rebuild_blocked(S):
        """
        Given a set S (list of indices), return a boolean array 'blocked' 
        where blocked[p] is True if p cannot be added to S.
        """
        blocked = [False] * TOTAL_POINTS
        S_list = list(S)
        n_s = len(S_list)
        for i in range(n_s):
            u = S_list[i]
            for j in range(i + 1, n_s):
                v = S_list[j]
                # Mark all conflicts of u,v as blocked
                for w in conflicts[u][v]:
                    blocked[w] = True
        return blocked

    def make_maximal(S, order):
        """
        Greedily add points from 'order' to S until S is maximal.
        Returns the new set S.
        """
        S_set = set(S)
        S_list = list(S)
        blocked = rebuild_blocked(S_list)
        
        # Optimization: We only consider points not already in S
        # The 'order' should ideally contain candidates not in S.
        for p in order:
            if p in S_set:
                continue
            if not blocked[p]:
                # Add p
                S_set.add(p)
                S_list.append(p)
                # Update blocked
                # p conflicts with all existing v in S_list (except p itself, but p is new)
                # For each v in S_list (old), mark conflicts[p][v] as blocked
                # Note: S_list includes p now. We need to iterate over the OLD S_list.
                # Let's keep track of the size before adding.
                # Actually, we can just iterate over all v in S_list where v != p.
                for v in S_list:
                    if v == p:
                        continue
                    for w in conflicts[p][v]:
                        blocked[w] = True
        return S_list

    # Initial Solution: Random Greedy
    # We want a small maximal set.
    # Random greedy with a specific order might yield different sizes.
    # Let's run a few initial random greedies to find a good starting point.
    
    best_S = None
    best_size = TOTAL_POINTS + 1
    
    all_indices = list(range(TOTAL_POINTS))
    
    # 1. Generate initial candidates
    for _ in range(20):
        if elapsed() > DEADLINE:
            break
        random.shuffle(all_indices)
        S = make_maximal([], all_indices)
        if len(S) < best_size:
            best_size = len(S)
            best_S = S[:]
            
    if best_S is None:
        # Fallback
        return []

    # 2. Local Search / Perturbation
    # Try to improve best_S
    # Strategy: Remove k points, refill greedily.
    
    while elapsed() < DEADLINE:
        # Determine perturbation size
        # Start with 1, occasionally 2
        k = 1
        if random.random() < 0.2:
            k = 2
            
        if len(best_S) <= k:
            # Can't remove more than we have
            k = len(best_S)
            if k == 0:
                break
                
        # Remove k random points
        S_temp = best_S[:]
        random.shuffle(S_temp)
        removed = S_temp[:k]
        S_remaining = S_temp[k:]
        
        # Refill
        # Create a random order of candidates
        # Candidates are all points NOT in S_remaining
        # To be efficient, we can just shuffle all_indices and filter, 
        # or create a list of candidates.
        # Since TOTAL_POINTS is small (256), shuffling all is fine.
        
        candidates = [p for p in all_indices if p not in S_remaining]
        random.shuffle(candidates)
        
        S_new = make_maximal(S_remaining, candidates)
        
        if len(S_new) < best_size:
            best_size = len(S_new)
            best_S = S_new[:]
            
        # Optional: If we find a set of the same size, we might keep it 
        # to explore different neighbors, but for now we only track minimum.
        
    return best_S

# Run precomputation at module load time? 
# No, inside construct is safer for timing, but precomputation is ~1-2s.
# The task allows 300s. Precomputation is worth it.

# One detail: _precompute is called inside construct. 
# This is fine.

# Let's double check the isosceles definition.
# "no-isosceles set (no further grid point can be added)"
# "no three points in the set form an isosceles triangle"
# My conflict check:
# d_ij == d_ik or d_ij == d_jk or d_ik == d_jk
# This covers all cases of isosceles triangles with vertices i,j,k.
# If i,j,k are distinct, this is correct.
# What if two points are the same? The set contains distinct points.
# What if the triangle is degenerate (collinear)?
# If i,j,k are collinear and distinct, can they form an isosceles triangle?
# Yes, if one is the midpoint. E.g., 0, 1, 2. d(0,1)=1, d(1,2)=1, d(0,2)=4.
# d(0,1) == d(1,2) is True. So it is detected.
# If they are collinear but not isosceles (e.g., 0, 1, 3), d(0,1)=1, d(1,3)=4, d(0,3)=9. No equal distances.
# So the check handles degeneracy correctly.

# The code looks solid.

# One potential issue: `make_maximal` takes `S` as a list and returns a list.
# `S_remaining` is a list.
# `candidates` is a list.
# The set `S_set` inside `make_maximal` is used for O(1) lookup.
# The `blocked` array is updated incrementally.
# This should be fast enough.

# Let's refine the perturbation.
# Sometimes removing 1 point and refilling might just put the same point back.
# To avoid this, we could exclude the removed points from the refill candidates?
# No, that would prevent finding the same set, but we want to find *different* sets.
# If we exclude them, we force the set to change.
# However, if we exclude them, we might miss a better solution that happens to use one of the removed points but a different one elsewhere.
# Standard local search usually allows re-adding.
# But to ensure progress, we can track if we are stuck.
# Given the small search space, random restarts of the *refill order* should be enough.

# Let's add a counter for stagnation.
# If we don't improve for X iterations, increase k or do a full restart.
# For now, simple random perturbation should work well.

# Final check on imports and structure.
import time
import random
import numpy as np

N = 16
TOTAL_POINTS = N * N
DEADLINE = 325.0

def _precompute():
    coords = [(x, y) for y in range(N) for x in range(N)]
    dist2 = np.zeros((TOTAL_POINTS, TOTAL_POINTS), dtype=np.int32)
    for i in range(TOTAL_POINTS):
        x1, y1 = coords[i]
        for j in range(i + 1, TOTAL_POINTS):
            x2, y2 = coords[j]
            d = (x1 - x2) ** 2 + (y1 - y2) ** 2
            dist2[i][j] = d
            dist2[j][i] = d
            
    conflicts = [[None for _ in range(TOTAL_POINTS)] for _ in range(TOTAL_POINTS)]
    for i in range(TOTAL_POINTS):
        for j in range(i + 1, TOTAL_POINTS):
            d_ij = dist2[i][j]
            conflict_list = []
            for k in range(TOTAL_POINTS):
                if k == i or k == j:
                    continue
                d_ik = dist2[i][k]
                d_jk = dist2[j][k]
                if d_ij == d_ik or d_ij == d_jk or d_ik == d_jk:
                    conflict_list.append(k)
            conflicts[i][j] = conflict_list
            conflicts[j][i] = conflict_list
    return coords, dist2, conflicts

def construct():
    t0 = time.time()
    coords, dist2, conflicts = _precompute()
    
    def elapsed():
        return time.time() - t0

    def rebuild_blocked(S_list):
        blocked = [False] * TOTAL_POINTS
        n_s = len(S_list)
        for i in range(n_s):
            u = S_list[i]
            for j in range(i + 1, n_s):
                v = S_list[j]
                for w in conflicts[u][v]:
                    blocked[w] = True
        return blocked

    def make_maximal(S_list, order):
        S_set = set(S_list)
        blocked = rebuild_blocked(S_list)
        
        for p in order:
            if p in S_set:
                continue
            if not blocked[p]:
                S_set.add(p)
                S_list.append(p)
                for v in S_list:
                    if v == p:
                        continue
                    for w in conflicts[p][v]:
                        blocked[w] = True
        return S_list

    all_indices = list(range(TOTAL_POINTS))
    best_S = None
    best_size = TOTAL_POINTS + 1
    
    # Initial Random Greedy
    for _ in range(50):
        if elapsed() > DEADLINE:
            break
        random.shuffle(all_indices)
        S = make_maximal([], all_indices)
        if len(S) < best_size:
            best_size = len(S)
            best_S = S[:]
            
    if best_S is None:
        return []

    # Local Search
    stagnation_count = 0
    last_improvement = time.time()
    
    while elapsed() < DEADLINE:
        # Adaptive k
        k = 1
        if random.random() < 0.3:
            k = 2
        if random.random() < 0.1:
            k = 3
            
        if len(best_S) <= k:
            k = max(1, len(best_S))
            if k == 0:
                break
                
        S_temp = best_S[:]
        random.shuffle(S_temp)
        removed = S_temp[:k]
        S_remaining = S_temp[k:]
        
        candidates = [p for p in all_indices if p not in S_remaining]
        random.shuffle(candidates)
        
        S_new = make_maximal(S_remaining, candidates)
        
        if len(S_new) < best_size:
            best_size = len(S_new)
            best_S = S_new[:]
            stagnation_count = 0
            last_improvement = time.time()
        else:
            stagnation_count += 1
            
        # If stuck for a long time, do a full restart?
        # Or just keep trying.
        # Given 300s, we have plenty of time.
        
    return best_S
