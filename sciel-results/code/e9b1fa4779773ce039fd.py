import random
import time
import math
import numpy as np

N = 64

def construct():
    start_time = time.time()
    deadline = start_time + 280  # Leave 20s margin
    
    # Precompute all grid points
    ALL_POINTS = [(x, y) for x in range(N) for y in range(N)]
    n_points = len(ALL_POINTS)
    
    # Build conflict graph: two points conflict if they can form an isosceles triangle
    # with any third grid point
    # Two points A, B conflict if there exists C such that:
    # - dist(A,C) == dist(B,C) (C equidistant from A and B) - isosceles with base AB
    # - dist(A,C) == dist(A,B) - isosceles with apex A
    # - dist(B,C) == dist(A,B) - isosceles with apex B
    
    # Precompute squared distances between all pairs
    coords = np.array(ALL_POINTS, dtype=np.int32)
    
    # Compute pairwise squared distances efficiently
    # diff[i,j] = coords[i] - coords[j]
    diff = coords[:, np.newaxis, :] - coords[np.newaxis, :, :]
    sq_dists = np.sum(diff * diff, axis=2)
    
    # For each pair (i,j), check if they conflict
    # Conflict if exists k such that:
    # sq_dists[i,k] == sq_dists[j,k] (k equidistant)
    # sq_dists[i,k] == sq_dists[i,j] (apex i)
    # sq_dists[j,k] == sq_dists[i,j] (apex j)
    
    # This is O(n^3) which is too slow for n=4096
    # Instead, use a different approach: for each pair, check conflicts incrementally
    
    # Better approach: Use the fact that we're building an independent set greedily
    # Precompute for each point, which other points it conflicts with
    
    # Actually, let's use a smarter approach:
    # For a set S, point p is valid if:
    # 1. All distances from p to points in S are distinct
    # 2. No distance from p to S matches any distance within S
    
    # Let's precompute distance signatures more efficiently
    
    # For the greedy minimum-degree approach, we need to know conflicts dynamically
    # Let's use a hybrid: precompute some structure, then greedy
    
    # Precompute for each point p, and each possible squared distance d,
    # how many points in ALL_POINTS are at distance d from p
    # This helps us quickly check if adding p creates a conflict
    
    # Actually, the key insight: two points A,B in our set conflict if there's a C in the set
    # such that triangle ABC is isosceles.
    
    # For minimum degree greedy on independent set:
    # We want to select vertices that have fewest conflicts with remaining vertices
    # But "conflict" here is not pairwise - it's about forming isosceles triangles
    
    # Let me reconsider: The problem is NOT a standard independent set on a pairwise conflict graph
    # because three points form an isosceles triangle, not two.
    
    # However, we can define: points A and B are "incompatible" if there exists ANY point C
    # in the grid such that A,B,C form an isosceles triangle. Then we need an independent set
    # in this incompatibility graph.
    
    # But this is overly restrictive! A and B might be incompatible due to C, but if C is not
    # in our set, A and B can both be in the set.
    
    # So the pairwise conflict graph approach is too conservative.
    
    # Let's stick with the dynamic check but optimize it:
    # Use numpy for faster distance computations
    
    best_set = []
    num_trials = 0
    
    while time.time() < deadline:
        num_trials += 1
        
        # Shuffle points for diversity
        points = ALL_POINTS[:]
        random.shuffle(points)
        
        S = []
        S_coords = []
        dist_set = set()
        
        for p in points:
            m = len(S)
            
            if m < 2:
                S.append(p)
                S_coords.append(p)
                if m == 1:
                    d = (p[0]-S[0][0])**2 + (p[1]-S[0][1])**2
                    dist_set.add(d)
                continue
            
            # Calculate distances to all existing points
            # Use vectorized computation for speed
            p_arr = np.array(p)
            s_arr = np.array(S_coords)
            diffs = p_arr - s_arr
            dists = np.sum(diffs * diffs, axis=1)
            
            # Check 1: Distances from p to S must be distinct
            if len(set(dists.tolist())) != m:
                continue
            
            # Check 2: No distance from p to S should match any distance within S
            conflict = False
            for d in dists:
                if d in dist_set:
                    conflict = True
                    break
            
            if conflict:
                continue
            
            # Valid point
            dist_set.update(dists.tolist())
            S.append(p)
            S_coords.append(p)
        
        if len(S) > len(best_set):
            best_set = S[:]
            # If we've found a good solution, we can try local search improvements
            if len(best_set) > 100:
                # Try to improve via local search
                improved = _local_search(best_set, time.time(), deadline)
                if len(improved) > len(best_set):
                    best_set = improved
    
    return best_set

def _local_search(initial_set, start_time, deadline):
    """Try to improve the set by adding/removing points"""
    S = initial_set[:]
    S_set = set(S)
    dist_set = set()
    
    # Rebuild dist_set
    for i in range(len(S)):
        for j in range(i+1, len(S)):
            d = (S[i][0]-S[j][0])**2 + (S[i][1]-S[j][1])**2
            dist_set.add(d)
    
    improved = False
    max_iters = 50
    
    for _ in range(max_iters):
        if time.time() > deadline - 5:
            break
        
        # Try to add a random point
        candidates = [p for p in ALL_POINTS if p not in S_set]
        if not candidates:
            break
        
        random.shuffle(candidates)
        
        added = False
        for p in candidates[:100]:  # Try up to 100 candidates
            m = len(S)
            
            if m < 2:
                S.append(p)
                S_set.add(p)
                if m == 1:
                    d = (p[0]-S[-2][0])**2 + (p[1]-S[-2][1])**2
                    dist_set.add(d)
                added = True
                improved = True
                break
            
            # Check if p can be added
            dists = []
            for s in S:
                d = (p[0]-s[0])**2 + (p[1]-s[1])**2
                dists.append(d)
            
            if len(set(dists)) != m:
                continue
            
            conflict = False
            for d in dists:
                if d in dist_set:
                    conflict = True
                    break
            
            if conflict:
                continue
            
            # Add p
            S.append(p)
            S_set.add(p)
            dist_set.update(dists)
            added = True
            improved = True
            break
        
        if not added:
            # Try removing a point and re-adding
            if len(S) > 2:
                idx = random.randint(0, len(S)-1)
                removed = S.pop(idx)
                S_set.discard(removed)
                
                # Rebuild dist_set
                dist_set = set()
                for i in range(len(S)):
                    for j in range(i+1, len(S)):
                        d = (S[i][0]-S[j][0])**2 + (S[i][1]-S[j][1])**2
                        dist_set.add(d)
                
                # Try to add different points
                candidates = [p for p in ALL_POINTS if p not in S_set]
                random.shuffle(candidates)
                
                for p in candidates[:50]:
                    m = len(S)
                    
                    if m < 2:
                        S.append(p)
                        S_set.add(p)
                        if m == 1:
                            d = (p[0]-S[-2][0])**2 + (p[1]-S[-2][1])**2
                            dist_set.add(d)
                        improved = True
                        break
                    
                    dists = []
                    for s in S:
                        d = (p[0]-s[0])**2 + (p[1]-s[1])**2
                        dists.append(d)
                    
                    if len(set(dists)) != m:
                        continue
                    
                    conflict = False
                    for d in dists:
                        if d in dist_set:
                            conflict = True
                            break
                    
                    if conflict:
                        continue
                    
                    S.append(p)
                    S_set.add(p)
                    dist_set.update(dists)
                    improved = True
                    break
    
    return S

# Precompute ALL_POINTS at module level
ALL_POINTS = [(x, y) for x in range(N) for y in range(N)]
