"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 64
import time
import random
import numpy as np

# EVOLVE-BLOCK-START
def construct():
    deadline = time.time() + 280  # Safety margin
    
    # Grid points
    points = [(x, y) for y in range(N) for x in range(N)]
    num_points = N * N
    
    # Create coordinate arrays
    xs = np.array([p[0] for p in points], dtype=np.int32)
    ys = np.array([p[1] for p in points], dtype=np.int32)
    
    # Calculate squared distances
    dx = xs[:, None] - xs[None, :]
    dy = ys[:, None] - ys[None, :]
    dist2 = (dx*dx + dy*dy).astype(np.int32)
    
    # For each point i, for each distance d, store list of points at that distance
    # This allows fast checking: "is there already a point at distance d from i in S?"
    # We'll use a dictionary: dist_map[i][d] = set of points at distance d from i
    
    # But storing all sets is memory heavy. Instead, use a different approach:
    # For the greedy algorithm, maintain for each point i in S, a set of used distances.
    # When adding P, check: for each Q in S, d(P,Q) not in used_dists[Q], 
    # and all d(P,Q) for Q in S are distinct.
    
    def greedy_construct(order, time_limit):
        """Greedy construction following a given order."""
        S = []
        S_set = set()
        used_dists = {}  # point_idx -> set of distances from that point to other points in S
        
        for idx_P in order:
            if time.time() > time_limit:
                break
            if idx_P in S_set:
                continue
            
            # Check if P can be added
            new_dists = set()
            valid = True
            
            for idx_Q in S:
                d = int(dist2[idx_P, idx_Q])
                if d in new_dists:
                    valid = False
                    break
                new_dists.add(d)
                if d in used_dists[idx_Q]:
                    valid = False
                    break
            
            if valid:
                S.append(idx_P)
                S_set.add(idx_P)
                used_dists[idx_P] = new_dists
                for idx_Q in S:
                    if idx_Q != idx_P:
                        used_dists[idx_Q].add(int(dist2[idx_Q, idx_P]))
        
        return S, used_dists
    
    def local_search(S, used_dists, time_limit, rng):
        """Improve S via swaps."""
        S_set = set(S)
        complement = list(set(range(num_points)) - S_set)
        
        improved = True
        
        while improved and time.time() < time_limit:
            improved = False
            s_list = S[:]
            rng.shuffle(s_list)
            
            for idx_Q in s_list:
                if time.time() > time_limit:
                    break
                if idx_Q not in S_set:
                    continue
                
                # Try to find a P in complement that can replace Q
                rng.shuffle(complement)
                
                for idx_P in complement:
                    if idx_P in S_set:
                        continue
                    
                    # Check if swapping Q for P is valid
                    S_without_Q = [i for i in S if i != idx_Q]
                    
                    valid = True
                    new_dists_P = set()
                    
                    for idx_R in S_without_Q:
                        d_PR = int(dist2[idx_P, idx_R])
                        
                        if d_PR in new_dists_P:
                            valid = False
                            break
                        new_dists_P.add(d_PR)
                        
                        d_RQ = int(dist2[idx_R, idx_Q])
                        
                        if d_PR in used_dists[idx_R] and d_PR != d_RQ:
                            valid = False
                            break
                    
                    if valid:
                        # Perform the swap
                        S.remove(idx_Q)
                        S_set.discard(idx_Q)
                        complement.remove(idx_P)
                        complement.append(idx_Q)
                        
                        S.append(idx_P)
                        S_set.add(idx_P)
                        
                        # Update used_dists
                        for idx_R in S:
                            if idx_R != idx_P:
                                d_new = int(dist2[idx_R, idx_P])
                                d_old = int(dist2[idx_R, idx_Q])
                                used_dists[idx_R].discard(d_old)
                                used_dists[idx_R].add(d_new)
                        
                        used_dists[idx_P] = set()
                        for idx_R in S:
                            if idx_R != idx_P:
                                used_dists[idx_P].add(int(dist2[idx_P, idx_R]))
                        
                        del used_dists[idx_Q]
                        
                        improved = True
                        break
        
        return S, used_dists
    
    # Phase 1: Multiple greedy initializations with different orderings
    rng = random.Random(42)
    
    best_S = []
    best_used = {}
    
    # Generate diverse orderings
    orderings = []
    
    # Random shuffles
    for _ in range(6):
        order = list(range(num_points))
        rng.shuffle(order)
        orderings.append(order)
    
    # Diagonal scan
    order_diag = sorted(range(num_points), key=lambda i: (points[i][0] + points[i][1], points[i][0]))
    orderings.append(order_diag)
    
    # Anti-diagonal
    order_anti = sorted(range(num_points), key=lambda i: (points[i][0] - points[i][1], points[i][0]))
    orderings.append(order_anti)
    
    # Radial from center
    cx, cy = N/2.0, N/2.0
    order_radial = sorted(range(num_points), key=lambda i: (points[i][0]-cx)**2 + (points[i][1]-cy)**2)
    orderings.append(order_radial)
    
    # Row-major
    order_rows = list(range(num_points))
    orderings.append(order_rows)
    
    # Stride patterns
    for stride in [3, 5, 7, 11]:
        order_stride = list(range(0, num_points, stride))
        orderings.append(order_stride)
    
    # Run greedy on each ordering
    for order in orderings:
        S, used = greedy_construct(order, time_limit=deadline - 40)
        if len(S) > len(best_S):
            best_S = S[:]
            best_used = {k: v.copy() for k, v in used.items()}
    
    # Phase 2: Local search on best solution
    S, used = local_search(best_S, best_used, time_limit=deadline - 15, rng=rng)
    
    # Phase 3: Try a few more restarts if time remains
    if time.time() < deadline - 50:
        for _ in range(2):
            order = list(range(num_points))
            rng.shuffle(order)
            S_new, used_new = greedy_construct(order, time_limit=deadline - 30)
            if len(S_new) > len(S):
                S, used = S_new, used_new
            else:
                if len(S_new) >= len(S) - 2 and time.time() < deadline - 20:
                    S_try, used_try = local_search(S_new, used_new, time_limit=deadline - 10, rng=rng)
                    if len(S_try) > len(S):
                        S, used = S_try, used_try
    
    return [points[i] for i in S]
# EVOLVE-BLOCK-END
