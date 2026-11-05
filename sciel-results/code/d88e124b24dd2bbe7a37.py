"""Combined program: fast greedy + bitset conflict precomputation + local search."""
import time
import random
import math
import numpy as np

N = 64

def construct():
    start_time = time.time()
    deadline = start_time + 280  # Safety margin for 300s limit
    
    num_points = N * N
    
    # Precompute coordinates
    xs = np.arange(N, dtype=np.int32)
    ys = np.arange(N, dtype=np.int32)
    all_x = np.repeat(xs, N)
    all_y = np.tile(ys, N)
    
    # Compute squared distance matrix
    dx = all_x[:, None] - all_x[None, :]
    dy = all_y[:, None] - all_y[None, :]
    dist2 = (dx * dx + dy * dy).astype(np.int32)
    
    # Precompute distance maps: for each point P, map from distance to list of points at that distance
    # This helps quickly check if a point conflicts with existing set
    dist_map = [{} for _ in range(num_points)]
    for P in range(num_points):
        d_arr = dist2[P]
        for Q in range(num_points):
            if P != Q:
                d = int(d_arr[Q])
                if d not in dist_map[P]:
                    dist_map[P][d] = []
                dist_map[P][d].append(Q)
    
    # For fast conflict checking: precompute for each point P, the set of distances from P
    # and a bitset representation of which points are at each distance
    # Actually, let's use a simpler approach: for each point P, precompute the set of distances
    dist_set_all = [set() for _ in range(num_points)]
    for P in range(num_points):
        for Q in range(num_points):
            if P != Q:
                dist_set_all[P].add(int(dist2[P, Q]))
    
    # Precompute equidistant conflicts: for each pair (P,Q), check if there exists R
    # such that dist(R,P) == dist(R,Q). We'll use a bitset for this.
    # equi_conflict[P] is a bitset where bit Q is set if P and Q are equidistant from some R
    num_bits = num_points
    num_uint64 = (num_bits + 63) // 64
    
    # Initialize equi_conflict as list of bytearrays for efficiency
    equi_conflict = [bytearray(num_points) for _ in range(num_points)]
    
    # For each R, group points by distance from R
    # For each group, all pairs in the group are equidistant from R
    # We'll mark these pairs
    for R in range(num_points):
        # Group points by distance from R
        groups = {}
        for Q in range(num_points):
            if Q != R:
                d = int(dist2[R, Q])
                if d not in groups:
                    groups[d] = []
                groups[d].append(Q)
        
        # For each group with more than one point, mark all pairs as equidistant
        for d, pts in groups.items():
            if len(pts) > 1:
                for i in range(len(pts)):
                    for j in range(i + 1, len(pts)):
                        P = pts[i]
                        Q = pts[j]
                        equi_conflict[P][Q] = 1
                        equi_conflict[Q][P] = 1
    
    # Now we can check if P and Q conflict:
    # 1. dist(P,Q) is in dist_set_all[P] (P has another point at same distance as Q)
    #    AND that other point is not Q itself... wait, Q is at distance dist(P,Q) from P,
    #    so dist(P,Q) is always in dist_set_all[P]. We need to check if there's ANOTHER point
    #    at the same distance. So check if len(dist_map[P][dist(P,Q)]) > 1
    # 2. Similarly for Q
    # 3. equi_conflict[P][Q] is set
    
    def points_conflict(P, Q):
        if P == Q:
            return False
        d = int(dist2[P, Q])
        # Case 1: P has another point at distance d
        if len(dist_map[P].get(d, [])) > 1:
            return True
        # Case 2: Q has another point at distance d
        if len(dist_map[Q].get(d, [])) > 1:
            return True
        # Case 3: There exists R equidistant from P and Q
        if equi_conflict[P][Q]:
            return True
        return False
    
    def try_greedy(order):
        S = []
        S_set = set()
        # dist_sets[i] is a set of distances from S[i] to other points in S
        dist_sets = []
        
        for P in order:
            if P in S_set:
                continue
            
            # Check if P can be added
            valid = True
            if len(S) > 0:
                # Check each existing point in S
                for i, Q in enumerate(S):
                    if points_conflict(P, Q):
                        valid = False
                        break
            
            if valid:
                S.append(P)
                S_set.add(P)
                # Update dist_sets
                new_dist_set = set()
                if len(S) > 1:
                    d_vals = dist2[P, S[:-1]]
                    for i, d in enumerate(d_vals):
                        dist_sets[i].add(int(d))
                        new_dist_set.add(int(d))
                dist_sets.append(new_dist_set)
        
        return S, dist_sets
    
    def local_search(S, dist_sets, time_limit):
        best_S = S[:]
        best_dist_sets = [s.copy() for s in dist_sets]
        best_size = len(S)
        
        S_set = set(S)
        
        no_improve = 0
        max_no_improve = 50
        
        while time.time() < time_limit and no_improve < max_no_improve:
            # Try to add a random point not in S
            complement = [i for i in range(num_points) if i not in S_set]
            if not complement:
                break
            
            added = False
            batch_size = min(200, len(complement))
            candidates = random.sample(complement, batch_size)
            
            for P in candidates:
                if P in S_set:
                    continue
                    
                # Validate
                valid = True
                for Q in S:
                    if points_conflict(P, Q):
                        valid = False
                        break
                
                if valid:
                    S.append(P)
                    S_set.add(P)
                    
                    new_dist_set = set()
                    if len(S) > 1:
                        d_vals = dist2[P, S[:-1]]
                        for i, d in enumerate(d_vals):
                            dist_sets[i].add(int(d))
                            new_dist_set.add(int(d))
                    dist_sets.append(new_dist_set)
                    
                    if len(S) > best_size:
                        best_size = len(S)
                        best_S = S[:]
                        best_dist_sets = [s.copy() for s in dist_sets]
                        no_improve = 0
                    else:
                        no_improve += 1
                    
                    added = True
                    break
            
            if not added:
                no_improve += 1
                
                # If stuck, try a swap: remove a random point and try to add two new ones
                if len(S) > 2:
                    rem_idx = random.randrange(len(S))
                    rem_P = S[rem_idx]
                    
                    # Remove rem_P from S
                    for i, P_i in enumerate(S):
                        if i == rem_idx:
                            continue
                        d = int(dist2[rem_P, P_i])
                        dist_sets[i].discard(d)
                    
                    S.pop(rem_idx)
                    dist_sets.pop(rem_idx)
                    S_set.discard(rem_P)
                    
                    # Try to add 2 new points
                    added_count = 0
                    complement = [i for i in range(num_points) if i not in S_set]
                    if complement:
                        batch_size = min(100, len(complement))
                        candidates = random.sample(complement, batch_size)
                        
                        for P in candidates:
                            if P in S_set:
                                continue
                            valid = True
                            for Q in S:
                                if points_conflict(P, Q):
                                    valid = False
                                    break
                            
                            if valid:
                                S.append(P)
                                S_set.add(P)
                                new_dist_set = set()
                                if len(S) > 1:
                                    d_vals = dist2[P, S[:-1]]
                                    for i, d in enumerate(d_vals):
                                        dist_sets[i].add(int(d))
                                        new_dist_set.add(int(d))
                                dist_sets.append(new_dist_set)
                                added_count += 1
                                if len(S) > best_size:
                                    best_size = len(S)
                                    best_S = S[:]
                                    best_dist_sets = [s.copy() for s in dist_sets]
                                    no_improve = 0
                                if added_count == 2:
                                    break
                    
                    # Reset no_improve after perturbation
                    no_improve = 0
        
        return best_S, best_dist_sets
    
    best_S = []
    best_dist_sets = []
    best_size = 0
    
    rng = random.Random(42)
    
    # Multiple restarts with different orderings
    for restart in range(30):
        if time.time() > deadline - 30:
            break
            
        # Generate a random ordering
        order = list(range(num_points))
        rng.shuffle(order)
        
        # Sometimes use structured orderings
        if restart % 5 == 0:
            # Sort by x
            order = sorted(range(num_points), key=lambda i: all_x[i])
        elif restart % 5 == 1:
            # Sort by y
            order = sorted(range(num_points), key=lambda i: all_y[i])
        elif restart % 5 == 2:
            # Sort by x+y
            order = sorted(range(num_points), key=lambda i: all_x[i] + all_y[i])
        elif restart % 5 == 3:
            # Random
            order = list(range(num_points))
            rng.shuffle(order)
        else:
            # Random with bias towards center
            cx, cy = N/2.0, N/2.0
            order = sorted(range(num_points), key=lambda i: (all_x[i]-cx)**2 + (all_y[i]-cy)**2)
        
        S, dist_sets = try_greedy(order)
        
        if len(S) > best_size:
            best_size = len(S)
            best_S = S[:]
            best_dist_sets = [s.copy() for s in dist_sets]
        
        # Local search
        S_ls, dist_sets_ls = local_search(S, dist_sets, deadline - 30)
        
        if len(S_ls) > best_size:
            best_size = len(S_ls)
            best_S = S_ls[:]
            best_dist_sets = [s.copy() for s in dist_sets_ls]
        
        # Check time
        if time.time() > deadline - 10:
            break
    
    return [(int(all_x[i]), int(all_y[i])) for i in best_S]
