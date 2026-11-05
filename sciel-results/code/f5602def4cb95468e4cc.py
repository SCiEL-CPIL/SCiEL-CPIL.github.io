"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 64
import time
import random
import numpy as np

# EVOLVE-BLOCK-START
def construct():
    deadline = time.time() + 285  # Safety margin
    
    # Grid points
    points = [(x, y) for y in range(N) for x in range(N)]
    num_points = N * N
    
    # Create coordinate arrays
    xs = np.array([p[0] for p in points], dtype=np.int32)
    ys = np.array([p[1] for p in points], dtype=np.int32)
    
    # Calculate squared distances using int32 to save memory
    dx = xs[:, None] - xs[None, :]
    dy = ys[:, None] - ys[None, :]
    dist2 = (dx*dx + dy*dy).astype(np.int32)
    
    # Helper to check if a point can be added to a set
    def is_valid_addition(idx_P, S_indices, D_dict):
        if not S_indices:
            return True
        dists_from_P = set()
        # Get all distances from P to points in S at once using numpy
        S_arr = np.array(S_indices, dtype=np.int32)
        dists_to_S = dist2[idx_P, S_arr]
        
        for idx_Q, d in zip(S_indices, dists_to_S):
            d_val = int(d)
            if d_val in dists_from_P:
                return False
            dists_from_P.add(d_val)
            if d_val in D_dict[idx_Q]:
                return False
        return True

    def add_point(idx_P, S_indices, D_dict, S_set, complement):
        S_indices.append(idx_P)
        S_set.add(idx_P)
        complement.remove(idx_P)
        
        dists_from_P = set()
        for idx_Q in S_indices:
            if idx_Q == idx_P:
                continue
            d = int(dist2[idx_P, idx_Q])
            dists_from_P.add(d)
            D_dict[idx_Q].add(d)
        D_dict[idx_P] = dists_from_P

    def remove_point(idx_P, S_indices, D_dict, S_set, complement):
        if idx_P not in S_set: return
        S_indices.remove(idx_P)
        S_set.discard(idx_P)
        complement.append(idx_P)
        
        for idx_Q in S_indices:
            d = int(dist2[idx_P, idx_Q])
            D_dict[idx_Q].discard(d)
        del D_dict[idx_P]

    def swap_point(idx_P, idx_Q, S_indices, D_dict, S_set, complement):
        remove_point(idx_Q, S_indices, D_dict, S_set, complement)
        add_point(idx_P, S_indices, D_dict, S_set, complement)

    def greedy_construct(order, time_limit=None):
        S_indices = []
        D_dict = {}
        S_set = set()
        complement = list(range(num_points))
        
        for idx_P in order:
            if idx_P in S_set:
                continue
            if time_limit and time.time() > time_limit:
                break
            if is_valid_addition(idx_P, S_indices, D_dict):
                add_point(idx_P, S_indices, D_dict, S_set, complement)
                
        return S_indices, D_dict, S_set, complement

    def local_search(S_indices, D_dict, S_set, complement, time_limit=None):
        improved = True
        passes = 0
        max_passes = 100
        rng = random.Random(42)
        
        while improved and time.time() < time_limit and passes < max_passes:
            improved = False
            passes += 1
            
            # Shuffle the order of points in S to try
            s_list = S_indices[:]
            rng.shuffle(s_list)
            
            for idx_Q in s_list:
                if time.time() > time_limit:
                    break
                if idx_Q not in S_set:
                    continue
                
                # Try to find a valid P in complement to replace Q
                comp_list = complement[:]
                rng.shuffle(comp_list)
                
                # Limit candidates to speed up, but try more if set is small
                num_candidates = min(len(comp_list), 200)
                
                found = False
                for idx_P in comp_list[:num_candidates]:
                    if idx_P in S_set:
                        continue
                    
                    # Check if swapping Q for P is valid
                    # We need to check if P can be added to S \ {Q}
                    # and if removing Q doesn't break anything (it won't, as it's just removal)
                    
                    # Temporarily remove Q
                    S_without_Q = [i for i in S_indices if i != idx_Q]
                    
                    # Check if P is valid against S_without_Q
                    # We need a helper that doesn't modify state yet
                    valid = True
                    dists_from_P = set()
                    
                    # Optimization: use numpy for distance check
                    if S_without_Q:
                        S_arr = np.array(S_without_Q, dtype=np.int32)
                        dists_to_S = dist2[idx_P, S_arr]
                        
                        for idx_R, d in zip(S_without_Q, dists_to_S):
                            d_val = int(d)
                            if d_val in dists_from_P:
                                valid = False
                                break
                            dists_from_P.add(d_val)
                            
                            # Check R's perspective: after swap, R has d_PR instead of d_RQ
                            # R's other distances (excluding d_RQ) must not include d_PR
                            d_RQ = int(dist2[idx_R, idx_Q])
                            
                            if d_val in D_dict[idx_R] and d_val != d_RQ:
                                valid = False
                                break
                    
                    if valid:
                        # Perform the swap
                        swap_point(idx_P, idx_Q, S_indices, D_dict, S_set, complement)
                        improved = True
                        found = True
                        break
                
                if not found:
                    # If no swap found for Q, continue to next Q
                    pass
                    
        return S_indices, D_dict, S_set, complement

    # Main Search
    best_S_indices = []
    best_D = {}
    best_S_set = set()
    best_complement = list(range(num_points))
    
    rng = random.Random(42)
    
    # Phase 1: Multiple Greedy Initializations
    orderings = []
    
    # 1. Random shuffles
    for _ in range(5):
        order = list(range(num_points))
        rng.shuffle(order)
        orderings.append(order)
        
    # 2. Diagonal scan
    order_diag = sorted(range(num_points), key=lambda i: (points[i][0] + points[i][1], points[i][0]))
    orderings.append(order_diag)
    
    # 3. Anti-diagonal
    order_anti = sorted(range(num_points), key=lambda i: (points[i][0] - points[i][1], points[i][0]))
    orderings.append(order_anti)
    
    # 4. Center-out
    cx, cy = N/2.0, N/2.0
    order_radial = sorted(range(num_points), key=lambda i: (points[i][0]-cx)**2 + (points[i][1]-cy)**2)
    orderings.append(order_radial)
    
    # 5. Row-major
    order_rows = list(range(num_points))
    orderings.append(order_rows)
    
    # 6. Column-major (transposed)
    order_cols = [y*N + x for y in range(N) for x in range(N)]
    orderings.append(order_cols)
    
    # 7. Stride patterns
    for stride in [2, 3, 5, 7]:
        order_stride = list(range(0, num_points, stride))
        orderings.append(order_stride)
        
    # Run greedy on each ordering
    for order in orderings:
        S_indices, D_dict, S_set, complement = greedy_construct(order, time_limit=deadline - 30)
        if len(S_indices) > len(best_S_indices):
            best_S_indices = S_indices[:]
            best_D = {k: v.copy() for k, v in D_dict.items()}
            best_S_set = S_set.copy()
            best_complement = complement[:]
            
    # Phase 2: Local Search on best solution
    S_indices = best_S_indices[:]
    D_dict = {k: v.copy() for k, v in best_D.items()}
    S_set = best_S_set.copy()
    complement = best_complement[:]
    
    S_indices, D_dict, S_set, complement = local_search(S_indices, D_dict, S_set, complement, time_limit=deadline - 10)
    
    # Phase 3: Try a few more restarts if time remains
    if time.time() < deadline - 60:
        for _ in range(3):
            order = list(range(num_points))
            rng.shuffle(order)
            S_new, D_new, S_set_new, comp_new = greedy_construct(order, time_limit=deadline - 30)
            
            if len(S_new) > len(S_indices):
                S_indices = S_new
                D_dict = D_new
                S_set = S_set_new
                complement = comp_new
            else:
                # Try local search on this too if it's close
                if len(S_new) >= len(S_indices) - 2 and time.time() < deadline - 20:
                    S_try, D_try, S_set_try, comp_try = local_search(S_new, D_new, S_set_new, comp_new, time_limit=deadline - 10)
                    if len(S_try) > len(S_indices):
                        S_indices = S_try
                        D_dict = D_try
                        S_set = S_set_try
                        complement = comp_try
                        
    return [points[i] for i in S_indices]
# EVOLVE-BLOCK-END
