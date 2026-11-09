import time
import random
import numpy as np

N = 64
M = N * N

def construct():
    start_time = time.time()
    deadline = start_time + 280  # Safety margin for 300s limit
    
    # Precompute all points as a numpy array for fast distance calculations
    # Shape: (M, 2)
    points = np.array([(x, y) for x in range(N) for y in range(N)], dtype=np.int32)
    
    best_size = 0
    best_S_list = []
    best_D = {}
    
    def try_add(S, D, p_idx):
        """
        Attempts to add point p_idx to set S.
        Returns True if successful, updating S and D.
        """
        if p_idx in S:
            return False
            
        p = points[p_idx]
        
        if len(S) == 0:
            S.add(p_idx)
            D[p_idx] = set()
            return True
            
        S_list = list(S)
        S_pts = points[S_list]
        
        # Vectorized distance calculation
        # diff has shape (|S|, 2)
        diff = p - S_pts
        dists = np.sum(diff * diff, axis=1)
        
        # Check for duplicate distances from p to any two points in S
        # This handles the case where p is the apex of an isosceles triangle
        if len(set(dists)) != len(dists):
            return False
            
        # Check if any distance is already present in the distance set of the target point
        # This handles the case where the target point is the apex of an isosceles triangle
        # involving p and another point already in S
        for i, s_idx in enumerate(S_list):
            d_val = int(dists[i])
            if d_val in D[s_idx]:
                return False
                
        # Validity confirmed. Update data structures.
        S.add(p_idx)
        D[p_idx] = set(dists.tolist()) # Store as list of ints
        
        # Update distance sets for all existing points
        for i, s_idx in enumerate(S_list):
            D[s_idx].add(int(dists[i]))
            
        return True

    def remove_point(S, D, p_idx):
        """
        Removes point p_idx from set S and updates D.
        """
        if p_idx not in S:
            return
        S.discard(p_idx)
        D.pop(p_idx, None)
        
        # Update D for remaining points
        for s_idx in S:
            dx = points[p_idx][0] - points[s_idx][0]
            dy = points[p_idx][1] - points[s_idx][1]
            d_sq = int(dx*dx + dy*dy)
            if d_sq in D[s_idx]:
                D[s_idx].discard(d_sq)

    def greedy_pass(order, time_limit):
        """
        Runs a greedy pass with the given order of points.
        """
        S = set()
        D = {}
        
        for p_idx in order:
            if time.time() > time_limit:
                break
            try_add(S, D, p_idx)
                
        return S, D

    # Phase 1: Random Greedy Restarts
    # The goal is to find a good initial solution quickly.
    t_phase1_end = start_time + 60
    
    while time.time() < t_phase1_end:
        order = list(range(M))
        random.shuffle(order)
        
        S, D = greedy_pass(order, deadline)
        
        if len(S) > best_size:
            best_size = len(S)
            best_S_list = list(S)
            best_D = {k: set(v) for k, v in D.items()}

    # Phase 2: Local Search
    # Refine the best solution found so far.
    S = set(best_S_list)
    D = {k: set(v) for k, v in best_D.items()}
    
    no_improve_count = 0
    max_no_improve = 100
    last_improve_time = time.time()
    
    while time.time() < deadline - 5:
        improved = False
        
        # Create list of candidates not in S
        # Optimization: If S is large, this list is small.
        # If S is small, it's large.
        candidates = [i for i in range(M) if i not in S]
        random.shuffle(candidates)
        
        for p_idx in candidates:
            if try_add(S, D, p_idx):
                improved = True
                break # Restart candidate loop to check new possibilities
        
        if improved:
            no_improve_count = 0
            last_improve_time = time.time()
            if len(S) > best_size:
                best_size = len(S)
                best_S_list = list(S)
                best_D = {k: set(v) for k, v in D.items()}
        else:
            no_improve_count += 1
            
            # If we haven't improved in a while, perturb
            if no_improve_count > max_no_improve:
                # Aggressive Perturbation: Remove 2-5 points
                if len(S) > 5:
                    k = random.randint(2, 5)
                    to_remove = random.sample(list(S), min(k, len(S)))
                    for r in to_remove:
                        remove_point(S, D, r)
                else:
                    # If set is very small, full restart
                    order = list(range(M))
                    random.shuffle(order)
                    S, D = greedy_pass(order, deadline)
                
                no_improve_count = 0
                last_improve_time = time.time()

    # Return the best solution found
    # Ensure we return tuples of (x, y)
    return [tuple(points[i]) for i in best_S_list]
