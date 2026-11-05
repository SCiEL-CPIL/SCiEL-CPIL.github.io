import time
import random
import numpy as np

N = 64

def construct():
    start_time = time.time()
    deadline = start_time + 290  # Safety margin for 300s limit
    
    # Precompute squared distances between all pairs of grid points
    # Indexing: idx = y * N + x
    num_points = N * N
    xs = np.arange(N, dtype=np.int32)
    ys = np.arange(N, dtype=np.int32)
    
    # Create coordinate arrays for all points
    all_x = np.repeat(xs, N)
    all_y = np.tile(ys, N)
    
    # Compute squared distance matrix (num_points x num_points)
    dx = all_x[:, None] - all_x[None, :]
    dy = all_y[:, None] - all_y[None, :]
    dist2 = (dx * dx + dy * dy).astype(np.int32)
    
    # Helper function to check if a point can be added to the current set
    # S: list of indices in current set
    # dist_sets: list of sets of squared distances from S[i] to other points in S
    def can_add(P, S, dist_sets):
        if len(S) == 0:
            return True
        
        # Get distances from P to all points in S
        d_vals = dist2[P, S]
        
        # Check if P forms an isosceles triangle with any two points in S
        # This happens if any distance is repeated (P is the apex)
        if len(np.unique(d_vals)) != len(d_vals):
            return False
        
        # Check if any point in S forms an isosceles triangle with P and another point in S
        # This happens if the distance from S[i] to P is already in dist_sets[i]
        # (meaning S[i] already has another point at that distance)
        for i, d in enumerate(d_vals):
            if d in dist_sets[i]:
                return False
        
        return True
    
    def add_point(P, S, dist_sets):
        S.append(P)
        new_dist_set = set()
        if len(S) > 1:
            d_vals = dist2[P, S[:-1]]
            for i, d in enumerate(d_vals):
                dist_sets[i].add(d)
                new_dist_set.add(d)
        dist_sets.append(new_dist_set)
    
    def remove_point(P, S, dist_sets):
        idx = S.index(P)
        # Remove distances from P to all other points in S
        for i, P_i in enumerate(S):
            if i == idx:
                continue
            d = int(dist2[P, P_i])
            dist_sets[i].discard(d)
        S.pop(idx)
        dist_sets.pop(idx)
        
    def greedy_construction(order):
        S = []
        dist_sets = []
        in_S = np.zeros(num_points, dtype=bool)
        
        for P in order:
            if in_S[P]:
                continue
            
            if can_add(P, S, dist_sets):
                in_S[P] = True
                add_point(P, S, dist_sets)
        
        return S, dist_sets, in_S
    
    def local_search(S, dist_sets, in_S, time_limit):
        best_S = S[:]
        best_dist_sets = [s.copy() for s in dist_sets]
        best_size = len(S)
        
        no_improve = 0
        max_no_improve = 50
        
        while time.time() < time_limit and no_improve < max_no_improve:
            # Try to add a random point not in S
            # We sample a few random points from the complement
            added = False
            
            # Try a batch of random candidates
            batch_size = min(200, num_points - len(S))
            if batch_size > 0:
                candidates = random.sample(range(num_points), batch_size)
                for P in candidates:
                    if in_S[P]:
                        continue
                    
                    if can_add(P, S, dist_sets):
                        in_S[P] = True
                        add_point(P, S, dist_sets)
                        
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
                
                # If stuck, try a perturbation: remove a random point and try to add 2
                if len(S) > 2:
                    rem_idx = random.randrange(len(S))
                    rem_P = S[rem_idx]
                    
                    in_S[rem_P] = False
                    remove_point(rem_P, S, dist_sets)
                    
                    # Try to add 2 new points
                    added_count = 0
                    batch_size = min(100, num_points - len(S))
                    if batch_size > 0:
                        candidates = random.sample(range(num_points), batch_size)
                        for P in candidates:
                            if in_S[P]:
                                continue
                            
                            if can_add(P, S, dist_sets):
                                in_S[P] = True
                                add_point(P, S, dist_sets)
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
    
    # Main loop: run multiple rounds of greedy + local search
    best_set = []
    best_size = 0
    
    # Initial greedy construction with a few different orderings
    for _ in range(3):
        order = list(range(num_points))
        random.shuffle(order)
        
        S, dist_sets, in_S = greedy_construction(order)
        
        if len(S) > best_size:
            best_size = len(S)
            best_set = S[:]
        
        # Run local search
        remaining_time = deadline - time.time()
        if remaining_time > 10:
            S, dist_sets = local_search(S, dist_sets, in_S, time.time() + min(20, remaining_time))
            if len(S) > best_size:
                best_size = len(S)
                best_set = S[:]
        
        if time.time() > deadline:
            break
    
    # Continue with more rounds if time permits
    while time.time() < deadline - 5:
        order = list(range(num_points))
        random.shuffle(order)
        
        S, dist_sets, in_S = greedy_construction(order)
        
        if len(S) > best_size:
            best_size = len(S)
            best_set = S[:]
        
        # Run local search with a shorter time limit per round
        remaining_time = deadline - time.time()
        if remaining_time > 5:
            search_time = min(10, remaining_time / 2)
            S, dist_sets = local_search(S, dist_sets, in_S, time.time() + search_time)
            if len(S) > best_size:
                best_size = len(S)
                best_set = S[:]
        
        # Early exit if we found a very good set
        if best_size > 50:
            break
    
    # Convert indices back to (x, y) coordinates
    result = [(idx % N, idx // N) for idx in best_set]
    
    return result

if __name__ == "__main__":
    points = construct()
    print(f"Found set of size {len(points)}")
    print(points[:10])
