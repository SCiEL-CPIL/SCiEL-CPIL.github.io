"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 64
import time
import random
import numpy as np

# EVOLVE-BLOCK-START
def construct():
    deadline = time.time() + 290  # Safety margin before 330s kill
    
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
    
    # Precompute for each point, the set of distances to all other points
    # This is already in dist2
    
    # Key data structure: for each point i, dists_from_i[j] = dist2[i,j]
    # We'll use a different approach: track conflicts via distance multisets
    
    def greedy_construct(order, time_limit):
        """Greedy construction following a given order."""
        S = []
        S_set = set()
        # For each point in S, track distances to other points in S
        dist_sets = {}  # point_idx -> set of squared distances to other points in S
        
        for idx_P in order:
            if time.time() > time_limit:
                break
            if idx_P in S_set:
                continue
            
            # Check if P can be added: all distances from P to S must be unique,
            # and for each Q in S, dist(P,Q) must not be in dist_sets[Q]
            new_dists = set()
            valid = True
            
            for idx_Q in S:
                d = int(dist2[idx_P, idx_Q])
                if d in new_dists:
                    valid = False
                    break
                new_dists.add(d)
                if d in dist_sets[idx_Q]:
                    valid = False
                    break
            
            if valid:
                S.append(idx_P)
                S_set.add(idx_P)
                dist_sets[idx_P] = new_dists
                for idx_Q in S:
                    if idx_Q != idx_P:
                        dist_sets[idx_Q].add(int(dist2[idx_Q, idx_P]))
        
        return S, dist_sets
    
    def can_add_point(idx_P, S_list, dist_sets):
        """Check if idx_P can be added to S_list."""
        if idx_P in dist_sets:
            return False
        
        new_dists = set()
        for idx_Q in S_list:
            d = int(dist2[idx_P, idx_Q])
            if d in new_dists:
                return False
            new_dists.add(d)
            if d in dist_sets[idx_Q]:
                return False
        return True
    
    def add_point(idx_P, S_list, dist_sets, S_set):
        """Add idx_P to the set."""
        S_list.append(idx_P)
        S_set.add(idx_P)
        dist_sets[idx_P] = set()
        for idx_Q in S_list:
            if idx_Q != idx_P:
                d = int(dist2[idx_P, idx_Q])
                dist_sets[idx_P].add(d)
                dist_sets[idx_Q].add(d)
    
    def remove_point(idx_P, S_list, dist_sets, S_set):
        """Remove idx_P from the set."""
        S_list.remove(idx_P)
        S_set.discard(idx_P)
        for idx_Q in S_list:
            dist_sets[idx_Q].discard(int(dist2[idx_Q, idx_P]))
        del dist_sets[idx_P]
    
    def try_add_random(S_list, dist_sets, S_set, complement, rng, max_tries=100):
        """Try to add a random point from complement."""
        for _ in range(max_tries):
            if not complement:
                break
            idx_P = rng.choice(complement)
            if can_add_point(idx_P, S_list, dist_sets):
                add_point(idx_P, S_list, dist_sets, S_set)
                complement.remove(idx_P)
                return True
        return False
    
    def try_swap(S_list, dist_sets, S_set, complement, rng, max_tries=50):
        """Try a random swap: remove a random point, add a random other point."""
        if not S_list or not complement:
            return False
        
        idx_Q = rng.choice(S_list)
        
        # Temporarily remove Q
        S_list.remove(idx_Q)
        S_set.discard(idx_Q)
        temp_dists = {k: v.copy() for k, v in dist_sets.items() if k != idx_Q}
        for idx_R in S_list:
            temp_dists[idx_R].discard(int(dist2[idx_R, idx_Q]))
        
        # Try to add a point from complement
        success = False
        for _ in range(max_tries):
            if not complement:
                break
            idx_P = rng.choice(complement)
            if idx_P == idx_Q:
                continue
            if can_add_point(idx_P, S_list, temp_dists):
                # Add P
                S_list.append(idx_P)
                S_set.add(idx_P)
                temp_dists[idx_P] = set()
                for idx_R in S_list:
                    if idx_R != idx_P:
                        d = int(dist2[idx_P, idx_R])
                        temp_dists[idx_P].add(d)
                        temp_dists[idx_R].add(d)
                
                # Commit the swap
                dist_sets.clear()
                dist_sets.update(temp_dists)
                complement.remove(idx_P)
                complement.append(idx_Q)
                success = True
                break
        
        if not success:
            # Revert: put Q back
            S_list.append(idx_Q)
            S_set.add(idx_Q)
            for idx_R in S_list:
                if idx_R != idx_Q:
                    dist_sets[idx_R].add(int(dist2[idx_R, idx_Q]))
            dist_sets[idx_Q] = set()
            for idx_R in S_list:
                if idx_R != idx_Q:
                    dist_sets[idx_Q].add(int(dist2[idx_Q, idx_R]))
        
        return success
    
    # Phase 1: Greedy initialization
    rng = random.Random(42)
    
    best_S = []
    best_dist_sets = {}
    
    # Generate diverse orderings
    orderings = []
    
    # Random shuffles
    for _ in range(5):
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
    greedy_time_limit = deadline - 80  # Leave time for local search
    for order in orderings:
        S, dist_sets = greedy_construct(order, time_limit=greedy_time_limit)
        if len(S) > len(best_S):
            best_S = S[:]
            best_dist_sets = {k: v.copy() for k, v in dist_sets.items()}
    
    # Phase 2: Local search with simulated annealing style
    S = best_S[:]
    dist_sets = {k: v.copy() for k, v in best_dist_sets.items()}
    S_set = set(S)
    complement = list(set(range(num_points)) - S_set)
    
    ls_time_limit = deadline - 20
    
    # Simulated annealing inspired local search
    current_temp = 1.0
    cooling_rate = 0.999
    
    best_local = S[:]
    best_local_dist = {k: v.copy() for k, v in dist_sets.items()}
    best_local_size = len(S)
    
    step_count = 0
    no_improve_count = 0
    
    while time.time() < ls_time_limit:
        step_count += 1
        
        # Choose move type
        move_type = rng.random()
        
        if move_type < 0.3:
            # Try to add a point
            if try_add_random(S, dist_sets, S_set, complement, rng, max_tries=20):
                if len(S) > best_local_size:
                    best_local = S[:]
                    best_local_dist = {k: v.copy() for k, v in dist_sets.items()}
                    best_local_size = len(S)
                    no_improve_count = 0
                else:
                    no_improve_count += 1
            else:
                no_improve_count += 1
        elif move_type < 0.7:
            # Try a swap
            if try_swap(S, dist_sets, S_set, complement, rng, max_tries=30):
                if len(S) > best_local_size:
                    best_local = S[:]
                    best_local_dist = {k: v.copy() for k, v in dist_sets.items()}
                    best_local_size = len(S)
                    no_improve_count = 0
                else:
                    no_improve_count += 1
            else:
                no_improve_count += 1
        else:
            # Perturbation: remove 1-2 random points
            num_remove = rng.randint(1, 2)
            for _ in range(min(num_remove, len(S))):
                if S:
                    idx_Q = rng.choice(S)
                    remove_point(idx_Q, S, dist_sets, S_set)
                    complement.append(idx_Q)
            
            # Then try to add back
            for _ in range(num_remove + 1):
                try_add_random(S, dist_sets, S_set, complement, rng, max_tries=30)
            
            if len(S) > best_local_size:
                best_local = S[:]
                best_local_dist = {k: v.copy() for k, v in dist_sets.items()}
                best_local_size = len(S)
                no_improve_count = 0
            else:
                no_improve_count += 1
        
        # Cool down
        current_temp *= cooling_rate
        
        # If stuck for too long, restart from best
        if no_improve_count > 200:
            S = best_local[:]
            dist_sets = {k: v.copy() for k, v in best_local_dist.items()}
            S_set = set(S)
            complement = list(set(range(num_points)) - S_set)
            no_improve_count = 0
            current_temp = max(0.1, current_temp * 2)
    
    # Use the best found
    S = best_local
    dist_sets = best_local_dist
    
    # Phase 3: Final greedy fill
    if time.time() < deadline - 10:
        S_set = set(S)
        complement = list(set(range(num_points)) - S_set)
        # Try to add more points greedily
        order = list(complement)
        rng.shuffle(order)
        for idx_P in order:
            if time.time() > deadline - 10:
                break
            if can_add_point(idx_P, S, dist_sets):
                add_point(idx_P, S, dist_sets, S_set)
                complement.remove(idx_P)
    
    return [points[i] for i in S]
# EVOLVE-BLOCK-END
