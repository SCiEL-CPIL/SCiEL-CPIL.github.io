"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 64
import time
import random
import numpy as np

# EVOLVE-BLOCK-START
def construct():
    deadline = time.time() + 295  # Safety margin before 330s kill
    
    # Precompute all points and distances
    points = [(x, y) for y in range(N) for x in range(N)]
    num_points = N * N  # 4096
    
    xs = np.array([p[0] for p in points], dtype=np.int32)
    ys = np.array([p[1] for p in points], dtype=np.int32)
    
    dx = xs[:, None] - xs[None, :]
    dy = ys[:, None] - ys[None, :]
    dist2 = (dx * dx + dy * dy).astype(np.int32)
    
    # For each point, precompute the list of squared distances to all other points
    # We'll use a different approach: track for each point which distances are "used"
    
    def initialize_greedy(order, seed=42):
        """Greedy initialization with given order."""
        rng = random.Random(seed)
        S_set = set()
        # For each point in S, track the set of squared distances to other points in S
        D_dict = {}
        
        for idx in order:
            valid = True
            dists_from_idx = set()
            for other in S_set:
                d = int(dist2[idx, other])
                if d in dists_from_idx:
                    valid = False
                    break
                if d in D_dict[other]:
                    valid = False
                    break
                dists_from_idx.add(d)
            
            if valid:
                S_set.add(idx)
                for other in S_set:
                    if other != idx:
                        d = int(dist2[idx, other])
                        D_dict[other].add(d)
                D_dict[idx] = dists_from_idx
        
        return S_set, D_dict
    
    def is_valid_addition(idx_P, S_set, D_dict):
        """Check if idx_P can be added to S."""
        if idx_P in S_set:
            return False
        dists_from_P = set()
        for idx_Q in S_set:
            d = int(dist2[idx_P, idx_Q])
            if d in dists_from_P:
                return False
            if d in D_dict[idx_Q]:
                return False
            dists_from_P.add(d)
        return True
    
    def add_point(idx_P, S_set, D_dict):
        """Add idx_P to the set."""
        S_set.add(idx_P)
        dists_from_P = set()
        for idx_Q in S_set:
            if idx_Q == idx_P:
                continue
            d = int(dist2[idx_P, idx_Q])
            dists_from_P.add(d)
            D_dict[idx_Q].add(d)
        D_dict[idx_P] = dists_from_P
    
    def remove_point(idx_P, S_set, D_dict):
        """Remove idx_P from the set."""
        if idx_P not in S_set:
            return
        for idx_Q in S_set:
            if idx_Q != idx_P:
                d = int(dist2[idx_P, idx_Q])
                D_dict[idx_Q].discard(d)
        S_set.remove(idx_P)
        del D_dict[idx_P]
    
    # Phase 1: Multiple diverse initializations
    best_S_set = set()
    best_D_dict = {}
    best_size = 0
    
    rng = random.Random(123)
    
    # Try various structured initializations
    # 1. Diagonal lines: points where x - y = c for various c
    for c in range(-N+1, N):
        candidates = []
        for y in range(N):
            x = y + c
            if 0 <= x < N:
                candidates.append(y * N + x)
        rng.shuffle(candidates)
        S_set, D_dict = initialize_greedy(candidates, seed=42 + abs(c))
        if len(S_set) > best_size:
            best_size = len(S_set)
            best_S_set = S_set.copy()
            best_D_dict = {k: v.copy() for k, v in D_dict.items()}
    
    # 2. Anti-diagonal lines: points where x + y = c
    for c in range(2*N - 1):
        candidates = []
        for y in range(N):
            x = c - y
            if 0 <= x < N:
                candidates.append(y * N + x)
        rng.shuffle(candidates)
        S_set, D_dict = initialize_greedy(candidates, seed=100 + c)
        if len(S_set) > best_size:
            best_size = len(S_set)
            best_S_set = S_set.copy()
            best_D_dict = {k: v.copy() for k, v in D_dict.items()}
    
    # 3. Random orders
    for trial in range(20):
        order = list(range(num_points))
        rng.shuffle(order)
        S_set, D_dict = initialize_greedy(order, seed=200 + trial)
        if len(S_set) > best_size:
            best_size = len(S_set)
            best_S_set = S_set.copy()
            best_D_dict = {k: v.copy() for k, v in D_dict.items()}
    
    # 4. Row-based: pick points from different rows with spacing
    for start_row in range(0, N, 3):
        candidates = []
        for y in range(start_row, N, 3):
            for x in range(0, N, 3):
                candidates.append(y * N + x)
        rng.shuffle(candidates)
        S_set, D_dict = initialize_greedy(candidates, seed=300 + start_row)
        if len(S_set) > best_size:
            best_size = len(S_set)
            best_S_set = S_set.copy()
            best_D_dict = {k: v.copy() for k, v in D_dict.items()}
    
    # 5. Sparse grid with varying offsets
    for scale_x in [2, 3, 4, 5]:
        for scale_y in [2, 3, 4, 5]:
            candidates = []
            for y in range(0, N, scale_y):
                for x in range(0, N, scale_x):
                    candidates.append(y * N + x)
            rng.shuffle(candidates)
            S_set, D_dict = initialize_greedy(candidates, seed=400 + scale_x * 10 + scale_y)
            if len(S_set) > best_size:
                best_size = len(S_set)
                best_S_set = S_set.copy()
                best_D_dict = {k: v.copy() for k, v in D_dict.items()}
    
    # Phase 2: Local search with add/remove/swap moves
    S_set = best_S_set.copy()
    D_dict = {k: v.copy() for k, v in best_D_dict.items()}
    current_size = len(S_set)
    best_size_ls = current_size
    best_S_set_ls = S_set.copy()
    best_D_dict_ls = {k: v.copy() for k, v in D_dict.items()}
    
    rng_ls = random.Random(789)
    start_time = time.time()
    no_improve_count = 0
    max_no_improve = 500
    
    while time.time() < deadline - 20:
        if no_improve_count > max_no_improve:
            break
        
        move_type = rng_ls.random()
        
        if move_type < 0.6:
            # Try to add a random point
            complement = [i for i in range(num_points) if i not in S_set]
            if not complement:
                continue
            idx_add = rng_ls.choice(complement)
            
            if is_valid_addition(idx_add, S_set, D_dict):
                add_point(idx_add, S_set, D_dict)
                new_size = len(S_set)
                if new_size > best_size_ls:
                    best_size_ls = new_size
                    best_S_set_ls = S_set.copy()
                    best_D_dict_ls = {k: v.copy() for k, v in D_dict.items()}
                    no_improve_count = 0
                else:
                    no_improve_count += 1
            else:
                no_improve_count += 1
        
        elif move_type < 0.85:
            # Try a swap: remove one, add another
            if not S_set:
                continue
            idx_out = rng_ls.choice(list(S_set))
            complement = [i for i in range(num_points) if i not in S_set]
            if not complement:
                continue
            idx_in = rng_ls.choice(complement)
            
            # Check if swap is valid
            # Temporarily remove idx_out
            remove_point(idx_out, S_set, D_dict)
            
            if is_valid_addition(idx_in, S_set, D_dict):
                add_point(idx_in, S_set, D_dict)
                new_size = len(S_set)
                if new_size > best_size_ls:
                    best_size_ls = new_size
                    best_S_set_ls = S_set.copy()
                    best_D_dict_ls = {k: v.copy() for k, v in D_dict.items()}
                    no_improve_count = 0
                else:
                    no_improve_count += 1
            else:
                # Invalid swap, add back idx_out
                add_point(idx_out, S_set, D_dict)
                no_improve_count += 1
        
        else:
            # Try to remove a point (to explore different configurations)
            if len(S_set) > 1:
                idx_remove = rng_ls.choice(list(S_set))
                remove_point(idx_remove, S_set, D_dict)
                no_improve_count += 1
        
        # Periodic check
        if (no_improve_count % 100 == 0) and time.time() > deadline - 20:
            break
    
    # Use the best found
    if best_size_ls > best_size:
        best_S_set = best_S_set_ls
        best_D_dict = best_D_dict_ls
    
    # Convert to list of points
    return [points[i] for i in best_S_set]
# EVOLVE-BLOCK-END
