"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 64
import time
import random
import numpy as np

# EVOLVE-BLOCK-START
def construct():
    deadline = time.time() + 290  # Safety margin before 330s kill
    
    # Precompute all points and distances
    points = [(x, y) for y in range(N) for x in range(N)]
    num_points = N * N
    
    xs = np.array([p[0] for p in points], dtype=np.int32)
    ys = np.array([p[1] for p in points], dtype=np.int32)
    
    dx = xs[:, None] - xs[None, :]
    dy = ys[:, None] - ys[None, :]
    dist2 = (dx * dx + dy * dy).astype(np.int32)
    
    # State representation:
    # S_set: set of indices in current solution
    # D_dict: dict mapping point index -> set of squared distances to other points in S
    
    def initialize_lattice(scale, offset_x=0, offset_y=0, seed=42):
        """Initialize with a scaled lattice that avoids isosceles triangles."""
        rng = random.Random(seed)
        S_set = set()
        D_dict = {}
        
        # Use a diagonal lattice: points where (x - y) % scale == offset
        # This creates a 1D-like structure embedded in 2D
        candidates = []
        for y in range(N):
            for x in range(N):
                if (x - y) % scale == offset_x % scale:
                    candidates.append(y * N + x)
        
        # Sort by a non-trivial order
        rng.shuffle(candidates)
        
        for idx in candidates:
            # Check if valid to add
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
    
    def initialize_random(order, seed=42):
        """Greedy initialization with given order."""
        rng = random.Random(seed)
        S_set = set()
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
    
    def is_valid_swap(idx_in, idx_out, S_set, D_dict):
        """Check if swapping idx_out for idx_in keeps the set valid."""
        if idx_in in S_set or idx_out not in S_set:
            return False
        
        # Temporarily consider S without idx_out, with idx_in added
        # For each point R in S\{idx_out}:
        #   - dist(R, idx_in) must not conflict with other distances of R (excluding dist to idx_out)
        #   - dist(idx_in, R) must be unique among all distances from idx_in
        
        dists_from_in = set()
        for idx_R in S_set:
            if idx_R == idx_out:
                continue
            
            d_in_R = int(dist2[idx_in, idx_R])
            
            # Check uniqueness from idx_in's perspective
            if d_in_R in dists_from_in:
                return False
            dists_from_in.add(d_in_R)
            
            # Check from idx_R's perspective:
            # Current distances of R include d_R_out = dist(R, idx_out)
            # New distances will replace d_R_out with d_in_R
            # So d_in_R must not be in D_dict[R] unless it equals d_R_out
            d_R_out = int(dist2[idx_R, idx_out])
            if d_in_R in D_dict[idx_R] and d_in_R != d_R_out:
                return False
        
        return True
    
    def perform_swap(idx_in, idx_out, S_set, D_dict):
        """Perform a swap: remove idx_out, add idx_in."""
        remove_point(idx_out, S_set, D_dict)
        add_point(idx_in, S_set, D_dict)
    
    # Phase 1: Multiple initializations
    best_S_set = set()
    best_D_dict = {}
    best_size = 0
    
    rng = random.Random(123)
    
    # Try various lattice initializations
    for scale in [2, 3, 4, 5, 6]:
        for offset in range(scale):
            S_set, D_dict = initialize_lattice(scale, offset, seed=42 + scale * 10 + offset)
            if len(S_set) > best_size:
                best_size = len(S_set)
                best_S_set = S_set.copy()
                best_D_dict = {k: v.copy() for k, v in D_dict.items()}
    
    # Try random greedy initializations
    for trial in range(10):
        order = list(range(num_points))
        rng.shuffle(order)
        S_set, D_dict = initialize_random(order, seed=100 + trial)
        if len(S_set) > best_size:
            best_size = len(S_set)
            best_S_set = S_set.copy()
            best_D_dict = {k: v.copy() for k, v in D_dict.items()}
    
    # Diagonal scan
    order_diag = sorted(range(num_points), key=lambda i: (points[i][0] + points[i][1], points[i][0]))
    S_set, D_dict = initialize_random(order_diag, seed=200)
    if len(S_set) > best_size:
        best_size = len(S_set)
        best_S_set = S_set.copy()
        best_D_dict = {k: v.copy() for k, v in D_dict.items()}
    
    # Phase 2: Simulated Annealing
    S_set = best_S_set.copy()
    D_dict = {k: v.copy() for k, v in best_D_dict.items()}
    current_size = len(S_set)
    best_size_sa = current_size
    best_S_set_sa = S_set.copy()
    best_D_dict_sa = {k: v.copy() for k, v in D_dict.items()}
    
    T = 50.0  # Initial temperature
    cooling_rate = 0.995
    min_T = 0.1
    
    rng_sa = random.Random(456)
    
    start_time = time.time()
    steps = 0
    
    while time.time() < deadline - 15:
        # Generate a random move
        move_type = rng_sa.random()
        
        if move_type < 0.5:
            # 1-for-1 swap
            if not S_set or not (num_points - len(S_set)):
                continue
            idx_out = rng_sa.choice(list(S_set))
            complement = [i for i in range(num_points) if i not in S_set]
            if not complement:
                continue
            idx_in = rng_sa.choice(complement)
            
            if is_valid_swap(idx_in, idx_out, S_set, D_dict):
                # Accept (always valid)
                perform_swap(idx_in, idx_out, S_set, D_dict)
                new_size = len(S_set)
                if new_size > best_size_sa:
                    best_size_sa = new_size
                    best_S_set_sa = S_set.copy()
                    best_D_dict_sa = {k: v.copy() for k, v in D_dict.items()}
            else:
                # Invalid swap - skip (we don't allow invalid states)
                pass
        
        elif move_type < 0.75:
            # Try to add a point (if we have room conceptually)
            complement = [i for i in range(num_points) if i not in S_set]
            if not complement:
                continue
            idx_add = rng_sa.choice(complement)
            
            if is_valid_addition(idx_add, S_set, D_dict):
                add_point(idx_add, S_set, D_dict)
                new_size = len(S_set)
                if new_size > best_size_sa:
                    best_size_sa = new_size
                    best_S_set_sa = S_set.copy()
                    best_D_dict_sa = {k: v.copy() for k, v in D_dict.items()}
        
        else:
            # Try to remove a point (to allow future swaps)
            if len(S_set) > 1:
                idx_remove = rng_sa.choice(list(S_set))
                remove_point(idx_remove, S_set, D_dict)
        
        # Cool down
        T *= cooling_rate
        if T < min_T:
            T = min_T
        
        steps += 1
        # Check time periodically
        if steps % 1000 == 0 and time.time() > deadline - 15:
            break
    
    # Use the best found
    if best_size_sa > best_size:
        best_S_set = best_S_set_sa
        best_D_dict = best_D_dict_sa
    
    # Convert to list of points
    return [points[i] for i in best_S_set]
# EVOLVE-BLOCK-END
