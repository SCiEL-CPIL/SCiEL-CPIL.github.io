"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 64
import time
import random
import numpy as np

# EVOLVE-BLOCK-START
def construct():
    deadline = time.time() + 290  # Safety margin before 330s kill
    
    # Precompute all points and squared distances
    points = [(x, y) for y in range(N) for x in range(N)]
    num_points = N * N
    
    xs = np.array([p[0] for p in points], dtype=np.int32)
    ys = np.array([p[1] for p in points], dtype=np.int32)
    
    dx = xs[:, None] - xs[None, :]
    dy = ys[:, None] - ys[None, :]
    dist2 = (dx * dx + dy * dy).astype(np.int32)
    
    # Max squared distance is 2*(N-1)^2
    max_dist2 = int(dist2.max())
    
    # State:
    # S: set of indices
    # masks: list of Python ints, where bit d is set if distance d is used by point i
    S = set()
    masks = [0] * num_points
    
    def is_valid_addition(idx_P):
        """Check if idx_P can be added to S."""
        if idx_P in S:
            return False
            
        # Get indices of points in S
        S_list = list(S)
        if not S_list:
            return True
            
        # Vectorized distance calculation from P to all points in S
        # dist2[idx_P, S_list] gives an array of distances
        dists = dist2[idx_P, S_list]
        
        # 1. Check uniqueness of distances from P
        # If there are duplicates, isosceles triangle with apex P
        if len(np.unique(dists)) != len(dists):
            return False
            
        # 2. Check if any distance conflicts with existing masks
        # For each q in S, check if masks[q] has bit dist2[P, q] set
        # We can iterate through S_list and dists
        for i, q in enumerate(S_list):
            d = dists[i]
            if masks[q] & (1 << d):
                return False
                
        return True

    def add_point(idx_P):
        """Add idx_P to S and update masks."""
        S_list = list(S)
        dists = dist2[idx_P, S_list] if S_list else np.array([])
        
        # Create mask for P
        mask_P = 0
        for d in dists:
            mask_P |= (1 << d)
        masks[idx_P] = mask_P
        
        # Update masks for existing points
        for i, q in enumerate(S_list):
            d = dists[i]
            masks[q] |= (1 << d)
            
        S.add(idx_P)

    def remove_point(idx_P):
        """Remove idx_P from S and update masks."""
        if idx_P not in S:
            return
        S_list = [x for x in S if x != idx_P]
        dists = dist2[idx_P, S_list] if S_list else np.array([])
        
        # Clear mask for P
        masks[idx_P] = 0
        
        # Remove bits from existing points' masks
        for i, q in enumerate(S_list):
            d = dists[i]
            masks[q] &= ~(1 << d)
            
        S.remove(idx_P)

    def is_valid_swap(idx_in, idx_out):
        """Check if swapping idx_out for idx_in keeps the set valid."""
        if idx_in in S or idx_out not in S:
            return False
            
        # Temporarily remove idx_out to check validity of adding idx_in
        # We can simulate this without modifying state if we are careful, 
        # but modifying and reverting is simpler and fast enough.
        
        # 1. Remove idx_out
        remove_point(idx_out)
        
        # 2. Check if idx_in is valid
        valid = is_valid_addition(idx_in)
        
        # 3. Revert: Add idx_out back, remove idx_in if it was added
        if valid:
            add_point(idx_in)
            # We want to return True, but the state is now modified.
            # We should handle the swap atomically in the main loop or 
            # revert here and let the caller apply it.
            # Let's revert here to keep state consistent until 'perform_swap' is called.
            remove_point(idx_in)
            add_point(idx_out)
            return True
        else:
            add_point(idx_out)
            return False

    def perform_swap(idx_in, idx_out):
        """Perform the swap."""
        remove_point(idx_out)
        add_point(idx_in)

    # Phase 1: Greedy Initialization
    # Try a few different orders
    best_S = set()
    best_masks = [0] * num_points
    best_size = 0
    
    orders = []
    # Random orders
    rng_init = random.Random(123)
    for _ in range(5):
        order = list(range(num_points))
        rng_init.shuffle(order)
        orders.append(order)
        
    # Structured orders
    orders.append(sorted(range(num_points), key=lambda i: (points[i][0] + points[i][1], points[i][0])))
    orders.append(sorted(range(num_points), key=lambda i: (points[i][0] - points[i][1], points[i][0])))
    
    for order in orders:
        S = set()
        masks = [0] * num_points
        for idx in order:
            if is_valid_addition(idx):
                add_point(idx)
        
        if len(S) > best_size:
            best_size = len(S)
            best_S = S.copy()
            best_masks = masks[:]
            
    # Phase 2: Simulated Annealing / Local Search
    S = best_S.copy()
    masks = best_masks[:]
    current_size = len(S)
    best_size_sa = current_size
    best_S_sa = S.copy()
    best_masks_sa = masks[:]
    
    T = 50.0
    cooling_rate = 0.999
    min_T = 0.1
    
    rng_sa = random.Random(456)
    start_time = time.time()
    
    while time.time() < deadline - 10:
        move_type = rng_sa.random()
        
        if move_type < 0.4:
            # Try to add a point
            complement = [i for i in range(num_points) if i not in S]
            if complement:
                idx_add = rng_sa.choice(complement)
                if is_valid_addition(idx_add):
                    add_point(idx_add)
                    if len(S) > best_size_sa:
                        best_size_sa = len(S)
                        best_S_sa = S.copy()
                        best_masks_sa = masks[:]
                else:
                    # No change, just cool down
                    pass
                    
        elif move_type < 0.7:
            # Try to remove a point (to allow future swaps)
            if len(S) > 1:
                idx_remove = rng_sa.choice(list(S))
                remove_point(idx_remove)
                
        else:
            # Try a swap
            if len(S) > 0:
                idx_out = rng_sa.choice(list(S))
                complement = [i for i in range(num_points) if i not in S]
                if complement:
                    idx_in = rng_sa.choice(complement)
                    if is_valid_swap(idx_in, idx_out):
                        perform_swap(idx_in, idx_out)
                        if len(S) > best_size_sa:
                            best_size_sa = len(S)
                            best_S_sa = S.copy()
                            best_masks_sa = masks[:]

        # Cool down
        T *= cooling_rate
        if T < min_T:
            T = min_T

    # Return the best found
    if best_size_sa > best_size:
        S = best_S_sa
    else:
        S = best_S

    return [points[i] for i in S]
# EVOLVE-BLOCK-END
