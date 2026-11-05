"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 64
import time
import random
import math

# EVOLVE-BLOCK-START
def construct():
    deadline = time.time() + 290  # Safety margin
    
    # Grid points
    points = [(x, y) for y in range(N) for x in range(N)]
    num_points = N * N
    
    # Precompute indices and distances
    idx_map = {}
    for i, p in enumerate(points):
        idx_map[p] = i
        
    # Precompute squared distances
    # dist2[i][j] = (x_i - x_j)^2 + (y_i - y_j)^2
    # Using a list of lists might be slow/memory heavy if not careful, but 4096x4096 is ~16M entries.
    # 16M integers is ~128MB. Might be okay.
    # Let's use a flat list or numpy? NumPy is available.
    import numpy as np
    
    # Create coordinate arrays
    xs = np.array([p[0] for p in points])
    ys = np.array([p[1] for p in points])
    
    # Calculate squared distances
    # dist2[i, j] = (xs[i]-xs[j])^2 + (ys[i]-ys[j])^2
    # This is a 4096x4096 matrix.
    # Memory: 4096 * 4096 * 8 bytes = 134 MB.
    # Time: O(N^2) vectorized operations.
    
    dx = xs[:, None] - xs[None, :]
    dy = ys[:, None] - ys[None, :]
    dist2 = dx*dx + dy*dy
    
    def get_dist(i, j):
        return dist2[i, j]

    def is_valid_addition(idx_P, S_indices, D_dict):
        """
        Check if point idx_P can be added to set S_indices.
        S_indices: list of indices of points in S.
        D_dict: dict {idx: set of squared distances to other points in S}
        """
        if not S_indices:
            return True
        dists_from_P = set()
        for idx_Q in S_indices:
            d = dist2[idx_P, idx_Q]
            if d in dists_from_P:
                return False
            dists_from_P.add(d)
            if d in D_dict[idx_Q]:
                return False
        return True

    def is_valid_swap(idx_P, idx_Q, S_indices, D_dict):
        """
        Check if swapping idx_Q for idx_P keeps the set valid.
        """
        if idx_P in D_dict:
            return False
            
        dists_from_P = set()
        for idx_R in S_indices:
            if idx_R == idx_Q:
                continue
            d_PR = dist2[idx_P, idx_R]
            
            if d_PR in dists_from_P:
                return False
            dists_from_P.add(d_PR)
            
            if d_PR in D_dict[idx_R]:
                d_RQ = dist2[idx_R, idx_Q]
                if d_PR != d_RQ:
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
            d = dist2[idx_P, idx_Q]
            dists_from_P.add(d)
            D_dict[idx_Q].add(d)
        D_dict[idx_P] = dists_from_P

    def remove_point(idx_P, S_indices, D_dict, S_set, complement):
        S_indices.remove(idx_P)
        S_set.discard(idx_P)
        complement.append(idx_P)
        
        dists_from_P = D_dict[idx_P]
        for idx_Q in S_indices:
            d = dist2[idx_P, idx_Q]
            D_dict[idx_Q].discard(d)
        del D_dict[idx_P]

    def swap_point(idx_P, idx_Q, S_indices, D_dict, S_set, complement):
        remove_point(idx_Q, S_indices, D_dict, S_set, complement)
        add_point(idx_P, S_indices, D_dict, S_set, complement)

    # Main Search
    best_S_indices = []
    best_D = {}
    best_S_set = set()
    best_complement = list(range(num_points))
    
    rng = random.Random(42)
    
    # Phase 1: Multiple Greedy Initializations
    # Try different orderings
    
    orderings = []
    # Random
    orderings.append(list(range(num_points)))
    rng.shuffle(orderings[0])
    # Sorted by X
    orderings.append(sorted(range(num_points), key=lambda i: points[i][0]))
    # Sorted by Y
    orderings.append(sorted(range(num_points), key=lambda i: points[i][1]))
    # Sorted by X+Y
    orderings.append(sorted(range(num_points), key=lambda i: points[i][0] + points[i][1]))
    # Sorted by distance from center
    cx, cy = N/2, N/2
    orderings.append(sorted(range(num_points), key=lambda i: (points[i][0]-cx)**2 + (points[i][1]-cy)**2))
    
    for order in orderings:
        S_indices = []
        D_dict = {}
        S_set = set()
        complement = list(range(num_points))
        
        for idx_P in order:
            if idx_P in S_set:
                continue
            if is_valid_addition(idx_P, S_indices, D_dict):
                add_point(idx_P, S_indices, D_dict, S_set, complement)
                
        if len(S_indices) > len(best_S_indices):
            best_S_indices = S_indices[:]
            best_D = {k: v.copy() for k, v in D_dict.items()}
            best_S_set = S_set.copy()
            best_complement = complement[:]
            
    # Phase 2: Local Search
    S_indices = best_S_indices[:]
    D_dict = {k: v.copy() for k, v in best_D.items()}
    S_set = best_S_set.copy()
    complement = best_complement[:]
    
    no_improve_count = 0
    max_no_improve = 500
    
    while time.time() < deadline - 10:
        if not complement:
            break
            
        # Action: Add or Swap
        # Bias towards Add if possible
        action = 'add' if (len(S_indices) < 20 or rng.random() < 0.3) else 'swap'
        
        if action == 'add':
            # Try to find a valid add
            # Sample a few candidates
            found = False
            for _ in range(10):
                if not complement: break
                idx_P = rng.choice(complement)
                if is_valid_addition(idx_P, S_indices, D_dict):
                    add_point(idx_P, S_indices, D_dict, S_set, complement)
                    found = True
                    if len(S_indices) > len(best_S_indices):
                        best_S_indices = S_indices[:]
                        best_D = {k: v.copy() for k, v in D_dict.items()}
                        best_S_set = S_set.copy()
                        best_complement = complement[:]
                        no_improve_count = 0
                    else:
                        no_improve_count += 1
                    break
            if not found:
                no_improve_count += 1
        else:
            # Swap
            # Try to find a valid swap
            found = False
            for _ in range(10):
                if not complement or not S_indices: break
                idx_P = rng.choice(complement)
                idx_Q = rng.choice(S_indices)
                if is_valid_swap(idx_P, idx_Q, S_indices, D_dict):
                    swap_point(idx_P, idx_Q, S_indices, D_dict, S_set, complement)
                    found = True
                    # Swap doesn't increase size, but might enable future adds
                    # Check if we can add now
                    if rng.random() < 0.5 and complement:
                         idx_P_add = rng.choice(complement)
                         if is_valid_addition(idx_P_add, S_indices, D_dict):
                             add_point(idx_P_add, S_indices, D_dict, S_set, complement)
                             if len(S_indices) > len(best_S_indices):
                                 best_S_indices = S_indices[:]
                                 best_D = {k: v.copy() for k, v in D_dict.items()}
                                 best_S_set = S_set.copy()
                                 best_complement = complement[:]
                                 no_improve_count = 0
                    break
            if not found:
                no_improve_count += 1
                
        if no_improve_count > max_no_improve:
            # Perturb
            S_indices = best_S_indices[:]
            D_dict = {k: v.copy() for k, v in best_D.items()}
            S_set = best_S_set.copy()
            complement = best_complement[:]
            no_improve_count = 0
            # Remove 2 random points
            if len(S_indices) > 2:
                idx1 = rng.randrange(len(S_indices))
                idx2 = rng.randrange(len(S_indices))
                if idx1 == idx2:
                    idx2 = (idx2 + 1) % len(S_indices)
                P1 = S_indices[idx1]
                P2 = S_indices[idx2]
                remove_point(P1, S_indices, D_dict, S_set, complement)
                remove_point(P2, S_indices, D_dict, S_set, complement)
                
    # Convert indices back to points
    return [points[i] for i in best_S_indices]
# EVOLVE-BLOCK-END
