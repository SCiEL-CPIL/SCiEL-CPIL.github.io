"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 64
import time
import random
import math
import numpy as np

# EVOLVE-BLOCK-START
def construct():
    deadline = time.time() + 285  # Safety margin
    
    # Grid points
    points = [(x, y) for y in range(N) for x in range(N)]
    num_points = N * N
    
    # Precompute indices and distances using NumPy for speed
    xs = np.array([p[0] for p in points])
    ys = np.array([p[1] for p in points])
    
    # Calculate squared distances matrix
    # dist2[i, j] = (xs[i]-xs[j])^2 + (ys[i]-ys[j])^2
    dx = xs[:, None] - xs[None, :]
    dy = ys[:, None] - ys[None, :]
    dist2 = dx*dx + dy*dy
    
    # Helper to check validity of adding point P to set S
    # S_indices: list of indices
    # D_dict: dict {idx: set of squared distances to other points in S}
    def is_valid_addition(idx_P, S_indices, D_dict):
        if not S_indices:
            return True
        
        dists_from_P = set()
        for idx_Q in S_indices:
            d = dist2[idx_P, idx_Q]
            # Check if distance from P to Q is already in distances from P to others
            if d in dists_from_P:
                return False
            dists_from_P.add(d)
            
            # Check if distance from P to Q is already in distances from Q to others
            if d in D_dict[idx_Q]:
                return False
        
        return True

    # Helper to check validity of swapping Q for P in set S
    # S' = S \ {Q} U {P}
    def is_valid_swap(idx_P, idx_Q, S_indices, D_dict):
        if idx_P in D_dict:
            return False
            
        dists_from_P = set()
        for idx_R in S_indices:
            if idx_R == idx_Q:
                continue
            d_PR = dist2[idx_P, idx_R]
            
            # Check uniqueness among P to others in S'
            if d_PR in dists_from_P:
                return False
            dists_from_P.add(d_PR)
            
            # Check if d_PR is in D'_R
            # D'_R contains d_PR if d_PR in D_R AND d_PR != dist(R, Q)
            if d_PR in D_dict[idx_R]:
                d_RQ = dist2[idx_R, idx_Q]
                if d_PR != d_RQ:
                    return False
            # If d_PR == d_RQ, then d_PR was removed from D_R to form D'_R, so it's valid.
            # If d_PR not in D_R, it's not in D'_R, so it's valid.
            
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
        # Find index of idx_P in S_indices for removal
        try:
            pos = S_indices.index(idx_P)
            S_indices.pop(pos)
        except ValueError:
            return # Not found, should not happen
        
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
    # Try different orderings to escape different local optima
    
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
    # Another random
    orderings.append(list(range(num_points)))
    rng.shuffle(orderings[5])
    
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
    # Use the best found so far
    S_indices = best_S_indices[:]
    D_dict = {k: v.copy() for k, v in best_D.items()}
    S_set = best_S_set.copy()
    complement = best_complement[:]
    
    no_improve_count = 0
    max_no_improve = 100  # Lower threshold to trigger perturbations more often
    
    while time.time() < deadline - 5:
        if not complement:
            break
            
        # Action: Add or Swap
        # Bias towards Add if possible, but swap is crucial for breaking plateaus
        if len(S_indices) < 25:
            action = 'add' if rng.random() < 0.8 else 'swap'
        else:
            action = 'add' if rng.random() < 0.3 else 'swap'
        
        if action == 'add':
            # Try to find a valid add
            found = False
            # Try a few random candidates
            for _ in range(5):
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
            found = False
            for _ in range(5):
                if not complement or not S_indices: break
                idx_P = rng.choice(complement)
                idx_Q = rng.choice(S_indices)
                if is_valid_swap(idx_P, idx_Q, S_indices, D_dict):
                    swap_point(idx_P, idx_Q, S_indices, D_dict, S_set, complement)
                    found = True
                    # After a swap, try to add a point immediately
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
                             else:
                                 no_improve_count += 1
                    else:
                         no_improve_count += 1
                    break
            if not found:
                no_improve_count += 1
                
        if no_improve_count > max_no_improve:
            # Perturb: revert to best and remove a few random points to open space
            S_indices = best_S_indices[:]
            D_dict = {k: v.copy() for k, v in best_D.items()}
            S_set = best_S_set.copy()
            complement = best_complement[:]
            no_improve_count = 0
            
            # Remove 2-3 random points
            num_remove = 2 + (1 if rng.random() < 0.5 else 0)
            if len(S_indices) > num_remove:
                indices_to_remove = rng.sample(S_indices, num_remove)
                for idx in indices_to_remove:
                    remove_point(idx, S_indices, D_dict, S_set, complement)
                    
    # Convert indices back to points
    return [points[i] for i in best_S_indices]
# EVOLVE-BLOCK-END
