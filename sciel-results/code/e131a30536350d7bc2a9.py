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
    
    # Create coordinate arrays
    xs = np.array([p[0] for p in points], dtype=np.int32)
    ys = np.array([p[1] for p in points], dtype=np.int32)
    
    # Calculate squared distances using int32 to save memory and speed up
    dx = xs[:, None] - xs[None, :]
    dy = ys[:, None] - ys[None, :]
    dist2 = (dx*dx + dy*dy).astype(np.int32)
    
    # Precompute compatibility: two points i,j are compatible if no third point k
    # forms an isosceles triangle with i and j.
    # For a set S to be valid, for every pair (i,j) in S, there must be no k in S
    # such that dist(i,k) == dist(j,k) or dist(i,k) == dist(i,j) or dist(j,k) == dist(i,j).
    #
    # Actually, the constraint is: no three points form an isosceles triangle.
    # Points i, j, k form an isosceles triangle if:
    #   dist(i,j) == dist(i,k)  (isosceles with apex at i)
    #   dist(i,j) == dist(j,k)  (isosceles with apex at j)  
    #   dist(i,k) == dist(j,k)  (isosceles with apex at k)
    #
    # So for a set S, we need: for all distinct i,j,k in S, none of the above hold.
    #
    # This means: for any point i in S, all distances from i to other points in S must be distinct.
    # AND: for any two points i,j in S, the distance d = dist(i,j) must not equal
    #       dist(i,k) for any k in S\{j}, and must not equal dist(j,k) for any k in S\{i}.
    #
    # The current program's check is: when adding point P to set S, for each Q in S,
    # dist(P,Q) must be new (not seen before from P) AND not in D_dict[Q].
    # This is equivalent to the full isosceles constraint.
    #
    # For the bitset approach, I'll precompute for each pair (i,j) whether they can coexist
    # in a valid set. Two points i,j CAN coexist if there's no k such that {i,j,k} forms
    # an isosceles triangle. But that's not sufficient - we need the whole set to be valid.
    #
    # Actually, the constraint is NOT pairwise. Three points form an isosceles triangle
    # based on all three distances. So we can't just check pairs independently.
    #
    # However, we CAN reformulate: a set S is valid iff for every point i in S,
    # the set of distances {dist(i,j) : j in S, j != i} has no duplicates, AND
    # for every pair i,j in S, dist(i,j) is not equal to any other distance from i
    # or from j to a third point in S.
    #
    # The second condition is automatically satisfied if the first holds for all points.
    # Wait, no. If dist(i,j) = dist(i,k), that's caught by i's distance set having duplicates.
    # If dist(i,j) = dist(j,k), that's caught by j's distance set having duplicates.
    # If dist(i,k) = dist(j,k), that's caught by k's distance set having duplicates.
    #
    # So the constraint IS: for every point i in S, all distances from i to other points
    # in S are distinct!
    #
    # This is a pairwise constraint in disguise: for each i, no two other points in S
    # can be at the same distance from i.
    #
    # So we can define a conflict graph: edge (j,k) exists for a fixed i if dist(i,j) == dist(i,k).
    # But this depends on i, so it's not a simple pairwise conflict.
    #
    # Let me reconsider. The constraint "for each i in S, all dist(i,j) for j in S\{i} are distinct"
    # means: for each i, and for each pair j,k in S\{i}, j != k, we need dist(i,j) != dist(i,k).
    #
    # This is a 3-way constraint, not pairwise. The bitset approach needs to handle this.
    #
    # Alternative: For each point i, precompute which other points are at the same distance.
    # For point i, group other points by distance. Within each group, at most one point can be in S (along with i).
    #
    # This is still complex for a simple bitset greedy. Let me keep the current approach
    # but optimize the inner loops using numpy for the distance checks.
    
    # Precompute distance groups for each point
    # For point i, dist_groups[i][d] = list of points at distance d from i
    # This allows fast checking: if we want to add P to S, for each Q in S,
    # check if dist(P,Q) is already used by P (i.e., another R in S has dist(P,R) == dist(P,Q))
    # and check if dist(P,Q) is in D_dict[Q].
    
    # Actually, let me optimize the existing approach by using numpy for batch operations.
    
    # Precompute: for each point i, a dict mapping distance -> list of points at that distance
    dist_groups = [None] * num_points
    for i in range(num_points):
        d = dist2[i, :].copy()
        d[i] = -1  # exclude self
        # Group by distance
        groups = {}
        for j in range(num_points):
            if j == i:
                continue
            dval = int(d[j])
            if dval not in groups:
                groups[dval] = []
            groups[dval].append(j)
        dist_groups[i] = groups
    
    # State variables
    # S_set: set of indices currently in the solution
    # D_dict: {idx: set of squared distances from idx to other points in S}
    # complement: list of indices NOT in S
    
    def is_valid_addition_fast(idx_P, S_indices, D_dict):
        """
        Check if point idx_P can be added to set S_indices.
        Returns True if valid, False otherwise.
        """
        if not S_indices:
            return True
        
        dists_from_P = set()
        for idx_Q in S_indices:
            d_val = int(dist2[idx_P, idx_Q])
            if d_val in dists_from_P:
                return False
            dists_from_P.add(d_val)
            if d_val in D_dict[idx_Q]:
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
            d_PR = int(dist2[idx_P, idx_R])
            
            if d_PR in dists_from_P:
                return False
            dists_from_P.add(d_PR)
            
            if d_PR in D_dict[idx_R]:
                d_RQ = int(dist2[idx_R, idx_Q])
                if d_PR != d_RQ:
                    return False
        return True

    def add_point(idx_P, S_indices, D_dict, S_set, complement):
        S_indices.append(idx_P)
        S_set.add(idx_P)
        if idx_P in complement:
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
        if idx_P in S_indices:
            S_indices.remove(idx_P)
        S_set.discard(idx_P)
        complement.append(idx_P)
        
        if idx_P in D_dict:
            for idx_Q in S_indices:
                d = int(dist2[idx_P, idx_Q])
                if d in D_dict[idx_Q]:
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
    orderings = []
    orderings.append(list(range(num_points)))
    rng.shuffle(orderings[0])
    orderings.append(sorted(range(num_points), key=lambda i: points[i][0]))
    orderings.append(sorted(range(num_points), key=lambda i: points[i][1]))
    orderings.append(sorted(range(num_points), key=lambda i: points[i][0] + points[i][1]))
    cx, cy = N/2, N/2
    orderings.append(sorted(range(num_points), key=lambda i: (points[i][0]-cx)**2 + (points[i][1]-cy)**2))
    
    # Add a few more random orderings
    for _ in range(5):
        orderings.append(list(range(num_points)))
        rng.shuffle(orderings[-1])
        
    for order in orderings:
        S_indices = []
        D_dict = {}
        S_set = set()
        complement = list(range(num_points))
        
        for idx_P in order:
            if idx_P in S_set:
                continue
            if is_valid_addition_fast(idx_P, S_indices, D_dict):
                add_point(idx_P, S_indices, D_dict, S_set, complement)
                
        if len(S_indices) > len(best_S_indices):
            best_S_indices = S_indices[:]
            best_D = {k: v.copy() for k, v in D_dict.items()}
            best_S_set = S_set.copy()
            best_complement = complement[:]
            
    # Phase 2: Local Search with Simulated Annealing
    S_indices = best_S_indices[:]
    D_dict = {k: v.copy() for k, v in best_D.items()}
    S_set = best_S_set.copy()
    complement = best_complement[:]
    
    no_improve_count = 0
    max_no_improve = 500
    
    # Annealing parameters
    current_temp = 1.0
    cooling_rate = 0.999
    min_temp = 0.01
    
    # Track recent history for action selection
    add_attempts = 0
    swap_attempts = 0
    
    while time.time() < deadline - 15:
        if not complement:
            break
            
        # Dynamically choose action
        # Bias towards add if set is small, otherwise mix
        if len(S_indices) < 10:
            action = 'add'
        else:
            # Use history to balance
            if add_attempts > swap_attempts * 2:
                action = 'swap'
            elif swap_attempts > add_attempts * 2:
                action = 'add'
            else:
                action = 'add' if rng.random() < 0.5 else 'swap'
        
        if action == 'add':
            add_attempts += 1
            swap_attempts = max(0, swap_attempts - 1)
            found = False
            # Try multiple candidates
            for _ in range(15):
                if not complement: break
                idx_P = rng.choice(complement)
                if is_valid_addition_fast(idx_P, S_indices, D_dict):
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
            swap_attempts += 1
            add_attempts = max(0, add_attempts - 1)
            found = False
            # Try to find a valid swap, possibly accepting worse ones
            for _ in range(15):
                if not complement or not S_indices: break
                idx_P = rng.choice(complement)
                idx_Q = rng.choice(S_indices)
                
                if is_valid_swap(idx_P, idx_Q, S_indices, D_dict):
                    # Perform swap
                    swap_point(idx_P, idx_Q, S_indices, D_dict, S_set, complement)
                    found = True
                    
                    # After swap, try to add a point to grow the set
                    # Greedy fill: try a few random adds
                    for _ in range(5):
                        if not complement: break
                        idx_P_add = rng.choice(complement)
                        if is_valid_addition_fast(idx_P_add, S_indices, D_dict):
                            add_point(idx_P_add, S_indices, D_dict, S_set, complement)
                            if len(S_indices) > len(best_S_indices):
                                best_S_indices = S_indices[:]
                                best_D = {k: v.copy() for k, v in D_dict.items()}
                                best_S_set = S_set.copy()
                                best_complement = complement[:]
                                no_improve_count = 0
                            break
                    break
            
            if not found:
                no_improve_count += 1
        
        # Cooling
        current_temp *= cooling_rate
        
        if no_improve_count > max_no_improve:
            # Perturb: restart from best and remove some points
            S_indices = best_S_indices[:]
            D_dict = {k: v.copy() for k, v in best_D.items()}
            S_set = best_S_set.copy()
            complement = best_complement[:]
            no_improve_count = 0
            add_attempts = 0
            swap_attempts = 0
            current_temp = 1.0 # Reset temperature
            
            # Remove 1-3 random points to create space
            num_remove = rng.randint(1, 3)
            if len(S_indices) > num_remove:
                to_remove = rng.sample(S_indices, num_remove)
                for P in to_remove:
                    remove_point(P, S_indices, D_dict, S_set, complement)
                
    # Phase 3: Final Polish - Try to add more points to the best solution
    S_indices = best_S_indices[:]
    D_dict = {k: v.copy() for k, v in best_D.items()}
    S_set = best_S_set.copy()
    complement = best_complement[:]
    
    # Try all remaining points to see if any can be added
    for idx_P in complement[:]:
        if time.time() > deadline - 5:
            break
        if is_valid_addition_fast(idx_P, S_indices, D_dict):
            add_point(idx_P, S_indices, D_dict, S_set, complement)
    
    if len(S_indices) > len(best_S_indices):
        best_S_indices = S_indices[:]
    
    # Convert indices back to points
    return [points[i] for i in best_S_indices]
# EVOLVE-BLOCK-END
