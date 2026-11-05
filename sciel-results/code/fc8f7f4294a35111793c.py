"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 64
import time
import random
import math
import numpy as np

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 280  # Safety margin

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

    # For each point i, we need to check if adding i to set S is valid.
    # This requires:
    # 1. For all j, k in S, j != k, dist(i, j) != dist(i, k)  -> distances from i to S are all distinct
    # 2. For all j in S, dist(i, j) not in {dist(j, k) : k in S, k != i, k != j}
    #    This is equivalent to: for all j in S, dist(i, j) not in D[j] where D[j] is the set of distances from j to S\{j}
    #    BUT we also need to ensure that when we add i, the new distance dist(i, j) doesn't conflict with existing distances.
    #    
    #    Actually, the condition "no three points form an isosceles triangle" is equivalent to:
    #    For every point p in S, the set of distances {dist(p, q) : q in S, q != p} has no duplicates.
    #    
    #    So when adding i to S, we need:
    #    A. For every j in S: dist(i, j) is not equal to any dist(i, k) for k in S, k != j  -> all dist(i, j) for j in S are distinct
    #    B. For every j in S: dist(i, j) is not equal to any dist(j, k) for k in S, k != i  -> dist(i, j) not in {dist(j, k) : k in S, k != i}
    #       Since i is not in S yet, {dist(j, k) : k in S, k != i} = {dist(j, k) : k in S, k != j} = D[j]
    #    
    #    So both conditions must hold.

    # Precompute for each point i, a list of points at each distance from i
    # This helps with condition A: we can quickly check if two points in S are at the same distance from i
    # But for now, let's use the direct approach with sets.

    # State:
    # S_set: set of indices in S
    # D_dict: {idx: set of squared distances from idx to other points in S}
    # For each idx in S, D_dict[idx] = {dist(idx, j) : j in S, j != idx}

    def is_valid_addition(idx_P, S_indices, D_dict):
        """Check if point idx_P can be added to set S_indices."""
        if not S_indices:
            return True
        
        dists_from_P = set()
        for idx_Q in S_indices:
            d_val = int(dist2[idx_P, idx_Q])
            # Condition A: distances from P to all Q in S must be distinct
            if d_val in dists_from_P:
                return False
            dists_from_P.add(d_val)
            # Condition B: d_val must not be in D_dict[idx_Q]
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

    # Phase 1: Greedy with multiple orderings
    rng = random.Random(123)
    best_size = 0
    best_solution = []
    best_D = {}
    best_S_set = set()
    best_complement = list(range(num_points))

    # Generate several orderings
    orderings = []
    
    # Random orderings
    for _ in range(8):
        order = list(range(num_points))
        rng.shuffle(order)
        orderings.append(order)
    
    # Sorted by x
    orderings.append(sorted(range(num_points), key=lambda i: (points[i][0], points[i][1])))
    # Sorted by y
    orderings.append(sorted(range(num_points), key=lambda i: (points[i][1], points[i][0])))
    # Sorted by x+y
    orderings.append(sorted(range(num_points), key=lambda i: points[i][0] + points[i][1]))
    # Sorted by x-y
    orderings.append(sorted(range(num_points), key=lambda i: points[i][0] - points[i][1]))
    # Sorted by distance from center
    cx, cy = N/2 - 0.5, N/2 - 0.5
    orderings.append(sorted(range(num_points), key=lambda i: (points[i][0]-cx)**2 + (points[i][1]-cy)**2))
    # Sorted by distance from corner
    orderings.append(sorted(range(num_points), key=lambda i: points[i][0]**2 + points[i][1]**2))

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
        
        if len(S_indices) > best_size:
            best_size = len(S_indices)
            best_solution = S_indices[:]
            best_D = {k: v.copy() for k, v in D_dict.items()}
            best_S_set = S_set.copy()
            best_complement = complement[:]

    # Phase 2: Local search
    S_indices = best_solution[:]
    D_dict = {k: v.copy() for k, v in best_D.items()}
    S_set = best_S_set.copy()
    complement = best_complement[:]
    
    current_size = len(S_indices)
    best_size = current_size
    
    # Try to improve by swapping points
    no_improve = 0
    max_no_improve = 100
    temperature = 1.0
    cooling = 0.999
    
    while time.time() < deadline - 10:
        if not complement:
            break
            
        # Try a few random swaps
        improved = False
        for _ in range(20):
            if time.time() > deadline - 10:
                break
            if not complement or not S_indices:
                break
                
            idx_P = rng.choice(complement)
            idx_Q = rng.choice(S_indices)
            
            # Check if swapping Q for P is valid
            # First, check if P can replace Q
            # We need to check if P can be added to S\{Q}
            # Create a temporary copy of D_dict for S\{Q}
            temp_D = {k: v.copy() for k, v in D_dict.items()}
            # Remove Q from temp_D
            if idx_Q in temp_D:
                for idx_R in S_indices:
                    if idx_R != idx_Q:
                        d = int(dist2[idx_Q, idx_R])
                        if d in temp_D[idx_R]:
                            temp_D[idx_R].discard(d)
                del temp_D[idx_Q]
            
            # Check if P can be added to S\{Q}
            valid = True
            dists_from_P = set()
            for idx_R in S_indices:
                if idx_R == idx_Q:
                    continue
                d_val = int(dist2[idx_P, idx_R])
                if d_val in dists_from_P:
                    valid = False
                    break
                dists_from_P.add(d_val)
                if d_val in temp_D.get(idx_R, set()):
                    valid = False
                    break
            
            if valid:
                # Perform the swap
                # Remove Q
                if idx_Q in S_indices:
                    S_indices.remove(idx_Q)
                S_set.discard(idx_Q)
                complement.append(idx_Q)
                
                if idx_Q in D_dict:
                    for idx_R in S_indices:
                        d = int(dist2[idx_Q, idx_R])
                        if d in D_dict[idx_R]:
                            D_dict[idx_R].discard(d)
                    del D_dict[idx_Q]
                
                # Add P
                S_indices.append(idx_P)
                S_set.add(idx_P)
                complement.remove(idx_P)
                
                dists_from_P_new = set()
                for idx_R in S_indices:
                    if idx_R == idx_P:
                        continue
                    d = int(dist2[idx_P, idx_R])
                    dists_from_P_new.add(d)
                    D_dict[idx_R].add(d)
                D_dict[idx_P] = dists_from_P_new
                
                improved = True
                break
        
        if improved:
            no_improve = 0
            # Try to add more points after swap
            for _ in range(10):
                if time.time() > deadline - 10:
                    break
                if not complement:
                    break
                idx_P_add = rng.choice(complement)
                if is_valid_addition(idx_P_add, S_indices, D_dict):
                    add_point(idx_P_add, S_indices, D_dict, S_set, complement)
            
            if len(S_indices) > best_size:
                best_size = len(S_indices)
                best_solution = S_indices[:]
                best_D = {k: v.copy() for k, v in D_dict.items()}
                best_S_set = S_set.copy()
                best_complement = complement[:]
        else:
            no_improve += 1
            temperature *= cooling
        
        if no_improve > max_no_improve:
            # Restart from best
            S_indices = best_solution[:]
            D_dict = {k: v.copy() for k, v in best_D.items()}
            S_set = best_S_set.copy()
            complement = best_complement[:]
            no_improve = 0
            temperature = 1.0

    # Phase 3: Final polish
    S_indices = best_solution[:]
    D_dict = {k: v.copy() for k, v in best_D.items()}
    S_set = best_S_set.copy()
    complement = best_complement[:]
    
    # Try to add any remaining valid points
    for idx_P in complement[:]:
        if time.time() > deadline - 5:
            break
        if is_valid_addition(idx_P, S_indices, D_dict):
            add_point(idx_P, S_indices, D_dict, S_set, complement)
    
    if len(S_indices) > len(best_solution):
        best_solution = S_indices[:]

    return [points[i] for i in best_solution]
# EVOLVE-BLOCK-END
