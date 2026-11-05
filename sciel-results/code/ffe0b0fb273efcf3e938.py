import math
import time
import itertools

def construct():
    """
    Returns a list of 32 distinct integer pairs [a, b] representing points on the triangular lattice
    that maximize the number of unit-distance pairs (edges).
    The optimal number of edges is 76.
    """
    # Deadline guard
    deadline = time.time() + 135
    
    def dist_sq(p, q):
        da = p[0] - q[0]
        db = p[1] - q[1]
        return da*da + db*db + da*db

    def is_adjacent(p, q):
        return dist_sq(p, q) == 1

    # 1. Generate Hexagon of radius 2 (Core)
    # Condition: max(|a|, |b|, |a+b|) <= 2
    core_points = []
    for a in range(-2, 3):
        for b in range(-2, 3):
            if max(abs(a), abs(b), abs(a+b)) <= 2:
                core_points.append([a, b])
    
    # 2. Generate Shell of radius 3 (Boundary candidates)
    # Condition: max(|a|, |b|, |a+b|) == 3
    boundary_points = []
    for a in range(-3, 4):
        for b in range(-3, 4):
            if max(abs(a), abs(b), abs(a+b)) == 3:
                boundary_points.append([a, b])
                
    n_core = len(core_points) # 19
    n_boundary = len(boundary_points) # 18
    target_count = 32
    num_to_pick = target_count - n_core # 13

    if len(core_points) != 19 or len(boundary_points) != 18:
        # Fallback if generation logic fails (should not happen)
        return [[a, b] for a in range(16) for b in range(2)]

    # 3. Pre-compute edges
    # Edges within core
    core_edges = 0
    for i in range(n_core):
        for j in range(i+1, n_core):
            if is_adjacent(core_points[i], core_points[j]):
                core_edges += 1

    # Edges between core and each boundary point
    b_core_edges = [0] * n_boundary
    for i, bp in enumerate(boundary_points):
        for cp in core_points:
            if is_adjacent(bp, cp):
                b_core_edges[i] += 1

    # Edges between boundary points
    b_b_adj = [[0]*n_boundary for _ in range(n_boundary)]
    for i in range(n_boundary):
        for j in range(i+1, n_boundary):
            if is_adjacent(boundary_points[i], boundary_points[j]):
                b_b_adj[i][j] = 1
                b_b_adj[j][i] = 1

    best_edges = -1
    best_combo = None
    
    # 4. Brute force the best subset of 13 boundary points
    # C(18, 13) = 8568 combinations. Very fast.
    # Optimization: Pre-compute the sum of b_core_edges for any subset?
    # Not necessary for 8568 combos.
    
    # To speed up, we can iterate and break early if possible? No.
    # Just standard loop.
    for combo in itertools.combinations(range(n_boundary), num_to_pick):
        if time.time() > deadline - 5:
            break
            
        edges = core_edges
        # Add edges from selected boundary points to core
        for i in combo:
            edges += b_core_edges[i]
            
        # Add edges between selected boundary points
        # b_b_adj is a list of lists.
        for idx in range(len(combo)):
            i = combo[idx]
            # Access row i of adjacency matrix
            row = b_b_adj[i]
            for jdx in range(idx+1, len(combo)):
                j = combo[jdx]
                edges += row[j]
                
        if edges > best_edges:
            best_edges = edges
            best_combo = combo
            
            # Early termination if we hit the theoretical maximum
            if best_edges == 76:
                break

    if best_combo is None:
        # Fallback: just take the first 13
        best_combo = tuple(range(num_to_pick))
        # Recalculate edges for fallback
        best_edges = core_edges
        for i in best_combo:
            best_edges += b_core_edges[i]
        for idx in range(len(best_combo)):
            i = best_combo[idx]
            for jdx in range(idx+1, len(best_combo)):
                j = best_combo[jdx]
                best_edges += b_b_adj[i][j]

    final_points = core_points + [boundary_points[i] for i in best_combo]
    
    # 5. Verification and Local Search Safety Net
    def count_edges(points):
        count = 0
        n = len(points)
        for i in range(n):
            for j in range(i+1, n):
                if is_adjacent(points[i], points[j]):
                    count += 1
        return count

    current_score = count_edges(final_points)
    
    # If we didn't reach 76, run a limited local search to try to improve.
    if current_score < 76:
        # Generate a larger pool for swaps: Hexagon radius 4
        R = 4
        pool = []
        for a in range(-R, R+1):
            for b in range(-R, R+1):
                if max(abs(a), abs(b), abs(a+b)) <= R:
                    pool.append((a, b))
        
        current_set = set(map(tuple, final_points))
        points_list = [list(p) for p in current_set]
        
        improved = True
        while improved and current_score < 76 and time.time() < deadline - 10:
            improved = False
            best_swap_score = current_score
            best_swap_in_idx = -1
            best_swap_out_pt = None
            
            # Precompute degrees for all points in the set
            degrees = [0] * len(points_list)
            for i in range(len(points_list)):
                for j in range(i+1, len(points_list)):
                    if is_adjacent(points_list[i], points_list[j]):
                        degrees[i] += 1
                        degrees[j] += 1
            
            # Try swapping each point in the set with each point in the pool
            for i in range(len(points_list)):
                deg_in = degrees[i]
                p_in = points_list[i]
                
                for p_out in pool:
                    if tuple(p_out) in current_set:
                        continue
                    
                    # Calculate degree of p_out to the rest of the set (excluding p_in)
                    deg_out = 0
                    for j in range(len(points_list)):
                        if i != j and is_adjacent(p_out, points_list[j]):
                            deg_out += 1
                    
                    new_score = current_score - deg_in + deg_out
                    
                    if new_score > best_swap_score:
                        best_swap_score = new_score
                        best_swap_in_idx = i
                        best_swap_out_pt = p_out
                        
            if best_swap_in_idx != -1:
                # Perform swap
                old_pt = points_list[best_swap_in_idx]
                current_set.remove(tuple(old_pt))
                current_set.add(tuple(best_swap_out_pt))
                points_list[best_swap_in_idx] = list(best_swap_out_pt)
                current_score = best_swap_score
                improved = True
        
        if current_score > best_edges:
            best_edges = current_score
            final_points = points_list

    return final_points
