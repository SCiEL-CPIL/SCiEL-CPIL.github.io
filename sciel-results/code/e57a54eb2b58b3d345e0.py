"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import math
import numpy as np

N = 64

# EVOLVE-BLOCK-START

def _sq_dist(p1, p2):
    dx = p1[0] - p2[0]
    dy = p1[1] - p2[1]
    return dx * dx + dy * dy

def _check_valid_fast(pos, points_arr, dist_sets):
    """
    Fast validity check using NumPy.
    points_arr: (k, 2) int array of current points
    dist_sets: list of sets, dist_sets[i] contains squared distances from point i to all other points
    Returns (valid, dists_list) where dists_list[i] = squared dist from pos to points_arr[i]
    """
    k = len(points_arr)
    if k == 0:
        return True, []
    
    # Vectorized distance computation
    dx = pos[0] - points_arr[:, 0]
    dy = pos[1] - points_arr[:, 1]
    dists_list = (dx * dx + dy * dy).tolist()
    
    # Check if pos is apex: all distances from pos must be distinct
    if len(set(dists_list)) != k:
        return False, dists_list
    
    # Check if any existing point is apex: for each i, dists_list[i] must not be in dist_sets[i]
    for i, d2 in enumerate(dists_list):
        if d2 in dist_sets[i]:
            return False, dists_list
    
    return True, dists_list

def _greedy_construct_fast(timeout):
    """
    Fast greedy construction using vectorized validity check with smart ordering.
    """
    start_time = time.time()
    best_points = []
    
    # Pre-generate all grid points as list and array
    all_positions = [(x, y) for x in range(N) for y in range(N)]
    
    # Run multiple random greedy attempts
    while time.time() - start_time < timeout:
        current_points = []
        current_points_arr = np.empty((0, 2), dtype=np.int32)
        current_dist_sets = []
        
        # Use different ordering strategies for diversity
        strategy = random.random()
        if strategy < 0.3:
            shuffled = all_positions[:]
            random.shuffle(shuffled)
        elif strategy < 0.5:
            # Spiral from center
            cx, cy = N // 2, N // 2
            shuffled = sorted(all_positions, key=lambda p: (p[0]-cx)**2 + (p[1]-cy)**2)
        elif strategy < 0.7:
            # Row-major with random row offset
            shuffled = []
            row_offset = random.randint(0, N-1)
            for i in range(N):
                y = (row_offset + i) % N
                row = [(x, y) for x in range(N)]
                if random.random() < 0.5:
                    row.reverse()
                shuffled.extend(row)
        else:
            # Random with slight bias toward corners
            shuffled = all_positions[:]
            random.shuffle(shuffled)
            corners = [(0,0), (N-1,0), (0,N-1), (N-1,N-1)]
            shuffled = [c for c in shuffled if c in corners] + [p for p in shuffled if p not in corners]
        
        for pos in shuffled:
            if time.time() - start_time > timeout:
                break
            
            valid, dists_list = _check_valid_fast(pos, current_points_arr, current_dist_sets)
            
            if valid:
                current_points.append(pos)
                # Update distance sets
                new_dists = set(dists_list)
                for i, d2 in enumerate(dists_list):
                    current_dist_sets[i].add(d2)
                current_dist_sets.append(new_dists)
                # Update array
                current_points_arr = np.vstack([current_points_arr, np.array(pos, dtype=np.int32)])
                
                if len(current_points) > len(best_points):
                    best_points = current_points[:]

    return best_points

def _local_search_improve(points, timeout):
    """
    Try to add more points to an existing set via systematic search with vectorized checks.
    """
    start_time = time.time()
    current_points = list(points)
    current_points_arr = np.array(current_points, dtype=np.int32).reshape(-1, 2) if current_points else np.empty((0, 2), dtype=np.int32)
    
    # Build distance sets from existing points
    current_dist_sets = []
    for i, p in enumerate(current_points):
        dists = set()
        for j, q in enumerate(current_points):
            if i != j:
                dists.add(_sq_dist(p, q))
        current_dist_sets.append(dists)
    
    all_positions = [(x, y) for x in range(N) for y in range(N)]
    existing_set = set(current_points)
    candidates = [p for p in all_positions if p not in existing_set]
    random.shuffle(candidates)
    
    improved = True
    while improved and time.time() - start_time < timeout:
        improved = False
        for pos in candidates[:]:
            if time.time() - start_time > timeout:
                break
            
            valid, dists_list = _check_valid_fast(pos, current_points_arr, current_dist_sets)
            
            if valid:
                current_points.append(pos)
                new_dists = set(dists_list)
                for i, d2 in enumerate(dists_list):
                    current_dist_sets[i].add(d2)
                current_dist_sets.append(new_dists)
                current_points_arr = np.vstack([current_points_arr, np.array(pos, dtype=np.int32)])
                candidates.remove(pos)
                improved = True
    
    return current_points

def _sidon_line():
    """Construct points on a line using Sidon-like positions."""
    xs, used = [], set()
    x = 0
    while x < N:
        diffs = {abs(x - y) for y in xs}
        if not (diffs & used):
            used |= diffs
            xs.append(x)
        x += 1
    return [(x, 0) for x in xs]

def _two_row_construction():
    """
    Construct points on two rows using a Sidon-like approach.
    Place points on rows y=0 and y=h such that no three form isosceles triangle.
    """
    h = N // 2
    # Greedily add points to row 0
    row0 = []
    used_diffs_0 = set()
    for x in range(N):
        diffs = {abs(x - y) for y in row0}
        if not (diffs & used_diffs_0):
            used_diffs_0 |= diffs
            row0.append(x)
    
    # Greedily add points to row h
    row1 = []
    all_points = [(x, 0) for x in row0]
    used_diffs_1 = set()
    for x in range(N):
        # Check distances to all existing points
        valid = True
        dists_from_new = set()
        for px, py in all_points:
            d2 = (x - px)**2 + (h - py)**2
            if d2 in dists_from_new:
                valid = False
                break
            dists_from_new.add(d2)
        # Check if any existing point is apex
        if valid:
            for i, (px, py) in enumerate(all_points):
                d2 = (x - px)**2 + (h - py)**2
                # Check if this distance already exists from point i
                for j, (qx, qy) in enumerate(all_points):
                    if i == j:
                        continue
                    d2_ij = (px - qx)**2 + (py - qy)**2
                    if d2 == d2_ij:
                        valid = False
                        break
                if not valid:
                    break
        
        if valid:
            row1.append(x)
            all_points.append((x, h))
    
    return all_points

def construct():
    start_time = time.time()
    deadline = start_time + 290  # Leave safety margin
    
    # Start with deterministic constructions
    candidates = []
    
    # Sidon line
    sidon_pts = _sidon_line()
    candidates.append(sidon_pts)
    
    # Two-row construction
    two_row_pts = _two_row_construction()
    candidates.append(two_row_pts)
    
    best_points = max(candidates, key=len)
    
    # Try fast greedy randomized search
    time_remaining = deadline - time.time()
    if time_remaining > 10:
        # Split time: 70% for greedy, 30% for local search polish
        greedy_time = (time_remaining - 2) * 0.7
        polish_time = (time_remaining - 2) * 0.3
        
        greedy_pts = _greedy_construct_fast(greedy_time)
        if len(greedy_pts) > len(best_points):
            best_points = greedy_pts
        
        # Local search polish on the best solution
        if len(best_points) > 0 and deadline - time.time() > 1:
            remaining = deadline - time.time()
            polished = _local_search_improve(best_points, min(polish_time, remaining))
            if len(polished) > len(best_points):
                best_points = polished
    
    return best_points

# EVOLVE-BLOCK-END
