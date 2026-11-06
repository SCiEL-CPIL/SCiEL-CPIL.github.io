"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import random
import time
import sys

N = 64
NUM_POINTS = N * N

# EVOLVE-BLOCK-START

def construct():
    """
    Find a large set of grid points in [0, N)^2 with no three points forming an isosceles triangle.
    Uses a local search / min-conflicts approach with random restarts and a deadline guard.
    """
    deadline = time.time() + 280  # Leave margin before 300s limit
    
    # Pre-compute all points
    points = np.array([(x, y) for y in range(N) for x in range(N)], dtype=np.int32)
    xs = points[:, 0]
    ys = points[:, 1]
    
    # Pre-compute squared distances between all pairs
    # D[i, j] = squared distance between point i and point j
    # Use broadcasting for efficiency
    dx = xs[:, None] - xs[None, :]
    dy = ys[:, None] - ys[None, :]
    D2 = dx * dx + dy * dy  # (NUM_POINTS, NUM_POINTS)
    
    # For each pair (i, j), we need to check if there exists a k such that
    # dist(i,k) == dist(j,k) or dist(i,k) == dist(i,j) or dist(j,k) == dist(i,j)
    # Actually, three points i, j, k form an isosceles triangle if:
    #   dist(i,j) == dist(i,k)  (vertex at i)
    #   dist(i,j) == dist(j,k)  (vertex at j)
    #   dist(i,k) == dist(j,k)  (vertex at k)
    
    # For a set S to be valid, for every triple (i,j,k) in S, none of the above should hold.
    # Equivalently, for each pair (i,j) in S, there should be no k in S such that:
    #   D2[i,k] == D2[i,j]  (isosceles with vertex i)
    #   D2[j,k] == D2[i,j]  (isosceles with vertex j)
    #   D2[i,k] == D2[j,k]  (isosceles with vertex k)
    
    # Strategy: Use a greedy + local search approach.
    # Represent the set as a boolean array of size NUM_POINTS.
    
    def count_conflicts(selected):
        """Count the number of isosceles triangles in the selected set."""
        K = np.sum(selected)
        if K < 3:
            return 0
        
        # Get indices of selected points
        idx = np.where(selected)[0]
        
        # Compute pairwise squared distances among selected points
        # D2_sel[i, j] = D2[idx[i], idx[j]]
        D2_sel = D2[np.ix_(idx, idx)]
        
        # For each triple, check if isosceles
        # This is O(K^3) which is too slow for large K
        # Instead, for each pair (i,j), check if there's a k with:
        #   D2_sel[i,k] == D2_sel[i,j]  (k != i, j)
        #   D2_sel[j,k] == D2_sel[i,j]  (k != i, j)
        #   D2_sel[i,k] == D2_sel[j,k]  (k != i, j)
        
        conflict_count = 0
        K = len(idx)
        
        # For each pair (i, j) with i < j
        for i in range(K):
            for j in range(i + 1, K):
                dij2 = D2_sel[i, j]
                
                # Check if there exists k != i, j such that D2_sel[i,k] == dij2
                # i.e., k is at the same distance from i as j is
                # Exclude i and j themselves
                row_i = D2_sel[i, :].copy()
                row_i[i] = -1  # Exclude self
                row_i[j] = -1  # Exclude j
                if np.any(row_i == dij2):
                    conflict_count += 1
                
                # Check if there exists k != i, j such that D2_sel[j,k] == dij2
                row_j = D2_sel[j, :].copy()
                row_j[i] = -1
                row_j[j] = -1
                if np.any(row_j == dij2):
                    conflict_count += 1
                
                # Check if there exists k != i, j such that D2_sel[i,k] == D2_sel[j,k]
                # This means k is equidistant from i and j
                # D2_sel[i, k] == D2_sel[j, k] for some k != i, j
                diff = D2_sel[i, :] - D2_sel[j, :]
                diff[i] = -1
                diff[j] = -1
                if np.any(diff == 0):
                    conflict_count += 1
        
        return conflict_count
    
    def try_add_point(selected, p):
        """Check if adding point p to selected creates any isosceles triangle."""
        if selected[p]:
            return False  # Already in set
        
        idx = np.where(selected)[0]
        K = len(idx)
        
        if K == 0:
            return True
        if K == 1:
            # Two points never form an isosceles triangle
            return True
        
        # For the new point p and each pair (i, j) in selected:
        # Check if (i, j, p) forms an isosceles triangle
        for i_idx in range(K):
            i = idx[i_idx]
            for j_idx in range(i_idx + 1, K):
                j = idx[j_idx]
                
                dij2 = D2[i, j]
                dip2 = D2[i, p]
                djp2 = D2[j, p]
                
                # Isosceles if any two distances are equal
                if dij2 == dip2 or dij2 == djp2 or dip2 == djp2:
                    return False
        
        return True
    
    def remove_point(selected, p):
        """Remove point p from selected."""
        selected[p] = False
    
    def add_point(selected, p):
        """Add point p to selected."""
        selected[p] = True
    
    # Local search: start with a random subset, then try to add/remove/swap points
    best_selected = np.zeros(NUM_POINTS, dtype=bool)
    best_size = 0
    
    random.seed(42)
    
    # Multiple restarts
    for restart in range(200):
        if time.time() > deadline - 5:
            break
        
        # Initialize with a random sparse set
        selected = np.zeros(NUM_POINTS, dtype=bool)
        
        # Greedy initialization: try to add points in random order
        order = list(range(NUM_POINTS))
        random.shuffle(order)
        
        for p in order:
            if time.time() > deadline - 10:
                break
            if try_add_point(selected, p):
                add_point(selected, p)
        
        current_size = np.sum(selected)
        if current_size > best_size:
            best_selected = selected.copy()
            best_size = current_size
        
        # Local search: try to improve by adding, removing, or swapping
        improved = True
        max_iterations = 1000
        iteration = 0
        
        while improved and iteration < max_iterations:
            if time.time() > deadline - 10:
                break
            
            improved = False
            iteration += 1
            
            # Try to add a random point
            p = random.randint(0, NUM_POINTS - 1)
            if not selected[p]:
                if try_add_point(selected, p):
                    add_point(selected, p)
                    improved = True
                    current_size += 1
                    if current_size > best_size:
                        best_selected = selected.copy()
                        best_size = current_size
                    continue
            
            # Try to swap: remove a random selected point and add an unselected one
            if current_size > 0:
                selected_idx = np.where(selected)[0]
                if len(selected_idx) > 0:
                    remove_p = selected_idx[random.randint(0, len(selected_idx) - 1)]
                    # Try to find a point to add after removing remove_p
                    old_selected = selected.copy()
                    selected[remove_p] = False
                    current_size -= 1
                    
                    # Try to add a random unselected point
                    added = False
                    for _ in range(20):  # Try 20 random candidates
                        if time.time() > deadline - 10:
                            break
                        add_p = random.randint(0, NUM_POINTS - 1)
                        if not selected[add_p]:
                            if try_add_point(selected, add_p):
                                add_point(selected, add_p)
                                current_size += 1
                                added = True
                                if current_size > best_size:
                                    best_selected = selected.copy()
                                    best_size = current_size
                                improved = True
                                break
                    
                    if not added:
                        # Revert
                        selected[:] = old_selected
                        current_size += 1
        
        # If we found a better solution, continue from there
        if best_size > 0:
            # Optionally, continue local search from the best solution
            pass
    
    # Return the best solution found
    result_idx = np.where(best_selected)[0]
    result = [(int(points[i, 0]), int(points[i, 1])) for i in result_idx]
    
    return result

# EVOLVE-BLOCK-END
