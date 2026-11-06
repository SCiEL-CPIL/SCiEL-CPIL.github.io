"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import random
import time


def _v_shaped_initial(processing_times):
    """
    Generate an initial permutation using the V-shaped heuristic.
    Jobs are sorted by total processing time. The job with the largest total
    is placed in the middle, then alternating jobs with next largest totals
    are placed to the left and right.
    """
    n = processing_times.shape[0]
    if n == 0:
        return []
    if n == 1:
        return [0]
    
    totals = processing_times.sum(axis=1)
    # Sort indices by total processing time descending
    sorted_indices = np.argsort(-totals)
    
    perm = [None] * n
    mid = n // 2
    perm[mid] = sorted_indices[0]
    
    left = mid - 1
    right = mid + 1
    for i in range(1, n):
        if i % 2 == 1:
            # Place on left
            if left >= 0:
                perm[left] = sorted_indices[i]
                left -= 1
        else:
            # Place on right
            if right < n:
                perm[right] = sorted_indices[i]
                right += 1
    
    # Fill any remaining (shouldn't happen with proper logic, but just in case)
    for i in range(n):
        if perm[i] is None:
            # Find first unused index
            used = set(perm)
            for j in range(n):
                if j not in used:
                    perm[i] = j
                    break
    
    return perm


def _compute_makespan(processing_times, perm):
    """
    Compute the makespan for a given permutation.
    Uses the standard flow-shop makespan formula:
    C_j = max(C_{j-1}, C_{j-1,m-1} + p_{j,m}) ... actually more simply:
    Let T[i][j] be the completion time of job i on machine j.
    T[i][0] = T[i-1][0] + p[i][0]
    T[i][j] = max(T[i-1][j], T[i][j-1]) + p[i][j]
    Makespan = T[n-1][m-1]
    """
    n = len(perm)
    if n == 0:
        return 0.0
    m = processing_times.shape[1]
    if m == 0:
        return 0.0
    
    # Use a 2D array or just track the last row
    # For efficiency, we can use a 1D array of size m
    # prev_row[j] = completion time of previous job on machine j
    # curr_row[j] = completion time of current job on machine j
    
    prev_row = np.zeros(m)
    curr_row = np.zeros(m)
    
    for job in perm:
        p = processing_times[job]
        for j in range(m):
            if j == 0:
                curr_row[j] = prev_row[j] + p[j]
            else:
                curr_row[j] = max(prev_row[j], curr_row[j-1]) + p[j]
        prev_row = curr_row.copy()
    
    return prev_row[-1]


def _neighborhood(perm, n):
    """
    Generate neighbors by swapping two adjacent jobs (or non-adjacent? 
    For tabu search, let's use adjacent swaps for efficiency, 
    but also consider inserting a job at a different position for more diversity.
    Actually, let's use pairwise swaps for simplicity and speed, 
    but that's O(n^2) neighbors. For large n, this might be slow.
    
    Alternative: use insertion moves. Pick a job and insert it at a different position.
    This is also O(n^2) but might be better.
    
    For tabu search, we'll generate neighbors on the fly.
    Let's use adjacent swaps first for simplicity, and if time permits, expand.
    Actually, for a 120s budget, we can afford O(n^2) per iteration if n is moderate.
    
    Let's use a combination: for each position i, swap with i+1 (adjacent).
    This gives n-1 neighbors per iteration.
    """
    neighbors = []
    for i in range(n - 1):
        new_perm = perm.copy()
        new_perm[i], new_perm[i+1] = new_perm[i+1], new_perm[i]
        neighbors.append((i, i+1, new_perm))
    return neighbors


def order_jobs(processing_times):
    """
    Main function: V-shaped initial solution + Tabu Search with dynamic tabu list.
    """
    processing_times = np.asarray(processing_times)
    n = processing_times.shape[0]
    m = processing_times.shape[1]
    
    if n <= 1:
        return list(range(n))
    
    start_time = time.time()
    time_budget = 115.0  # Leave 5s buffer
    
    # Generate initial V-shaped solution
    perm = _v_shaped_initial(processing_times)
    best_perm = perm.copy()
    best_makespan = _compute_makespan(processing_times, best_perm)
    
    # Tabu Search parameters
    tabu_tenure = max(5, n // 3)  # Dynamic tabu list size
    max_iterations = 50000
    no_improve_limit = 1000  # Reset tabu list if no improvement for this many iterations
    
    current_makespan = best_makespan
    current_perm = perm.copy()
    
    iteration = 0
    no_improve_count = 0
    
    # Tabu list: store (i, j) pairs that are tabu
    tabu_list = {}  # (i, j) -> iteration when it becomes non-tabu
    
    while iteration < max_iterations:
        # Check time
        if time.time() - start_time > time_budget:
            break
        
        # Clean up tabu list
        for key in list(tabu_list.keys()):
            if tabu_list[key] <= iteration:
                del tabu_list[key]
        
        # Generate neighbors (adjacent swaps)
        best_neighbor_perm = None
        best_neighbor_makespan = float('inf')
        best_neighbor_move = None
        
        for i in range(n - 1):
            # Swap i and i+1
            new_perm = current_perm.copy()
            new_perm[i], new_perm[i+1] = new_perm[i+1], new_perm[i]
            
            move_key = (i, i+1)
            # Check if this move is tabu
            is_tabu = move_key in tabu_list
            
            # Compute makespan
            makespan = _compute_makespan(processing_times, new_perm)
            
            # Aspiration criterion: accept if it improves the best known
            if not is_tabu or makespan < best_makespan:
                if makespan < best_neighbor_makespan:
                    best_neighbor_makespan = makespan
                    best_neighbor_perm = new_perm
                    best_neighbor_move = move_key
        
        if best_neighbor_perm is None:
            # No non-tabu moves found, break
            break
        
        # Accept the best neighbor
        current_perm = best_neighbor_perm
        current_makespan = best_neighbor_makespan
        
        # Add the reverse move to tabu list
        if best_neighbor_move is not None:
            i, j = best_neighbor_move
            tabu_list[(i, j)] = iteration + tabu_tenure
            # Also tabu the reverse move (same thing for adjacent swaps)
            tabu_list[(j, i)] = iteration + tabu_tenure
        
        # Update best
        if current_makespan < best_makespan:
            best_makespan = current_makespan
            best_perm = current_perm.copy()
            no_improve_count = 0
        else:
            no_improve_count += 1
        
        # Dynamic tabu list: increase tenure if stuck
        if no_improve_count > no_improve_limit:
            tabu_tenure = min(tabu_tenure * 2, n)
            no_improve_count = 0
        
        iteration += 1
    
    return best_perm
# EVOLVE-BLOCK-END
