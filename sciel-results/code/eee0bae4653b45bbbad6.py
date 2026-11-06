"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from typing import List, Tuple


def _v_shape_initial(processing_times: np.ndarray) -> List[int]:
    """
    Generate an initial permutation using a V-shaped heuristic.
    Jobs are sorted by total processing time. The longest jobs are placed
    first (ascending order of total time for the left side) and the shortest
    jobs are placed last (descending order of total time for the right side).
    Specifically:
    - Sort jobs by total processing time.
    - The first half (largest totals) go to the beginning in increasing order of total time? 
      Actually, standard V-shape: 
      - Sort jobs by total processing time.
      - The jobs with the largest total times are placed at the ends of the permutation.
      - The jobs with the smallest total times are placed in the middle.
      A common heuristic: 
        1. Calculate total processing time for each job.
        2. Sort jobs by total processing time.
        3. Alternate placing the next job with the largest remaining total time at the front or back,
           and the next job with the smallest remaining total time in the middle? 
      
      Let's use a simpler, well-known V-shaped construction:
      - Sort jobs by total processing time.
      - The job with the largest total time goes to the first position.
      - The job with the second largest total time goes to the last position.
      - The job with the third largest total time goes to the second position.
      - The job with the fourth largest total time goes to the second-to-last position.
      - And so on.
      
      Wait, that's just alternating from ends. Let's try:
      - Sort jobs by total processing time in descending order.
      - Take the largest, put it at index 0.
      - Take the next largest, put it at the end.
      - Take the next largest, put it at index 1.
      - Take the next largest, put it at the end-1.
      - Continue until all jobs are placed.
    """
    n_jobs = processing_times.shape[0]
    totals = processing_times.sum(axis=1)
    # Sort job indices by total processing time in descending order
    sorted_indices = np.argsort(-totals)
    
    perm = [None] * n_jobs
    left = 0
    right = n_jobs - 1
    for idx in sorted_indices:
        if left <= right:
            if (left + right) % 2 == 0:
                # Place at left
                perm[left] = idx
                left += 1
            else:
                # Place at right
                perm[right] = idx
                right -= 1
    
    return perm


def _makespan(processing_times: np.ndarray, perm: List[int]) -> float:
    """
    Calculate the makespan for a given permutation.
    """
    n_machines = processing_times.shape[1]
    # C[m][j] is the completion time of job j on machine m
    # We can compute this efficiently
    # C[0][j] = sum of processing times of jobs 0..j on machine 0
    # C[m][j] = max(C[m-1][j], C[m][j-1]) + p[m][j]
    
    # Let's use a 2D array for completion times
    # C[m][j] for m in range(n_machines), j in range(n_jobs)
    C = np.zeros((n_machines, len(perm)))
    
    # First machine
    for j in range(len(perm)):
        if j == 0:
            C[0][j] = processing_times[perm[j]][0]
        else:
            C[0][j] = C[0][j-1] + processing_times[perm[j]][0]
    
    # Other machines
    for m in range(1, n_machines):
        for j in range(len(perm)):
            if j == 0:
                C[m][j] = C[m-1][j] + processing_times[perm[j]][m]
            else:
                C[m][j] = max(C[m-1][j], C[m][j-1]) + processing_times[perm[j]][m]
    
    return C[n_machines-1][len(perm)-1]


def order_jobs(processing_times: np.ndarray) -> List[int]:
    """
    Construct a job permutation minimizing makespan using a V-shaped initial
    solution and Tabu Search with a dynamic tabu list.
    """
    n_jobs = processing_times.shape[0]
    n_machines = processing_times.shape[1]
    
    if n_jobs == 0:
        return []
    if n_jobs == 1:
        return [0]
    
    # 1. Generate initial solution using V-shaped heuristic
    current_perm = _v_shape_initial(processing_times)
    current_cost = _makespan(processing_times, current_perm)
    
    # 2. Tabu Search
    import time
    start_time = time.time()
    time_limit = 115.0  # Leave 5 seconds buffer
    
    # Dynamic tabu list length
    base_tabu_len = max(1, int(np.sqrt(n_jobs)))
    tabu_len = base_tabu_len
    
    # Tabu list: store (i, j, iteration) where move swapping i and j is tabu until iteration
    tabu_list = []
    iteration = 0
    
    best_perm = current_perm[:]
    best_cost = current_cost
    
    # Aspiration criterion: if a tabu move leads to a solution better than best_cost, allow it
    no_improve_count = 0
    max_no_improve = 50  # Diversification trigger
    
    while time.time() - start_time < time_limit:
        # Dynamic tabu list length: increase if no improvement, decrease if improvement
        if no_improve_count > max_no_improve:
            tabu_len = int(base_tabu_len * (1 + no_improve_count / max_no_improve))
        else:
            tabu_len = base_tabu_len
        
        # Generate neighborhood: all possible swaps
        # To speed up, we can use incremental makespan calculation, but for simplicity
        # and given n_jobs is likely not huge, we'll evaluate swaps.
        # However, for larger n_jobs, we need to be careful.
        # Let's try to limit the neighborhood or use a faster evaluation.
        
        # For efficiency, let's consider a subset of moves or use a faster heuristic.
        # Actually, for n_jobs up to ~100, O(n^2) swaps per iteration might be slow in Python.
        # Let's estimate: 100^2 = 10,000 swaps per iteration. Each makespan calc is O(n*m).
        # If m=10, n=100, that's 1,000,000 operations per iteration. In Python, this might be too slow.
        
        # We need a faster approach or limit the search.
        # Let's try to use a random sample of moves if n_jobs is large.
        
        if n_jobs > 50:
            # Sample a subset of moves
            num_moves = min(100, n_jobs * (n_jobs - 1) // 2)
            indices = np.random.choice(n_jobs * (n_jobs - 1) // 2, size=num_moves, replace=False)
            # Convert linear index to (i, j)
            # This is tricky. Let's just pick random pairs.
            i_list, j_list = [], []
            for _ in range(num_moves):
                i = np.random.randint(n_jobs)
                j = np.random.randint(n_jobs)
                if i != j:
                    i_list.append(i)
                    j_list.append(j)
        else:
            i_list, j_list = [], []
            for i in range(n_jobs):
                for j in range(i + 1, n_jobs):
                    i_list.append(i)
                    j_list.append(j)
        
        best_move_cost = float('inf')
        best_move_i, best_move_j = -1, -1
        best_move_perm = None
        
        for i, j in zip(i_list, j_list):
            # Check if move is tabu
            is_tabu = False
            for (ti, tj, t_iter) in tabu_list:
                if (ti == i and tj == j) or (ti == j and tj == i):
                    if iteration - t_iter < tabu_len:
                        is_tabu = True
                        break
            
            # Perform swap
            new_perm = current_perm[:]
            new_perm[i], new_perm[j] = new_perm[j], new_perm[i]
            
            # Calculate new cost
            new_cost = _makespan(processing_times, new_perm)
            
            # Aspiration criterion
            if is_tabu and new_cost >= best_cost:
                continue
            
            if new_cost < best_move_cost:
                best_move_cost = new_cost
                best_move_i, best_move_j = i, j
                best_move_perm = new_perm
        
        if best_move_i != -1:
            # Update tabu list
            tabu_list.append((best_move_i, best_move_j, iteration))
            
            # Remove old tabu entries
            tabu_list = [(ti, tj, t_iter) for (ti, tj, t_iter) in tabu_list if iteration - t_iter < tabu_len]
            
            # Update current solution
            current_perm = best_move_perm
            current_cost = best_move_cost
            
            if current_cost < best_cost:
                best_cost = current_cost
                best_perm = current_perm[:]
                no_improve_count = 0
            else:
                no_improve_count += 1
        
        iteration += 1
    
    return best_perm
# EVOLVE-BLOCK-END
