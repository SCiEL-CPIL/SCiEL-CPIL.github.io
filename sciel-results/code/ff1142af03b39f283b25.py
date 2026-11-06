import numpy as np
import time

def order_jobs(processing_times):
    start_time = time.time()
    deadline = start_time + 115.0  # 5s safety margin
    
    n_jobs, n_machines = processing_times.shape
    
    # Helper to calculate makespan for a given permutation (list of job indices)
    def calc_makespan(perm):
        completion = np.zeros(n_machines)
        for j in perm:
            job_times = processing_times[j]
            for m in range(n_machines):
                if m == 0:
                    st = completion[0]
                else:
                    st = completion[m]
                    prev = completion[m-1]
                    if prev > st:
                        st = prev
                completion[m] = st + job_times[m]
        return completion[-1]

    # 1. V-shaped heuristic construction
    # Split jobs into two groups: high total processing time first, then low
    total_times = processing_times.sum(axis=1)
    sorted_by_total = np.argsort(-total_times)
    
    # V-shaped: start with jobs having high total time, end with low total time
    # This creates a V-shape in the processing time profile
    seq = [sorted_by_total[0]]
    
    for j in sorted_by_total[1:]:
        best_pos = 0
        best_makespan = float('inf')
        
        # Try inserting job j at each position
        # Optimization: If n_jobs is large, this O(N^3) step is expensive.
        # We limit the insertion search to a subset if N is large to save time for search.
        # However, for small N, we do full search.
        # Let's stick to full search for construction as it's a one-time cost, 
        # but we must ensure it doesn't exceed 10s.
        if n_jobs > 50 and time.time() > start_time + 5.0:
             # Fallback to simple sort if construction is taking too long
             seq = list(sorted_by_total)
             break
             
        for pos in range(len(seq) + 1):
            new_seq = seq[:pos] + [j] + seq[pos:]
            ms = calc_makespan(new_seq)
            if ms < best_makespan:
                best_makespan = ms
                best_pos = pos
        
        seq.insert(best_pos, j)
        
        if time.time() > deadline - 20.0: # Reserve 20s for search
            return list(seq)

    best_seq = list(seq)
    best_ms = calc_makespan(best_seq)
    current_seq = list(best_seq)
    current_ms = best_ms

    # 2. Tabu Search with dynamic tabu list
    # Tabu tenure: number of iterations a move stays tabu
    # Dynamic: adjust based on performance
    
    tabu_tenure = max(5, n_jobs // 3)  # Initial tabu tenure
    min_tenure = 3
    max_tenure = max(10, n_jobs // 2)
    
    # Track tabu moves: (i, j) -> iteration when it becomes non-tabu
    tabu_list = {}
    iteration = 0
    
    # Diversification: if no improvement for X iterations, perturb
    no_improve_count = 0
    perturbation_threshold = 50
    
    # Limit the number of neighbors to evaluate per iteration to ensure speed
    # We sample random swaps instead of checking all O(N^2)
    max_neighbors_per_iter = 100 if n_jobs < 20 else 50
    
    while time.time() < deadline:
        iteration += 1
        best_move = None
        best_move_ms = float('inf')
        best_move_is_tabu = False
        best_move_i = -1
        best_move_j = -1
        
        # Generate a set of candidate moves (random swaps)
        # This reduces complexity from O(N^2) to O(K) per iteration
        candidates = set()
        while len(candidates) < max_neighbors_per_iter:
            i = np.random.randint(0, n_jobs - 1)
            j = np.random.randint(i + 1, n_jobs)
            candidates.add((i, j))
            
        for i, j in candidates:
            if time.time() > deadline:
                break
                
            # Check if this move is tabu
            move_key = (i, j)
            is_tabu = move_key in tabu_list and tabu_list[move_key] > iteration
            
            # Try the swap
            current_seq[i], current_seq[j] = current_seq[j], current_seq[i]
            ms = calc_makespan(current_seq)
            current_seq[i], current_seq[j] = current_seq[j], current_seq[i]
            
            # Accept if: not tabu and improving, OR tabu but better than best found so far
            if not is_tabu and ms < best_move_ms:
                best_move_ms = ms
                best_move_i = i
                best_move_j = j
                best_move_is_tabu = False
            elif is_tabu and ms < best_move_ms:
                # Allow tabu moves if they lead to a solution better than current best
                if ms < best_ms:
                    best_move_ms = ms
                    best_move_i = i
                    best_move_j = j
                    best_move_is_tabu = True
        
        if best_move_i == -1:
            # No improving move found in sample
            # To avoid infinite loop, we can accept a random non-improving move or break
            # Let's accept a random move to diversify if stuck
            i = np.random.randint(0, n_jobs - 1)
            j = np.random.randint(i + 1, n_jobs)
            current_seq[i], current_seq[j] = current_seq[j], current_seq[i]
            current_ms = calc_makespan(current_seq)
            best_move_i = i
            best_move_j = j
        else:
            # Apply the best move
            current_seq[best_move_i], current_seq[best_move_j] = current_seq[best_move_j], current_seq[best_move_i]
            current_ms = best_move_ms
        
        # Update tabu list
        tabu_list[(best_move_i, best_move_j)] = iteration + tabu_tenure
        # Also add reverse move
        tabu_list[(best_move_j, best_move_i)] = iteration + tabu_tenure
        
        # Check if we found a new best
        if current_ms < best_ms:
            best_ms = current_ms
            best_seq = list(current_seq)
            no_improve_count = 0
            # Slightly increase tabu tenure when making progress
            tabu_tenure = min(max_tenure, tabu_tenure + 1)
        else:
            no_improve_count += 1
            # Decrease tabu tenure when stuck to allow more exploration
            tabu_tenure = max(min_tenure, tabu_tenure - 1)
        
        # Diversification: if stuck for too long, perturb the solution
        if no_improve_count >= perturbation_threshold:
            # Randomly swap 2-3 pairs of jobs
            n_perturb = min(3, n_jobs // 4)
            for _ in range(n_perturb):
                i = np.random.randint(0, n_jobs - 1)
                j = np.random.randint(i + 1, n_jobs)
                current_seq[i], current_seq[j] = current_seq[j], current_seq[i]
            current_ms = calc_makespan(current_seq)
            
            # Clear tabu list to allow fresh exploration
            tabu_list.clear()
            no_improve_count = 0
            tabu_tenure = max(5, n_jobs // 3)
    
    return best_seq
