"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from itertools import combinations


def _makespan(perm, processing_times):
    """Compute the makespan for a given permutation of job indices."""
    n_jobs, n_machines = processing_times.shape
    # C[j, m] = completion time of job j on machine m
    # We process jobs in the order given by perm
    # For each machine, the start time is max(prev job completion on this machine, prev machine completion for this job)
    
    # Efficient vectorized-ish computation
    # Let's use a loop over jobs in the permutation
    completion = np.zeros((n_jobs, n_machines))
    for idx, job in enumerate(perm):
        for m in range(n_machines):
            p = processing_times[job, m]
            if m == 0:
                # First machine: constrained by previous job on machine 0
                if idx == 0:
                    completion[job, m] = p
                else:
                    prev_job = perm[idx - 1]
                    completion[job, m] = completion[prev_job, m] + p
            else:
                # Constrained by previous job on this machine AND previous machine for this job
                if idx == 0:
                    # First job in permutation
                    completion[job, m] = completion[job, m - 1] + p
                else:
                    prev_job = perm[idx - 1]
                    # Start time is max of:
                    # 1. Completion of previous job on this machine
                    # 2. Completion of this job on previous machine
                    start = max(completion[prev_job, m], completion[job, m - 1])
                    completion[job, m] = start + p
    
    # Makespan is the maximum completion time over all jobs and machines
    # Specifically, it's the completion time of the last job on the last machine
    last_job = perm[-1]
    return completion[last_job, -1]


def _composite_key(processing_times):
    """
    Compute a composite key for each job based on the longest path in the job-machine dependency graph.
    For a flow shop, the longest path through the graph for a job starting at time 0 on machine 0
    is simply the sum of processing times for that job across all machines.
    However, to make it more sophisticated, we can consider the impact on the critical path.
    
    A better heuristic: Sort by a key that approximates the critical path contribution.
    One common approach is to use the sum, but let's try a weighted sum where machines in the middle
    might have different weights, or use the NEH rule which sorts by total processing time in descending order.
    
    The directive says: "composite key derived from the longest path in the job-machine dependency graph".
    In a flow shop, the longest path from source to sink goes through all machines. The total length is the sum of all processing times for a specific job if it were the only job.
    But since jobs interact, a better initial heuristic is often the NEH rule.
    
    Let's stick to the NEH rule for the initial solution as it's robust, but the directive asks for a "composite key derived from the longest path".
    For a single job j, the longest path through its own nodes is sum(processing_times[j, :]).
    Let's use this sum as the key, sorting in descending order.
    """
    total_processing = processing_times.sum(axis=1)
    # Sort indices by total processing time in descending order
    return np.argsort(-total_processing)


def _tabu_search(initial_perm, processing_times, max_iters=500, tabu_tenure=10):
    """
    Refine the permutation using tabu search.
    Moves: Swap two jobs in the permutation.
    Tabu: Prohibit swapping back to the previous state for a certain number of iterations.
    """
    n_jobs = len(initial_perm)
    if n_jobs <= 1:
        return initial_perm
    
    current_perm = list(initial_perm)
    current_makespan = _makespan(current_perm, processing_times)
    best_perm = list(current_perm)
    best_makespan = current_makespan
    
    # Tabu list: store (i, j) pairs that are tabu, with their expiration iteration
    tabu_list = {}
    # To save time, we can limit the neighborhood or use a faster makespan calculation
    
    for iteration in range(max_iters):
        best_neighbor_perm = None
        best_neighbor_makespan = current_makespan
        best_neighbor_move = None
        
        # Generate neighbors by swapping two jobs
        # To limit computation, we can sample a subset of swaps if n_jobs is large
        # For small n_jobs, we can do all pairs
        if n_jobs <= 20:
            pairs = list(combinations(range(n_jobs), 2))
        else:
            # Sample 100 random pairs
            pairs = [(np.random.randint(0, n_jobs), np.random.randint(0, n_jobs)) for _ in range(100)]
            # Ensure i != j
            pairs = [(i, j) for i, j in pairs if i != j]
        
        for i, j in pairs:
            if i == j:
                continue
            
            # Check if this move is tabu
            move_key = (min(i, j), max(i, j))
            if move_key in tabu_list and tabu_list[move_key] > iteration:
                continue
            
            # Apply swap
            new_perm = current_perm.copy()
            new_perm[i], new_perm[j] = new_perm[j], new_perm[i]
            
            # Evaluate
            new_makespan = _makespan(new_perm, processing_times)
            
            if new_makespan < best_neighbor_makespan:
                best_neighbor_makespan = new_makespan
                best_neighbor_perm = new_perm
                best_neighbor_move = move_key
        
        if best_neighbor_perm is None:
            # No non-tabu moves found, or all moves are worse
            # Break or continue with diversification
            break
        
        # Update tabu list
        tabu_list[best_neighbor_move] = iteration + tabu_tenure
        
        # Remove expired tabu entries
        for key in list(tabu_list.keys()):
            if tabu_list[key] <= iteration:
                del tabu_list[key]
        
        # Accept the move
        current_perm = best_neighbor_perm
        current_makespan = best_neighbor_makespan
        
        if current_makespan < best_makespan:
            best_makespan = current_makespan
            best_perm = list(current_perm)
    
    return best_perm


def order_jobs(processing_times):
    """
    Construct a job permutation minimizing makespan on local synthetic flow-shop instances.
    Uses a composite key derived from the longest path (total processing time) for initial ordering,
    then refines using tabu search.
    """
    if len(processing_times) == 0:
        return []
    
    # Initial solution: NEH rule (sort by total processing time descending)
    initial_perm = _composite_key(processing_times)
    
    # Refine with tabu search
    # Adjust max_iters based on problem size to stay within time limits
    n_jobs = len(processing_times)
    if n_jobs <= 10:
        max_iters = 1000
    elif n_jobs <= 50:
        max_iters = 500
    else:
        max_iters = 100
    
    final_perm = _tabu_search(initial_perm, processing_times, max_iters=max_iters, tabu_tenure=10)
    
    return list(final_perm)
# EVOLVE-BLOCK-END
