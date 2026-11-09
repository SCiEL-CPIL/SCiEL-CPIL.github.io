"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def _makespan(perm, processing_times):
    """
    Compute the makespan for a given permutation of job indices.
    Optimized for speed using Python lists and local variables.
    """
    n_jobs, n_machines = processing_times.shape
    if n_jobs == 0:
        return 0
    
    prev_job_completion = np.zeros(n_machines)
    
    for job in perm:
        current_job_completion_on_prev_machine = 0.0
        for m in range(n_machines):
            p = processing_times[job, m]
            if m == 0:
                start = prev_job_completion[0]
                current_job_completion_on_prev_machine = start + p
            else:
                start = max(current_job_completion_on_prev_machine, prev_job_completion[m])
                current_job_completion_on_prev_machine = start + p
            
            prev_job_completion[m] = current_job_completion_on_prev_machine
    
    return prev_job_completion[-1]


def _longest_path_initialization(processing_times):
    """
    Initialize permutation using longest path in job-machine dependency graph.
    
    The flow shop can be modeled as a DAG where:
    - Each job-machine pair (j, m) is a node with weight P[j, m]
    - Edges: (j, m) -> (j, m+1) [same job, next machine]
    - Edges: (j, m) -> (j+1, m) [next job, same machine] - but this depends on permutation
    
    For initialization, we compute a "criticality" score for each job:
    - Forward pass: earliest start time if job were first
    - Backward pass: latest finish time considering all jobs
    
    We sort jobs by a composite key that reflects their contribution to the critical path.
    """
    n_jobs, n_machines = processing_times.shape
    if n_jobs == 0:
        return []
    
    # Compute total processing time per job
    total_processing = processing_times.sum(axis=1)
    
    # Compute "criticality" score:
    # For each job, estimate how much it contributes to the makespan.
    # A job is critical if it has high processing times on machines that are bottlenecks.
    
    # Machine load: total time each machine is busy (sum over all jobs)
    machine_load = processing_times.sum(axis=0)
    
    # For each job, compute weighted sum: sum over machines of P[j,m] * machine_load[m]
    # This approximates the "delay" a job causes on each machine
    job_criticality = np.zeros(n_jobs)
    for j in range(n_jobs):
        for m in range(n_machines):
            job_criticality[j] += processing_times[j, m] * machine_load[m]
    
    # Also consider the "span" of the job: difference between max and min processing time
    # Jobs with high span may cause more idle time
    job_span = processing_times.max(axis=1) - processing_times.min(axis=1)
    
    # Composite key: primarily criticality, secondarily span
    # We want to schedule critical jobs early to avoid them blocking later jobs
    # But also consider that jobs with high span should be scheduled to minimize idle time
    
    # Sort by criticality descending, then by span descending
    # This gives us an initial ordering
    sorted_indices = np.lexsort((-job_span, -job_criticality))
    
    return list(sorted_indices)


def _tabu_search(initial_perm, processing_times, time_limit=10.0):
    """
    Tabu search with swaps and insertions.
    """
    n_jobs = len(initial_perm)
    if n_jobs <= 1:
        return initial_perm
    
    current_perm = list(initial_perm)
    current_makespan = _makespan(current_perm, processing_times)
    best_perm = list(current_perm)
    best_makespan = current_makespan
    
    # Tabu list: dictionary mapping move tuple to expiration time
    # Move types: ('swap', i, j) or ('move', i, j) where i is removed from i and inserted at j
    tabu_list = {}
    tabu_tenure = 10
    
    start_time = time.time()
    iteration = 0
    no_improve_count = 0
    max_no_improve = 50  # Restart or stop if no improvement for a while
    
    while time.time() - start_time < time_limit:
        iteration += 1
        best_neighbor_perm = None
        best_neighbor_makespan = current_makespan
        best_neighbor_move = None
        
        # Neighborhood: 
        # 1. Adjacent swaps
        # 2. Moves (remove job i, insert at j)
        
        # To limit computation, we sample moves for large n_jobs
        if n_jobs <= 30:
            # All adjacent swaps
            moves_to_try = []
            for i in range(n_jobs - 1):
                moves_to_try.append(('swap', i, i + 1))
            # Some random moves
            for _ in range(50):
                i = np.random.randint(0, n_jobs)
                j = np.random.randint(0, n_jobs)
                if i != j:
                    moves_to_try.append(('move', i, j))
        else:
            # Sample moves
            moves_to_try = []
            for _ in range(200):
                if np.random.rand() < 0.5:
                    i = np.random.randint(0, n_jobs - 1)
                    moves_to_try.append(('swap', i, i + 1))
                else:
                    i = np.random.randint(0, n_jobs)
                    j = np.random.randint(0, n_jobs)
                    if i != j:
                        moves_to_try.append(('move', i, j))
        
        for move in moves_to_try:
            if move[0] == 'swap':
                i, j = move[1], move[2]
                move_key = ('swap', min(i, j), max(i, j))
            else:
                i, j = move[1], move[2]
                if i == j:
                    continue
                move_key = ('move', i, j)
            
            # Check tabu
            if move_key in tabu_list and tabu_list[move_key] > iteration:
                # Aspiration: accept if it improves best
                # We don't know the makespan yet, so we can't check aspiration here easily without evaluating.
                # We'll just skip tabu moves for now to keep it simple, or evaluate and then check.
                # Let's evaluate first, then check tabu/aspiration.
                pass
            
            # Generate new permutation
            new_perm = current_perm.copy()
            if move[0] == 'swap':
                new_perm[i], new_perm[j] = new_perm[j], new_perm[i]
            else:
                # Move job at i to position j
                job_to_move = new_perm.pop(i)
                # Adjust j if i < j
                if i < j:
                    j -= 1
                new_perm.insert(j, job_to_move)
            
            new_makespan = _makespan(new_perm, processing_times)
            
            # Check tabu and aspiration
            is_tabu = move_key in tabu_list and tabu_list[move_key] > iteration
            is_aspiration = new_makespan < best_makespan
            
            if is_tabu and not is_aspiration:
                continue
            
            if new_makespan < best_neighbor_makespan:
                best_neighbor_makespan = new_makespan
                best_neighbor_perm = new_perm
                best_neighbor_move = move_key
        
        if best_neighbor_perm is None:
            # No improving move found
            no_improve_count += 1
            if no_improve_count > max_no_improve:
                break
            # Perturb: pick a random move
            i = np.random.randint(0, n_jobs)
            j = np.random.randint(0, n_jobs)
            if i != j:
                new_perm = current_perm.copy()
                job_to_move = new_perm.pop(i)
                if i < j:
                    j -= 1
                new_perm.insert(j, job_to_move)
                current_perm = new_perm
                current_makespan = _makespan(current_perm, processing_times)
        else:
            no_improve_count = 0
            # Update tabu list
            tabu_list[best_neighbor_move] = iteration + tabu_tenure
            
            # Clean up tabu list
            for key in list(tabu_list.keys()):
                if tabu_list[key] <= iteration:
                    del tabu_list[key]
            
            # Accept move
            current_perm = best_neighbor_perm
            current_makespan = best_neighbor_makespan
            
            if current_makespan < best_makespan:
                best_makespan = current_makespan
                best_perm = list(current_perm)
    
    return best_perm


def order_jobs(processing_times):
    """
    Construct a job permutation minimizing makespan on local synthetic flow-shop instances.
    Uses longest-path-based initialization and tabu search with aspiration.
    """
    if len(processing_times) == 0:
        return []
    
    # Initial solution: longest path based
    initial_perm = _longest_path_initialization(processing_times)
    
    # Refine with tabu search
    # Adjust time limit based on problem size
    n_jobs = len(processing_times)
    if n_jobs <= 20:
        time_limit = 15.0
    elif n_jobs <= 50:
        time_limit = 10.0
    else:
        time_limit = 5.0
    
    final_perm = _tabu_search(initial_perm, processing_times, time_limit=time_limit)
    
    return list(final_perm)
# EVOLVE-BLOCK-END
