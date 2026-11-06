# EVOLVE-BLOCK-START
import numpy as np
import time
import random


def _makespan(processing_times, perm):
    """Compute makespan for a given permutation."""
    n_jobs, n_machines = processing_times.shape
    if n_jobs == 0:
        return 0.0
    # completion[job_index_in_perm][machine]
    # We simulate the flow shop
    # C[m] = completion time on machine m for the current job
    C = np.zeros(n_machines, dtype=np.float64)
    for job in perm:
        for m in range(n_machines):
            C[m] = max(C[m], C[m-1] if m > 0 else 0) + processing_times[job, m]
    return C[-1]


def _v_shape_heuristic(processing_times):
    """
    Generate an initial permutation using a V-shaped heuristic.
    The idea: jobs with high processing times on early machines go to the front,
    jobs with high processing times on late machines go to the back.
    Middle jobs are placed in between.
    
    A common V-shaped approach:
    1. Split jobs into two sets: those that should be early and those late.
    2. Early jobs sorted by decreasing sum of first half machines.
    3. Late jobs sorted by increasing sum of last half machines.
    4. Combine: early jobs + late jobs.
    
    Alternatively, use the NEH heuristic which is known to be strong.
    Let's implement a proper NEH heuristic as the initial solution, then apply V-shape refinement.
    Actually, the directive says "V-shaped heuristic". Let's implement a standard V-shape:
    - Compute total processing time for each job.
    - Sort by total processing time descending.
    - Place the job with the largest total processing time first.
    - For remaining jobs, place each next job at the position that minimizes makespan,
      but bias towards the ends to create a V-shape.
    
    Actually, a simpler and effective V-shape heuristic:
    1. Sort jobs by total processing time descending.
    2. The first job goes to position 0.
    3. The second job goes to the last position.
    4. The third job goes to position 1.
    5. The fourth job goes to the second-to-last position.
    6. And so on, alternating from the left and right ends.
    This creates a V-shape in terms of job sizes.
    """
    n_jobs, n_machines = processing_times.shape
    if n_jobs == 0:
        return []
    
    # Sort by total processing time descending
    totals = processing_times.sum(axis=1)
    sorted_jobs = np.argsort(-totals)
    
    perm = [-1] * n_jobs
    left = 0
    right = n_jobs - 1
    for i, job in enumerate(sorted_jobs):
        if i % 2 == 0:
            perm[left] = job
            left += 1
        else:
            perm[right] = job
            right -= 1
    
    return perm


def _neighborhood(perm):
    """
    Generate a neighborhood of permutations via adjacent swaps and insertions.
    For efficiency, we'll just yield swaps of adjacent elements and some random insertions.
    """
    n = len(perm)
    if n <= 1:
        return
    
    # Adjacent swaps
    for i in range(n - 1):
        new_perm = perm.copy()
        new_perm[i], new_perm[i+1] = new_perm[i+1], new_perm[i]
        yield new_perm
    
    # Insertions: move a job to a random other position
    for i in range(n):
        for j in range(n):
            if i == j or abs(i - j) == 1:
                continue
            new_perm = perm.copy()
            job = new_perm[i]
            del new_perm[i]
            # Adjust index if j > i
            if j > i:
                j -= 1
            new_perm.insert(j, job)
            yield new_perm


def order_jobs(processing_times):
    """
    Construct a job permutation minimizing makespan using V-shaped heuristic
    initialization followed by Tabu Search with a dynamic tabu list.
    """
    processing_times = np.asarray(processing_times, dtype=np.float64)
    n_jobs, n_machines = processing_times.shape
    
    if n_jobs == 0:
        return []
    if n_jobs == 1:
        return [0]
    
    # Time budget management
    start_time = time.time()
    time_limit = 115.0  # Leave some buffer before 120s
    
    # Initial solution using V-shaped heuristic
    current_perm = _v_shape_heuristic(processing_times)
    current_cost = _makespan(processing_times, current_perm)
    
    # Also try NEH-like initial solution and pick the best
    # NEH heuristic: sort by total processing time descending, insert each job at best position
    neeh_perm = list(np.argsort(-processing_times.sum(axis=1)))
    neeh_cost = _makespan(processing_times, neeh_perm)
    
    # Try a few random restarts for initial solution
    best_perm = current_perm.copy()
    best_cost = min(current_cost, neeh_cost)
    if neeh_cost < current_cost:
        best_perm = neeh_perm
        best_cost = neeh_cost
    
    # Tabu Search parameters
    tabu_tenure = max(5, n_jobs // 2)
    min_tabu_tenure = 3
    max_tabu_tenure = n_jobs + 5
    
    # Track tabu status: for each pair (i, j) where we swapped, record the time it became tabu
    # We'll use a dictionary: key = (min(i,j), max(i,j)) for swap moves, value = iteration when it becomes non-tabu
    tabu_dict = {}
    iteration = 0
    no_improve_count = 0
    best_perm_found = best_perm.copy()
    best_cost_found = best_cost
    
    # Dynamic tabu list: adjust tenure based on diversity
    last_tabu_adjustment = 0
    
    while time.time() - start_time < time_limit:
        iteration += 1
        
        # Dynamic tabu tenure adjustment
        if iteration - last_tabu_adjustment >= 100:
            if no_improve_count > 50:
                tabu_tenure = max(min_tabu_tenure, tabu_tenure - 2)
            elif no_improve_count < 10:
                tabu_tenure = min(max_tabu_tenure, tabu_tenure + 1)
            last_tabu_adjustment = iteration
        
        # Clean up expired tabu entries
        expired_keys = [k for k, v in tabu_dict.items() if v <= iteration]
        for k in expired_keys:
            del tabu_dict[k]
        
        # Generate neighborhood and find best non-tabu move
        best_move = None
        best_move_cost = current_cost
        best_new_perm = None
        
        # For large n_jobs, full neighborhood is too expensive. Use a subset.
        n_neighbors = min(len(perm) * 2 + 10, 500)
        random.seed(iteration)
        
        # Sample moves
        moves_to_try = []
        # Always include adjacent swaps
        for i in range(len(current_perm) - 1):
            moves_to_try.append(('swap', i, i + 1))
        
        # Add some random insertion moves
        n_inserts = max(10, len(current_perm) // 2)
        for _ in range(n_inserts):
            i = random.randint(0, len(current_perm) - 1)
            j = random.randint(0, len(current_perm) - 1)
            if i != j and abs(i - j) != 1:
                moves_to_try.append(('insert', i, j))
        
        # Shuffle and limit
        random.shuffle(moves_to_try)
        moves_to_try = moves_to_try[:n_neighbors]
        
        for move in moves_to_try:
            if move[0] == 'swap':
                i, j = move[1], move[2]
                key = (min(i, j), max(i, j))
                if key in tabu_dict and tabu_dict[key] > iteration:
                    # Check aspiration criterion: if it improves best, allow it
                    new_perm = current_perm.copy()
                    new_perm[i], new_perm[j] = new_perm[j], new_perm[i]
                    cost = _makespan(processing_times, new_perm)
                    if cost < best_cost_found:
                        # Aspiration: allow tabu move if it improves global best
                        if cost < best_move_cost:
                            best_move_cost = cost
                            best_new_perm = new_perm
                            best_move = move
                    continue
                new_perm = current_perm.copy()
                new_perm[i], new_perm[j] = new_perm[j], new_perm[i]
                cost = _makespan(processing_times, new_perm)
                if cost < best_move_cost:
                    best_move_cost = cost
                    best_new_perm = new_perm
                    best_move = move
            else:  # insert
                i, j = move[1], move[2]
                # For insertion moves, we don't track tabu as strictly, or we can
                # Let's just allow all insertion moves for diversity
                new_perm = current_perm.copy()
                job = new_perm[i]
                del new_perm[i]
                if j > i:
                    j -= 1
                new_perm.insert(j, job)
                cost = _makespan(processing_times, new_perm)
                if cost < best_move_cost:
                    best_move_cost = cost
                    best_new_perm = new_perm
                    best_move = move
        
        if best_new_perm is None:
            # No improving move found, try a random perturbation to escape local optimum
            if no_improve_count > 20:
                # Diversification: random restart or large perturbation
                if random.random() < 0.3:
                    # Random restart
                    current_perm = _v_shape_heuristic(processing_times)
                    current_cost = _makespan(processing_times, current_perm)
                else:
                    # Large perturbation: swap several pairs
                    for _ in range(min(5, len(current_perm) // 2)):
                        i = random.randint(0, len(current_perm) - 2)
                        j = random.randint(i + 1, len(current_perm) - 1)
                        current_perm[i], current_perm[j] = current_perm[j], current_perm[i]
                    current_cost = _makespan(processing_times, current_perm)
                no_improve_count = 0
                continue
            else:
                no_improve_count += 1
                # Try the best move even if it's not improving (to explore)
                if best_new_perm is not None:
                    current_perm = best_new_perm
                    current_cost = best_move_cost
                else:
                    break
                continue
        
        # Accept the best move
        if best_move[0] == 'swap':
            i, j = best_move[1], best_move[2]
            key = (min(i, j), max(i, j))
            tabu_dict[key] = iteration + tabu_tenure
        
        current_perm = best_new_perm
        current_cost = best_move_cost
        
        if current_cost < best_cost_found:
            best_cost_found = current_cost
            best_perm_found = current_perm.copy()
            no_improve_count = 0
        else:
            no_improve_count += 1
    
    return list(best_perm_found)
# EVOLVE-BLOCK-END
