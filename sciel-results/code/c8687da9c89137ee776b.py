"""Flow-shop permutation construction using NEH variants + insertion/Or-opt local search."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import random


def _makespan(order, P):
    """Compute makespan for a given job order on processing_times matrix P."""
    if len(order) == 0:
        return 0.0
    n_jobs = len(order)
    n_machines = P.shape[1]
    
    if n_machines == 0:
        return 0.0
        
    # Use list-based approach for small n_machines to reduce NumPy overhead
    if n_machines <= 10:
        first_job = order[0]
        row = P[first_job]
        curr_row = [0.0] * n_machines
        curr_row[0] = row[0]
        for m in range(1, n_machines):
            curr_row[m] = curr_row[m-1] + row[m]
            
        for idx in range(1, n_jobs):
            j = order[idx]
            row = P[j]
            new_row = [0.0] * n_machines
            new_row[0] = curr_row[0] + row[0]
            for m in range(1, n_machines):
                prev_val = curr_row[m]
                left_val = new_row[m-1]
                new_row[m] = (prev_val if prev_val > left_val else left_val) + row[m]
            curr_row = new_row
        return curr_row[-1]
    else:
        first_job = order[0]
        curr_row = np.cumsum(P[first_job])
        
        for idx in range(1, n_jobs):
            j = order[idx]
            prev_row = curr_row
            new_row = np.empty(n_machines)
            new_row[0] = prev_row[0] + P[j, 0]
            if n_machines > 1:
                for m in range(1, n_machines):
                    new_row[m] = max(prev_row[m], new_row[m-1]) + P[j, m]
            curr_row = new_row
        
        return curr_row[-1]


def _makespan_incremental(order, P, completion):
    """Compute makespan using a pre-allocated completion array."""
    if len(order) == 0:
        return 0.0
    n_machines = P.shape[1]
    
    if n_machines == 0:
        return 0.0
    
    completion[:] = 0.0
    first_job = order[0]
    row = P[first_job]
    completion[0] = row[0]
    for m in range(1, n_machines):
        completion[m] = completion[m-1] + row[m]
    
    for idx in range(1, len(order)):
        j = order[idx]
        row = P[j]
        new_row = completion  # reuse
        prev_val = completion[0]
        completion[0] = prev_val + row[0]
        for m in range(1, n_machines):
            prev_val = completion[m]
            left_val = completion[m-1]
            completion[m] = (prev_val if prev_val > left_val else left_val) + row[m]
    
    return completion[-1]


def _neh_standard(P):
    """Standard NEH heuristic."""
    n_jobs = P.shape[0]
    totals = P.sum(axis=1)
    sorted_jobs = np.argsort(-totals)
    
    perm = []
    for j in sorted_jobs:
        best_pos = 0
        best_ms = float('inf')
        for pos in range(len(perm) + 1):
            candidate = perm[:pos] + [j] + perm[pos:]
            ms = _makespan(candidate, P)
            if ms < best_ms:
                best_ms = ms
                best_pos = pos
        perm.insert(best_pos, int(j))
    
    return perm


def _neh_first_last(P):
    """NEH variant using first machine processing time for sorting."""
    n_jobs = P.shape[0]
    sort_key = P[:, 0]
    sorted_jobs = np.argsort(-sort_key)
    
    perm = []
    for j in sorted_jobs:
        best_pos = 0
        best_ms = float('inf')
        for pos in range(len(perm) + 1):
            candidate = perm[:pos] + [j] + perm[pos:]
            ms = _makespan(candidate, P)
            if ms < best_ms:
                best_ms = ms
                best_pos = pos
        perm.insert(best_pos, int(j))
    
    return perm


def _neh_last_machine(P):
    """NEH variant using last machine processing time for sorting."""
    n_jobs = P.shape[0]
    n_machines = P.shape[1]
    sort_key = P[:, n_machines - 1]
    sorted_jobs = np.argsort(-sort_key)
    
    perm = []
    for j in sorted_jobs:
        best_pos = 0
        best_ms = float('inf')
        for pos in range(len(perm) + 1):
            candidate = perm[:pos] + [j] + perm[pos:]
            ms = _makespan(candidate, P)
            if ms < best_ms:
                best_ms = ms
                best_pos = pos
        perm.insert(best_pos, int(j))
    
    return perm


def _neh_random(P, rng):
    """NEH variant with random sorting."""
    n_jobs = P.shape[0]
    sorted_jobs = rng.permutation(n_jobs)
    
    perm = []
    for j in sorted_jobs:
        best_pos = 0
        best_ms = float('inf')
        for pos in range(len(perm) + 1):
            candidate = perm[:pos] + [j] + perm[pos:]
            ms = _makespan(candidate, P)
            if ms < best_ms:
                best_ms = ms
                best_pos = pos
        perm.insert(best_pos, int(j))
    
    return perm


def _insertion_search(order, P, max_passes=20, time_limit=None, start_time=None):
    """Insertion local search: try moving each job to every other position."""
    n = len(order)
    if n <= 2:
        return list(order), _makespan(order, P)
    
    best_order = list(order)
    best_ms = _makespan(best_order, P)
    
    for _ in range(max_passes):
        if time_limit is not None and start_time is not None:
            if time.time() > start_time + time_limit:
                break
        
        improved = False
        for i in range(n):
            if time_limit is not None and start_time is not None:
                if time.time() > start_time + time_limit:
                    return best_order, best_ms
            job = best_order[i]
            # Remove job from position i
            remaining = best_order[:i] + best_order[i+1:]
            best_pos = i
            best_candidate_ms = best_ms
            
            for pos in range(len(remaining) + 1):
                if pos == i or pos == i + 1:
                    continue
                candidate = remaining[:pos] + [job] + remaining[pos:]
                ms = _makespan(candidate, P)
                if ms < best_candidate_ms - 1e-10:
                    best_candidate_ms = ms
                    best_pos = pos
            
            if best_pos != i:
                remaining = best_order[:i] + best_order[i+1:]
                best_order = remaining[:best_pos] + [job] + remaining[best_pos:]
                best_ms = best_candidate_ms
                improved = True
        
        if not improved:
            break
    
    return best_order, best_ms


def _or_opt_search(order, P, max_passes=10, time_limit=None, start_time=None):
    """Or-opt local search: try moving blocks of size 1-3 to other positions."""
    n = len(order)
    if n <= 3:
        return list(order), _makespan(order, P)
    
    best_order = list(order)
    best_ms = _makespan(best_order, P)
    
    for _ in range(max_passes):
        if time_limit is not None and start_time is not None:
            if time.time() > start_time + time_limit:
                break
        
        improved = False
        # Try blocks of size 1, 2, 3
        for block_size in [1, 2, 3]:
            if n < block_size + 1:
                continue
            for i in range(n - block_size + 1):
                if time_limit is not None and start_time is not None:
                    if time.time() > start_time + time_limit:
                        return best_order, best_ms
                block = best_order[i:i+block_size]
                # Remove block
                remaining = best_order[:i] + best_order[i+block_size:]
                # Try inserting at all other positions
                for pos in range(len(remaining) + 1):
                    # Skip positions that would result in same or adjacent placement
                    if pos == i or (block_size == 1 and pos == i + 1) or \
                       (block_size > 1 and pos >= i and pos < i + block_size):
                        continue
                    candidate = remaining[:pos] + block + remaining[pos:]
                    ms = _makespan(candidate, P)
                    if ms < best_ms - 1e-10:
                        best_order = candidate
                        best_ms = ms
                        improved = True
                        break
                if improved:
                    break
            if improved:
                break
        
        if not improved:
            break
    
    return best_order, best_ms


def _swap_search(order, P, max_passes=10, time_limit=None, start_time=None):
    """Swap local search: try swapping pairs of jobs."""
    n = len(order)
    if n <= 2:
        return list(order), _makespan(order, P)
    
    best_order = list(order)
    best_ms = _makespan(best_order, P)
    
    for _ in range(max_passes):
        if time_limit is not None and start_time is not None:
            if time.time() > start_time + time_limit:
                break
        
        improved = False
        for i in range(n - 1):
            if time_limit is not None and start_time is not None:
                if time.time() > start_time + time_limit:
                    return best_order, best_ms
            for j in range(i + 1, n):
                candidate = best_order[:]
                candidate[i], candidate[j] = candidate[j], candidate[i]
                ms = _makespan(candidate, P)
                if ms < best_ms - 1e-10:
                    best_order = candidate
                    best_ms = ms
                    improved = True
                    break
            if improved:
                break
        
        if not improved:
            break
    
    return best_order, best_ms


def _simulated_annealing(order, P, start_time, time_limit):
    """Simulated annealing with swap and or-opt moves."""
    n_jobs = len(order)
    if n_jobs < 2:
        return list(order), _makespan(order, P)
    
    current_order = list(order)
    current_makespan = _makespan(current_order, P)
    
    best_order = current_order[:]
    best_makespan = current_makespan
    
    # Simulated Annealing Parameters
    T = 100.0  # Initial temperature
    cooling_rate = 0.99
    min_temperature = 0.1
    max_steps = 50000
    
    rng = random.Random(42)
    
    step = 0
    while (time.time() - start_time) < time_limit and T > min_temperature and step < max_steps:
        step += 1
        
        # Generate a neighbor move
        move_type = rng.choice(['swap', 'or_opt'])
        
        if move_type == 'swap' and n_jobs >= 2:
            i, j = rng.sample(range(n_jobs), 2)
            new_order = current_order[:]
            new_order[i], new_order[j] = new_order[j], new_order[i]
        elif move_type == 'or_opt' and n_jobs >= 3:
            # Or-opt: move a block of 1, 2, or 3 jobs to a new position
            block_len = rng.choice([1, 2, 3])
            if n_jobs < block_len + 1:
                block_len = max(1, n_jobs - 1)
            
            i = rng.randint(0, n_jobs - block_len)
            block = current_order[i:i+block_len]
            
            # Remove block
            new_order = current_order[:i] + current_order[i+block_len:]
            
            # Insert block at random position (excluding original position to ensure a move)
            j = rng.randint(0, len(new_order))
            
            new_order = new_order[:j] + block + new_order[j:]
            
            if new_order == current_order:
                continue
        else:
            continue
        
        new_makespan = _makespan(new_order, P)
        delta = new_makespan - current_makespan
        
        if delta < 0:
            # Accept improvement
            current_order = new_order
            current_makespan = new_makespan
            
            if current_makespan < best_makespan:
                best_order = current_order[:]
                best_makespan = current_makespan
        else:
            # Accept worse solution with probability exp(-delta / T)
            if rng.random() < np.exp(-delta / T):
                current_order = new_order
                current_makespan = new_makespan
        
        T *= cooling_rate
    
    return best_order, best_makespan


def _tabu_search(order, P, start_time, time_limit, tabu_tenure=7):
    """Tabu search with swap and or-opt moves."""
    n_jobs = len(order)
    if n_jobs < 2:
        return list(order), _makespan(order, P)
    
    current_order = list(order)
    current_makespan = _makespan(current_order, P)
    
    best_order = current_order[:]
    best_makespan = current_makespan
    
    # Tabu list: store (i, j) pairs for swaps and (i, block_len, pos) for or-opt
    tabu_list = []
    rng = random.Random(123)
    
    max_iterations = 20000
    no_improve_count = 0
    max_no_improve = 200
    
    for it in range(max_iterations):
        if time.time() > start_time + time_limit:
            break
        
        # Remove expired tabu entries
        tabu_list = [(move, expiry) for move, expiry in tabu_list if expiry > it]
        
        # Generate a neighbor move
        move_type = rng.choice(['swap', 'or_opt'])
        
        if move_type == 'swap' and n_jobs >= 2:
            i, j = rng.sample(range(n_jobs), 2)
            move_key = ('swap', i, j)
            if move_key in [m[0] for m in tabu_list]:
                continue
            new_order = current_order[:]
            new_order[i], new_order[j] = new_order[j], new_order[i]
        elif move_type == 'or_opt' and n_jobs >= 3:
            block_len = rng.choice([1, 2, 3])
            if n_jobs < block_len + 1:
                block_len = max(1, n_jobs - 1)
            
            i = rng.randint(0, n_jobs - block_len)
            block = current_order[i:i+block_len]
            
            # Remove block
            new_order = current_order[:i] + current_order[i+block_len:]
            
            # Insert block at random position (excluding original position to ensure a move)
            j = rng.randint(0, len(new_order))
            
            new_order = new_order[:j] + block + new_order[j:]
            
            if new_order == current_order:
                continue
            move_key = ('or_opt', i, block_len, j)
            if move_key in [m[0] for m in tabu_list]:
                continue
        else:
            continue
        
        new_makespan = _makespan(new_order, P)
        
        if new_makespan < best_makespan - 1e-10:
            # Aspiration criterion: accept if it improves best
            current_order = new_order
            current_makespan = new_makespan
            best_order = current_order[:]
            best_makespan = current_makespan
            no_improve_count = 0
            tabu_list.append((move_key, it + tabu_tenure))
        elif new_makespan < current_makespan - 1e-10:
            # Accept improvement
            current_order = new_order
            current_makespan = new_makespan
            no_improve_count += 1
            tabu_list.append((move_key, it + tabu_tenure))
        else:
            # Accept worse solution with some probability
            if rng.random() < 0.3:
                current_order = new_order
                current_makespan = new_makespan
                no_improve_count += 1
                tabu_list.append((move_key, it + tabu_tenure))
            else:
                no_improve_count += 1
        
        if no_improve_count > max_no_improve:
            break
    
    return best_order, best_makespan


def order_jobs(processing_times):
    """Construct a job permutation minimizing makespan on flow-shop instances."""
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin
    
    P = np.asarray(processing_times, dtype=np.float64)
    n_jobs, n_machines = P.shape
    
    if n_jobs <= 1:
        return list(range(n_jobs))
    
    rng = np.random.RandomState(42)
    
    # Generate multiple NEH variants
    candidates = []
    candidates.append(_neh_standard(P))
    candidates.append(_neh_first_last(P))
    candidates.append(_neh_last_machine(P))
    
    # Add more random NEH variants for diversity
    for _ in range(4):
        candidates.append(_neh_random(P, rng))
    
    # Evaluate all candidates
    best_order = None
    best_ms = float('inf')
    for cand in candidates:
        ms = _makespan(cand, P)
        if ms < best_ms:
            best_ms = ms
            best_order = cand
    
    # Check time
    elapsed = time.time() - start_time
    if elapsed > deadline - 5:
        return best_order
    
    # Phase 1: Insertion search on best candidate
    remaining_time = deadline - (time.time() - start_time) - 2.0
    best_order, best_ms = _insertion_search(
        best_order, P, max_passes=30, time_limit=remaining_time, start_time=start_time
    )
    
    # Check time
    elapsed = time.time() - start_time
    if elapsed > deadline - 5:
        return best_order
    
    # Phase 2: Or-opt search (block moves)
    remaining_time = deadline - (time.time() - start_time) - 2.0
    best_order, best_ms = _or_opt_search(
        best_order, P, max_passes=10, time_limit=remaining_time, start_time=start_time
    )
    
    # Check time
    elapsed = time.time() - start_time
    if elapsed > deadline - 5:
        return best_order
    
    # Phase 3: Swap search for additional refinement
    remaining_time = deadline - (time.time() - start_time) - 2.0
    if remaining_time > 5:
        best_order, best_ms = _swap_search(
            best_order, P, max_passes=10, time_limit=remaining_time, start_time=start_time
        )
    
    # Check time
    elapsed = time.time() - start_time
    if elapsed > deadline - 5:
        return best_order
    
    # Phase 4: Simulated Annealing for further refinement
    remaining_time = deadline - (time.time() - start_time) - 2.0
    if remaining_time > 10:
        best_order, best_ms = _simulated_annealing(
            best_order, P, start_time, remaining_time
        )
    
    # Check time
    elapsed = time.time() - start_time
    if elapsed > deadline - 5:
        return best_order
    
    # Phase 5: Tabu search for further refinement
    remaining_time = deadline - (time.time() - start_time) - 2.0
    if remaining_time > 10:
        best_order, best_ms = _tabu_search(
            best_order, P, start_time, remaining_time
        )
    
    # Check time
    elapsed = time.time() - start_time
    if elapsed > deadline - 5:
        return best_order
    
    # Phase 6: Final insertion polish
    remaining_time = deadline - (time.time() - start_time) - 1.0
    if remaining_time > 2:
        best_order, best_ms = _insertion_search(
            best_order, P, max_passes=10, time_limit=remaining_time, start_time=start_time
        )
    
    return best_order
# EVOLVE-BLOCK-END
