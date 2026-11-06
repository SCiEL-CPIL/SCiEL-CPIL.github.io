"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import random


def _process_job(current, p_row):
    """Apply one job to a machine-completion vector. Returns new vector."""
    if p_row.shape[0] == 1:
        return np.array([current[0] + p_row[0]])
    c0 = current[0] + p_row[0]
    out = np.empty_like(current)
    out[0] = c0
    out[1:] = np.maximum(c0, current[1:]) + p_row[1:]
    return out


def _makespan(order, processing_times):
    """
    Calculate the makespan for a given job order in a flow shop.
    order: list of job indices
    processing_times: (n_jobs, n_machines) array
    Returns the makespan (total completion time on the last machine).
    """
    n_machines = processing_times.shape[1]
    if n_machines == 0:
        return 0.0
    if len(order) == 0:
        return 0.0

    if n_machines == 1:
        return float(processing_times[order, 0].sum())

    current = np.zeros(n_machines)
    for j in order:
        current = _process_job(current, processing_times[j])
    return float(current[-1])


def _makespan_array(order, processing_times):
    """
    Vectorized makespan calculation returning completion times per machine.
    """
    n_machines = processing_times.shape[1]
    if n_machines == 0:
        return np.array([0.0])
    if len(order) == 0:
        return np.zeros(n_machines)

    current = np.zeros(n_machines)
    for j in order:
        current = _process_job(current, processing_times[j])
    return current


def _neh_heuristic(processing_times, randomize_order=False):
    """
    Generate an initial permutation using the NEH (Nawaz-Enscore-Ham) heuristic.
    Optionally perturbs the sorted job order before insertion to diversify.
    """
    n_jobs, n_machines = processing_times.shape

    if n_jobs == 0:
        return []
    if n_jobs == 1:
        return [0]

    job_sums = processing_times.sum(axis=1)
    sorted_jobs = list(np.argsort(-job_sums))

    if randomize_order:
        random.shuffle(sorted_jobs)

    current_sequence = []
    current_vec = np.zeros(n_machines) if n_machines > 0 else np.array([0.0])

    for job in sorted_jobs:
        best_pos = 0
        best_ms = float('inf')
        p_row = processing_times[job]

        if n_machines == 1:
            # For one machine, any insertion yields the same makespan (sum).
            best_pos = 0
        else:
            # Incrementally evaluate all insertion positions using the prefix vector.
            # We scan positions 0..len(current_sequence).
            # For pos k, the state before inserting job is prefix[k].
            # We maintain prefix_vec as we iterate k.
            prefix_vec = np.zeros(n_machines)
            for k in range(len(current_sequence) + 1):
                # Insert job at position k: process job on prefix_vec, then process
                # jobs current_sequence[k:] to get final makespan.
                v = _process_job(prefix_vec, p_row)
                for idx in range(k, len(current_sequence)):
                    v = _process_job(v, processing_times[current_sequence[idx]])
                ms = float(v[-1])
                if ms < best_ms:
                    best_ms = ms
                    best_pos = k
                # Advance prefix_vec to include current_sequence[k]
                if k < len(current_sequence):
                    prefix_vec = _process_job(prefix_vec, processing_times[current_sequence[k]])

        current_sequence.insert(best_pos, job)

    return current_sequence


def _compute_prefix_completions(order, processing_times):
    """
    Precompute prefix completion times for efficient incremental evaluation.
    Returns a list where prefix[i] is the completion vector after processing jobs order[0:i].
    """
    n = len(order)
    n_machines = processing_times.shape[1]

    if n_machines == 0:
        return [np.array([0.0])] * (n + 1)

    prefix = [np.zeros(n_machines)]
    current = np.zeros(n_machines)

    for i in range(n):
        j = order[i]
        current = _process_job(current, processing_times[j])
        prefix.append(current.copy())

    return prefix


def _eval_insertion(order, processing_times, prefix, i, j):
    """
    Evaluate makespan when moving job at position i to position j (i < j case).
    New order: order[0:i] + order[i+1:j+1] + [order[i]] + order[j+1:]
    """
    n = len(order)
    n_machines = processing_times.shape[1]

    if n_machines == 0:
        return 0.0

    current = prefix[i].copy()

    # Jobs that shift left: order[i+1] ... order[j]
    for k in range(i + 1, j + 1):
        current = _process_job(current, processing_times[order[k]])

    # The moved job
    current = _process_job(current, processing_times[order[i]])

    # Remaining jobs
    for k in range(j + 1, n):
        current = _process_job(current, processing_times[order[k]])

    return float(current[-1])


def _eval_insertion_reverse(order, processing_times, prefix, i, j):
    """
    Evaluate makespan when moving job at position i to position j (i > j case).
    New order: order[0:j] + [order[i]] + order[j:i] + order[i+1:]
    """
    n = len(order)
    n_machines = processing_times.shape[1]

    if n_machines == 0:
        return 0.0

    current = prefix[j].copy()

    # The moved job
    current = _process_job(current, processing_times[order[i]])

    # Jobs that shift right: order[j] ... order[i-1]
    for k in range(j, i):
        current = _process_job(current, processing_times[order[k]])

    # Remaining jobs
    for k in range(i + 1, n):
        current = _process_job(current, processing_times[order[k]])

    return float(current[-1])


def _local_search_insertion(order, processing_times, deadline, best_order, best_makespan,
                            tabu_list=None, tabu_tenure=7):
    """
    Local search using insertion moves with prefix precomputation.
    Optionally uses a short-term tabu list keyed by (job, dest_position) to escape
    shallow local minima. Returns (best_order, best_makespan).
    """
    n = len(order)
    n_machines = processing_times.shape[1]

    if n <= 1 or n_machines == 0:
        return order[:], best_makespan

    current_order = order[:]
    current_ms = best_makespan

    use_tabu = tabu_list is not None
    if use_tabu:
        tabu_iter = 0
    else:
        tabu_iter = 0

    improved = True
    while improved and time.time() < deadline:
        improved = False

        prefix = _compute_prefix_completions(current_order, processing_times)

        for i in range(n):
            if time.time() > deadline:
                break

            # Try all insertion positions
            best_pos = i
            best_eval_ms = current_ms

            for j in range(n):
                if j == i:
                    continue

                if use_tabu:
                    key = (current_order[i], j)
                    if key in tabu_list and tabu_list[key] > tabu_iter:
                        continue

                if i < j:
                    ms = _eval_insertion(current_order, processing_times, prefix, i, j)
                else:
                    ms = _eval_insertion_reverse(current_order, processing_times, prefix, i, j)

                if ms < best_eval_ms:
                    best_eval_ms = ms
                    best_pos = j

            if best_pos != i:
                job = current_order.pop(i)
                current_order.insert(best_pos, job)
                current_ms = best_eval_ms

                if use_tabu:
                    tabu_list[(job, best_pos)] = tabu_iter + tabu_tenure

                if current_ms < best_makespan:
                    best_makespan = current_ms
                    best_order = current_order[:]

                improved = True
                break  # Restart the outer loop with new prefix

        if use_tabu:
            tabu_iter += 1

    return best_order, best_makespan


def _tabu_search(order, processing_times, deadline, best_order, best_makespan):
    """
    Tabu search with insertion moves. Uses a dictionary as the tabu list,
    periodically pruned to keep memory bounded.
    """
    n = len(order)
    n_machines = processing_times.shape[1]

    if n <= 1 or n_machines == 0:
        return best_order, best_makespan

    current_order = order[:]
    current_ms = _makespan(current_order, processing_times)
    if current_ms < best_makespan:
        best_makespan = current_ms
        best_order = current_order[:]

    tabu_tenure = max(5, min(20, n // 2))
    tabu_list = {}
    tabu_iter = 0
    no_improve_count = 0
    max_no_improve = 50

    while time.time() < deadline:
        prefix = _compute_prefix_completions(current_order, processing_times)

        best_move = None
        best_move_ms = current_ms

        for i in range(n):
            if time.time() > deadline:
                break
            for j in range(n):
                if j == i:
                    continue
                key = (current_order[i], j)
                if key in tabu_list and tabu_list[key] > tabu_iter:
                    continue
                if i < j:
                    ms = _eval_insertion(current_order, processing_times, prefix, i, j)
                else:
                    ms = _eval_insertion_reverse(current_order, processing_times, prefix, i, j)
                if ms < best_move_ms:
                    best_move_ms = ms
                    best_move = (i, j)

        if best_move is None:
            # No non-tabu improving move; accept the best tabu move if it improves
            # the best-so-far (aspiration), otherwise break.
            best_move = None
            best_move_ms = float('inf')
            for i in range(n):
                for j in range(n):
                    if j == i:
                        continue
                    if i < j:
                        ms = _eval_insertion(current_order, processing_times, prefix, i, j)
                    else:
                        ms = _eval_insertion_reverse(current_order, processing_times, prefix, i, j)
                    if ms < best_move_ms:
                        best_move_ms = ms
                        best_move = (i, j)
            if best_move is None or best_move_ms >= best_makespan:
                break

        i, j = best_move
        job = current_order.pop(i)
        current_order.insert(j, job)
        tabu_list[(job, j)] = tabu_iter + tabu_tenure
        tabu_iter += 1

        current_ms = best_move_ms
        if current_ms < best_makespan:
            best_makespan = current_ms
            best_order = current_order[:]
            no_improve_count = 0
        else:
            no_improve_count += 1
            if no_improve_count > max_no_improve:
                break

        # Prune tabu list to keep it bounded
        if len(tabu_list) > 5000:
            cutoff = tabu_iter
            tabu_list = {k: v for k, v in tabu_list.items() if v > cutoff}

    return best_order, best_makespan


def _simulated_annealing(order, processing_times, deadline, best_order, best_makespan):
    """
    Simulated annealing with insertion moves to escape local minima.
    Kept as a fallback diversification step, but tabu search is the primary
    escape mechanism.
    """
    n = len(order)
    n_machines = processing_times.shape[1]

    if n <= 1 or n_machines == 0:
        return best_order, best_makespan

    current_order = order[:]
    current_ms = _makespan(current_order, processing_times)

    T = max(1.0, current_ms * 0.05)
    T_min = max(0.01, current_ms * 0.001)
    cooling_rate = 0.97

    while T > T_min and time.time() < deadline:
        i = random.randint(0, n - 1)
        j = random.randint(0, n - 1)

        if i == j:
            continue

        job = current_order[i]
        remaining = current_order[:i] + current_order[i+1:]

        if j > i:
            j -= 1
        j = max(0, min(j, len(remaining)))

        new_order = remaining[:j] + [job] + remaining[j:]
        new_ms = _makespan(new_order, processing_times)

        delta = new_ms - current_ms

        if delta < 0:
            current_order = new_order
            current_ms = new_ms

            if current_ms < best_makespan:
                best_makespan = current_ms
                best_order = current_order[:]
        else:
            if random.random() < np.exp(-delta / T):
                current_order = new_order
                current_ms = new_ms

        T *= cooling_rate

    return best_order, best_makespan


def order_jobs(processing_times):
    """
    Construct a job permutation minimizing makespan.
    """
    start_time = time.time()
    deadline = start_time + 115.0  # 5 second safety margin

    n_jobs = processing_times.shape[0]
    n_machines = processing_times.shape[1]

    if n_jobs == 0:
        return []
    if n_jobs == 1:
        return [0]

    # 1. Generate initial solution using NEH heuristic (deterministic + one randomized restart)
    current_order = _neh_heuristic(processing_times, randomize_order=False)
    current_makespan = _makespan(current_order, processing_times)

    best_order = current_order[:]
    best_makespan = current_makespan

    # Try a randomized NEH restart if there is time
    if time.time() < deadline - 30.0:
        alt_order = _neh_heuristic(processing_times, randomize_order=True)
        alt_ms = _makespan(alt_order, processing_times)
        if alt_ms < best_makespan:
            best_makespan = alt_ms
            best_order = alt_order[:]

    # 2. Local search with insertion moves (no tabu, pure steepest descent)
    best_order, best_makespan = _local_search_insertion(
        best_order, processing_times, deadline, best_order, best_makespan
    )

    # 3. Tabu search to escape local minima (primary diversification)
    if time.time() < deadline - 10.0:
        best_order, best_makespan = _tabu_search(
            best_order, processing_times, deadline, best_order, best_makespan
        )

    # 4. Final local search polish
    if time.time() < deadline - 2.0:
        best_order, best_makespan = _local_search_insertion(
            best_order, processing_times, deadline, best_order, best_makespan
        )

    return best_order

# EVOLVE-BLOCK-END
