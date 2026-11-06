"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def _makespan(order, processing_times):
    """Compute makespan for a given job order."""
    n_machines = processing_times.shape[1]
    if n_machines == 0 or len(order) == 0:
        return 0.0
    completion = np.zeros(n_machines)
    for job in order:
        p = processing_times[job]
        prev = 0.0
        for m in range(n_machines):
            prev = max(prev, completion[m]) + p[m]
            completion[m] = prev
    return float(completion[-1])


def _compute_completion_matrix(order, processing_times):
    """
    Compute the full completion time matrix C[j, m] for the given order.
    Returns (completion_matrix, makespan).
    """
    n_jobs = len(order)
    n_machines = processing_times.shape[1]
    if n_jobs == 0 or n_machines == 0:
        return np.zeros((0, n_machines)), 0.0
    
    C = np.zeros((n_jobs, n_machines))
    for j, job in enumerate(order):
        p = processing_times[job]
        for m in range(n_machines):
            if m == 0:
                C[j, m] = C[j-1, m] + p[m] if j > 0 else p[m]
            else:
                C[j, m] = max(C[j-1, m], C[j, m-1]) + p[m]
    makespan = float(C[-1, -1]) if n_jobs > 0 else 0.0
    return C, makespan


def _eval_insertion_makespan(order, job, pos, processing_times):
    """
    Quickly evaluate makespan if we insert job at position pos in order.
    Uses incremental computation from the completion matrix concept.
    """
    n_machines = processing_times.shape[1]
    if n_machines == 0:
        return 0.0
    
    # Build new order
    new_order = order[:pos] + [job] + order[pos:]
    
    # Compute makespan incrementally
    completion = np.zeros(n_machines)
    for i, j in enumerate(new_order):
        p = processing_times[j]
        prev = 0.0
        for m in range(n_machines):
            prev = max(prev, completion[m]) + p[m]
            completion[m] = prev
    return float(completion[-1])


def _neh_initialization(processing_times, seed=None):
    """
    NEH heuristic with optional randomization for diversity.
    """
    n_jobs, n_machines = processing_times.shape
    if n_jobs == 0:
        return []
    
    if seed is not None:
        rng = np.random.RandomState(seed)
    else:
        rng = np.random.RandomState()
    
    # Order jobs by decreasing total processing time, with some randomization
    totals = processing_times.sum(axis=1)
    
    # Add small random perturbation to create diverse orderings
    perturbation = rng.rand(n_jobs) * 0.1 * (totals.max() - totals.min() + 1e-9)
    perturbed_totals = totals + perturbation
    
    job_order = list(np.argsort(-perturbed_totals))
    
    # Build sequence greedily
    seq = [job_order[0]]
    
    for job in job_order[1:]:
        best_pos = 0
        best_ms = float('inf')
        for pos in range(len(seq) + 1):
            ms = _eval_insertion_makespan(seq, job, pos, processing_times)
            if ms < best_ms:
                best_ms = ms
                best_pos = pos
        seq = seq[:best_pos] + [job] + seq[best_pos:]
    
    return seq


def _fast_local_search(order, processing_times, time_limit=50.0):
    """
    Fast local search using insertion moves with early termination.
    """
    start_time = time.time()
    n_jobs = len(order)
    if n_jobs <= 1:
        return order, _makespan(order, processing_times)
    
    best_order = list(order)
    best_ms = _makespan(best_order, processing_times)
    
    while time.time() - start_time < time_limit:
        improved = False
        
        # Try all possible insertion moves
        for i in range(n_jobs):
            if time.time() - start_time >= time_limit:
                break
            
            job = best_order[i]
            remaining = best_order[:i] + best_order[i+1:]
            
            for pos in range(len(remaining) + 1):
                if pos == i or pos == i + 1:
                    continue  # Skip no-op moves
                
                ms = _eval_insertion_makespan(remaining, job, pos, processing_times)
                if ms < best_ms - 1e-9:
                    best_order = remaining[:pos] + [job] + remaining[pos:]
                    best_ms = ms
                    improved = True
                    # Restart the i loop with new position
                    i = pos
                    break
        
        if not improved:
            break
    
    return best_order, best_ms


def _block_decomposition(processing_times, n_blocks=None):
    """
    Partition jobs into blocks based on processing time similarity.
    Returns list of blocks, each a list of job indices.
    """
    n_jobs = processing_times.shape[0]
    if n_jobs == 0:
        return []
    
    if n_blocks is None:
        n_blocks = max(2, int(np.sqrt(n_jobs)))
    
    # Compute job profiles
    totals = processing_times.sum(axis=1)
    means = processing_times.mean(axis=1)
    
    # Sort by total processing time
    sorted_jobs = np.argsort(totals)
    
    # Divide into roughly equal-sized blocks
    block_size = n_jobs // n_blocks
    blocks = []
    for b in range(n_blocks):
        start = b * block_size
        end = start + block_size if b < n_blocks - 1 else n_jobs
        blocks.append(sorted_jobs[start:end].tolist())
    
    return blocks


def _optimize_blocks(blocks, processing_times, time_limit=50.0):
    """
    Optimize internal order of each block using NEH, then optimize block order.
    """
    start_time = time.time()
    
    # Step 1: Optimize internal order of each block
    optimized_blocks = []
    for block in blocks:
        if len(block) <= 1:
            optimized_blocks.append(block)
            continue
        
        # Create sub-matrix for this block
        block_jobs = np.array(block)
        sub_pt = processing_times[block_jobs]
        
        # Use NEH on the sub-problem
        sub_order = _neh_initialization(sub_pt)
        # Map back to original job indices
        optimized_blocks.append([block_jobs[j] for j in sub_order])
    
    # Step 2: Determine optimal block order using 2-opt
    n_blocks = len(optimized_blocks)
    if n_blocks <= 1:
        return optimized_blocks[0] if n_blocks == 1 else []
    
    # Compute block makespan contribution
    block_totals = np.array([
        sum(processing_times[j] for j in block) for block in optimized_blocks
    ])
    
    # Start with blocks in current order
    block_order = list(range(n_blocks))
    
    # 2-opt on block order
    def _block_makespan(order_of_blocks):
        full_order = []
        for b_idx in order_of_blocks:
            full_order.extend(optimized_blocks[b_idx])
        return _makespan(full_order, processing_times)
    
    best_block_order = list(block_order)
    best_ms = _block_makespan(best_block_order)
    
    while time.time() - start_time < time_limit:
        improved = False
        for i in range(n_blocks - 1):
            for j in range(i + 1, n_blocks):
                if time.time() - start_time >= time_limit:
                    break
                
                # Reverse blocks i..j
                new_order = best_block_order[:]
                new_order[i:j+1] = reversed(new_order[i:j+1])
                
                ms = _block_makespan(new_order)
                if ms < best_ms - 1e-9:
                    best_block_order = new_order
                    best_ms = ms
                    improved = True
        
        if not improved:
            break
    
    # Reconstruct final order
    final_order = []
    for b_idx in best_block_order:
        final_order.extend(optimized_blocks[b_idx])
    
    return final_order


def order_jobs(processing_times):
    """
    Main function: construct a job permutation minimizing makespan.
    Uses multiple NEH restarts, block decomposition, and local search.
    """
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin (5s before 120s limit)
    
    n_jobs, n_machines = processing_times.shape
    if n_jobs == 0:
        return []
    if n_jobs == 1:
        return [0]
    
    best_order = None
    best_ms = float('inf')
    
    # Strategy 1: Multiple NEH restarts with local search
    n_restarts = 5
    time_per_restart = (deadline - start_time - 10.0) / max(1, n_restarts)
    
    for r in range(n_restarts):
        if time.time() >= deadline - 10.0:
            break
        
        # Generate diverse initial solution
        seed = r * 1000 + 42
        initial = _neh_initialization(processing_times, seed=seed)
        
        # Apply local search
        remaining = min(time_per_restart, deadline - time.time() - 10.0)
        if remaining > 5.0:
            improved, ms = _fast_local_search(initial, processing_times, time_limit=remaining)
            if ms < best_ms:
                best_ms = ms
                best_order = improved
    
    # Strategy 2: Block-based decomposition (if time remains)
    if time.time() < deadline - 5.0 and n_jobs >= 4:
        remaining = deadline - time.time() - 5.0
        blocks = _block_decomposition(processing_times)
        block_solution = _optimize_blocks(blocks, processing_times, time_limit=remaining)
        
        if block_solution and len(block_solution) == n_jobs:
            ms = _makespan(block_solution, processing_times)
            if ms < best_ms:
                best_ms = ms
                best_order = block_solution
    
    # Final polish on best solution found
    if best_order is not None and time.time() < deadline - 2.0:
        remaining = deadline - time.time() - 2.0
        if remaining > 2.0:
            polished, ms = _fast_local_search(best_order, processing_times, time_limit=remaining)
            if ms < best_ms:
                best_ms = ms
                best_order = polished
    
    if best_order is None:
        # Fallback: simple NEH
        best_order = _neh_initialization(processing_times)
    
    return best_order
# EVOLVE-BLOCK-END
