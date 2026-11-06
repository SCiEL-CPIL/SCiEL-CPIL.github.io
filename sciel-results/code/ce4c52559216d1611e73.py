"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import random


def _makespan(order, pt_list):
    """
    Compute makespan for a given job order.
    pt_list is a list of lists for faster access.
    """
    n_machines = len(pt_list[0]) if pt_list else 0
    if n_machines == 0 or len(order) == 0:
        return 0.0
    
    completion = [0.0] * n_machines
    
    for job in order:
        p = pt_list[job]
        prev = 0.0
        for m in range(n_machines):
            if prev < completion[m]:
                prev = completion[m]
            prev += p[m]
            completion[m] = prev
    return completion[-1]


def _compute_completion_vector(order, pt_list):
    """
    Compute the completion time vector (one entry per machine) for a given order.
    """
    n_machines = len(pt_list[0]) if pt_list else 0
    if n_machines == 0 or len(order) == 0:
        return [0.0] * n_machines
    
    completion = [0.0] * n_machines
    
    for job in order:
        p = pt_list[job]
        prev = 0.0
        for m in range(n_machines):
            if prev < completion[m]:
                prev = completion[m]
            prev += p[m]
            completion[m] = prev
    return completion


def _makespan_from_completion(completion):
    """Get makespan from completion vector."""
    return completion[-1] if completion else 0.0


def _neh_block_internal(block_jobs, pt_list, n_machines):
    """
    Optimize the internal order of jobs within a block using NEH heuristic.
    """
    if len(block_jobs) <= 1:
        return list(block_jobs)
    
    # Sort by total processing time (descending)
    totals = [sum(pt_list[j]) for j in block_jobs]
    sorted_indices = sorted(range(len(block_jobs)), key=lambda i: totals[i], reverse=True)
    
    seq = [block_jobs[sorted_indices[0]]]
    
    for idx in sorted_indices[1:]:
        job = block_jobs[idx]
        best_pos = 0
        best_ms = float('inf')
        for pos in range(len(seq) + 1):
            new_seq = seq[:pos] + [job] + seq[pos:]
            ms = _makespan(new_seq, pt_list)
            if ms < best_ms:
                best_ms = ms
                best_pos = pos
        seq = seq[:best_pos] + [job] + seq[best_pos:]
    
    return seq


def _partition_into_blocks(processing_times, n_blocks=None, seed=None):
    """
    Partition jobs into contiguous blocks based on processing time similarity.
    Uses a greedy clustering approach based on Euclidean distance in processing time space.
    """
    n_jobs, n_machines = processing_times.shape
    if n_jobs == 0:
        return []
    if n_jobs <= 3:
        return [list(range(n_jobs))]
    
    if n_blocks is None:
        # Heuristic: aim for blocks of size 3-7
        target_size = max(3, min(7, n_jobs // 4))
        n_blocks = max(2, n_jobs // target_size)
    
    if seed is not None:
        rng = random.Random(seed)
    else:
        rng = random.Random()
    
    # Compute pairwise distances
    pt = processing_times
    # Normalize for distance computation
    pt_norm = pt / (pt.max(axis=0) + 1e-9)
    
    # Greedy clustering: start with job 0, add most similar job to current block
    used = [False] * n_jobs
    blocks = []
    
    # Start with the job with highest total processing time
    totals = pt.sum(axis=1)
    start_job = int(np.argmax(totals))
    used[start_job] = True
    current_block = [start_job]
    
    for _ in range(n_blocks - 1):
        # Find the most similar unused job to the current block's centroid
        block_pt = pt[current_block]
        centroid = block_pt.mean(axis=0)
        
        best_job = -1
        best_dist = float('inf')
        
        for j in range(n_jobs):
            if used[j]:
                continue
            dist = np.linalg.norm(pt_norm[j] - centroid / (pt.max(axis=0) + 1e-9))
            if dist < best_dist:
                best_dist = dist
                best_job = j
        
        if best_job == -1:
            break
        
        used[best_job] = True
        current_block.append(best_job)
        
        # If block is large enough, finalize it
        if len(current_block) >= max(2, n_jobs // n_blocks):
            blocks.append(current_block)
            current_block = []
        
        # If no jobs left, finalize
        if all(used):
            break
    
    # Add remaining jobs to the last block or create new blocks
    remaining = [j for j in range(n_jobs) if not used[j]]
    if remaining:
        if blocks:
            blocks[-1].extend(remaining)
        else:
            blocks.append(remaining)
    
    # Ensure we have at least 2 blocks if possible
    if len(blocks) == 1 and n_jobs > 3:
        # Split the single block
        mid = len(blocks[0]) // 2
        blocks = [blocks[0][:mid], blocks[0][mid:]]
    
    return blocks


def _optimize_block_order(blocks, pt_list, n_machines, time_limit=10.0):
    """
    Optimize the order of blocks using insertion-based local search.
    Each block is treated as a unit; we optimize the sequence of blocks.
    """
    start_time = time.time()
    n_blocks = len(blocks)
    
    if n_blocks <= 1:
        return [0]
    
    # Initial order: sort blocks by their total processing time (descending)
    block_totals = []
    for b in blocks:
        total = sum(sum(pt_list[j]) for j in b)
        block_totals.append(total)
    
    block_order = list(range(n_blocks))
    block_order.sort(key=lambda i: block_totals[i], reverse=True)
    
    # Compute makespan for a given block order
    def _block_order_makespan(b_order):
        # Build full job order from block order
        full_order = []
        for bi in b_order:
            full_order.extend(blocks[bi])
        return _makespan(full_order, pt_list)
    
    best_order = list(block_order)
    best_ms = _block_order_makespan(best_order)
    
    improved = True
    while improved and (time.time() - start_time < time_limit):
        improved = False
        for i in range(n_blocks):
            if time.time() - start_time >= time_limit:
                break
            for j in range(n_blocks):
                if j == i or j == i + 1:
                    continue
                if j < i:
                    new_order = best_order[:j] + [best_order[i]] + best_order[j:i] + best_order[i+1:]
                else:
                    new_order = best_order[:i] + best_order[i+1:j+1] + [best_order[i]] + best_order[j+1:]
                
                ms = _block_order_makespan(new_order)
                if ms < best_ms - 1e-9:
                    best_order = new_order
                    best_ms = ms
                    improved = True
                    break
    
    return best_order


def _local_search(order, pt_list, time_limit=50.0):
    """
    Local search using insertion moves with First Improvement strategy.
    """
    start_time = time.time()
    n_jobs = len(order)
    if n_jobs <= 1:
        return order, _makespan(order, pt_list)
    
    best_order = list(order)
    best_ms = _makespan(best_order, pt_list)
    
    improved = True
    while improved and (time.time() - start_time < time_limit):
        improved = False
        for i in range(n_jobs):
            if time.time() - start_time >= time_limit:
                return best_order, best_ms
            
            job = best_order[i]
            
            for j in range(n_jobs):
                if j == i or j == i + 1:
                    continue
                
                if j < i:
                    new_order = best_order[:j] + [job] + best_order[j:i] + best_order[i+1:]
                else:
                    new_order = best_order[:i] + best_order[i+1:j+1] + [job] + best_order[j+1:]
                
                ms = _makespan(new_order, pt_list)
                
                if ms < best_ms - 1e-9:
                    best_order = new_order
                    best_ms = ms
                    improved = True
                    break
            
            if improved:
                pass

    return best_order, best_ms


def order_jobs(processing_times):
    """
    Main function: construct a job permutation minimizing makespan.
    Uses block-based decomposition: partition jobs into blocks, optimize internal
    block orderings, then optimize block order, followed by full local search.
    """
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin
    
    n_jobs, n_machines = processing_times.shape
    if n_jobs == 0:
        return []
    if n_jobs == 1:
        return [0]
    
    pt_list = processing_times.tolist()
    
    best_order = None
    best_ms = float('inf')
    
    # Determine number of restarts
    n_restarts = 15 if n_jobs < 30 else 8
    
    for r in range(n_restarts):
        if time.time() >= deadline - 5.0:
            break
        
        seed = r * 12345 + 99
        
        # Step 1: Partition into blocks
        blocks = _partition_into_blocks(processing_times, seed=seed)
        
        # Step 2: Optimize internal order of each block
        optimized_blocks = []
        for block in blocks:
            opt_block = _neh_block_internal(block, pt_list, n_machines)
            optimized_blocks.append(opt_block)
        
        # Step 3: Optimize block order
        remaining_for_block = deadline - time.time() - 30.0
        if remaining_for_block > 5.0:
            block_order = _optimize_block_order(optimized_blocks, pt_list, n_machines, 
                                                 time_limit=min(remaining_for_block, 15.0))
        else:
            # Fallback: sort by total processing time
            block_totals = [sum(sum(pt_list[j]) for j in b) for b in optimized_blocks]
            block_order = sorted(range(len(optimized_blocks)), key=lambda i: block_totals[i], reverse=True)
        
        # Step 4: Build full order from block order
        full_order = []
        for bi in block_order:
            full_order.extend(optimized_blocks[bi])
        
        # Step 5: Full local search to refine
        remaining_for_ls = deadline - time.time() - 2.0
        if remaining_for_ls > 1.0:
            improved, ms = _local_search(full_order, pt_list, time_limit=remaining_for_ls)
            if ms < best_ms:
                best_ms = ms
                best_order = improved
        else:
            ms = _makespan(full_order, pt_list)
            if ms < best_ms:
                best_ms = ms
                best_order = full_order
    
    if best_order is None:
        # Fallback: simple NEH
        totals = processing_times.sum(axis=1)
        best_order = list(np.argsort(-totals))
    
    return best_order
# EVOLVE-BLOCK-END
