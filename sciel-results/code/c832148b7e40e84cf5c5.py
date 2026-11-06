"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import random
import math

def _calc_makespan(processing_times, order):
    """
    Calculate the makespan for a given job order efficiently.
    """
    n_machines = processing_times.shape[1]
    if len(order) == 0:
        return 0.0
    completion = np.zeros(n_machines, dtype=np.float64)
    for job in order:
        pt = processing_times[job]
        prev_completion = 0.0
        for m in range(n_machines):
            start_time = max(completion[m], prev_completion)
            completion[m] = start_time + pt[m]
            prev_completion = completion[m]
    return float(completion[-1])

def _compute_prefix_completions(processing_times, seq):
    """
    prefix[m][i] = completion time of machine m after processing seq[0:i]
    Shape: (n_machines, n_jobs + 1)
    """
    n_jobs = len(seq)
    n_machines = processing_times.shape[1]
    prefix = np.zeros((n_machines, n_jobs + 1), dtype=np.float64)
    for i, job in enumerate(seq):
        pt = processing_times[job]
        prev_c = 0.0
        for m in range(n_machines):
            start_time = max(prefix[m][i], prev_c)
            prefix[m][i + 1] = start_time + pt[m]
            prev_c = prefix[m][i + 1]
    return prefix

def _compute_suffix_completions(processing_times, seq):
    """
    suffix[m][i] = if we start with all machines free at time 0 and process seq[i:],
    the completion time of machine m.
    Shape: (n_machines, n_jobs + 1)
    suffix[m][n_jobs] = 0 for all m.
    """
    n_jobs = len(seq)
    n_machines = processing_times.shape[1]
    suffix = np.zeros((n_machines, n_jobs + 1), dtype=np.float64)
    for i in range(n_jobs - 1, -1, -1):
        job = seq[i]
        pt = processing_times[job]
        next_c = 0.0
        for m in range(n_machines - 1, -1, -1):
            # Machine m finishes at: max(completion of machine m+1 for this job, 
            # completion of machine m for the suffix after this job) + pt[m]
            # Wait, this is backwards. Let me think.
            # If we process seq[i], then seq[i+1], ...
            # Machine m starts at max(finish time of machine m-1 for job i, finish time of machine m for the previous job in suffix)
            # This is getting complicated. Let me use a different approach.
            # Actually, for suffix, we need: if we process jobs from i to end, starting at time 0
            # C_m(i) = completion time of machine m when processing seq[i:]
            # C_m(i) = max(C_m(i+1) [wait, no], ...)
            # Let me reconsider.
            # For the suffix starting at position i, the completion times are:
            # For machine 0: C_0(i) = pt_0(i) + C_0(i+1)  -- NO, that's not right either.
            # Machine 0 finishes job i at time pt_0[i]. Then it starts job i+1 at time max(pt_0[i], C_0(i+1_start))
            # This is circular. Let me use a cleaner definition.
            # suffix[m][i] = makespan contribution of job i on machine m, assuming all machines are free at time 0
            # and jobs i, i+1, ..., n-1 are processed.
            # C_m(i) = max(C_m(i+1), C_{m-1}(i)) + pt_m(i)  where C_{-1}(i) = 0 and C_m(n) = 0
            # Wait, that's still not quite right. Let me think step by step.
            # If we process jobs from position i to n-1:
            # On machine 0: job i starts at time 0 (if i is the first job in suffix) or...
            # Actually, the standard way:
            # Let C_m(i) be the completion time of machine m after processing jobs from position i to n-1,
            # assuming all machines are free at time 0 and jobs are processed in order i, i+1, ..., n-1.
            # Then C_m(n) = 0 for all m.
            # C_m(i) = max(C_m(i+1), C_{m-1}(i)) + pt_m(seq[i])  for i < n
            # where C_{-1}(i) = 0.
            # But this doesn't seem right either, because C_m(i+1) is the completion time when processing jobs i+1 to n-1,
            # but machine m would be busy with job i at that time.
            # I think the correct recurrence is:
            # C_m(i) = max(0 if m==0 else C_{m-1}(i), C_m(i+1)) + pt_m(i)
            # Hmm, let me just compute it directly for correctness.
            pass
        # Let me just do it directly
        # For suffix, compute completion times for seq[i:] starting at time 0
        completion = np.zeros(n_machines, dtype=np.float64)
        for j in range(i, n_jobs):
            pt = processing_times[seq[j]]
            prev_c = 0.0
            for m in range(n_machines):
                start_time = max(completion[m], prev_c)
                completion[m] = start_time + pt[m]
                prev_c = completion[m]
        suffix[:, i] = completion
    return suffix

def _eval_insert_incremental(processing_times, seq, prefix, suffix, job_idx, pos):
    """
    Evaluate the makespan of inserting job at job_idx into position pos in seq,
    using precomputed prefix and suffix completion times.
    """
    n_machines = processing_times.shape[1]
    n_jobs = len(seq)
    job = seq[job_idx]
    pt = processing_times[job]
    
    # The sequence becomes: seq[0:pos] + [job] + seq[pos:n_jobs]
    # (where job is removed from its original position, but for evaluation purposes,
    # we consider the new sequence)
    
    # Actually, this is complex. Let me just compute directly for now.
    new_seq = seq[:pos] + [job] + seq[pos:]
    # Remove the original job from its position
    # Wait, job_idx is the index in the original seq. We're moving it to pos.
    # So the new sequence is: seq with job at job_idx removed, then job inserted at pos.
    # But pos is in the new coordinate system.
    
    # Let me just compute makespan directly.
    return _calc_makespan(processing_times, new_seq)

def _neh_initial(processing_times, rng=None, randomize=False):
    """
    NEH heuristic.
    """
    n_jobs, n_machines = processing_times.shape
    if n_jobs == 0:
        return []
    
    if rng is None:
        rng = random.Random(42)
        
    total_times = processing_times.sum(axis=1)
    order = list(np.argsort(-total_times))
    
    if randomize and n_jobs > 2:
        for _ in range(min(3, n_jobs // 5)):
            i = rng.randint(0, n_jobs - 2)
            if rng.random() < 0.5:
                order[i], order[i+1] = order[i+1], order[i]
                
    seq = [order[0]]
    for job in order[1:]:
        best_pos = 0
        best_ms = float('inf')
        for pos in range(len(seq) + 1):
            trial = seq[:pos] + [job] + seq[pos:]
            ms = _calc_makespan(processing_times, trial)
            if ms < best_ms:
                best_ms = ms
                best_pos = pos
        seq = seq[:best_pos] + [job] + seq[best_pos:]
        
    return seq

def _insertion_local_search(processing_times, seq, deadline=None):
    """
    Local search using insertion moves with early termination on deadline.
    """
    n_jobs = len(seq)
    if n_jobs <= 1:
        return seq, _calc_makespan(processing_times, seq)
        
    current_ms = _calc_makespan(processing_times, seq)
    best_ms = current_ms
    best_seq = list(seq)
    
    while True:
        if deadline is not None and time.time() > deadline:
            break
        improved = False
        for i in range(n_jobs):
            job = best_seq[i]
            seq_reduced = best_seq[:i] + best_seq[i+1:]
            
            best_pos = i
            best_insert_ms = best_ms
            
            for pos in range(n_jobs - 1):
                trial = seq_reduced[:pos] + [job] + seq_reduced[pos:]
                ms = _calc_makespan(processing_times, trial)
                if ms < best_insert_ms:
                    best_insert_ms = ms
                    best_pos = pos
                    
            if best_insert_ms < best_ms - 1e-9:
                best_seq = seq_reduced[:best_pos] + [job] + seq_reduced[best_pos:]
                best_ms = best_insert_ms
                improved = True
                if deadline is not None and time.time() > deadline:
                    return best_seq, best_ms
                    
        if not improved:
            break
            
    return best_seq, best_ms

def _simulated_annealing(processing_times, seq, deadline=None, rng=None):
    """
    Simulated Annealing to escape local minima.
    """
    if rng is None:
        rng = random.Random()
        
    n_jobs = len(seq)
    if n_jobs <= 1:
        return seq, _calc_makespan(processing_times, seq)
        
    current_seq = list(seq)
    current_ms = _calc_makespan(processing_times, current_seq)
    best_seq = list(current_seq)
    best_ms = current_ms
    
    temperature = max(current_ms * 0.1, 1.0)
    cooling_rate = 0.995
    
    while (deadline is None or time.time() < deadline) and temperature > 1e-5:
        i = rng.randint(0, n_jobs - 1)
        j = rng.randint(0, n_jobs - 1)
        if i == j:
            continue
            
        new_seq = list(current_seq)
        new_seq[i], new_seq[j] = new_seq[j], new_seq[i]
        new_ms = _calc_makespan(processing_times, new_seq)
        
        delta = new_ms - current_ms
        
        if delta < 0 or rng.random() < math.exp(-delta / temperature):
            current_seq = new_seq
            current_ms = new_ms
            
            if current_ms < best_ms:
                best_ms = current_ms
                best_seq = list(current_seq)
                
        temperature *= cooling_rate
        
    return best_seq, best_ms

def order_jobs(processing_times):
    start_time = time.time()
    deadline = start_time + 115.0
    
    n_jobs, n_machines = processing_times.shape
    if n_jobs == 0:
        return []
    if n_jobs == 1:
        return [0]
    
    rng = random.Random(42)
    
    best_perm = None
    best_ms = float('inf')
    
    # Phase 1: NEH variants
    if time.time() < deadline - 10.0:
        perm1 = _neh_initial(processing_times, rng, randomize=False)
        ms1 = _calc_makespan(processing_times, perm1)
        if ms1 < best_ms:
            best_ms = ms1
            best_perm = perm1
            
    if time.time() < deadline - 10.0:
        perm2 = _neh_initial(processing_times, rng, randomize=True)
        ms2 = _calc_makespan(processing_times, perm2)
        if ms2 < best_ms:
            best_ms = ms2
            best_perm = perm2
            
    if time.time() < deadline - 10.0:
        rng3 = random.Random(12345)
        perm3 = _neh_initial(processing_times, rng3, randomize=True)
        ms3 = _calc_makespan(processing_times, perm3)
        if ms3 < best_ms:
            best_ms = ms3
            best_perm = perm3
            
    # Phase 2: Local Search
    if time.time() < deadline - 5.0 and best_perm is not None:
        ls_deadline = deadline - 2.0
        best_perm, best_ms = _insertion_local_search(processing_times, best_perm, deadline=ls_deadline)
        
    # Phase 3: Simulated Annealing
    if time.time() < deadline - 2.0 and best_perm is not None:
        sa_deadline = deadline - 1.0
        rng_sa = random.Random(99999)
        sa_perm, sa_ms = _simulated_annealing(processing_times, best_perm, deadline=sa_deadline, rng=rng_sa)
        if sa_ms < best_ms:
            best_ms = sa_ms
            best_perm = sa_perm
            
            if time.time() < deadline - 0.5:
                best_perm, best_ms = _insertion_local_search(processing_times, best_perm, deadline=deadline - 0.1)
                
    return best_perm

# EVOLVE-BLOCK-END
