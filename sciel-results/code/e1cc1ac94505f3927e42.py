"""Flow-shop makespan minimisation using NEH + fast insertion local search with time budget."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import random


def _makespan_from(processing_times, order, start_idx, completion_prev):
    """
    Compute makespan for order[start_idx:] given completion_prev (array of length M)
    which holds the completion times on each machine just before job order[start_idx].
    Returns (final_completion_array, makespan).
    """
    n_machines = processing_times.shape[1]
    c = completion_prev.copy()
    for i in range(start_idx, len(order)):
        job = order[i]
        p_row = processing_times[job]
        if n_machines > 0:
            # c[0] = c[0] + p[0]
            c[0] = c[0] + p_row[0]
            for j in range(1, n_machines):
                c[j] = max(c[j-1], c[j]) + p_row[j]
    return c, c[-1]


def _makespan(processing_times, order):
    """Compute makespan for a full permutation."""
    _, mk = _makespan_from(processing_times, order, 0, np.zeros(processing_times.shape[1]))
    return mk


def _neh_rule(processing_times):
    """NEH heuristic: sort by total processing time descending, insert each job at best position."""
    n_jobs, n_machines = processing_times.shape
    totals = processing_times.sum(axis=1)
    # Sort by total processing time descending
    order = list(np.argsort(-totals))
    
    # Build sequence by inserting each job at the position minimizing makespan
    seq = [order[0]]
    for job in order[1:]:
        best_pos = 0
        best_mk = float('inf')
        for pos in range(len(seq) + 1):
            trial = seq[:pos] + [job] + seq[pos:]
            # Quick makespan computation for trial
            c = np.zeros(n_machines)
            for j_idx in trial:
                p_row = processing_times[j_idx]
                if n_machines > 0:
                    c[0] = c[0] + p_row[0]
                    for j in range(1, n_machines):
                        c[j] = max(c[j-1], c[j]) + p_row[j]
            mk = c[-1] if n_machines > 0 else 0
            if mk < best_mk:
                best_mk = mk
                best_pos = pos
        seq.insert(best_pos, job)
    return seq


def _insertion_local_search(processing_times, seq, time_limit, deadline):
    """
    First-improvement insertion local search with incremental makespan update.
    Returns best sequence found and its makespan.
    """
    n_jobs, n_machines = processing_times.shape
    current_mk = _makespan(processing_times, seq)
    best_seq = list(seq)
    best_mk = current_mk
    improved = True
    
    while improved and time.time() < deadline:
        improved = False
        # Compute completion times for each prefix to enable fast insertion delta
        # c_prefix[i] = completion array after processing seq[0..i-1]
        # For insertion at position pos (0-indexed, meaning insert before seq[pos]):
        # We need completion times up to pos-1, then process the inserted job, then the rest.
        
        for pos in range(1, n_jobs):
            if time.time() >= deadline:
                break
            for i in range(pos):
                # Try inserting seq[i] at position pos (i < pos, so we're moving it right)
                # Remove seq[i] from position i, insert at position pos
                # The sequence becomes: seq[0..i-1], seq[i+1..pos], seq[i], seq[pos+1..]
                
                # We can compute makespan incrementally:
                # Pre-compute prefix completions up to i-1
                # Then process seq[i+1..pos], then seq[i], then seq[pos+1..]
                
                # For efficiency, let's just compute full makespan but with early termination
                # Actually, let's precompute prefix completion arrays once per iteration
                
                pass
        
        # Simpler approach: for each pair (i, j) with i < j, compute delta by
        # computing the makespan of the modified sequence
        # To keep it fast, we limit to adjacent insertions first, then broader
        
        for i in range(n_jobs - 1):
            if time.time() >= deadline:
                break
            for j in range(i + 1, n_jobs):
                if time.time() >= deadline:
                    break
                # Move seq[i] to position j (insert after seq[j])
                # New seq: seq[0..i-1], seq[i+1..j], seq[i], seq[j+1..]
                trial = list(seq)
                job = trial.pop(i)
                trial.insert(j, job)
                
                # Compute makespan for trial
                c = np.zeros(n_machines)
                for job_idx in trial:
                    p_row = processing_times[job_idx]
                    if n_machines > 0:
                        c[0] = c[0] + p_row[0]
                        for m in range(1, n_machines):
                            c[m] = max(c[m-1], c[m]) + p_row[m]
                mk = c[-1] if n_machines > 0 else 0
                
                if mk < current_mk - 1e-9:
                    seq = trial
                    current_mk = mk
                    improved = True
                    if current_mk < best_mk:
                        best_mk = current_mk
                        best_seq = list(seq)
                    break  # restart from beginning of j loop for this i
            if improved and time.time() >= deadline:
                break
    
    return best_seq, best_mk


def _simulated_annealing(processing_times, seq, current_mk, time_limit, deadline):
    """Simple SA with insertion moves."""
    n_jobs, n_machines = processing_times.shape
    best_seq = list(seq)
    best_mk = current_mk
    T = 1.0
    
    while time.time() < deadline:
        # Random insertion move
        i = random.randint(0, n_jobs - 1)
        j = random.randint(0, n_jobs - 1)
        if j == i:
            continue
        trial = list(seq)
        job = trial.pop(i)
        trial.insert(j, job)
        
        c = np.zeros(n_machines)
        for job_idx in trial:
            p_row = processing_times[job_idx]
            if n_machines > 0:
                c[0] = c[0] + p_row[0]
                for m in range(1, n_machines):
                    c[m] = max(c[m-1], c[m]) + p_row[m]
        trial_mk = c[-1] if n_machines > 0 else 0
        
        delta = trial_mk - current_mk
        if delta < 0 or random.random() < np.exp(-delta / max(T, 1e-9)):
            seq = trial
            current_mk = trial_mk
            if current_mk < best_mk:
                best_mk = current_mk
                best_seq = list(seq)
        
        T *= 0.999
    
    return best_seq, best_mk


def order_jobs(processing_times):
    processing_times = np.asarray(processing_times, dtype=float)
    n_jobs, n_machines = processing_times.shape
    
    if n_jobs <= 1:
        return list(range(n_jobs))
    
    deadline = time.time() + 115  # Safety margin before 120s limit
    
    # Start with NEH
    seq = _neh_rule(processing_times)
    current_mk = _makespan(processing_times, seq)
    
    # Insertion local search
    seq, current_mk = _insertion_local_search(processing_times, seq, time_limit=1, deadline=deadline)
    
    # If time remains, do some SA
    if time.time() < deadline - 10:
        seq, current_mk = _simulated_annealing(processing_times, seq, current_mk, time_limit=1, deadline=deadline)
    
    # Final local search polish
    seq, current_mk = _insertion_local_search(processing_times, seq, time_limit=1, deadline=deadline)
    
    return list(seq)
# EVOLVE-BLOCK-END
