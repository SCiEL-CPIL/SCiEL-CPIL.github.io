"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import random

def order_jobs(processing_times):
    """
    Construct a job permutation minimizing makespan using multi-start local search.
    Combines NEH, Palmer, LPT, SPT heuristics with 2-opt (segment reversal) local search.
    """
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin before 150s limit
    
    n_jobs, n_machines = processing_times.shape
    
    if n_jobs <= 1:
        return list(range(n_jobs))
    
    # Fast fallback for very large instances
    if n_jobs * n_machines > 500000:
        return list(np.argsort(-processing_times.sum(axis=1)))
    
    def makespan(seq):
        """Compute makespan for a given sequence of job indices."""
        if len(seq) == 0:
            return 0.0
        completion = np.zeros(n_machines)
        for job in seq:
            prev = completion[0]
            curr = prev + processing_times[job, 0]
            completion[0] = curr
            for m in range(1, n_machines):
                prev = curr
                curr = max(prev, completion[m]) + processing_times[job, m]
                completion[m] = curr
        return completion[-1]
    
    def makespan_incremental_update(seq, pos_i, pos_j):
        """
        Quick makespan estimate after reversing segment [i:j].
        For accuracy, we just recompute.
        """
        return makespan(seq)
    
    def local_search_2opt(seq, time_budget):
        """
        2-opt local search: try all segment reversals, accept best improvement.
        Repeat until no improvement or time budget exceeded.
        """
        seq = list(seq)
        n = len(seq)
        best_ms = makespan(seq)
        
        while time.time() < time_budget:
            improved = False
            best_delta = 0.0
            best_i, best_j = -1, -1
            
            # Try all segment reversals
            for i in range(n):
                for j in range(i + 2, n + 1):
                    # Reverse segment [i:j]
                    seq2 = seq[:i] + seq[i:j][::-1] + seq[j:]
                    ms = makespan(seq2)
                    delta = ms - best_ms
                    if delta < best_delta:
                        best_delta = delta
                        best_i, best_j = i, j
            
            if best_delta < -1e-10:
                # Apply the best reversal
                i, j = best_i, best_j
                seq[i:j] = seq[i:j][::-1]
                best_ms += best_delta
                improved = True
            else:
                break
        
        return seq, best_ms
    
    def neh_heuristic():
        """NEH (Nawaz, Enscore, Ham) heuristic."""
        # Sort jobs by total processing time descending
        totals = processing_times.sum(axis=1)
        order = list(np.argsort(-totals))
        
        seq = [order[0]]
        for job in order[1:]:
            best_pos = 0
            best_ms = float('inf')
            for pos in range(len(seq) + 1):
                trial = seq[:pos] + [job] + seq[pos:]
                ms = makespan(trial)
                if ms < best_ms:
                    best_ms = ms
                    best_pos = pos
            seq = seq[:best_pos] + [job] + seq[best_pos:]
        
        return seq
    
    def palmer_heuristic():
        """Palmer's priority rule: sort by (n * first_machine_time - last_machine_time)."""
        n = n_jobs
        priorities = []
        for job in range(n_jobs):
            p_first = processing_times[job, 0]
            p_last = processing_times[job, -1]
            priorities.append(n * p_first - p_last)
        priorities = np.array(priorities)
        return list(np.argsort(-priorities))
    
    def lpt_heuristic():
        """Longest Processing Time first."""
        return list(np.argsort(-processing_times.sum(axis=1)))
    
    def spt_heuristic():
        """Shortest Processing Time first."""
        return list(np.argsort(processing_times.sum(axis=1)))
    
    def random_perm():
        """Random permutation."""
        seq = list(range(n_jobs))
        random.shuffle(seq)
        return seq
    
    # Generate initial solutions from multiple heuristics
    candidates = []
    candidates.append(neh_heuristic())
    candidates.append(palmer_heuristic())
    candidates.append(lpt_heuristic())
    candidates.append(spt_heuristic())
    
    # Add a few random permutations
    n_random = min(3, max(1, n_jobs // 10))
    for _ in range(n_random):
        candidates.append(random_perm())
    
    best_seq = None
    best_ms = float('inf')
    
    # Apply local search to each candidate
    for i, seq in enumerate(candidates):
        if time.time() > deadline:
            break
        
        # Give more time to earlier candidates
        remaining = deadline - time.time()
        time_budget = min(remaining * 0.4, 30.0)
        
        seq_ls, ms = local_search_2opt(seq, time_budget)
        
        if ms < best_ms:
            best_ms = ms
            best_seq = seq_ls
    
    # If we have time left, do a few more random restarts with local search
    if best_seq is not None and time.time() < deadline:
        for _ in range(5):
            if time.time() > deadline:
                break
            
            seq = random_perm()
            remaining = deadline - time.time()
            time_budget = min(remaining * 0.3, 15.0)
            
            seq_ls, ms = local_search_2opt(seq, time_budget)
            
            if ms < best_ms:
                best_ms = ms
                best_seq = seq_ls
    
    if best_seq is None:
        best_seq = lpt_heuristic()
    
    return list(best_seq)
# EVOLVE-BLOCK-END
