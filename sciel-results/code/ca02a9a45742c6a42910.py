"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def _makespan(perm, pt):
    """Compute makespan for a given permutation of job indices."""
    n_jobs, n_machines = pt.shape
    if n_machines == 0 or n_jobs == 0:
        return 0.0
    # C[job_position][machine]
    # C[i][0] = sum_{k=0}^{i} pt[perm[k], 0]
    # C[i][m] = max(C[i][m-1], C[i-1][m]) + pt[perm[i], m]
    # We only need the last row's last element for makespan.
    # But we need the previous row for the recurrence.
    
    # For efficiency, we can compute row by row.
    # Let C_prev[m] be the completion time of the previous job on machine m.
    # Let C_curr[m] be the completion time of the current job on machine m.
    
    C_prev = np.zeros(n_machines)
    for i in range(n_jobs):
        job = perm[i]
        # C_curr[0] = C_prev[0] + pt[job, 0]  (since C_curr[-1][0] is effectively 0 for the first machine of the first job, but for i>0, C_prev[0] is the finish time of previous job on machine 0)
        # Actually, the recurrence is:
        # C[i][0] = C[i-1][0] + pt[perm[i], 0]
        # C[i][m] = max(C[i][m-1], C[i-1][m]) + pt[perm[i], m] for m > 0
        
        C_curr = np.zeros(n_machines)
        C_curr[0] = C_prev[0] + pt[job, 0]
        for m in range(1, n_machines):
            C_curr[m] = max(C_curr[m-1], C_prev[m]) + pt[job, m]
        C_prev = C_curr
    return C_prev[-1]


def order_jobs(processing_times):
    """
    Construct a job permutation minimizing makespan on local synthetic flow-shop instances.
    """
    try:
        start_time = time.time()
        deadline = 145.0  # Safety margin before 150s limit
        
        pt = np.asarray(processing_times, dtype=np.float64)
        if pt.ndim != 2:
            # Fallback for unexpected shapes
            if pt.size == 0:
                return []
            pt = pt.reshape(-1, 1)
            
        n_jobs, n_machines = pt.shape
        
        if n_jobs == 0:
            return []
        if n_jobs == 1:
            return [0]
        
        # 1. Initial heuristic: NEH (Nawaz-Enscore-Ham)
        # Sort jobs by total processing time in descending order.
        # Insert each job into the position in the current partial permutation that minimizes the makespan of the partial schedule.
        
        sums = pt.sum(axis=1)
        order = np.argsort(-sums)
        
        perm = [int(order[0])]
        for idx in range(1, n_jobs):
            job = int(order[idx])
            best_pos = 0
            best_makespan = float('inf')
            # Try inserting job at every position in the current perm
            for pos in range(len(perm) + 1):
                new_perm = perm[:pos] + [job] + perm[pos:]
                ms = _makespan(new_perm, pt)
                if ms < best_makespan:
                    best_makespan = ms
                    best_pos = pos
            perm.insert(best_pos, job)
        
        best_perm = perm[:]
        best_ms = _makespan(best_perm, pt)
        
        # 2. Local Search: Insertion neighborhood
        # Try moving each job to every other position.
        # Use a simple hill-climbing or tabu-like approach with time limit.
        
        improved = True
        while improved and (time.time() - start_time) < deadline:
            improved = False
            for i in range(n_jobs):
                if (time.time() - start_time) > deadline:
                    break
                job = best_perm[i]
                # Remove job from position i
                temp_perm = best_perm[:i] + best_perm[i+1:]
                # Try inserting at every position j in temp_perm
                for j in range(len(temp_perm) + 1):
                    if j == i:
                        continue
                    new_perm = temp_perm[:j] + [job] + temp_perm[j:]
                    ms = _makespan(new_perm, pt)
                    if ms < best_ms:
                        best_ms = ms
                        best_perm = new_perm
                        improved = True
                        # Restart the loop to check for further improvements from the new state
                        # Or just continue searching. Let's break and restart for simplicity of logic.
                        break
                if improved:
                    break
            # If no improvement in a full pass, we could try a perturbation to escape local optima,
            # but simple hill climbing often works well for these synthetic instances.
            # Let's add a simple 2-opt style swap if no improvement found.
            if not improved and (time.time() - start_time) < deadline:
                # Try swaps
                swapped = False
                for i in range(n_jobs - 1):
                    if (time.time() - start_time) > deadline:
                        break
                    for j in range(i + 1, n_jobs):
                        new_perm = best_perm[:]
                        new_perm[i], new_perm[j] = new_perm[j], new_perm[i]
                        ms = _makespan(new_perm, pt)
                        if ms < best_ms:
                            best_ms = ms
                            best_perm = new_perm
                            swapped = True
                            improved = True
                            break
                    if improved:
                        break
                if not improved:
                    # No improvement in insertion or swap, stop.
                    # We could add more complex neighborhoods, but let's stick to this for time.
                    break
        
        return list(best_perm)
    except Exception:
        # Fallback: return jobs sorted by total processing time
        try:
            pt = np.asarray(processing_times, dtype=np.float64)
            if pt.ndim != 2:
                if pt.size == 0:
                    return []
                pt = pt.reshape(-1, 1)
            sums = pt.sum(axis=1)
            return list(np.argsort(-sums))
        except Exception:
            return []
# EVOLVE-BLOCK-END
