<<<<<<< SEARCH
def _calc_makespan(perm, pt):
    """
    Calculate the makespan for a given permutation.
    perm: array of job indices
    pt: processing times (n_jobs, n_machines)
    Returns: makespan (float)
    """
    n_jobs = len(perm)
    n_machines = pt.shape[1]
    if n_jobs == 0:
        return 0.0
    
    # Select processing times for the jobs in the permutation order
    # p[j, m] is processing time of job perm[j] on machine m
    p = pt[perm]
    
    # c[j] stores the completion time of job j on the current machine
    c = np.zeros(n_jobs, dtype=float)
    
    for m in range(n_machines):
        for j in range(n_jobs):
            if j == 0:
                # First job on this machine: starts at 0 (or after previous machine, but c[0] is 0 for m=0)
                # Wait, c[j] currently holds completion time on machine m-1.
                # For m=0, c is zeros, so start is 0.
                # For m>0, c[j] is completion on m-1.
                start_time = c[j]
            else:
                # Must wait for previous job on this machine to finish
                # c[j-1] has already been updated to completion on machine m
                start_time = max(c[j], c[j-1])
            
            c[j] = start_time + p[j, m]
            
    return c[-1]
=======
def _calc_makespan(perm, pt):
    """
    Calculate the makespan for a given permutation.
    perm: array of job indices
    pt: processing times (n_jobs, n_machines)
    Returns: makespan (float)
    """
    n_jobs = len(perm)
    n_machines = pt.shape[1]
    if n_jobs == 0:
        return 0.0
    
    p = pt[perm]
    c = np.zeros(n_jobs, dtype=float)
    
    # Local variable binding for speed
    range_n = range(n_jobs)
    range_m = range(n_machines)
    
    for m in range_m:
        c_prev = c.copy() # We need the previous machine's completion times
        # Actually, we can just use the array c which holds m-1 values before update
        # But we update c in place. So we need the old values for max(c[j], c[j-1])?
        # No, c[j-1] is already updated for machine m.
        # c[j] is still the value for machine m-1.
        # So we need to be careful.
        
        # Let's trace:
        # m=0: c is zeros.
        # j=0: c[0] = max(0, 0) + p[0,0] = p[0,0]
        # j=1: c[1] = max(c[1]_old (0), c[0]_new (p[0,0])) + p[1,0]
        
        # This logic holds if we update c[j] sequentially.
        # c[j] before update is C[j, m-1].
        # c[j-1] after update is C[j-1, m].
        
        for j in range_n:
            if j == 0:
                c[0] = c[0] + p[0, m]
            else:
                # c[j] is C[j, m-1]
                # c[j-1] is C[j-1, m]
                if c[j] > c[j-1]:
                    c[j] = c[j] + p[j, m]
                else:
                    c[j] = c[j-1] + p[j, m]
            
    return c[-1]
>>>>>>> REPLACE

