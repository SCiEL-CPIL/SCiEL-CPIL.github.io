def _makespan(perm, processing_times):
    n_jobs = len(perm)
    n_machines = processing_times.shape[1]
    if n_jobs == 0:
        return 0.0
    
    # Using a list for machine times is faster than a numpy array for small M
    machine_times = [0.0] * n_machines
    pt = processing_times
    for j in perm:
        job_pt = pt[j]
        prev = 0.0
        for m in range(n_machines):
            if machine_times[m] > prev:
                new_t = machine_times[m] + job_pt[m]
            else:
                new_t = prev + job_pt[m]
            machine_times[m] = new_t
            prev = new_t
    return machine_times[-1]
