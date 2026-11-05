def order_jobs(processing_times):
    P = np.asarray(processing_times)
    if P.ndim == 1:
        # If 1D, assume 1 machine or N jobs 1 machine? 
        # The signature says [jobs, machines]. If it's 1D, it's likely [jobs] with 1 machine.
        P = P.reshape(-1, 1)
    elif P.ndim != 2:
        # Handle other cases, though unlikely
        P = P.reshape(P.shape[0], -1)
        
    n, m = P.shape
    if n == 0:
        return np.array([], dtype=int)
    if m == 0:
        # No machines? Makespan is 0? Or invalid? 
        # Return identity permutation
        return np.arange(n)
    # ... rest of code
