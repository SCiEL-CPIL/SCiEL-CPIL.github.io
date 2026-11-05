<<<<<<< SEARCH
    # Family 1: a = cycle, b with controlled defects
    a_cycle = [(i + 1) % N for i in range(N)]
    
    # Single defect: b[i] = i for all i except b[k] = v
    for k in range(N):
        for v in range(N):
            if v == k:
                continue
            if time.time() > deadline:
                break
            b = list(range(N))
            b[k] = v
            try_automaton(a_cycle, b)
=======
    # Family 1: a = cycle, b with controlled defects
    a_cycle = [(i + 1) % N for i in range(N)]
    
    # Single defect: b[i] = i for all i except b[k] = v
    for k in range(N):
        for v in range(N):
            if v == k:
                continue
            if time.time() > deadline:
                break
            b = list(range(N))
            b[k] = v
            try_automaton(a_cycle, b)
            
    # Explicitly test the "cycle-shift" variant: b[N-1] = 1
    # This is a known structure for high thresholds
    if time.time() < deadline:
        b_shift = list(range(N))
        b_shift[N-1] = 1
        try_automaton(a_cycle, b_shift)
>>>>>>> REPLACE
