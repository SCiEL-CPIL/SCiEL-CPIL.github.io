def construct():
    """
    Constructs a binary string of length N with all bits set to 1.
    This is the global optimum for the OneMax objective.
    Includes a time deadline guard to ensure safe termination.
    """
    start_time = time.time()
    deadline = start_time + 115.0
    
    best_result = "1" * N
    
    if time.time() > deadline:
        return best_result
    
    return best_result
