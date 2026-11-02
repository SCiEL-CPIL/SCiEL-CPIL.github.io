<<<<<<< SEARCH
# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 100  # Leave 20s safety margin
    
    # Try to find a lower-rank decomposition
    result = _structured_search(n, m, p, deadline)
    if result is not None:
        return result
    
    # Fall back to schoolbook
    return _schoolbook(n, m, p)
# EVOLVE-BLOCK-END
=======
# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 130  # Leave 20s safety margin before 150s limit
    
    # Try to find a lower-rank decomposition
    result = _structured_search(n, m, p, deadline)
    if result is not None:
        return result
    
    # Fall back to schoolbook
    return _schoolbook(n, m, p)
# EVOLVE-BLOCK-END
>>>>>>> REPLACE
