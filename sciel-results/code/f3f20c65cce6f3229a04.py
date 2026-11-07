def construct():
    n = 128
    L = set(range(65, n + 1))
    
    # T consists of odd numbers 1..33 and 64
    T = set(range(1, 34, 2))
    T.add(64)
    
    S = T | L
    
    # Verification step (optional but good for safety)
    # Check all triples
    # Since we are confident, we can skip heavy verification in final run
    # but for development, we might want it.
    # The task requires returning the list.
    
    return sorted(S)
