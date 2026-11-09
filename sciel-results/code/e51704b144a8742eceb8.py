def construct(k, g):
    start_time = time.time()
    deadline = start_time + 140
    
    if k != 3:
        return _luw_construct(k, g)
        
    # Try Cayley graph construction for k=3, g=16
    for q in [7, 8]:
        if time.time() > deadline:
            break
        try:
            result = _try_cayley_gl2(q, g, deadline)
            if result:
                return result
        except Exception:
            pass
            
    # Fallback to LUW
    return _luw_construct(k, g)
