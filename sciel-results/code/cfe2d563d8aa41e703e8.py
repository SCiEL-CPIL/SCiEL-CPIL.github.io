<<<<<<< SEARCH
def construct(k, g):
    start_time = time.time()
    deadline = start_time + 130.0
    
    if k == 3 and g == 16:
        result = _construct_cage_3_16()
        if result is not None:
            return result
        return _luw_construct(k, g)
    
    return _luw_construct(k, g)
=======
def construct(k, g):
    start_time = time.time()
    deadline = start_time + 115.0  # safety margin before 150s kill
    
    if k == 3 and g == 16:
        result = _construct_cage_3_16()
        if result is not None:
            return result
        return _luw_construct(k, g)
    
    return _luw_construct(k, g)
>>>>>>> REPLACE
