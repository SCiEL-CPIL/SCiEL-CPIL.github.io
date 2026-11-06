<<<<<<< SEARCH
def construct():
    start_time = time.time()
    deadline = 115.0
    
    # Try n=47, 46, 45, ...
    for n in range(47, 38, -1):
        if time.time() - start_time > deadline:
            break
        remaining = deadline - (time.time() - start_time)
        if remaining < 2:
            break
            
        min_deg_req = max(1, n - 39)
        
        # Allocate time: more for higher n
        if n >= 46:
            t_limit = min(45.0, remaining - 1)
        elif n >= 44:
            t_limit = min(25.0, remaining - 1)
        else:
            t_limit = min(15.0, remaining - 1)
        
        result = _construct_c4_free(n, min_deg_req, t_limit)
        if result:
            return result
            
    return {'n': 39, 'edges': []}
=======
def construct():
    start_time = time.time()
    deadline = 115.0
    
    # Try n=47, 46, 45, ...
    for n in range(47, 38, -1):
        if time.time() - start_time > deadline:
            break
        remaining = deadline - (time.time() - start_time)
        if remaining < 2:
            break
            
        min_deg_req = max(1, n - 39)
        
        # Allocate time: more for higher n
        if n >= 46:
            t_limit = min(50.0, remaining - 1)
        elif n >= 44:
            t_limit = min(30.0, remaining - 1)
        else:
            t_limit = min(15.0, remaining - 1)
        
        result = _construct_c4_free(n, min_deg_req, t_limit)
        if result:
            return result
            
    return {'n': 39, 'edges': []}
>>>>>>> REPLACE
