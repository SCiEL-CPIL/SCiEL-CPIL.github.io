def construct():
    try:
        start = time.time()
        deadline = start + 140.0
        
        result = _strassen_44()
        ok, err = _verify(result['U'], result['V'], result['W'])
        
        if not ok:
            return _schoolbook()
            
        best_result = result
        best_rank = len(result['U'][0])
        
        if time.time() < deadline - 5.0:
            reduced_result, reduced_rank = _try_reduce_by_linear_dependency(
                result['U'], result['V'], result['W'], deadline
            )
            if reduced_rank < best_rank:
                ok, err = _verify(reduced_result['U'], reduced_result['V'], reduced_result['W'])
                if ok:
                    best_result = reduced_result
                    best_rank = reduced_rank
                    
        return best_result
    except Exception:
        return _schoolbook()
