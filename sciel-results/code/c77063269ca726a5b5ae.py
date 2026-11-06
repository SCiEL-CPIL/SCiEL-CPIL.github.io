    if best_pts is None:
        result = [[x, x * x] for x in range(-10, 10)]
    else:
        result = [list(p) for p in best_pts]
        
    if not _is_valid_output(result):
        result = [[x, x * x] for x in range(-10, 10)]
        
    return result
