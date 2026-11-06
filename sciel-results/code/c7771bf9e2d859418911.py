def _search(T, indices, start, current_set, current_sums):
    if len(current_set) == T:
        return list(current_set)
    if len(current_set) + len(indices) - start < T:
        return None
        
    for i in range(start, len(indices)):
        idx = indices[i]
        w = weights[idx]
        
        # Check add
        new_sums = set()
        valid = True
        for s in current_sums:
            ns = s + w
            if ns in current_sums:
                valid = False
                break
            new_sums.add(ns)
            
        if not valid:
            continue
            
        res = _search(T, indices, i+1, current_set + [idx], current_sums.union(new_sums))
        if res:
            return res
    return None
