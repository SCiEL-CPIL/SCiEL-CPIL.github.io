for u in list(S):
    S.remove(u)
    added_v = None
    for v in range(1, LIMIT + 1):
        if v not in S and is_valid(v, S):
            S.add(v)
            added_v = v
            break
    
    if added_v is not None:
        # Try to add more
        added_more = False
        for w in range(1, LIMIT + 1):
            if w not in S and is_valid(w, S):
                S.add(w)
                added_more = True
        
        if not added_more:
            # Swap didn't lead to growth, revert to avoid cycle
            S.remove(added_v)
            S.add(u)
        else:
            improved = True
    else:
        S.add(u)
