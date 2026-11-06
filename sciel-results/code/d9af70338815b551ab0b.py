<<<<<<< SEARCH
def _local_search(initial_set):
    """
    Local search: try to improve the set by adding elements or swapping.
    """
    S = set(initial_set)
    best_set = list(S)
    best_size = len(S)
    
    improved = True
    while improved and not _deadline_hit():
        improved = False
        
        # Try adding one element
        for add in range(1, 25):
            if _deadline_hit():
                break
            if add in S:
                continue
            if _is_valid_addition(list(S), add):
                S.add(add)
                best_set = list(S)
                best_size = len(S)
                improved = True
        
        # Try removing one element and adding another (to potentially grow)
        for remove in list(S):
            if _deadline_hit():
                break
            S_without = S - {remove}
            for add in range(1, 25):
                if add in S:
                    continue
                if _is_valid_addition(list(S_without), add):
                    new_S = S_without | {add}
                    if len(new_S) > best_size:
                        S = new_S
                        best_set = list(S)
                        best_size = len(S)
                        improved = True
                        break
            if improved:
                break
    
    return best_set
=======
def _local_search(initial_set):
    """
    Local search: try to improve the set by adding elements, swapping, or remove-1-add-2.
    """
    S = set(initial_set)
    best_set = list(S)
    best_size = len(S)
    
    improved = True
    while improved and not _deadline_hit():
        improved = False
        
        # Try adding one element
        for add in range(1, 25):
            if _deadline_hit():
                break
            if add in S:
                continue
            if _is_valid_addition(list(S), add):
                S.add(add)
                best_set = list(S)
                best_size = len(S)
                improved = True
        
        # Try removing one element and adding another (swap)
        for remove in list(S):
            if _deadline_hit():
                break
            S_without = S - {remove}
            for add in range(1, 25):
                if add in S:
                    continue
                if _is_valid_addition(list(S_without), add):
                    new_S = S_without | {add}
                    if len(new_S) > best_size:
                        S = new_S
                        best_set = list(S)
                        best_size = len(S)
                        improved = True
                        break
            if improved:
                break
        
        # Try removing one element and adding two elements (remove-1-add-2)
        if not improved and not _deadline_hit():
            for remove in list(S):
                if _deadline_hit():
                    break
                S_without = S - {remove}
                found = False
                for add1 in range(1, 25):
                    if _deadline_hit():
                        break
                    if add1 in S:
                        continue
                    if not _is_valid_addition(list(S_without), add1):
                        continue
                    S_with_add1 = S_without | {add1}
                    for add2 in range(1, 25):
                        if _deadline_hit():
                            break
                        if add2 in S:
                            continue
                        if add2 == add1:
                            continue
                        if _is_valid_addition(list(S_with_add1), add2):
                            new_S = S_with_add1 | {add2}
                            if len(new_S) > best_size:
                                S = new_S
                                best_set = list(S)
                                best_size = len(S)
                                improved = True
                                found = True
                                break
                    if found:
                        break
                if improved:
                    break
    
    return best_set
>>>>>>> REPLACE

