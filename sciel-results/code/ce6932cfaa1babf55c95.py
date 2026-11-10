def search_clique(D, bound, target_size):
    candidates = []
    for x in range(bound + 1):
        for y in range(bound + 1):
            candidates.append((x, y))
    
    best_set = []
    for p in candidates:
        ok = True
        for q in best_set:
            if not is_perfect_square(dist_sq(p, q, D)):
                ok = False
                break
        if not ok:
            continue
        
        collinear_ok = True
        for i in range(len(best_set)):
            for j in range(i+1, len(best_set)):
                if are_collinear(best_set[i], best_set[j], p):
                    collinear_ok = False
                    break
            if not collinear_ok:
                break
        if not collinear_ok:
            continue
        
        concyclic_ok = True
        if len(best_set) >= 3:
            for i in range(len(best_set)):
                for j in range(i+1, len(best_set)):
                    for k in range(j+1, len(best_set)):
                        if are_concyclic(best_set[i], best_set[j], best_set[k], p, D):
                            concyclic_ok = False
                            break
                    if not concyclic_ok:
                        break
                if not concyclic_ok:
                    break
        if not concyclic_ok:
            continue
            
        best_set.append(p)
        if len(best_set) >= target_size:
            return best_set
    return best_set
