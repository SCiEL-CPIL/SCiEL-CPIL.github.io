<<<<<<< SEARCH
def _local_search(best, universe, deadline):
    """Local search: try swaps and additions to improve cardinality."""
    improved = True
    while improved and time.time() < deadline:
        improved = False
        in_set = set(best)
        remaining = [x for x in universe if x not in in_set]
        
        # Try adding elements
        for x in remaining:
            if time.time() > deadline:
                break
            candidate = best + [x]
            if _verify(candidate):
                best = candidate
                improved = True
                break
        
        # Try single swaps
        if not improved:
            in_set = set(best)
            remaining = [x for x in universe if x not in in_set]
            for i, a in enumerate(best):
                if time.time() > deadline:
                    break
                for b in remaining:
                    candidate = best[:]
                    candidate[i] = b
                    if _verify(candidate):
                        # Check if this swap allows adding more
                        new_in = set(candidate)
                        new_remaining = [x for x in universe if x not in new_in]
                        for c in new_remaining:
                            if time.time() > deadline:
                                break
                            if _verify(candidate + [c]):
                                best = candidate + [c]
                                improved = True
                                break
                        if improved:
                            break
                if improved:
                    break
    
    return best
=======
def _local_search(best, universe, deadline):
    """Local search: try additions, 1-for-2 swaps, and 2-for-2 swaps."""
    improved = True
    while improved and time.time() < deadline:
        improved = False
        in_set = set(best)
        remaining = [x for x in universe if x not in in_set]
        
        # 1. Try adding 1 element
        for x in remaining:
            if time.time() > deadline:
                break
            candidate = best + [x]
            if _verify(candidate):
                best = candidate
                improved = True
                break
        
        # 2. Try removing 1, adding 2 (net +1)
        if not improved and time.time() < deadline - 5:
            indices = list(range(len(best)))
            random.shuffle(indices)
            for i in indices:
                if time.time() > deadline:
                    break
                a = best[i]
                base = best[:i] + best[i+1:]
                base_set = set(base)
                avail = [x for x in universe if x not in base_set]
                random.shuffle(avail)
                attempts = 0
                for j in range(len(avail)):
                    if attempts > 20: break
                    if time.time() > deadline:
                        break
                    c1 = avail[j]
                    for k in range(j+1, len(avail)):
                        attempts += 1
                        if attempts > 20: break
                        c2 = avail[k]
                        candidate = base + [c1, c2]
                        if _verify(candidate):
                            best = candidate
                            improved = True
                            break
                    if improved:
                        break
                if improved:
                    break

        # 3. Try removing 2, adding 2 (net 0)
        if not improved and time.time() < deadline - 5:
            indices = list(range(len(best)))
            random.shuffle(indices)
            for i in indices:
                if time.time() > deadline:
                    break
                j_indices = [x for x in indices if x != i]
                random.shuffle(j_indices)
                for j in j_indices[:5]:
                    if time.time() > deadline:
                        break
                    rem_set = {best[i], best[j]}
                    base = [x for x in best if x not in rem_set]
                    base_set = set(base)
                    avail = [x for x in universe if x not in base_set]
                    random.shuffle(avail)
                    attempts = 0
                    for m in range(len(avail)):
                        if attempts > 10: break
                        if time.time() > deadline:
                            break
                        c1 = avail[m]
                        for n in range(m+1, len(avail)):
                            attempts += 1
                            if attempts > 10: break
                            c2 = avail[n]
                            candidate = base + [c1, c2]
                            if _verify(candidate):
                                best = candidate
                                improved = True
                                break
                        if improved:
                            break
                    if improved:
                        break
                if improved:
                    break
    return best
>>>>>>> REPLACE
