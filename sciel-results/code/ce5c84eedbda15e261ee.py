<<<<<<< SEARCH
def _multi_swap_search(best, not_in_set, deadline):
    """Try removing up to 2 elements and adding up to 2 from not_in_set."""
    current = list(best)
    in_set = set(current)
    remaining = [x for x in not_in_set if x not in in_set]
    best_found = list(current)
    
    # Try single swaps first (already done in _try_swaps, but do it here for consistency)
    for i, a in enumerate(current):
        if time.time() > deadline:
            break
        for b in remaining:
            candidate = current[:]
            candidate[i] = b
            candidate.sort()
            if _find_collision_fast(candidate) is None:
                if len(candidate) > len(best_found):
                    best_found = candidate
                else:
                    best_found = candidate
                in_set = set(best_found)
                remaining = [x for x in not_in_set if x not in in_set]
                break
    
    # Try double swaps: remove 2, add 2
    # Order remaining elements to try promising ones first
    # Prefer elements that are closer in magnitude to the ones being removed
    if time.time() < deadline - 5:
        n_cur = len(current)
        for i in range(n_cur):
            if time.time() > deadline:
                break
            for j in range(i + 1, n_cur):
                if time.time() > deadline:
                    break
                # Sort remaining by proximity to the elements being removed
                removed_vals = [current[i], current[j]]
                remaining_sorted = sorted(remaining, key=lambda x: min(abs(x - r) for r in removed_vals))
                for b1 in remaining_sorted:
                    if time.time() > deadline:
                        break
                    for b2 in remaining_sorted:
                        if b2 == b1:
                            continue
                        candidate = current[:]
                        # Remove j first (higher index) to keep indices stable
                        candidate.pop(j)
                        candidate.pop(i)
                        candidate.extend([b1, b2])
                        candidate.sort()
                        if _find_collision_fast(candidate) is None:
                            if len(candidate) > len(best_found):
                                best_found = candidate
                                in_set = set(best_found)
                                remaining = [x for x in not_in_set if x not in in_set]
                            break
    return best_found
=======
def _multi_swap_search(best, not_in_set, deadline):
    """Try removing up to 2 elements and adding up to 2 from not_in_set."""
    current = list(best)
    in_set = set(current)
    remaining = [x for x in not_in_set if x not in in_set]
    best_found = list(current)
    
    # Try double swaps: remove 2, add 2
    if time.time() < deadline - 3:
        n_cur = len(current)
        for i in range(n_cur):
            if time.time() > deadline:
                break
            for j in range(i + 1, n_cur):
                if time.time() > deadline:
                    break
                # Sort remaining by proximity to the elements being removed
                removed_vals = [current[i], current[j]]
                remaining_sorted = sorted(remaining, key=lambda x: min(abs(x - r) for r in removed_vals))
                for b1 in remaining_sorted:
                    if time.time() > deadline:
                        break
                    for b2 in remaining_sorted:
                        if b2 == b1:
                            continue
                        candidate = current[:]
                        # Remove j first (higher index) to keep indices stable
                        candidate.pop(j)
                        candidate.pop(i)
                        candidate.extend([b1, b2])
                        candidate.sort()
                        if _find_collision_fast(candidate) is None:
                            if len(candidate) > len(best_found):
                                best_found = candidate
                                in_set = set(best_found)
                                remaining = [x for x in not_in_set if x not in in_set]
                            break
    return best_found
>>>>>>> REPLACE

