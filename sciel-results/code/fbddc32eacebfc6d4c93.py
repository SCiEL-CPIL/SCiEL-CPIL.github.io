def can_form_target(target, candidates):
    if not candidates:
        return target == 0
    # Quick checks
    total = sum(candidates)
    if total < target:
        return False
    if min(candidates) > target:
        return False
    
    achievable = {0}
    for c in candidates:
        if c > target:
            continue
        new_sums = set()
        for s in achievable:
            ns = s + c
            if ns <= target:
                new_sums.add(ns)
        if target in new_sums:
            return True
        achievable |= new_sums
    return False
