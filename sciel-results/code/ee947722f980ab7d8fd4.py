def _gain_3ap(x, in_set):
    """Count 3-APs that would be formed by adding x to the set in_set."""
    count = 0
    # x is 1st: x, x+d, x+2d
    for d in range(1, (N - x) // 2 + 1):
        if in_set[x+d] and in_set[x+2*d]:
            count += 1
    # x is 2nd: x-d, x, x+d
    for d in range(1, min(x-1, N-x) + 1):
        if in_set[x-d] and in_set[x+d]:
            count += 1
    # x is 3rd: x-2d, x-d, x
    for d in range(1, (x-1) // 2 + 1):
        if in_set[x-2*d] and in_set[x-d]:
            count += 1
    return count

def _loss_3ap(x, in_set):
    """Count 3-APs that would be lost by removing x from the set in_set."""
    # Note: in_set should NOT contain x for this calculation to be accurate?
    # Or we just count APs involving x.
    # If x is in in_set, we count APs where x is a term.
    count = 0
    # x is 1st: x, x+d, x+2d
    for d in range(1, (N - x) // 2 + 1):
        if in_set[x+d] and in_set[x+2*d]:
            count += 1
    # x is 2nd: x-d, x, x+d
    for d in range(1, min(x-1, N-x) + 1):
        if in_set[x-d] and in_set[x+d]:
            count += 1
    # x is 3rd: x-2d, x-d, x
    for d in range(1, (x-1) // 2 + 1):
        if in_set[x-2*d] and in_set[x-d]:
            count += 1
    return count
