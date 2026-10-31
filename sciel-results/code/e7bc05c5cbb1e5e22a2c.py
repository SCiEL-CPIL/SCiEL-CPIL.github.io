def priority(vector, n):
    count_ones = sum(1 for v in vector if v == 1)
    count_zeros = sum(1 for v in vector if v == 0)
    # Primary: fewer 1s is better
    # Secondary: among same count_ones, more zeros (or twos) is better
    return -float(count_ones) + 0.1 * float(count_zeros)
