def priority(vector, n):
    # Prioritize vectors with fewer 2s.
    # The set {0, 1}^n is a cap set of size 2^n.
    # By sorting vectors with no 2s first, the greedy algorithm will pick all of them.
    # Vectors with 2s will be considered later and likely rejected due to collisions
    # with the already picked {0, 1}^n vectors.
    count_2 = sum(1 for x in vector if x == 2)
    return -count_2
