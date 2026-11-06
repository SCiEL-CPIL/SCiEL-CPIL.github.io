def priority(vector, n):
    counts = [0, 0, 0]
    for x in vector:
        counts[x] += 1
    
    # We want to maximize the priority for balanced vectors.
    # Let's return the negative of the sum of squared deviations from the mean count.
    # Mean is n/3.
    # Sum of (counts[i] - n/3)^2
    # Or simpler: minimize max(counts) - min(counts)
    # Let's use the sum of squared differences between counts.
    diff_01 = counts[0] - counts[1]
    diff_12 = counts[1] - counts[2]
    diff_20 = counts[2] - counts[0]
    
    # Return a value such that more balanced vectors have higher priority.
    # The sum of squares is always non-negative. We want to minimize it.
    # So return negative sum of squares.
    penalty = diff_01 * diff_01 + diff_12 * diff_12 + diff_20 * diff_20
    return -float(penalty)
