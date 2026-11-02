    # Rule for LUW
    # rule[c] = (a, b) for c = 2..d
    # Standard LUW recurrence: y[c] = x[c] + y[c-1] * x[1]
    # This corresponds to a = c-1, b = 1.
    # The previous code had incorrect rules for c >= 4.
    
    rule = {}
    for c in range(2, d + 1):
        rule[c] = (c - 1, 1)
