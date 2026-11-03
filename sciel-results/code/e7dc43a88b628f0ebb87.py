<<<<<<< SEARCH
    # Choose d so that the girth is at least d + 5 (for odd d) or d + 4 (even).
    d = g - 5 if (g - 5) % 2 else g - 4
    if d < 3:
        d = 3
=======
    # LUW graph L(q, d) has girth 2d + 1.
    # We need girth >= g, so 2d + 1 >= g => d >= (g-1)/2.
    d = (g - 1) // 2
    if d < 3:
        d = 3
>>>>>>> REPLACE
