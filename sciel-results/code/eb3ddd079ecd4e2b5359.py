<<<<<<< SEARCH
    q = k
    # Choose d so that the girth is at least d + 5 (for odd d) or d + 4 (even).
    d = g - 5 if (g - 5) % 2 else g - 4
    if d < 3:
        d = 3
=======
    q = k
    # Choose d so that the girth is at least 2d + 2.
    # For girth g, we need 2d + 2 >= g => d >= (g-2)/2.
    # Since d must be integer, d = ceil((g-2)/2) = (g-1)//2.
    d = (g - 1) // 2
    if d < 3:
        d = 3
>>>>>>> REPLACE
