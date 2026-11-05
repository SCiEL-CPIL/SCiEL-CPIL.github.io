<<<<<<< SEARCH
def _luw_construct(k, g):
    # Lazebnik-Ustimenko-Woldar graph CD(d, q)
    q = k
    d = max(3, g - 5 if (g - 5) % 2 else g - 4)
=======
def _luw_construct(k, g):
    # Lazebnik-Ustimenko-Woldar graph CD(d, q)
    q = k
    # The girth of CD(d, q) is at least 2d + 1.
    # We need 2d + 1 >= g => d >= (g - 1) / 2.
    # Using the minimal d minimizes the number of vertices N = 2 * q^d.
    d = max(3, (g - 1) // 2)
>>>>>>> REPLACE
