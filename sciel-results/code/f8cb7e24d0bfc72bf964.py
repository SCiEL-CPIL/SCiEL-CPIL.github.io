<<<<<<< SEARCH
def _construct_luw(k, g, deadline):
    q = k
    d = max(3, g - 5 if (g - 5) % 2 else g - 4)
=======
def _construct_luw(k, g, deadline):
    q = k
    # Minimum d such that 2d + 2 >= g
    d = max(3, (g - 2 + 1) // 2)
>>>>>>> REPLACE
