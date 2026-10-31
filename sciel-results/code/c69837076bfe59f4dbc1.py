<<<<<<< SEARCH
    # Safety shrink
    r = r * (1 - 1e-6)
=======
    # Safety shrink (slightly more aggressive to ensure validity)
    r = r * (1 - 1e-5)
>>>>>>> REPLACE
