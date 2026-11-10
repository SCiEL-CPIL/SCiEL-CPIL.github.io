        # Efficient construction
        # For each k, the values are floor(x/k) for x=k..10K
        # We can generate the rows, cols, data arrays for each k and concatenate?
        # Or just use the loop with list extension, it's probably fine for 2.4M items.
