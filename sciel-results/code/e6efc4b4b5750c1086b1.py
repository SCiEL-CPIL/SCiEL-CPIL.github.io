    for i_start in range(0, n, batch_size):
        # ...
        for j in range(n):
            # Vectorized ops for batch_i vs elements[j]
