    for N in [100_000, 1_000_000, 2_000_000, 5_000_000]:
        if time.time() - start_time > DEADLINE - 15:
            break
        # ... sieve and search ...
