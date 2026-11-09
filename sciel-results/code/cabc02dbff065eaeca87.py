    sp = np.ones(MAX_VAL + 1, dtype=np.int64)
    for p in PRIMES:
        power = p
        while power <= MAX_VAL:
            sp[::power] *= p
            power *= p
    