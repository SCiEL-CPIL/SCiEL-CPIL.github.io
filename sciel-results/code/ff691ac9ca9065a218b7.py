rem = np.arange(L, R+1)
for p in primes:
    idx = ...
    rem[idx] //= p
    # Check if still divisible
    mask = rem[idx] % p == 0
    rem[idx][mask] //= p
    # Repeat?
