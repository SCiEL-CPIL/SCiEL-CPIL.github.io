<<<<<<< SEARCH
    # Try some larger primes beyond the sieve
    if time.time() < deadline - 10:
        random.seed(42)
        for _ in range(100):
            if time.time() > deadline:
                break
            p_candidate = random.randint(10**7, 10**9)
=======
    # Try some larger primes beyond the sieve
    if time.time() < deadline - 5:
        random.seed(42)
        for _ in range(300):
            if time.time() > deadline:
                break
            p_candidate = random.randint(2 * 10**7, 10**9)
>>>>>>> REPLACE
