<<<<<<< SEARCH
    for p in primes_to_check:
        if time.time() > deadline:
            break
        base = p * p
        k = 0
        for i in range(1, 201):
            n = base + i
            if not is_p_smooth(n, p):
                break
            k += 1
        if k > best_k:
            best_k = k
            best_p = p
            if best_k == 200:
                return {'p': p, 'k': 200}
=======
    for p in primes_to_check:
        if time.time() > deadline:
            break
        k = check_prime_run(p)
        if k > best_k:
            best_k = k
            best_p = p
            if best_k == 200:
                return {'p': p, 'k': 200}
>>>>>>> REPLACE
