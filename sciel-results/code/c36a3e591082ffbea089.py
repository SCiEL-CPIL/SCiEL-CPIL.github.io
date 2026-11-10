"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time

def _is_prime(n):
    if n < 2:
        return False
    for d in range(2, int(n**0.5) + 1):
        if n % d == 0:
            return False
    return True

def _prime_factor_exponents(n, primes):
    exps = []
    for p in primes:
        e = 0
        while n % p == 0:
            n //= p
            e += 1
        exps.append(e)
    return tuple(exps)

def _vec_add(a, b):
    return tuple(x + y for x, y in zip(a, b))

def construct():
    start_time = time.time()
    deadline = start_time + 55
    
    primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37]
    num_primes = len(primes)
    
    vecs = {}
    for n in range(2, 41):
        vecs[n] = _prime_factor_exponents(n, primes)
    
    def check_collision(selected_vecs, new_vec):
        if not selected_vecs:
            return True
        current_sums = {tuple(0) * num_primes}
        for v in selected_vecs:
            new_sums = set()
            for s in current_sums:
                new_sums.add(s)
                new_sums.add(_vec_add(s, v))
            current_sums = new_sums
        new_sums = set(current_sums)
        for s in current_sums:
            new_sums.add(_vec_add(s, new_vec))
        return len(new_sums) == 2 * len(current_sums)
    
    best_result = []
    
    # Generate multiple orderings
    all_nums = list(range(2, 41))
    primes_only = [n for n in all_nums if _is_prime(n)]
    composites = [n for n in all_nums if not _is_prime(n)]
    
    def pf_count(n):
        c = 0
        t = n
        for p in primes:
            while t % p == 0:
                t //= p
                c += 1
        return c
    
    orderings = [
        primes_only + composites,
        composites + primes_only,
        sorted(all_nums),
        sorted(all_nums, reverse=True),
        sorted(all_nums, key=lambda x: (pf_count(x), x)),
        sorted(all_nums, key=lambda x: (-pf_count(x), x)),
        sorted(all_nums, key=lambda x: (sum(vecs[x]), x)),
        sorted(all_nums, key=lambda x: (-sum(vecs[x]), x)),
    ]
    
    for ordering in orderings:
        if time.time() > deadline:
            break
        selected = []
        selected_vecs = []
        for n in ordering:
            if time.time() > deadline:
                break
            if check_collision(selected_vecs, vecs[n]):
                selected.append(n)
                selected_vecs.append(vecs[n])
        if len(selected) > len(best_result):
            best_result = selected[:]
    
    return list(best_result)
# EVOLVE-BLOCK-END

