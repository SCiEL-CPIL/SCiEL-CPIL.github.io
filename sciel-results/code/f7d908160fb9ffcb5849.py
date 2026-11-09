import math
import time
import random

def _divisors_from_factors(factors):
    """factors: list of (prime, exponent). Return sorted divisors."""
    divs = [1]
    for p, e in factors:
        new_divs = []
        for dv in divs:
            m = dv
            for _ in range(e + 1):
                new_divs.append(m)
                m *= p
        divs = new_divs
    divs.sort()
    return divs

def _tau_perp_factors(factors):
    divs = _divisors_from_factors(factors)
    cnt = 0
    for i in range(len(divs) - 1):
        if math.gcd(divs[i], divs[i + 1]) == 1:
            cnt += 1
    return cnt

def _tau_perp(n):
    # factorize n by trial division (n <= 1e18, but we mostly call with known factors)
    temp = n
    factors = []
    d = 2
    while d * d <= temp:
        if temp % d == 0:
            cnt = 0
            while temp % d == 0:
                temp //= d
                cnt += 1
            factors.append((d, cnt))
        d += 1 if d == 2 else 2
    if temp > 1:
        factors.append((temp, 1))
    return _tau_perp_factors(factors)

def _prod(lst):
    r = 1
    for x in lst:
        r *= x
    return r

def _primes_up_to(limit):
    sieve = [True] * (limit + 1)
    sieve[0] = sieve[1] = False
    for i in range(2, int(limit**0.5) + 1):
        if sieve[i]:
            for j in range(i * i, limit + 1, i):
                sieve[j] = False
    return [i for i, v in enumerate(sieve) if v]

def construct():
    start = time.time()
    deadline = start + 140

    primes = _primes_up_to(300)
    first15 = primes[:15]

    def score_factors(factors):
        n = 1
        for p, e in factors:
            n *= p ** e
        if n > 10**18:
            return -1
        return _tau_perp_factors(factors)

    # baseline
    base_factors = [(p, 1) for p in first15]
    best_score = score_factors(base_factors)
    best_factors = base_factors[:]

    def update(factors):
        nonlocal best_score, best_factors
        s = score_factors(factors)
        if s > best_score:
            best_score = s
            best_factors = factors[:]
        return s

    # 1) Square-free 15-prime local search
    for restart in range(30):
        if time.time() > deadline:
            break
        if restart == 0:
            current = list(first15)
        else:
            current = list(first15)
            for _ in range(3):
                i = random.randrange(15)
                candidates = [p for p in primes if p not in current]
                if candidates:
                    current[i] = random.choice(candidates)
        cur_score = score_factors([(p, 1) for p in current])
        for it in range(5000):
            if time.time() > deadline:
                break
            i = random.randrange(15)
            candidates = [p for p in primes if p not in current]
            if not candidates:
                break
            j = random.randrange(len(candidates))
            new_set = current[:]
            new_set[i] = candidates[j]
            n = _prod(new_set)
            if n > 10**18:
                continue
            s = score_factors([(p, 1) for p in new_set])
            if s > cur_score:
                cur_score = s
                current = new_set
        update([(p, 1) for p in current])

    # 2) 2^a * 14 odd primes
    odd_primes = [p for p in primes if p != 2]
    for a in range(1, 22):
        if time.time() > deadline:
            break
        for restart in range(5):
            if time.time() > deadline:
                break
            if restart == 0:
                current = list(odd_primes[:14])
            else:
                current = list(odd_primes[:14])
                for _ in range(2):
                    i = random.randrange(14)
                    candidates = [p for p in odd_primes if p not in current]
                    if candidates:
                        current[i] = random.choice(candidates)
            factors = [(2, a)] + [(p, 1) for p in current]
            cur_score = score_factors(factors)
            for it in range(3000):
                if time.time() > deadline:
                    break
                i = random.randrange(14)
                candidates = [p for p in odd_primes if p not in current]
                if not candidates:
                    break
                j = random.randrange(len(candidates))
                new_set = current[:]
                new_set[i] = candidates[j]
                n = (2 ** a) * _prod(new_set)
                if n > 10**18:
                    continue
                s = score_factors([(2, a)] + [(p, 1) for p in new_set])
                if s > cur_score:
                    cur_score = s
                    current = new_set
            update([(2, a)] + [(p, 1) for p in current])

    # 3) 3^a * 14 other primes
    for a in range(1, 14):
        if time.time() > deadline:
            break
        others = [p for p in primes if p != 3]
        for restart in range(3):
            if time.time() > deadline:
                break
            if restart == 0:
                current = list(others[:14])
            else:
                current = list(others[:14])
                for _ in range(2):
                    i = random.randrange(14)
                    candidates = [p for p in others if p not in current]
                    if candidates:
                        current[i] = random.choice(candidates)
            factors = [(3, a)] + [(p, 1) for p in current]
            cur_score = score_factors(factors)
            for it in range(2000):
                if time.time() > deadline:
                    break
                i = random.randrange(14)
                candidates = [p for p in others if p not in current]
                if not candidates:
                    break
                j = random.randrange(len(candidates))
                new_set = current[:]
                new_set[i] = candidates[j]
                n = (3 ** a) * _prod(new_set)
                if n > 10**18:
                    continue
                s = score_factors([(3, a)] + [(p, 1) for p in new_set])
                if s > cur_score:
                    cur_score = s
                    current = new_set
            update([(3, a)] + [(p, 1) for p in current])

    # 4) 2^a * 3^b * 13 other primes
    for a in range(1, 12):
        if time.time() > deadline:
            break
        for b in range(1, 8):
            if time.time() > deadline:
                break
            others = [p for p in primes if p not in (2, 3)]
            current = list(others[:13])
            factors = [(2, a), (3, b)] + [(p, 1) for p in current]
            cur_score = score_factors(factors)
            for it in range(3000):
                if time.time() > deadline:
                    break
                i = random.randrange(13)
                candidates = [p for p in others if p not in current]
                if not candidates:
                    break
                j = random.randrange(len(candidates))
                new_set = current[:]
                new_set[i] = candidates[j]
                n = (2 ** a) * (3 ** b) * _prod(new_set)
                if n > 10**18:
                    continue
                s = score_factors([(2, a), (3, b)] + [(p, 1) for p in new_set])
                if s > cur_score:
                    cur_score = s
                    current = new_set
            update([(2, a), (3, b)] + [(p, 1) for p in current])

    # Final: compute best_n
    best_n = 1
    for p, e in best_factors:
        best_n *= p ** e
    return best_n

if __name__ == "__main__":
    n = construct()
    print("n =", n)
    print("tau_perp =", _tau_perp(n))
