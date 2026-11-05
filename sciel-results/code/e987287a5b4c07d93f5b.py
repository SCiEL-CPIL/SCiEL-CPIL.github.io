import math
import time

def construct():
    start_time = time.time()
    deadline = start_time + 115.0

    # Primes up to enough to cover product <= 10^18
    primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79]

    def get_divisors(powers):
        """powers: list of (prime, exponent). Returns sorted list of divisors."""
        divs = [1]
        for p, e in powers:
            if e == 0:
                continue
            current_len = len(divs)
            power = 1
            for _ in range(e):
                power *= p
                for i in range(current_len):
                    divs.append(divs[i] * power)
        divs.sort()
        return divs

    def calc_score(divs):
        if len(divs) < 2:
            return 0
        s = 0
        for i in range(len(divs) - 1):
            if math.gcd(divs[i], divs[i + 1]) == 1:
                s += 1
        return s

    def evaluate(powers):
        """Evaluate a factorization. Returns (score, n) or (-1, -1) if invalid."""
        n = 1
        for p, e in powers:
            n *= (p ** e)
            if n > 10**18:
                return -1, -1
        if n < 2:
            return -1, -1
        divs = get_divisors(powers)
        s = calc_score(divs)
        return s, n

    best_score = 0
    best_n = 2

    # Find max k for square-free product of first k primes
    prefix_prod = [1]
    for p in primes:
        prefix_prod.append(prefix_prod[-1] * p)

    max_k = 0
    for i in range(1, len(primes) + 1):
        if prefix_prod[i] <= 10**18:
            max_k = i
        else:
            break

    # Evaluate square-free products of first k primes for k near max_k
    for k in range(max(1, max_k - 5), max_k + 1):
        if time.time() > deadline:
            break
        powers = [(primes[i], 1) for i in range(k)]
        s, n = evaluate(powers)
        if s > best_score:
            best_score = s
            best_n = n

    # Try variations: increase exponent of small primes, drop large primes
    # For each base k, try bumping exponent of prime i and dropping largest primes to fit
    for k in range(max(2, max_k - 4), max_k + 1):
        if time.time() > deadline:
            break
        base_powers = [(primes[i], 1) for i in range(k)]
        base_n = prefix_prod[k]

        # Try increasing exponent of primes[0] (2) by 1, 2, 3... and dropping largest primes
        for target_idx in range(min(4, k)):
            for extra in range(1, 8):
                new_powers = [(primes[i], 1) for i in range(k)]
                new_powers[target_idx] = (primes[target_idx], 1 + extra)
                # Check if fits; if not, drop largest primes
                n_val = 1
                for p, e in new_powers:
                    n_val *= (p ** e)
                    if n_val > 10**18:
                        break
                if n_val <= 10**18:
                    s, n = evaluate(new_powers)
                    if s > best_score:
                        best_score = s
                        best_n = n
                else:
                    # Drop largest primes until it fits
                    # Try dropping from the end
                    for drop_count in range(1, k - target_idx):
                        trimmed = new_powers[:k - drop_count]
                        n_val = 1
                        for p, e in trimmed:
                            n_val *= (p ** e)
                            if n_val > 10**18:
                                break
                        if n_val <= 10**18:
                            s, n = evaluate(trimmed)
                            if s > best_score:
                                best_score = s
                                best_n = n
                            break  # Only try first valid drop count

    # Also try: for k = max_k, try replacing the largest prime with a power of 2 or 3
    if max_k >= 2:
        for extra_2 in range(1, 10):
            powers = [(primes[i], 1) for i in range(max_k - 1)]
            powers[0] = (primes[0], 1 + extra_2)
            n_val = 1
            for p, e in powers:
                n_val *= (p ** e)
            if n_val <= 10**18:
                s, n = evaluate(powers)
                if s > best_score:
                    best_score = s
                    best_n = n

        for extra_3 in range(1, 8):
            powers = [(primes[i], 1) for i in range(max_k - 1)]
            powers[1] = (primes[1], 1 + extra_3)
            n_val = 1
            for p, e in powers:
                n_val *= (p ** e)
            if n_val <= 10**18:
                s, n = evaluate(powers)
                if s > best_score:
                    best_score = s
                    best_n = n

    return best_n
