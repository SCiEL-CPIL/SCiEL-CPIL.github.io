import time
import math
import random

def construct():
    start_time = time.time()
    LIMIT = 10**18
    DEADLINE = start_time + 115.0  # Safety margin

    # Generate primes up to 200
    def sieve(n):
        is_prime = [True] * (n + 1)
        is_prime[0] = is_prime[1] = False
        for i in range(2, int(n**0.5) + 1):
            if is_prime[i]:
                for j in range(i*i, n+1, i):
                    is_prime[j] = False
        return [i for i, v in enumerate(is_prime) if v]

    primes = sieve(200)

    def calc_score(factors):
        """
        factors: list of (prime, exponent)
        Returns the score tau_perp(n)
        """
        # Generate divisors
        divs = [1]
        for p, e in factors:
            current_divs = list(divs)
            for _ in range(e):
                current_divs = [d * p for d in current_divs]
                divs.extend(current_divs)

        divs.sort()

        score = 0
        prev = divs[0]
        for curr in divs[1:]:
            if math.gcd(prev, curr) == 1:
                score += 1
            prev = curr
        return score

    def fits(factors):
        prod = 1
        for p, e in factors:
            prod *= (p ** e)
            if prod > LIMIT:
                return False
        return True

    best_score = 0
    best_factors = []

    def update_best(fac):
        nonlocal best_score, best_factors
        if not fits(fac):
            return
        s = calc_score(fac)
        if s > best_score:
            best_score = s
            best_factors = list(fac)

    # Generate initial candidates
    # 1. Square-free: first k primes
    for k in range(10, 16):
        fac = [(primes[i], 1) for i in range(k)]
        update_best(fac)

    # 2. Higher exponents on small primes
    for k in range(10, 15):
        base_fac = [(primes[i], 1) for i in range(k)]
        # Increase exponent of 2
        for e2 in range(2, 7):
            fac = list(base_fac)
            fac[0] = (primes[0], e2)
            update_best(fac)
        # Increase exponent of 3
        for e3 in range(2, 6):
            fac = list(base_fac)
            fac[1] = (primes[1], e3)
            update_best(fac)
        # Increase both
        for e2 in range(2, 6):
            for e3 in range(2, 5):
                fac = list(base_fac)
                fac[0] = (primes[0], e2)
                fac[1] = (primes[1], e3)
                update_best(fac)

    # 3. Specific insight structures
    # 2^4 * prod(primes 3..43) = 2^4 * 3*5*...*43
    fac_insight = [(2, 4)] + [(primes[i], 1) for i in range(1, 14)]
    update_best(fac_insight)
    fac_insight3 = [(2, 3)] + [(primes[i], 1) for i in range(1, 14)]
    update_best(fac_insight3)
    fac_insight2 = [(2, 2)] + [(primes[i], 1) for i in range(1, 14)]
    update_best(fac_insight2)

    # 4. Local search on exponents
    if best_factors and time.time() < DEADLINE - 10:
        current_fac = list(best_factors)
        improved = True
        while improved and time.time() < DEADLINE - 10:
            improved = False
            current_score = calc_score(current_fac)

            # Try changing exponents of small primes
            for idx in range(min(4, len(current_fac))):
                p_old, e_old = current_fac[idx]
                for e_new in range(max(1, e_old - 2), e_old + 3):
                    if e_new == e_old:
                        continue
                    trial = list(current_fac)
                    trial[idx] = (p_old, e_new)
                    if fits(trial):
                        s = calc_score(trial)
                        if s > current_score:
                            current_fac = trial
                            current_score = s
                            improved = True
                            break
                if improved:
                    break

            # Try swapping a prime with a nearby one
            for idx in range(len(current_fac)):
                p_old, e_old = current_fac[idx]
                for p_new in primes:
                    if p_new == p_old:
                        continue
                    if any(p_new == p for p, _ in current_fac):
                        continue
                    trial = list(current_fac)
                    trial[idx] = (p_new, e_old)
                    trial.sort(key=lambda x: x[0])
                    if fits(trial):
                        s = calc_score(trial)
                        if s > current_score:
                            current_fac = trial
                            current_score = s
                            improved = True
                            break
                if improved:
                    break

        if calc_score(current_fac) > best_score:
            best_score = calc_score(current_fac)
            best_factors = current_fac

    # 5. Random search for additional structures
    if time.time() < DEADLINE - 10:
        for _ in range(3000):
            if time.time() > DEADLINE - 5:
                break
            if not best_factors:
                break

            current_fac = list(best_factors)
            op = random.random()

            if op < 0.4:
                # Change exponent
                idx = random.randint(0, min(3, len(current_fac) - 1))
                p, e = current_fac[idx]
                new_e = e + random.randint(-1, 2)
                if new_e >= 1:
                    trial = list(current_fac)
                    trial[idx] = (p, new_e)
                    if fits(trial):
                        s = calc_score(trial)
                        if s > best_score:
                            best_score = s
                            best_factors = trial
            elif op < 0.7:
                # Swap prime
                idx = random.randint(0, len(current_fac) - 1)
                p_old, e_old = current_fac[idx]
                candidates = [p for p in primes if p != p_old and all(p != pp for pp, _ in current_fac)]
                if candidates:
                    p_new = random.choice(candidates[:15])
                    trial = list(current_fac)
                    trial[idx] = (p_new, e_old)
                    trial.sort(key=lambda x: x[0])
                    if fits(trial):
                        s = calc_score(trial)
                        if s > best_score:
                            best_score = s
                            best_factors = trial
            else:
                # Add/remove prime
                if len(current_fac) < 16:
                    max_p = max(p for p, _ in current_fac)
                    new_candidates = [p for p in primes if p > max_p]
                    if new_candidates:
                        new_p = random.choice(new_candidates[:5])
                        trial = current_fac + [(new_p, 1)]
                        if not fits(trial):
                            # Reduce exponent of 2 if possible
                            if trial[0][0] == 2 and trial[0][1] > 1:
                                trial[0] = (2, trial[0][1] - 1)
                        if fits(trial):
                            s = calc_score(trial)
                            if s > best_score:
                                best_score = s
                                best_factors = trial
                elif len(current_fac) > 10:
                    idx_rm = random.randint(len(current_fac) // 2, len(current_fac) - 1)
                    trial = current_fac[:idx_rm] + current_fac[idx_rm + 1:]
                    if fits(trial):
                        s = calc_score(trial)
                        if s > best_score:
                            best_score = s
                            best_factors = trial

    # Final construction
    if not best_factors:
        best_factors = [(primes[i], 1) for i in range(15)]

    n = 1
    for p, e in best_factors:
        n *= (p ** e)

    assert 2 <= n <= LIMIT, f"n={n} out of range"
    return n
