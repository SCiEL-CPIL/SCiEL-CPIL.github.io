import math
import time

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin
    
    # Precompute primes
    primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61]
    
    def get_divisors_fast(prime_powers):
        """Get sorted divisors from prime factorization as list of (prime, exponent) tuples."""
        divs = [1]
        for p, e in prime_powers:
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
    
    def calc_score_from_divisors(divs):
        if len(divs) < 2:
            return 0
        score = 0
        for i in range(len(divs) - 1):
            if math.gcd(divs[i], divs[i + 1]) == 1:
                score += 1
        return score
    
    def evaluate_factorization(powers):
        """powers: list of (prime, exponent). Returns (score, n) or (-1, -1) if invalid."""
        n = 1
        for p, e in powers:
            if e > 0:
                n *= (p ** e)
        if n > 10**18 or n < 2:
            return -1, -1
        divs = get_divisors_fast(powers)
        s = calc_score_from_divisors(divs)
        return s, n
    
    def local_search(pows, max_iters=50):
        """Perform local search around a starting factorization."""
        best_pows = pows[:]
        best_n = 1
        for p, e in best_pows:
            best_n *= (p ** e)
        best_s, _ = evaluate_factorization(best_pows)
        
        improved = True
        iters = 0
        while improved and iters < max_iters and time.time() < deadline:
            improved = False
            iters += 1
            # Try increasing or decreasing exponent of each prime
            for idx in range(len(best_pows)):
                p, e = best_pows[idx]
                # Try increasing
                if e < 60:
                    new_pows = best_pows[:]
                    new_pows[idx] = (p, e + 1)
                    s, n = evaluate_factorization(new_pows)
                    if s > best_s and n > 0:
                        best_pows = new_pows
                        best_s = s
                        best_n = n
                        improved = True
                        break
                # Try decreasing (only if e > 1)
                if e > 1:
                    new_pows = best_pows[:]
                    new_pows[idx] = (p, e - 1)
                    s, n = evaluate_factorization(new_pows)
                    if s > best_s and n > 0:
                        best_pows = new_pows
                        best_s = s
                        best_n = n
                        improved = True
                        break
            # Try adding a new prime
            if len(best_pows) < len(primes):
                for i in range(len(primes)):
                    if i >= len(best_pows):
                        # Check if this prime is already in the list
                        already = any(p == primes[i] for p, _ in best_pows)
                        if not already:
                            new_pows = best_pows + [(primes[i], 1)]
                            s, n = evaluate_factorization(new_pows)
                            if s > best_s and n > 0:
                                best_pows = new_pows
                                best_s = s
                                best_n = n
                                improved = True
                                break
            # Try removing a prime (if we have many)
            if len(best_pows) > 5:
                for idx in range(len(best_pows) - 1, -1, -1):
                    new_pows = best_pows[:idx] + best_pows[idx+1:]
                    s, n = evaluate_factorization(new_pows)
                    if s > best_s and n > 0:
                        best_pows = new_pows
                        best_s = s
                        best_n = n
                        improved = True
                        break
        
        return best_pows, best_s, best_n
    
    best_n = 2
    best_score = 0
    
    # Precompute prefix products of primes
    prefix_prod = [1]
    for p in primes:
        prefix_prod.append(prefix_prod[-1] * p)
    
    # Find max k such that product of first k primes <= 10^18
    max_k = 0
    for i in range(1, len(primes) + 1):
        if prefix_prod[i] <= 10**18:
            max_k = i
        else:
            break
    
    # 1. Evaluate square-free products of first k primes
    for k in range(1, max_k + 1):
        if time.time() > deadline:
            break
        pows = [(primes[i], 1) for i in range(k)]
        s, n = evaluate_factorization(pows)
        if s > best_score and n > 0:
            best_score = s
            best_n = n
    
    # 2. Structured search with exponent variations
    for k in range(max(1, max_k - 6), max_k + 1):
        if time.time() > deadline:
            break
        base_pows = [(primes[i], 1) for i in range(k)]
        
        for i in range(min(4, k)):
            for extra in range(1, 4):
                pows = [(primes[j], 1) for j in range(k)]
                pows[i] = (primes[i], 1 + extra)
                
                n_val = 1
                for p, e in pows:
                    n_val *= (p ** e)
                
                if n_val <= 10**18:
                    s, n = evaluate_factorization(pows)
                    if s > best_score and n > 0:
                        best_score = s
                        best_n = n
                else:
                    for drop_start in range(k - 1, i, -1):
                        if drop_start >= k:
                            continue
                        new_pows = pows[:drop_start]
                        if i >= drop_start:
                            continue
                        n_val = 1
                        for p, e in new_pows:
                            n_val *= (p ** e)
                        if n_val <= 10**18 and n_val >= 2:
                            s, n = evaluate_factorization(new_pows)
                            if s > best_score and n > 0:
                                best_score = s
                                best_n = n
                        break
    
    # 3. Try "almost square-free": drop one large prime, increase exponent of 2
    for k in range(max(2, max_k - 4), max_k + 1):
        if time.time() > deadline:
            break
        for a in range(1, 15):
            pows = [(primes[0], a)]
            for i in range(1, k - 1):
                pows.append((primes[i], 1))
            s, n = evaluate_factorization(pows)
            if s > best_score and n > 0:
                best_score = s
                best_n = n
    
    # 4. Try dropping two large primes, increase exponents of 2 and 3
    for k in range(max(3, max_k - 4), max_k + 1):
        if time.time() > deadline:
            break
        for a in range(1, 10):
            for b in range(1, 5):
                pows = [(primes[0], a), (primes[1], b)]
                for i in range(2, k - 2):
                    pows.append((primes[i], 1))
                s, n = evaluate_factorization(pows)
                if s > best_score and n > 0:
                    best_score = s
                    best_n = n
    
    # 5. Systematic search around known good region
    if max_k >= 14:
        for k in range(max(12, max_k - 3), max_k + 1):
            if time.time() > deadline:
                break
            for a in range(1, 8):
                for b in range(1, 4):
                    for c in range(1, 3):
                        if time.time() > deadline:
                            break
                        pows = [(primes[0], a), (primes[1], b), (primes[2], c)]
                        for i in range(3, k - 1):
                            pows.append((primes[i], 1))
                        s, n = evaluate_factorization(pows)
                        if s > best_score and n > 0:
                            best_score = s
                            best_n = n
    
    # 6. Try small primes with higher exponents
    if time.time() < deadline - 20:
        for a in range(1, 61):
            if time.time() > deadline:
                break
            for b in range(1, a + 1):
                if time.time() > deadline:
                    break
                for c in range(1, b + 1):
                    if time.time() > deadline:
                        break
                    for d in range(1, c + 1):
                        if time.time() > deadline:
                            break
                        pows = [(2, a), (3, b), (5, c), (7, d)]
                        s, n = evaluate_factorization(pows)
                        if s > best_score and n > 0:
                            best_score = s
                            best_n = n
    
    # 7. NEW: Local search from multiple starting points
    # Start from the best candidate found so far
    if time.time() < deadline - 10:
        # Reconstruct the factorization of best_n
        temp_n = best_n
        start_pows = []
        for p in primes:
            e = 0
            while temp_n % p == 0:
                temp_n //= p
                e += 1
            if e > 0:
                start_pows.append((p, e))
        
        if start_pows:
            best_pows, best_s, best_n_local = local_search(start_pows, max_iters=30)
            if best_s > best_score:
                best_score = best_s
                best_n = best_n_local
    
    # 8. NEW: Try more diverse starting points for local search
    if time.time() < deadline - 15:
        # Try starting from primorial with various exponent tweaks
        for k in [max_k, max_k - 1, max_k - 2]:
            if k < 2:
                continue
            if time.time() > deadline:
                break
            for a in [1, 2, 3]:
                for b in [1, 2]:
                    if time.time() > deadline:
                        break
                    pows = [(primes[0], a), (primes[1], b)]
                    for i in range(2, k):
                        pows.append((primes[i], 1))
                    s, n = evaluate_factorization(pows)
                    if s > best_score and n > 0:
                        best_score = s
                        best_n = n
                        # Do local search from this point
                        if time.time() < deadline - 5:
                            best_pows, best_s, best_n_local = local_search(pows, max_iters=20)
                            if best_s > best_score:
                                best_score = best_s
                                best_n = best_n_local
    
    # 9. NEW: Greedy construction - build up primes one at a time, choosing
    # the prime that maximizes score increase
    if time.time() < deadline - 20:
        current_pows = [(2, 1)]
        current_n = 2
        current_s, _ = evaluate_factorization(current_pows)
        
        for step in range(len(primes) - 1):
            if time.time() > deadline:
                break
            best_add = None
            best_add_score = current_s
            best_add_pows = None
            best_add_n = current_n
            
            for i in range(len(primes)):
                # Check if prime i is already in current_pows
                already = any(p == primes[i] for p, _ in current_pows)
                if already:
                    continue
                
                # Try adding prime i with exponent 1
                new_pows = current_pows + [(primes[i], 1)]
                s, n = evaluate_factorization(new_pows)
                if s > best_add_score and n > 0:
                    best_add_score = s
                    best_add_pows = new_pows
                    best_add_n = n
                
                # Also try adding with exponent 2 if it fits
                new_pows2 = current_pows + [(primes[i], 2)]
                s2, n2 = evaluate_factorization(new_pows2)
                if s2 > best_add_score and n2 > 0:
                    best_add_score = s2
                    best_add_pows = new_pows2
                    best_add_n = n2
            
            if best_add_pows is not None:
                current_pows = best_add_pows
                current_n = best_add_n
                current_s = best_add_score
                if current_s > best_score:
                    best_score = current_s
                    best_n = current_n
            else:
                break
        
        # Do local search from the greedy result
        if time.time() < deadline - 5:
            best_pows, best_s, best_n_local = local_search(current_pows, max_iters=25)
            if best_s > best_score:
                best_score = best_s
                best_n = best_n_local
    
    # Final check: ensure best_n is valid
    if best_n < 2:
        best_n = 2
        best_score = 1
    
    return best_n
