import time
import math

def construct():
    start_time = time.time()
    deadline = start_time + 135  # safety margin before 150s limit

    def compute_score(n):
        """Compute tau_perp(n): number of adjacent coprime pairs in sorted divisors."""
        if n < 2:
            return 0
        # Factorize n
        factors = []
        temp = n
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
        
        # Generate all divisors
        divisors = [1]
        for p, e in factors:
            new_divs = []
            for dv in divisors:
                val = dv
                for k in range(e + 1):
                    new_divs.append(val)
                    val *= p
            divisors = new_divs
        
        divisors.sort()
        
        # Count coprime adjacent pairs
        score = 0
        for i in range(len(divisors) - 1):
            if math.gcd(divisors[i], divisors[i + 1]) == 1:
                score += 1
        return score

    # Evaluate primorials of various sizes
    primes_list = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53]
    
    best_n = 2
    best_score = 0
    
    # Try primorials
    for k in range(2, 16):
        n = 1
        for i in range(k):
            n *= primes_list[i]
        if n > 10**18:
            break
        sc = compute_score(n)
        if sc > best_score:
            best_score = sc
            best_n = n

    if time.time() > deadline - 5:
        return best_n

    # Try prime power configurations - focus on high-divisor-count numbers
    test_configs = [
        [(2,1),(3,1),(5,1),(7,1),(11,1),(13,1),(17,1),(19,1),(23,1),(29,1),(31,1),(37,1)],
        [(2,1),(3,1),(5,1),(7,1),(11,1),(13,1),(17,1),(19,1),(23,1),(29,1),(31,1),(37,1),(41,1)],
        [(2,2),(3,1),(5,1),(7,1),(11,1),(13,1),(17,1),(19,1),(23,1),(29,1)],
        [(2,2),(3,2),(5,1),(7,1),(11,1),(13,1),(17,1),(19,1),(23,1)],
        [(2,2),(3,2),(5,2),(7,1),(11,1),(13,1),(17,1),(19,1)],
        [(2,3),(3,2),(5,1),(7,1),(11,1),(13,1),(17,1),(19,1)],
        [(2,3),(3,2),(5,2),(7,1),(11,1),(13,1),(17,1)],
        [(2,3),(3,3),(5,2),(7,1),(11,1),(13,1),(17,1)],
        [(2,4),(3,2),(5,2),(7,1),(11,1),(13,1),(17,1)],
        [(2,4),(3,3),(5,2),(7,1),(11,1),(13,1)],
        [(2,4),(3,3),(5,2),(7,2),(11,1),(13,1)],
        [(2,5),(3,3),(5,2),(7,1),(11,1),(13,1)],
        [(2,5),(3,3),(5,2),(7,2),(11,1)],
        [(2,5),(3,3),(5,2),(7,2),(11,1),(13,1)],
        [(2,6),(3,3),(5,2),(7,1),(11,1),(13,1)],
        [(2,6),(3,3),(5,2),(7,2),(11,1)],
        [(2,6),(3,4),(5,2),(7,1),(11,1),(13,1)],
        [(2,6),(3,4),(5,2),(7,2),(11,1)],
        [(2,7),(3,4),(5,2),(7,1),(11,1),(13,1)],
        [(2,7),(3,4),(5,2),(7,2),(11,1)],
        [(2,7),(3,4),(5,3),(7,1),(11,1),(13,1)],
        [(2,7),(3,4),(5,3),(7,1),(11,1)],
        [(2,8),(3,4),(5,2),(7,1),(11,1),(13,1)],
        [(2,8),(3,4),(5,2),(7,2),(11,1)],
        [(2,8),(3,4),(5,3),(7,1),(11,1)],
        [(2,8),(3,5),(5,2),(7,1),(11,1),(13,1)],
        [(2,8),(3,5),(5,2),(7,2),(11,1)],
        [(2,9),(3,4),(5,2),(7,1),(11,1),(13,1)],
        [(2,9),(3,4),(5,2),(7,2),(11,1)],
        [(2,9),(3,5),(5,2),(7,1),(11,1)],
        [(2,9),(3,5),(5,2),(7,2),(11,1)],
        [(2,10),(3,4),(5,2),(7,1),(11,1),(13,1)],
        [(2,10),(3,4),(5,2),(7,2),(11,1)],
        [(2,10),(3,5),(5,2),(7,1),(11,1)],
        [(2,10),(3,5),(5,2),(7,2),(11,1)],
        [(2,10),(3,5),(5,3),(7,1),(11,1)],
        [(2,11),(3,4),(5,2),(7,1),(11,1),(13,1)],
        [(2,11),(3,4),(5,2),(7,2),(11,1)],
        [(2,11),(3,5),(5,2),(7,1),(11,1)],
        [(2,11),(3,5),(5,2),(7,2),(11,1)],
        [(2,12),(3,4),(5,2),(7,1),(11,1),(13,1)],
        [(2,12),(3,4),(5,2),(7,2),(11,1)],
        [(2,12),(3,5),(5,2),(7,1),(11,1)],
        [(2,12),(3,5),(5,2),(7,2),(11,1)],
    ]
    
    for config in test_configs:
        n = 1
        valid = True
        for p, e in config:
            n *= p ** e
        if n > 10**18 or n < 2:
            continue
        sc = compute_score(n)
        if sc > best_score:
            best_score = sc
            best_n = n

    if time.time() > deadline - 5:
        return best_n

    # Local search from best_n
    def get_factor_dict(n):
        factors = {}
        temp = n
        d = 2
        while d * d <= temp:
            if temp % d == 0:
                cnt = 0
                while temp % d == 0:
                    temp //= d
                    cnt += 1
                factors[d] = cnt
            d += 1 if d == 2 else 2
        if temp > 1:
            factors[temp] = 1
        return factors

    current_factors = get_factor_dict(best_n)
    current_score = best_score
    current_n = best_n

    search_primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47]

    for iteration in range(200):
        if time.time() > deadline - 5:
            break
        
        improved = False
        
        # Try increasing exponent of existing prime
        for p in sorted(current_factors.keys()):
            new_n = current_n * p
            if new_n <= 10**18:
                sc = compute_score(new_n)
                if sc > current_score:
                    current_score = sc
                    current_n = new_n
                    current_factors[p] += 1
                    improved = True
                    break
        
        if improved:
            continue
        
        # Try adding a new prime
        for p in search_primes:
            if p not in current_factors:
                new_n = current_n * p
                if new_n <= 10**18:
                    sc = compute_score(new_n)
                    if sc > current_score:
                        current_score = sc
                        current_n = new_n
                        current_factors[p] = 1
                        improved = True
                        break
        
        if improved:
            continue
        
        # Try decreasing exponent
        for p in sorted(current_factors.keys(), reverse=True):
            if current_factors[p] > 1:
                new_n = current_n // p
                sc = compute_score(new_n)
                if sc > current_score:
                    current_score = sc
                    current_n = new_n
                    current_factors[p] -= 1
                    if current_factors[p] == 0:
                        del current_factors[p]
                    improved = True
                    break
        
        if improved:
            continue
        
        # Try replacing a prime (exponent 1) with another
        for p in list(current_factors.keys()):
            if current_factors[p] != 1:
                continue
            for q in search_primes:
                if q == p or q in current_factors:
                    continue
                new_n = (current_n // p) * q
                if 2 <= new_n <= 10**18:
                    sc = compute_score(new_n)
                    if sc > current_score:
                        current_score = sc
                        current_n = new_n
                        del current_factors[p]
                        current_factors[q] = 1
                        improved = True
                        break
            if improved:
                break
        
        if not improved:
            break

    return current_n
