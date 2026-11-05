"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import math
import sys

def _sieve(n):
    is_prime = [True] * (n + 1)
    if n >= 0: is_prime[0] = False
    if n >= 1: is_prime[1] = False
    for i in range(2, int(math.isqrt(n)) + 1):
        if is_prime[i]:
            for j in range(i * i, n + 1, i):
                is_prime[j] = False
    return [i for i in range(2, n + 1) if is_prime[i]]

def _divisors_from_exp(primes, exps):
    """Return sorted divisors of n = prod(primes[i]^exps[i])."""
    divs = [1]
    for p, e in zip(primes, exps):
        pw = [1]
        for _ in range(e):
            pw.append(pw[-1] * p)
        new_divs = []
        for d in divs:
            for w in pw:
                new_divs.append(d * w)
        divs = new_divs
    divs.sort()
    return divs

def _tau_perp(n):
    """Compute tau_perp(n): number of adjacent coprime pairs in sorted divisors."""
    if n < 2:
        return 0
    # factor n
    temp = n
    factors = []
    for p in range(2, int(math.isqrt(temp)) + 1):
        if temp % p == 0:
            e = 0
            while temp % p == 0:
                temp //= p
                e += 1
            factors.append((p, e))
        if temp == 1:
            break
    if temp > 1:
        factors.append((temp, 1))
    
    primes = [f[0] for f in factors]
    exps = [f[1] for f in factors]
    
    divs = _divisors_from_exp(primes, exps)
    score = 0
    for i in range(len(divs) - 1):
        if math.gcd(divs[i], divs[i + 1]) == 1:
            score += 1
    return score

def _generate_primes(limit):
    """Generate primes up to limit using simple sieve."""
    if limit < 2:
        return []
    is_prime = [True] * (limit + 1)
    is_prime[0] = is_prime[1] = False
    for i in range(2, int(math.isqrt(limit)) + 1):
        if is_prime[i]:
            for j in range(i * i, limit + 1, i):
                is_prime[j] = False
    return [i for i in range(2, limit + 1) if is_prime[i]]

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin for 150s limit
    
    # Generate primes up to a reasonable bound
    # We need primes up to maybe 1000 for local search
    primes = _generate_primes(500)
    
    # Start with product of first 15 primes (squarefree)
    # 2*3*5*7*11*13*17*19*23*29*31*37*41*43*47 = 614889782588491410
    initial_primes_idx = list(range(15))
    n_best = 1
    for idx in initial_primes_idx:
        n_best *= primes[idx]
    score_best = _tau_perp(n_best)
    
    # Local search: try to adjust exponents and prime selection
    # Strategy: start with 15 primes, try increasing exponents of smaller primes
    # by removing larger primes to stay under 10^18
    
    # First, let's try a few hand-crafted configurations
    candidates = []
    
    # Base: 15 primes, all exponent 1
    base_exps = [1] * 15
    candidates.append((initial_primes_idx, base_exps))
    
    # Try: remove largest primes, increase exponents of small primes
    # For example, keep first 13 primes, give 2 a higher exponent
    for start_k in range(10, 16):
        for extra_exp in range(1, 8):
            idxs = list(range(start_k))
            exps = [1] * start_k
            exps[0] += extra_exp  # increase exponent of 2
            # Check if product is within limit
            prod = 1
            for i in range(start_k):
                prod *= primes[i] ** exps[i]
                if prod > 10**18:
                    break
            if prod <= 10**18 and prod >= 2:
                candidates.append((idxs, exps))
            
            # Also try increasing exponent of 3
            exps3 = [1] * start_k
            exps3[1] += extra_exp
            prod = 1
            for i in range(start_k):
                prod *= primes[i] ** exps3[i]
                if prod > 10**18:
                    break
            if prod <= 10**18 and prod >= 2:
                candidates.append((idxs, exps3))
    
    # Try a more systematic local search
    # Start with 15 primes, all exp 1
    current_idx = list(range(15))
    current_exp = [1] * 15
    current_n = n_best
    current_score = score_best
    
    # Local search steps
    for step in range(5000):
        if time.time() > deadline - 5.0:
            break
        
        improved = False
        # Try increasing exponent of a prime
        for i in range(len(current_idx)):
            new_exp = current_exp[i] + 1
            new_n = current_n * primes[current_idx[i]]
            if new_n > 10**18:
                continue
            # Remove a larger prime to make room if needed
            # Actually, let's just try removing the largest prime if we exceed
            # But first check if we're already within limit
            if new_n <= 10**18:
                new_score = _tau_perp(new_n)
                if new_score > current_score:
                    current_exp[i] = new_exp
                    current_n = new_n
                    current_score = new_score
                    improved = True
                    break
        
        if improved:
            continue
        
        # Try replacing a large prime with a smaller exponent of a small prime
        # This is more complex, so let's try a different approach:
        # Try removing the largest prime and see if we can add it back with higher exponent
        if len(current_idx) > 1:
            largest_idx = len(current_idx) - 1
            p_largest = primes[current_idx[largest_idx]]
            # Remove largest prime
            temp_n = current_n // p_largest
            temp_idx = current_idx[:-1]
            temp_exp = current_exp[:-1]
            
            # Try increasing exponents of smaller primes
            for i in range(len(temp_idx)):
                for extra in range(1, 5):
                    new_n = temp_n * (primes[temp_idx[i]] ** extra)
                    if new_n > 10**18:
                        break
                    new_score = _tau_perp(new_n)
                    if new_score > current_score:
                        # Rebuild
                        new_idx = temp_idx[:]
                        new_exp = temp_exp[:]
                        new_exp[i] += extra
                        # Try to add back a prime
                        for j in range(len(temp_idx), 20):
                            if j < len(primes):
                                candidate_n = new_n * primes[j]
                                if candidate_n <= 10**18:
                                    cand_score = _tau_perp(candidate_n)
                                    if cand_score > new_score:
                                        new_n = candidate_n
                                        new_score = cand_score
                                        new_idx.append(j)
                                        new_exp.append(1)
                                        break
                        if new_score > current_score:
                            current_idx = new_idx
                            current_exp = new_exp
                            current_n = new_n
                            current_score = new_score
                            improved = True
                            break
                if improved:
                    break
        
        if not improved:
            # Try a random restart with different prime set
            if step % 500 == 0 and step > 0:
                # Try a random configuration
                k = 13 + (step % 3)
                if k <= len(primes):
                    rand_idx = list(range(k))
                    rand_exp = [1] * k
                    # Randomly increase some exponents
                    for i in range(k):
                        if i < 3:
                            rand_exp[i] = 1 + (step % 4)
                    rand_n = 1
                    for i in range(k):
                        rand_n *= primes[rand_idx[i]] ** rand_exp[i]
                    if 2 <= rand_n <= 10**18:
                        rand_score = _tau_perp(rand_n)
                        if rand_score > current_score:
                            current_idx = rand_idx
                            current_exp = rand_exp
                            current_n = rand_n
                            current_score = rand_score
            else:
                # No improvement, stop if we've been stuck
                if step > 1000:
                    break
    
    # Update best
    if current_n >= 2 and current_n <= 10**18:
        if current_score > score_best:
            n_best = current_n
            score_best = current_score
    
    return n_best
# EVOLVE-BLOCK-END
