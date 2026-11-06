"""
Structural approach: Model the problem as a dynamic programming / search over
prime exponent vectors, using the insight that square-free numbers with the
smallest primes (primorial) maximize coprime adjacent divisor pairs.

Key structural change: Instead of just returning a fixed primorial, we perform
a systematic local search over exponent vectors starting from the primorial,
using efficient score computation. The search explores:
1. Squaring individual primes
2. Swapping primes (removing one, adding another)
3. Removing primes and adjusting others
4. Iterative local improvement

The score function is computed by generating divisors and counting coprime
adjacent pairs. We use time management to stay within the 150s budget.
"""
import time
from math import gcd

def _primes_up_to(limit):
    """Generate primes up to limit using sieve."""
    if limit < 2:
        return []
    sieve = [True] * (limit + 1)
    sieve[0] = sieve[1] = False
    for i in range(2, int(limit**0.5) + 1):
        if sieve[i]:
            for j in range(i * i, limit + 1, i):
                sieve[j] = False
    return [i for i, v in enumerate(sieve) if v]

# Precompute primes up to 500 (enough for swapping)
_PRIMES = _primes_up_to(500)

def _generate_divisors(primes, exponents):
    """Generate all divisors from prime factorization."""
    divisors = [1]
    for p, e in zip(primes, exponents):
        if e == 0:
            continue
        current = list(divisors)
        pk = 1
        for k in range(1, e + 1):
            pk *= p
            for d in current:
                divisors.append(d * pk)
    return divisors

def _tau_perp(primes, exponents):
    """Compute tau_perp score from prime factorization."""
    divisors = _generate_divisors(primes, exponents)
    divisors.sort()
    if len(divisors) < 2:
        return 0
    count = 0
    for i in range(len(divisors) - 1):
        if gcd(divisors[i], divisors[i + 1]) == 1:
            count += 1
    return count

def _compute_n(primes, exponents):
    """Compute n from prime factorization."""
    n = 1
    for p, e in zip(primes, exponents):
        if e > 0:
            n *= p ** e
    return n

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s limit
    
    max_n = 10**18
    primes = _PRIMES
    
    # Find the largest primorial <= 10^18
    prod = 1
    max_k = 0
    for i, p in enumerate(primes):
        if prod * p <= max_n:
            prod *= p
            max_k = i + 1
        else:
            break
    
    # Base case: primorial of first max_k primes (all exponent 1)
    base_exps = [1] * max_k
    base_n = prod
    base_score = _tau_perp(primes[:max_k], base_exps)
    
    best_score = base_score
    best_n = base_n
    best_exps = base_exps[:]
    
    # Strategy 1: Try squaring one prime in the primorial
    for i in range(max_k):
        if time.time() > deadline:
            break
        new_exps = base_exps[:]
        new_exps[i] = 2
        new_n = base_n * primes[i]
        if new_n <= max_n:
            score = _tau_perp(primes[:max_k], new_exps)
            if score > best_score:
                best_score = score
                best_n = new_n
                best_exps = new_exps[:]
    
    # Strategy 2: Remove prime i, add prime j (j >= max_k)
    for i in range(max_k):
        if time.time() > deadline:
            break
        for j in range(max_k, min(max_k + 30, len(primes))):
            temp_n = base_n // primes[i] * primes[j]
            if temp_n > max_n:
                break
            # Build new prime/exponent lists
            used_primes = []
            used_exps = []
            for k in range(max_k):
                if k != i:
                    used_primes.append(primes[k])
                    used_exps.append(1)
            used_primes.append(primes[j])
            used_exps.append(1)
            score = _tau_perp(used_primes, used_exps)
            if score > best_score:
                best_score = score
                best_n = temp_n
                # Store in a canonical form: use the full primes array with zeros
                new_exps = [0] * len(primes)
                for k in range(max_k):
                    if k != i:
                        new_exps[k] = 1
                new_exps[j] = 1
                best_exps = new_exps
    
    # Strategy 3: Remove prime i, square prime j (j != i, both in first max_k)
    for i in range(max_k):
        if time.time() > deadline:
            break
        for j in range(max_k):
            if i == j:
                continue
            new_exps = [0] * len(primes)
            for k in range(max_k):
                if k != i:
                    new_exps[k] = 1
            new_exps[j] = 2
            new_n = 1
            for k in range(len(primes)):
                if new_exps[k] > 0:
                    new_n *= primes[k] ** new_exps[k]
            if new_n <= max_n:
                used_primes = [primes[k] for k in range(len(primes)) if new_exps[k] > 0]
                used_exps = [new_exps[k] for k in range(len(primes)) if new_exps[k] > 0]
                score = _tau_perp(used_primes, used_exps)
                if score > best_score:
                    best_score = score
                    best_n = new_n
                    best_exps = new_exps
    
    # Strategy 4: Try primorial with one fewer prime but two primes squared
    if max_k >= 3:
        for i in range(max_k - 1):
            if time.time() > deadline:
                break
            for j in range(i + 1, max_k - 1):
                new_exps = [0] * len(primes)
                for k in range(max_k - 1):
                    new_exps[k] = 1
                new_exps[i] = 2
                new_exps[j] = 2
                new_n = 1
                for k in range(len(primes)):
                    if new_exps[k] > 0:
                        new_n *= primes[k] ** new_exps[k]
                if new_n <= max_n:
                    used_primes = [primes[k] for k in range(len(primes)) if new_exps[k] > 0]
                    used_exps = [new_exps[k] for k in range(len(primes)) if new_exps[k] > 0]
                    score = _tau_perp(used_primes, used_exps)
                    if score > best_score:
                        best_score = score
                        best_n = new_n
                        best_exps = new_exps
    
    # Strategy 5: Local search from best found so far
    if time.time() < deadline:
        current_exps = best_exps[:]
        current_n = best_n
        current_score = best_score
        
        improved = True
        while improved and time.time() < deadline:
            improved = False
            
            # Try increasing an exponent
            for i in range(len(current_exps)):
                if current_exps[i] >= 4:
                    continue
                new_exps = current_exps[:]
                new_exps[i] += 1
                new_n = current_n * primes[i]
                if new_n <= max_n:
                    used_primes = [primes[k] for k in range(len(current_exps)) if new_exps[k] > 0]
                    used_exps = [new_exps[k] for k in range(len(current_exps)) if new_exps[k] > 0]
                    score = _tau_perp(used_primes, used_exps)
                    if score > current_score:
                        current_exps = new_exps
                        current_n = new_n
                        current_score = score
                        improved = True
                        if current_score > best_score:
                            best_score = current_score
                            best_n = current_n
                            best_exps = current_exps[:]
            
            # Try decreasing an exponent (if > 1)
            for i in range(len(current_exps)):
                if current_exps[i] <= 1:
                    continue
                new_exps = current_exps[:]
                new_exps[i] -= 1
                new_n = current_n // primes[i]
                if new_n >= 2:
                    used_primes = [primes[k] for k in range(len(current_exps)) if new_exps[k] > 0]
                    used_exps = [new_exps[k] for k in range(len(current_exps)) if new_exps[k] > 0]
                    score = _tau_perp(used_primes, used_exps)
                    if score > current_score:
                        current_exps = new_exps
                        current_n = new_n
                        current_score = score
                        improved = True
                        if current_score > best_score:
                            best_score = current_score
                            best_n = current_n
                            best_exps = current_exps[:]
            
            # Try swapping: remove prime i (set to 0), add prime j (set to 1)
            for i in range(len(current_exps)):
                if current_exps[i] == 0:
                    continue
                for j in range(len(current_exps)):
                    if j == i or current_exps[j] > 0:
                        continue
                    new_exps = current_exps[:]
                    new_exps[i] = 0
                    new_exps[j] = 1
                    new_n = current_n // primes[i] * primes[j]
                    if new_n <= max_n and new_n >= 2:
                        used_primes = [primes[k] for k in range(len(current_exps)) if new_exps[k] > 0]
                        used_exps = [new_exps[k] for k in range(len(current_exps)) if new_exps[k] > 0]
                        score = _tau_perp(used_primes, used_exps)
                        if score > current_score:
                            current_exps = new_exps
                            current_n = new_n
                            current_score = score
                            improved = True
                            if current_score > best_score:
                                best_score = current_score
                                best_n = current_n
                                best_exps = current_exps[:]
    
    # Strategy 6: Try some random restarts if time permits
    if time.time() < deadline - 10:
        import random
        random.seed(42)
        for _ in range(20):
            if time.time() > deadline - 5:
                break
            # Start from a random subset of primes
            num_primes = random.randint(8, min(15, max_k))
            indices = random.sample(range(max_k), num_primes)
            indices.sort()
            new_exps = [0] * len(primes)
            for idx in indices:
                new_exps[idx] = 1
            new_n = 1
            for k in range(len(primes)):
                if new_exps[k] > 0:
                    new_n *= primes[k]
            if new_n <= max_n and new_n >= 2:
                used_primes = [primes[k] for k in range(len(primes)) if new_exps[k] > 0]
                used_exps = [new_exps[k] for k in range(len(primes)) if new_exps[k] > 0]
                score = _tau_perp(used_primes, used_exps)
                if score > best_score:
                    best_score = score
                    best_n = new_n
                    best_exps = new_exps
    
    return best_n
