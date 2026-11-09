import math
import time

def _divisors_of_squarefree(primes):
    """Generate all divisors of a squarefree number (product of distinct primes)."""
    divisors = [1]
    for p in primes:
        new_divs = [d * p for d in divisors]
        divisors.extend(new_divs)
    divisors.sort()
    return divisors

def _tau_perp_squarefree(primes):
    """Count adjacent coprime pairs among divisors of a squarefree number."""
    divs = _divisors_of_squarefree(primes)
    count = 0
    for i in range(len(divs) - 1):
        if math.gcd(divs[i], divs[i + 1]) == 1:
            count += 1
    return count

def _score_primes(prime_list):
    """Score a list of primes as a squarefree number."""
    n = 1
    for p in prime_list:
        n *= p
    if n > 10**18:
        return -1, n
    return _tau_perp_squarefree(prime_list), n

def _is_prime(n):
    """Check if n is prime."""
    if n < 2:
        return False
    if n < 4:
        return True
    if n % 2 == 0 or n % 3 == 0:
        return False
    i = 5
    while i * i <= n:
        if n % i == 0 or n % (i + 2) == 0:
            return False
        i += 6
    return True

def construct():
    start_time = time.time()
    deadline = start_time + 145  # Safety margin before 150s kill
    
    # Generate primes up to ~200 for a wider search space
    all_primes = []
    for x in range(2, 250):
        if _is_prime(x):
            all_primes.append(x)
    
    best_score = -1
    best_n = 2
    best_primes = [2]
    
    def try_candidate(prime_list):
        nonlocal best_score, best_n, best_primes
        if time.time() > deadline:
            return False
        s, n = _score_primes(prime_list)
        if n <= 10**18 and s > best_score:
            best_score = s
            best_n = n
            best_primes = list(prime_list)
        return True
    
    # Phase 1: Try primorials of first k primes for k=2..15
    for k in range(2, 16):
        primes_k = all_primes[:k]
        n_prod = 1
        for p in primes_k:
            n_prod *= p
        if n_prod > 10**18:
            break
        try_candidate(primes_k)
    
    # Phase 2: Iterated local search with multiple restarts
    # Use the best found so far as starting point, then do hill climbing
    
    def local_search(initial_primes, max_iter=500):
        nonlocal best_score, best_n, best_primes
        current_primes = list(initial_primes)
        current_n = 1
        for p in current_primes:
            current_n *= p
        
        for iteration in range(max_iter):
            if time.time() > deadline:
                return False
            improved = False
            
            # Try single swaps: remove one prime, add another
            for i in range(len(current_primes)):
                for j in range(len(all_primes)):
                    if all_primes[j] in current_primes:
                        continue
                    new_n = current_n // current_primes[i] * all_primes[j]
                    if new_n > 10**18:
                        continue
                    candidate = current_primes[:i] + [all_primes[j]] + current_primes[i+1:]
                    candidate.sort()
                    s, n = _score_primes(candidate)
                    if n <= 10**18 and s > best_score:
                        best_score = s
                        best_n = n
                        best_primes = list(candidate)
                    if n <= 10**18 and s >= _tau_perp_squarefree(current_primes):
                        current_primes = candidate
                        current_n = n
                        improved = True
                        break
                if improved:
                    break
            if not improved:
                break
        return True
    
    # Phase 3: Multiple local search restarts from different starting points
    # Start from primorials of various sizes
    for start_k in range(10, 16):
        if time.time() > deadline:
            return best_n
        primes_k = all_primes[:start_k]
        n_prod = 1
        for p in primes_k:
            n_prod *= p
        if n_prod <= 10**18:
            local_search(primes_k, max_iter=200)
    
    # Phase 4: Local search from best found
    if time.time() < deadline:
        local_search(best_primes, max_iter=1000)
    
    # Phase 5: Try adding/removing primes (change count)
    if time.time() < deadline:
        for target_size in range(12, 17):
            if time.time() > deadline:
                return best_n
            # Try to find a good set of target_size primes
            # Start with smallest target_size primes
            base = all_primes[:target_size]
            n_prod = 1
            for p in base:
                n_prod *= p
            if n_prod <= 10**18:
                local_search(base, max_iter=300)
    
    # Phase 6: Final local search polish
    if time.time() < deadline:
        local_search(best_primes, max_iter=500)
    
    return best_n
