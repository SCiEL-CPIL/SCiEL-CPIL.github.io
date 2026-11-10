import time
import math
import random

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # 145s deadline, 5s safety margin

    # 1. Generate primes up to a large limit
    def sieve(n):
        is_prime = [True] * (n + 1)
        is_prime[0] = is_prime[1] = False
        for i in range(2, int(n**0.5) + 1):
            if is_prime[i]:
                for j in range(i*i, n+1, i):
                    is_prime[j] = False
        return [i for i, v in enumerate(is_prime) if v]

    primes = sieve(100000)
    prime_set_global = set(primes)

    # 2. Efficient tau_perp calculation using bitmask tracking
    def tau_perp(prime_factors):
        # Generate divisors with their prime-support masks in sorted order
        pairs = [(1, 0)]  # (divisor, bitmask)
        for idx, p in enumerate(prime_factors):
            bit = 1 << idx
            merged = []
            i = 0
            j = 0
            len_p = len(pairs)
            while i < len_p or j < len_p:
                if j >= len_p:
                    merged.append(pairs[i])
                    i += 1
                elif i >= len_p:
                    d, m = pairs[j]
                    merged.append((d * p, m | bit))
                    j += 1
                else:
                    a_d, a_m = pairs[i]
                    b_d, b_m = (pairs[j][0] * p, pairs[j][1] | bit)
                    if a_d <= b_d:
                        merged.append((a_d, a_m))
                        i += 1
                    else:
                        merged.append((b_d, b_m))
                        j += 1
            pairs = merged
        
        # Count adjacent coprime pairs (disjoint prime supports)
        score = 0
        for i in range(len(pairs) - 1):
            if (pairs[i][1] & pairs[i+1][1]) == 0:
                score += 1
        return score

    # 3. Greedy set-packing heuristic:
    #    Model the divisor lattice as a hypercube. Each vertex is a subset of primes.
    #    Two adjacent vertices in sorted numerical order have coprime gcd=1 iff
    #    their prime-support masks are disjoint.
    #    
    #    Key insight: The sorted order of divisors is determined by the numerical
    #    values, which depend on the actual prime values. We want to choose primes
    #    such that when divisors are sorted, many adjacent pairs have disjoint masks.
    #
    #    Strategy: Start with the smallest primes (maximizes k), then use a global
    #    optimization that evaluates the "disjoint-neighbor density" of the sorted
    #    divisor sequence. We perform a simulated annealing-style search over the
    #    prime set, but with a different move structure: we swap primes based on
    #    their position in the sorted divisor sequence and their contribution to
    #    disjoint-adjacency.
    
    # Find maximum k
    prod = 1
    max_k = 0
    for p in primes:
        if prod * p > 10**18:
            break
        prod *= p
        max_k += 1
    
    # Initialize with smallest primes
    current_primes = list(primes[:max_k])
    current_n = 1
    for p in current_primes:
        current_n *= p
    
    best_n = current_n
    current_score = tau_perp(current_primes)
    best_score = current_score
    
    # 4. Global combinatorial search:
    #    Instead of local swaps, we use a different representation:
    #    - For each prime in the set, compute its "disjoint contribution" to the
    #      sorted divisor sequence.
    #    - Primes that appear in many disjoint-adjacent pairs are valuable.
    #    - We perform moves that replace low-contribution primes with high-potential
    #      candidates, evaluated via the full tau_perp score.
    
    # Precompute candidate primes (those that could replace a current prime)
    # For each current prime p_old, max_p_new = 10^18 * p_old // current_n
    
    # Simulated annealing with global moves
    temp = 10.0
    cooling = 0.999
    
    improved = True
    no_improve = 0
    
    while time.time() < deadline - 10.0:
        improved = False
        no_improve += 1
        
        # Try a batch of random swaps
        batch_size = min(50, len(current_primes) * 2)
        for _ in range(batch_size):
            if time.time() > deadline - 5.0:
                return best_n
            
            idx = random.randint(0, len(current_primes) - 1)
            p_old = current_primes[idx]
            max_p_new = (10**18 * p_old) // current_n
            
            # Find candidate primes not in current set, <= max_p_new
            candidates = [p for p in primes if p not in set(current_primes) and p <= max_p_new]
            if not candidates:
                continue
            
            p_new = random.choice(candidates)
            new_primes = list(current_primes)
            new_primes[idx] = p_new
            
            s_new = tau_perp(new_primes)
            
            # Accept if better, or with probability exp(-delta/temp) for worse
            if s_new > current_score:
                current_score = s_new
                current_primes = new_primes
                current_n = 1
                for p in current_primes:
                    current_n *= p
                if s_new > best_score:
                    best_score = s_new
                    best_n = current_n
                improved = True
                no_improve = 0
            elif temp > 1.0:
                delta = current_score - s_new
                if random.random() < math.exp(-delta / temp):
                    current_score = s_new
                    current_primes = new_primes
                    current_n = 1
                    for p in current_primes:
                        current_n *= p
                    if s_new > best_score:
                        best_score = s_new
                        best_n = current_n
        
        temp *= cooling
        
        if not improved and no_improve >= 10 and time.time() < deadline - 20.0:
            # Restart: go back to best known, then perturb
            current_primes = []
            prod = 1
            for p in primes:
                if prod * p > 10**18:
                    break
                prod *= p
                current_primes.append(p)
            current_n = 1
            for p in current_primes:
                current_n *= p
            current_score = tau_perp(current_primes)
            no_improve = 0
            temp = 10.0
    
    return best_n
