"""Combined program: bitmask-accelerated greedy + drop-and-re-greedy local search + swap polish."""
# EVOLVE-BLOCK-START
import math
import time

def _sieve_primes(limit):
    """Return list of primes <= limit."""
    sieve = [True] * (limit + 1)
    sieve[0] = sieve[1] = False
    for i in range(2, int(limit**0.5) + 1):
        if sieve[i]:
            for j in range(i*i, limit + 1, i):
                sieve[j] = False
    return [i for i, is_p in enumerate(sieve) if is_p]

def _factorize(n, primes):
    """Return list of distinct prime factors of n."""
    factors = []
    temp = n
    for p in primes:
        if p * p > temp:
            break
        if temp % p == 0:
            factors.append(p)
            while temp % p == 0:
                temp //= p
    if temp > 1:
        factors.append(temp)
    return factors

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 140s safety margin within 150s limit

    N = 512
    primes = _sieve_primes(N - 1)

    # Precompute prime factors for each number 1..511
    factor_cache = {}
    for a in range(1, N):
        if a == 1:
            factor_cache[a] = []
        else:
            factor_cache[a] = _factorize(a, primes)

    # Precompute conflict masks: mask[a] has bit b set if gcd(a, b) != 1
    # Index by value 1..511
    mask = [0] * N
    for a in range(1, N):
        m = 0
        fa = set(factor_cache[a])
        for b in range(1, N):
            if a != b:
                fb = set(factor_cache[b])
                if fa & fb:
                    m |= (1 << b)
        mask[a] = m

    def score(S):
        return sum(1.0 / (N - a) for a in S)

    def greedy(order, exclude=None):
        """Greedy selection in given order, optionally excluding a set of elements."""
        S = []
        chosen_mask = 0
        for a in order:
            if exclude is not None and a in exclude:
                continue
            if (chosen_mask & mask[a]) == 0:
                S.append(a)
                chosen_mask |= (1 << a)
        return S

    # Initial greedy in descending order (prefers large a -> large 1/(512-a))
    best = greedy(range(N - 1, 0, -1))
    best_score = score(best)

    # Local search phase 1: drop one chosen element, then re-greedy descending
    improved = True
    while improved and time.time() < deadline:
        improved = False
        for x in list(best):
            if time.time() >= deadline:
                break
            S2 = greedy(range(N - 1, 0, -1), exclude={x})
            s2 = score(S2)
            if s2 > best_score + 1e-12:
                best = S2
                best_score = s2
                improved = True
                break

    # Local search phase 2: swap-based improvement
    # Try removing one element and adding a better one that fits
    improved2 = True
    while improved2 and time.time() < deadline:
        improved2 = False
        best_set = set(best)
        best_mask = 0
        for a in best:
            best_mask |= (1 << a)
        
        for x in list(best):
            if time.time() >= deadline:
                break
            # Remove x
            rest_mask = best_mask & ~(1 << x)
            best_without_x = [a for a in best if a != x]
            best_without_x_score = sum(1.0 / (N - a) for a in best_without_x)
            
            # Try adding each candidate not in best that is coprime to (best - {x})
            for y in range(N - 1, 0, -1):
                if y in best_set:
                    continue
                if (rest_mask & mask[y]) != 0:
                    continue
                new_score = best_without_x_score + 1.0 / (N - y)
                if new_score > best_score + 1e-12:
                    best = best_without_x + [y]
                    best_score = new_score
                    best_set = set(best)
                    best_mask = 0
                    for a in best:
                        best_mask |= (1 << a)
                    improved2 = True
                    break

    # Local search phase 3: try adding any element that fits (no removal)
    improved3 = True
    while improved3 and time.time() < deadline:
        improved3 = False
        best_set = set(best)
        best_mask = 0
        for a in best:
            best_mask |= (1 << a)
        
        for a in range(N - 1, 0, -1):
            if time.time() >= deadline:
                break
            if a in best_set:
                continue
            if (best_mask & mask[a]) == 0:
                # a is coprime to all in best
                new_score = best_score + 1.0 / (N - a)
                if new_score > best_score + 1e-12:
                    best.append(a)
                    best_score = new_score
                    best_set.add(a)
                    best_mask |= (1 << a)
                    improved3 = True
                    break

    # Final polish: try multiple random restarts with different orderings
    import random
    random.seed(42)
    
    for _ in range(10):  # Up to 10 restarts if time permits
        if time.time() >= deadline:
            break
        
        # Create a shuffled ordering with bias towards high values
        order = list(range(N - 1, 0, -1))
        # Shuffle but keep high values somewhat prioritized
        for i in range(len(order) - 1, 0, -1):
            if random.random() < 0.3:  # 30% chance to swap
                j = random.randint(0, i)
                order[i], order[j] = order[j], order[i]
        
        S_new = greedy(order)
        s_new = score(S_new)
        if s_new > best_score + 1e-12:
            best = S_new
            best_score = s_new

    return best
# EVOLVE-BLOCK-END
