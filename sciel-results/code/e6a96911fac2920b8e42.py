"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
from itertools import combinations

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = 85.0

    def time_exceeded():
        return time.time() - start_time > deadline - 2.0

    # Precompute primes up to 20000 (max sum is 2*10000)
    limit = 20000
    is_prime = [True] * (limit + 1)
    is_prime[0] = is_prime[1] = False
    for i in range(2, int(limit**0.5) + 1):
        if is_prime[i]:
            for j in range(i * i, limit + 1, i):
                is_prime[j] = False
    primes = [i for i in range(2, limit + 1) if is_prime[i]]

    def prime_factors(n):
        """Return set of prime factors of n."""
        factors = set()
        if n <= 1:
            return factors
        for p in primes:
            if p * p > n:
                break
            if n % p == 0:
                factors.add(p)
                while n % p == 0:
                    n //= p
        if n > 1:
            factors.add(n)
        return factors

    def count_distinct_primes(arr):
        """Count distinct prime factors across all pairwise sums."""
        all_primes = set()
        n = len(arr)
        for i in range(n):
            for j in range(i + 1, n):
                s = arr[i] + arr[j]
                all_primes |= prime_factors(s)
        return len(all_primes)

    def is_smooth(n, prime_set):
        """Check if n has only prime factors in prime_set."""
        if n <= 1:
            return True
        for p in prime_set:
            while n % p == 0:
                n //= p
        return n == 1

    # Strategy 1: Powers of 2 - structural insight
    # Sums of powers of 2: 2^i + 2^j = 2^min(i,j) * (1 + 2^|i-j|)
    # The odd factor (1 + 2^k) for k=1..11 has limited prime factors
    powers_of_2 = [2**i for i in range(12)]
    best_result = list(powers_of_2)
    best_count = count_distinct_primes(powers_of_2)

    # Strategy 2: Clique search in P-smooth compatibility graph
    # For a given prime set P, two numbers x,y are "compatible" if x+y is P-smooth
    # We want to find a clique of size 12 in this graph
    def find_clique_greedy(prime_set, max_val=10000):
        """Greedy clique search: pick numbers whose pairwise sums are all P-smooth."""
        candidates = []
        for x in range(1, max_val + 1):
            if time_exceeded():
                return None
            valid = True
            for y in candidates:
                if not is_smooth(x + y, prime_set):
                    valid = False
                    break
            if valid:
                candidates.append(x)
            if len(candidates) >= 12:
                return candidates[:12]
        if len(candidates) >= 12:
            return candidates[:12]
        return None

    # Try various prime sets, ordered by size then by likelihood
    prime_set_candidates = [
        {2},
        {2, 3},
        {3},
        {3, 5},
        {2, 3, 5},
        {5},
        {5, 7},
        {2, 3, 5, 7},
        {7},
        {2, 5},
        {3, 7},
        {2, 3, 7},
        {2, 5, 7},
        {3, 5, 7},
    ]

    for ps in prime_set_candidates:
        if time_exceeded():
            break
        result = find_clique_greedy(ps)
        if result is not None and len(result) == 12:
            if len(set(result)) == 12 and all(1 <= x <= 10000 for x in result):
                cnt = count_distinct_primes(result)
                if cnt < best_count:
                    best_count = cnt
                    best_result = result
                if best_count <= len(ps):
                    break

    # Strategy 3: Local search refinement
    # Try to improve the best result by swapping/adjusting elements
    random.seed(42)
    improved = True
    while improved and not time_exceeded():
        improved = False
        for i in range(len(best_result)):
            if time_exceeded():
                break
            orig = best_result[i]
            # Try nearby values and some random values
            deltas = list(range(-30, 31)) + [random.randint(-500, 500) for _ in range(20)]
            random.shuffle(deltas)
            for delta in deltas:
                if delta == 0:
                    continue
                new_val = orig + delta
                if new_val < 1 or new_val > 10000:
                    continue
                if new_val in best_result:
                    continue
                trial = list(best_result)
                trial[i] = new_val
                score = count_distinct_primes(trial)
                if score < best_count:
                    best_result = trial
                    best_count = score
                    improved = True
                    break

    # Strategy 4: Structured sets
    # Odd numbers: sums are even, limiting to 2 and odd factors of small even numbers
    odds = list(range(1, 24, 2))[:12]
    cnt = count_distinct_primes(odds)
    if cnt < best_count:
        best_count = cnt
        best_result = odds

    # Multiples of small numbers
    for base in [2, 3, 4, 5, 6, 8, 10, 12]:
        if time_exceeded():
            break
        mults = [base * (i + 1) for i in range(12)]
        if all(x <= 10000 for x in mults):
            cnt = count_distinct_primes(mults)
            if cnt < best_count:
                best_count = cnt
                best_result = mults

    # Close together numbers
    close = list(range(100, 112))
    cnt = count_distinct_primes(close)
    if cnt < best_count:
        best_count = cnt
        best_result = close

    # Small numbers
    small = list(range(1, 13))
    cnt = count_distinct_primes(small)
    if cnt < best_count:
        best_count = cnt
        best_result = small

    # Strategy 5: Bitset-based clique search for {2,3,5,7}-smooth
    # Build adjacency structure for the 7-smooth compatibility graph
    if not time_exceeded():
        P7 = {2, 3, 5, 7}
        # For each number x in 1..10000, find which y in 1..10000 have x+y 7-smooth
        # Use a different approach: generate all 7-smooth numbers up to 20000
        smooth_numbers = set()
        for a in range(1, 15):  # 2^14 > 20000
            v = 2**a
            if v > 20000:
                break
            for b in range(1, 10):  # 3^9 > 20000
                v2 = v * (3**b)
                if v2 > 20000:
                    break
                for c in range(1, 8):  # 5^7 > 20000
                    v3 = v2 * (5**c)
                    if v3 > 20000:
                        break
                    for d in range(1, 7):  # 7^6 > 20000
                        v4 = v3 * (7**d)
                        if v4 > 20000:
                            break
                        smooth_numbers.add(v4)
        
        # For each x, the compatible y values are those where x+y is in smooth_numbers
        # i.e., y = s - x for s in smooth_numbers, 1 <= y <= 10000, y != x
        # Build a mapping: for each x, the set of compatible y's
        # This is too large for full storage, so use greedy with this knowledge
        
        # Greedy with 7-smooth: pick x, then for each candidate y, check if x+y is 7-smooth
        def is_7smooth(n):
            if n <= 0:
                return False
            return is_smooth(n, P7)
        
        # More targeted greedy: prefer numbers that have many compatible partners
        # Start with small numbers that are 7-smooth themselves
        seven_smooth_small = [x for x in range(1, 10001) if is_7smooth(x)]
        
        if len(seven_smooth_small) >= 12:
            # Try greedy starting from 7-smooth numbers
            candidates = []
            for x in seven_smooth_small:
                if time_exceeded():
                    break
                valid = True
                for y in candidates:
                    if not is_7smooth(x + y):
                        valid = False
                        break
                if valid:
                    candidates.append(x)
                if len(candidates) >= 12:
                    break
            
            if len(candidates) == 12:
                cnt = count_distinct_primes(candidates)
                if cnt < best_count:
                    best_count = cnt
                    best_result = candidates

    return best_result
# EVOLVE-BLOCK-END
