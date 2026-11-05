"""
Strategy: Instead of local search on a fixed set of primes, use a global
optimization approach:

1. The key insight: for a squarefree number with k primes, the score is
   (2^k - 1) - (number of adjacent pairs sharing a prime).
   For non-squarefree, the structure is more complex.

2. Key realization: The optimal n is likely close to the product of the
   smallest primes, but with some exponents > 1. The local search in the
   current program is slow because it regenerates all divisors from scratch
   each time.

3. NEW APPROACH: Use a different scoring method.
   - For a given set of primes and exponents, we can compute the score
     WITHOUT generating all divisors, by using the structure of the divisor
     lattice.
   - Actually, generating divisors is O(tau(n)), which is unavoidable if we
     want the exact score. But we can make it faster.

4. Actually, let me think about this differently. The score depends on the
   SORTED order of divisors. Two adjacent divisors d_i < d_{i+1} are coprime
   iff they share no common prime factor.

5. For a number n = p1^e1 * p2^e2 * ... * pk^ek, the divisors can be
   represented as tuples (a1, a2, ..., ak) where 0 <= ai <= ei.
   The divisor value is prod(pi^ai).
   Two divisors are coprime iff for all j, min(aj, bj) = 0.

6. The challenge is that the SORTED order by value is not the same as
   lexicographic order of the tuples.

7. Let me try a completely different approach: SIMULATED ANNEALING.
   - Start from the product of the first ~14 primes (squarefree).
   - Moves: increase/decrease exponent of a random prime, or swap primes.
   - Use a cooling schedule.
   - This explores the space more broadly than greedy local search.

8. Even better: I'll use a hybrid.
   - Phase 1: Greedy construction of a good squarefree seed.
   - Phase 2: Fast local search using INCREMENTAL score updates where possible.
   - Phase 3: Simulated annealing for broader exploration.

Actually, the simplest big improvement: the current program's local search
is limited to 20 primes and only does +/- 1 moves. Let me:
- Allow larger moves (change exponent by more than 1)
- Use a smarter initial configuration
- Vectorize the scoring more aggressively

Let me focus on making the scoring as fast as possible, then do more iterations.

Key speedup: Precompute prime powers. When we change exponent of prime i from
e to e', the divisors change in a structured way. But it's complex.

Simplest fast approach: Use numpy for everything.
- Generate divisors as a numpy array
- Sort with numpy
- Compute gcd with numpy

This should be 10-50x faster than the Python loop version.
"""

import time
import math
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin for 150s limit

    # Generate primes up to 200 (plenty for our needs)
    def sieve(n):
        is_prime = [True] * (n + 1)
        is_prime[0] = is_prime[1] = False
        for i in range(2, int(math.isqrt(n)) + 1):
            if is_prime[i]:
                for j in range(i * i, n + 1, i):
                    is_prime[j] = False
        return [i for i, prime in enumerate(is_prime) if prime]

    primes = sieve(200)
    num_primes = len(primes)

    def calc_score_v2(exps):
        """Calculate tau_perp score using numpy for speed."""
        # Check product constraint
        n_val = 1
        for i, e in enumerate(exps):
            if e > 0:
                n_val *= primes[i] ** e
        if n_val > 10**18 or n_val < 2:
            return -1

        # Generate all divisors
        divs = np.array([1], dtype=np.int64)
        for i, e in enumerate(exps):
            if e == 0:
                continue
            p = primes[i]
            p_pows = np.array([p ** k for k in range(e + 1)], dtype=np.int64)
            # Outer product: each existing divisor times each power
            divs = np.outer(divs, p_pows).ravel()

        if len(divs) < 2:
            return 0

        divs.sort()

        # Vectorized gcd of adjacent pairs
        g = np.gcd(divs[:-1], divs[1:])
        score = int(np.sum(g == 1))
        return score

    # Phase 1: Greedy squarefree seed
    current_divs = [1]
    current_n = 1
    current_score = 0
    seed_exps = [0] * num_primes

    for i, p in enumerate(primes):
        if time.time() > deadline:
            break
        new_n = current_n * p
        if new_n > 10**18:
            break

        # Quick merge for squarefree
        old_divs = current_divs
        new_divs_part = [d * p for d in old_divs]
        merged = []
        idx_i, idx_j = 0, 0
        while idx_i < len(old_divs) and idx_j < len(new_divs_part):
            if old_divs[idx_i] < new_divs_part[idx_j]:
                merged.append(old_divs[idx_i])
                idx_i += 1
            else:
                merged.append(new_divs_part[idx_j])
                idx_j += 1
        if idx_i < len(old_divs):
            merged.extend(old_divs[idx_i:])
        if idx_j < len(new_divs_part):
            merged.extend(new_divs_part[idx_j:])

        s = 0
        for k in range(len(merged) - 1):
            if math.gcd(merged[k], merged[k + 1]) == 1:
                s += 1

        if s > current_score:
            current_divs = merged
            current_n = new_n
            current_score = s
            seed_exps[i] = 1

    # Phase 2: Local search with fast scoring
    # Use first 18 primes (product of first 15 is ~6e17, leaving room for exponents)
    search_primes = 18
    best_exps = seed_exps[:search_primes]
    if len(best_exps) < search_primes:
        best_exps.extend([0] * (search_primes - len(best_exps)))

    best_score = calc_score_v2(best_exps)

    # Local search: try all +/- 1 moves, plus some larger moves
    improved = True
    max_iter = 0
    while improved and time.time() < deadline:
        improved = False
        max_iter += 1
        for i in range(search_primes):
            if time.time() > deadline:
                break

            # Try increasing exponent
            for delta in [1, 2]:
                if best_exps[i] + delta <= 25:
                    new_exps = best_exps[:]
                    new_exps[i] += delta
                    s = calc_score_v2(new_exps)
                    if s > best_score:
                        best_score = s
                        best_exps = new_exps
                        improved = True
                        break  # Restart scan from this prime

            # Try decreasing exponent
            if best_exps[i] > 0:
                for delta in [1, 2]:
                    if best_exps[i] - delta >= 0:
                        new_exps = best_exps[:]
                        new_exps[i] -= delta
                        s = calc_score_v2(new_exps)
                        if s > best_score:
                            best_score = s
                            best_exps = new_exps
                            improved = True
                            break

            # Try adding a new prime (if currently zero)
            if best_exps[i] == 0:
                new_exps = best_exps[:]
                new_exps[i] = 1
                s = calc_score_v2(new_exps)
                if s > best_score:
                    best_score = s
                    best_exps = new_exps
                    improved = True

    # Phase 3: Simulated annealing for broader exploration
    if time.time() < deadline - 20:
        current_exps = best_exps[:]
        current_score = best_score
        T = 50.0
        cooling = 0.995
        min_T = 0.1

        while T > min_T and time.time() < deadline - 5:
            # Random move
            i = int(np.random.randint(search_primes))
            move = int(np.random.randint(-2, 3))  # -2, -1, 0, 1, 2
            if move == 0:
                T *= cooling
                continue
            if current_exps[i] + move < 0 or current_exps[i] + move > 25:
                T *= cooling
                continue

            new_exps = current_exps[:]
            new_exps[i] += move
            new_score = calc_score_v2(new_exps)

            if new_score < 0:
                T *= cooling
                continue

            delta_score = new_score - current_score
            if delta_score > 0 or (T > 0 and np.random.random() < math.exp(delta_score / T)):
                current_exps = new_exps
                current_score = new_score
                if current_score > best_score:
                    best_score = current_score
                    best_exps = current_exps[:]

            T *= cooling

    # Final result
    final_n = 1
    for i in range(search_primes):
        if best_exps[i] > 0:
            final_n *= primes[i] ** best_exps[i]

    if final_n > 10**18 or final_n < 2:
        # Fallback: product of first 15 primes
        final_n = 1
        for i in range(15):
            final_n *= primes[i]

    return final_n
