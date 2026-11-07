import math
import time
import random
from functools import lru_cache

def construct():
    start_time = time.time()
    time_limit = 82.0  # Leave 8s margin for 90s limit

    N = 256

    # Precompute prime factors for each number 1..N
    prime_factors = [set() for _ in range(N + 1)]
    prime_factors[1] = set()
    for i in range(2, N + 1):
        x = i
        d = 2
        while d * d <= x:
            if x % d == 0:
                prime_factors[i].add(d)
                while x % d == 0:
                    x //= d
            d += 1
        if x > 1:
            prime_factors[i].add(x)

    def get_primes(n):
        sieve = [True] * (n + 1)
        sieve[0] = sieve[1] = False
        for i in range(2, int(n**0.5) + 1):
            if sieve[i]:
                for j in range(i*i, n+1, i):
                    sieve[j] = False
        return [i for i, is_prime in enumerate(sieve) if is_prime]

    primes = get_primes(N)
    prime_to_idx = {p: i for i, p in enumerate(primes)}
    num_primes = len(primes)
    all_numbers = list(range(1, N + 1))
    num_numbers = N

    # Precompute which numbers are coprime to which
    # For each number, precompute the set of numbers it's NOT coprime to (share a prime factor)
    conflict = [set() for _ in range(N + 1)]
    for i in range(1, N + 1):
        for j in range(i + 1, N + 1):
            if prime_factors[i] & prime_factors[j]:
                conflict[i].add(j)
                conflict[j].add(i)

    def is_coprime_to_set(x, S):
        if x == 1:
            return True
        for y in S:
            if y == 1:
                continue
            if y in conflict[x]:
                return False
        return True

    # Key insight: The problem is equivalent to assigning each prime to at most one number.
    # If we select number n, all primes dividing n are "used".
    # 
    # Better approach: Think of it as a weighted independent set in a conflict graph.
    # But that's NP-hard. Instead, use the structure:
    # - Large primes (> 128) can only appear in themselves (since 2*p > 256)
    # - So we should always include all primes > 128
    # - For primes <= 128, we need to decide whether to use them alone or combine them
    #
    # Actually, let's use a smarter DP/branch-and-bound approach.
    #
    # Observation: If we sort numbers by value descending, we can use branch and bound.
    # But 256 numbers is too many for pure B&B.
    #
    # Better: Use the fact that the conflict graph has special structure.
    # Two numbers conflict iff they share a prime factor.
    #
    # Let's use a different strategy:
    # 1. Start with a good initial solution
    # 2. Use local search with multiple move types
    # 3. Use simulated annealing for escaping local optima
    #
    # First, let's build a strong initial solution using a greedy approach
    # that considers composites.

    # Strategy: For each prime, decide whether to use it as a prime power
    # or combine it with other small primes to form a larger composite.
    #
    # Key: It's often beneficial to combine small primes into composites with large primes.
    # E.g., instead of using 2 and 127 separately (sum=129), use 2*127=254 (sum=254).
    #
    # Let's use a matching-based approach:
    # - Left nodes: small primes (primes <= 31, say)
    # - Right nodes: large primes (primes > 31)
    # - Edge weight: p*q - p - q (gain from combining)
    # - Find max weight matching
    # - Then fill in remaining with prime powers

    small_primes = [p for p in primes if p <= 31]
    large_primes = [p for p in primes if p > 31]
    
    # For each pair (small, large), compute the composite and gain
    # But we need to be careful: the composite must be <= 256
    # and we need to check that it doesn't conflict with other selected numbers

    # Actually, let's use a more systematic approach:
    # Build initial solution by considering all numbers sorted by value,
    # and greedily adding if coprime to current set.
    # Then improve with local search.

    # Greedy initial solution (sorted by value descending)
    def greedy_initial():
        S = set()
        # Always include 1
        S.add(1)
        s_sum = 1
        # Try numbers from largest to smallest
        for x in range(N, 1, -1):
            if x not in S and is_coprime_to_set(x, S):
                S.add(x)
                s_sum += x
        return S, s_sum

    # Better greedy: consider composites first
    def greedy_initial_v2():
        S = set()
        S.add(1)
        s_sum = 1
        # First, add all large primes (they can't be part of any composite > themselves)
        for p in primes:
            if p > 128:  # 2*p > 256
                if p not in S and is_coprime_to_set(p, S):
                    S.add(p)
                    s_sum += p
        # Then, for remaining numbers, try from largest to smallest
        for x in range(N, 1, -1):
            if x not in S and is_coprime_to_set(x, S):
                S.add(x)
                s_sum += x
        return S, s_sum

    # Even better: use a priority queue approach
    # For each prime, the best "use" is either the prime power or a composite
    # Let's precompute for each prime p, the best number that is a multiple of p
    # and is coprime to all other selected numbers... but that's circular.
    
    # Let's just use greedy_initial_v2 and then do extensive local search.
    
    current_set, current_sum = greedy_initial_v2()

    def local_search(S, s_sum, max_time, randomize=False):
        improved = True
        while improved and time.time() - start_time < max_time:
            improved = False
            
            # Phase 1: Add any number that's coprime to S
            for x in range(N, 0, -1):
                if x not in S and is_coprime_to_set(x, S):
                    S.add(x)
                    s_sum += x
                    improved = True
                    break
            
            # Phase 2: Swap 1 for 1
            if not improved:
                s_list = sorted(S, reverse=True)
                candidates_in = [y for y in range(N, 0, -1) if y not in S]
                for a in s_list:
                    if a == 1:
                        continue
                    temp_set = S - {a}
                    for b in candidates_in:
                        if b <= a:
                            break
                        if is_coprime_to_set(b, temp_set):
                            S.discard(a)
                            S.add(b)
                            s_sum += b - a
                            improved = True
                            break
                    if improved:
                        break
                        
            # Phase 3: Swap 2 for 1
            if not improved:
                s_list = sorted(S, reverse=True)
                n_s = len(s_list)
                for i in range(n_s):
                    if time.time() - start_time > max_time - 2:
                        break
                    for j in range(i + 1, n_s):
                        a, c = s_list[i], s_list[j]
                        if a + c >= N:
                            continue
                        temp_set = S - {a, c}
                        for b in range(N, a + c, -1):
                            if b in S:
                                continue
                            if is_coprime_to_set(b, temp_set):
                                S.discard(a)
                                S.discard(c)
                                S.add(b)
                                s_sum += b - a - c
                                improved = True
                                break
                        if improved:
                            break
                    if improved:
                        break
                        
            # Phase 4: Swap 1 for 2
            if not improved:
                s_list = sorted(S, reverse=True)
                for a in s_list:
                    if a == 1 or time.time() - start_time > max_time - 2:
                        continue
                    temp_set = S - {a}
                    # Find two numbers b, d not in S, coprime to temp_set and each other, with b+d > a
                    candidates = [y for y in range(N, 0, -1) if y not in S and is_coprime_to_set(y, temp_set)]
                    found = False
                    for i in range(len(candidates)):
                        if time.time() - start_time > max_time - 2:
                            break
                        b = candidates[i]
                        if b <= a:
                            break
                        for j in range(i + 1, len(candidates)):
                            d = candidates[j]
                            if b + d <= a:
                                break
                            if is_coprime_to_set(d, temp_set | {b}):
                                S.discard(a)
                                S.add(b)
                                S.add(d)
                                s_sum += b + d - a
                                improved = True
                                found = True
                                break
                        if found:
                            break
                    if improved:
                        break
                        
            # Phase 5: Swap 2 for 2
            if not improved:
                s_list = sorted(S, reverse=True)
                n_s = len(s_list)
                for i in range(n_s):
                    if time.time() - start_time > max_time - 2:
                        break
                    for j in range(i + 1, n_s):
                        a, c = s_list[i], s_list[j]
                        temp_set = S - {a, c}
                        a_c_sum = a + c
                        candidates = [y for y in range(N, 0, -1) if y not in S and is_coprime_to_set(y, temp_set)]
                        found = False
                        for ii in range(len(candidates)):
                            if time.time() - start_time > max_time - 2:
                                break
                            b = candidates[ii]
                            if b <= a_c_sum:
                                break
                            for jj in range(ii + 1, len(candidates)):
                                d = candidates[jj]
                                if b + d <= a_c_sum:
                                    break
                                if is_coprime_to_set(d, temp_set | {b}):
                                    S.discard(a)
                                    S.discard(c)
                                    S.add(b)
                                    S.add(d)
                                    s_sum += b + d - a - c
                                    improved = True
                                    found = True
                                    break
                            if found:
                                break
                        if found:
                            break
                    if improved:
                        break
                        
            # Phase 6: Swap 3 for 2 (remove 3, add 2)
            if not improved and time.time() - start_time < max_time - 5:
                s_list = sorted(S, reverse=True)
                n_s = len(s_list)
                # Only try if we have enough elements
                if n_s >= 3:
                    for i in range(n_s):
                        if time.time() - start_time > max_time - 5:
                            break
                        for j in range(i + 1, n_s):
                            if time.time() - start_time > max_time - 5:
                                break
                            for k in range(j + 1, n_s):
                                a, c, e = s_list[i], s_list[j], s_list[k]
                                if a + c + e >= 2 * N:
                                    continue
                                temp_set = S - {a, c, e}
                                a_c_e_sum = a + c + e
                                candidates = [y for y in range(N, 0, -1) if y not in S and is_coprime_to_set(y, temp_set)]
                                found = False
                                for ii in range(len(candidates)):
                                    if time.time() - start_time > max_time - 5:
                                        break
                                    b = candidates[ii]
                                    if b <= a_c_e_sum:
                                        break
                                    for jj in range(ii + 1, len(candidates)):
                                        d = candidates[jj]
                                        if b + d <= a_c_e_sum:
                                            break
                                        if is_coprime_to_set(d, temp_set | {b}):
                                            S.discard(a)
                                            S.discard(c)
                                            S.discard(e)
                                            S.add(b)
                                            S.add(d)
                                            s_sum += b + d - a - c - e
                                            improved = True
                                            found = True
                                            break
                                    if found:
                                        break
                                if found:
                                    break
                            if improved:
                                break
                        if improved:
                            break

        return S, s_sum

    best_sum = current_sum
    best_set = set(current_set)

    # Run local search from the initial solution
    current_set, current_sum = local_search(current_set, current_sum, time_limit - 15)
    if current_sum > best_sum:
        best_sum = current_sum
        best_set = set(current_set)

    # Simulated annealing / random restarts
    remaining_time = time_limit - (time.time() - start_time)
    if remaining_time > 10:
        # Multiple restarts with different initial solutions
        for restart in range(int(remaining_time / 5)):
            if time.time() - start_time > time_limit - 8:
                break
            
            # Create a perturbed version of the best solution
            random_set = set(best_set)
            num_remove = random.randint(1, 3)
            for _ in range(num_remove):
                if random_set:
                    random_set.remove(random.choice(list(random_set)))
            
            # Greedy fill
            random_sum = sum(random_set)
            for x in range(N, 0, -1):
                if x not in random_set and is_coprime_to_set(x, random_set):
                    random_set.add(x)
                    random_sum += x
            
            # Local search
            random_set, random_sum = local_search(random_set, random_sum, time_limit - 8)
            if random_sum > best_sum:
                best_sum = random_sum
                best_set = set(random_set)
            
            # Also try a completely different initial solution
            if time.time() - start_time < time_limit - 10:
                alt_set, alt_sum = greedy_initial()
                alt_set, alt_sum = local_search(alt_set, alt_sum, time_limit - 10)
                if alt_sum > best_sum:
                    best_sum = alt_sum
                    best_set = set(alt_set)

    # Final polish: try to add 1 if not present
    if 1 not in best_set:
        best_set.add(1)
        best_sum += 1

    result = list(best_set)
    result = [int(x) for x in result]
    return result
