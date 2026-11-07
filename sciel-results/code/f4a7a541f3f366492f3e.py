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

    # Precompute conflict masks
    # conflict_mask[i] has bit j set if gcd(i, j) > 1
    conflict_mask = [0] * (N + 1)
    for i in range(1, N + 1):
        mask = 0
        for j in range(1, N + 1):
            if i != j and (prime_factors[i] & prime_factors[j]):
                mask |= (1 << j)
        conflict_mask[i] = mask

    def is_coprime_to_set(x, S_mask):
        # S_mask is an integer bitmask of selected numbers
        # x is an integer
        # Check if any selected number conflicts with x
        return (conflict_mask[x] & S_mask) == 0

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
        S_mask = 0
        # Always include 1
        S.add(1)
        S_mask |= (1 << 1)
        s_sum = 1
        # Try numbers from largest to smallest
        for x in range(N, 1, -1):
            if not (S_mask & (1 << x)) and is_coprime_to_set(x, S_mask):
                S.add(x)
                S_mask |= (1 << x)
                s_sum += x
        return S, S_mask, s_sum

    # Better greedy: consider composites first
    def greedy_initial_v2():
        S = set()
        S_mask = 0
        S.add(1)
        S_mask |= (1 << 1)
        s_sum = 1
        # First, add all large primes (they can't be part of any composite > themselves)
        for p in primes:
            if p > 128:  # 2*p > 256
                if not (S_mask & (1 << p)) and is_coprime_to_set(p, S_mask):
                    S.add(p)
                    S_mask |= (1 << p)
                    s_sum += p
        # Then, for remaining numbers, try from largest to smallest
        for x in range(N, 1, -1):
            if not (S_mask & (1 << x)) and is_coprime_to_set(x, S_mask):
                S.add(x)
                S_mask |= (1 << x)
                s_sum += x
        return S, S_mask, s_sum

    # Even better: use a priority queue approach
    # For each prime, the best "use" is either the prime power or a composite
    # Let's precompute for each prime p, the best number that is a multiple of p
    # and is coprime to all other selected numbers... but that's circular.
    
    # Let's just use greedy_initial_v2 and then do extensive local search.
    
    current_set, current_mask, current_sum = greedy_initial_v2()

    def local_search(S, S_mask, s_sum, max_time):
        improved = True
        while improved and time.time() - start_time < max_time:
            improved = False
            
            # Phase 1: Add any number that's coprime to S
            for x in range(N, 0, -1):
                if not (S_mask & (1 << x)):
                    if is_coprime_to_set(x, S_mask):
                        S.add(x)
                        S_mask |= (1 << x)
                        s_sum += x
                        improved = True
                        break
            
            if not improved:
                s_list = sorted(S, reverse=True)
                
                # Phase 2: Swap 1 for 1
                found_swap = False
                for a in s_list:
                    if a == 1:
                        continue
                    temp_mask = S_mask ^ (1 << a)
                    for b in range(N, a + 1, -1):
                        if S_mask & (1 << b):
                            continue
                        if is_coprime_to_set(b, temp_mask):
                            S.discard(a)
                            S.add(b)
                            S_mask = temp_mask | (1 << b)
                            s_sum += b - a
                            improved = True
                            found_swap = True
                            break
                    if found_swap:
                        break
                            
                if not improved:
                    # Phase 3: Swap 2 for 1 (Remove 2, Add 1)
                    s_list = sorted(S, reverse=True)
                    n_s = len(s_list)
                    found_swap = False
                    for i in range(n_s):
                        if time.time() - start_time > max_time - 2:
                            break
                        for j in range(i + 1, n_s):
                            a, c = s_list[i], s_list[j]
                            if a + c >= N:
                                continue
                            temp_mask = S_mask ^ (1 << a) ^ (1 << c)
                            for b in range(N, a + c, -1):
                                if S_mask & (1 << b):
                                    continue
                                if is_coprime_to_set(b, temp_mask):
                                    S.discard(a)
                                    S.discard(c)
                                    S.add(b)
                                    S_mask = temp_mask | (1 << b)
                                    s_sum += b - a - c
                                    improved = True
                                    found_swap = True
                                    break
                            if found_swap:
                                break
                        if found_swap:
                            break
                            
                if not improved:
                    # Phase 4: Swap 1 for 2 (Remove 1, Add 2)
                    s_list = sorted(S, reverse=True)
                    found_swap = False
                    for a in s_list:
                        if a == 1 or time.time() - start_time > max_time - 2:
                            continue
                        temp_mask = S_mask ^ (1 << a)
                        found = False
                        for b in range(N, a + 1, -1):
                            if S_mask & (1 << b):
                                continue
                            if not is_coprime_to_set(b, temp_mask):
                                continue
                            for d in range(b - 1, 0, -1):
                                if S_mask & (1 << d):
                                    continue
                                if not is_coprime_to_set(d, temp_mask | (1 << b)):
                                    continue
                                if b + d <= a:
                                    break
                                S.discard(a)
                                S.add(b)
                                S.add(d)
                                S_mask = temp_mask | (1 << b) | (1 << d)
                                s_sum += b + d - a
                                improved = True
                                found = True
                                break
                            if found:
                                break
                        if found:
                            found_swap = True
                            break
                    improved = found_swap

                if not improved:
                    # Phase 5: Swap 2 for 2 (Remove 2, Add 2)
                    s_list = sorted(S, reverse=True)
                    n_s = len(s_list)
                    found_swap = False
                    for i in range(n_s):
                        if time.time() - start_time > max_time - 2:
                            break
                        for j in range(i + 1, n_s):
                            a, c = s_list[i], s_list[j]
                            temp_mask = S_mask ^ (1 << a) ^ (1 << c)
                            a_c_sum = a + c
                            found = False
                            for b in range(N, a_c_sum + 1, -1):
                                if S_mask & (1 << b):
                                    continue
                                if not is_coprime_to_set(b, temp_mask):
                                    continue
                                for d in range(b - 1, 0, -1):
                                    if S_mask & (1 << d):
                                        continue
                                    if not is_coprime_to_set(d, temp_mask | (1 << b)):
                                        continue
                                    if b + d <= a_c_sum:
                                        break
                                    S.discard(a)
                                    S.discard(c)
                                    S.add(b)
                                    S.add(d)
                                    S_mask = temp_mask | (1 << b) | (1 << d)
                                    s_sum += b + d - a - c
                                    improved = True
                                    found = True
                                    break
                                if found:
                                    break
                            if found:
                                found_swap = True
                                break
                        if found_swap:
                            break
                    improved = found_swap

                if not improved and time.time() - start_time < max_time - 5:
                    # Phase 6: Swap 3 for 2 (Remove 3, Add 2)
                    s_list = sorted(S, reverse=True)
                    n_s = len(s_list)
                    found_swap = False
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
                                    temp_mask = S_mask ^ (1 << a) ^ (1 << c) ^ (1 << e)
                                    a_c_e_sum = a + c + e
                                    found = False
                                    for b in range(N, a_c_e_sum + 1, -1):
                                        if S_mask & (1 << b):
                                            continue
                                        if not is_coprime_to_set(b, temp_mask):
                                            continue
                                        for d in range(b - 1, 0, -1):
                                            if S_mask & (1 << d):
                                                continue
                                            if not is_coprime_to_set(d, temp_mask | (1 << b)):
                                                continue
                                            if b + d <= a_c_e_sum:
                                                break
                                            S.discard(a)
                                            S.discard(c)
                                            S.discard(e)
                                            S.add(b)
                                            S.add(d)
                                            S_mask = temp_mask | (1 << b) | (1 << d)
                                            s_sum += b + d - a - c - e
                                            improved = True
                                            found = True
                                            break
                                        if found:
                                            break
                                    if found:
                                        found_swap = True
                                        break
                                if found_swap:
                                    break
                            if found_swap:
                                break
                    improved = found_swap

        return S, S_mask, s_sum

    best_sum = current_sum
    best_set = set(current_set)
    best_mask = current_mask

    # Run local search from the initial solution
    current_set, current_mask, current_sum = local_search(current_set, current_mask, current_sum, time_limit - 15)
    if current_sum > best_sum:
        best_sum = current_sum
        best_set = set(current_set)
        best_mask = current_mask

    # Simulated annealing / random restarts
    remaining_time = time_limit - (time.time() - start_time)
    if remaining_time > 10:
        # Multiple restarts with different initial solutions
        for restart in range(int(remaining_time / 5)):
            if time.time() - start_time > time_limit - 8:
                break
            
            # Create a perturbed version of the best solution
            random_set = set(best_set)
            random_mask = best_mask
            random_sum = best_sum
            num_remove = random.randint(1, 3)
            for _ in range(num_remove):
                if random_set:
                    item = random.choice(list(random_set))
                    random_set.remove(item)
                    random_mask ^= (1 << item)
                    random_sum -= item
            
            # Greedy fill
            for x in range(N, 0, -1):
                if not (random_mask & (1 << x)):
                    if is_coprime_to_set(x, random_mask):
                        random_set.add(x)
                        random_mask |= (1 << x)
                        random_sum += x
            
            # Local search
            random_set, random_mask, random_sum = local_search(random_set, random_mask, random_sum, time_limit - 8)
            if random_sum > best_sum:
                best_sum = random_sum
                best_set = set(random_set)
                best_mask = random_mask
            
            # Also try a completely different initial solution
            if time.time() - start_time < time_limit - 10:
                alt_set, alt_mask, alt_sum = greedy_initial()
                alt_set, alt_mask, alt_sum = local_search(alt_set, alt_mask, alt_sum, time_limit - 10)
                if alt_sum > best_sum:
                    best_sum = alt_sum
                    best_set = set(alt_set)
                    best_mask = alt_mask

    # Final polish: try to add 1 if not present
    if not (best_mask & (1 << 1)):
        best_set.add(1)
        best_mask |= (1 << 1)
        best_sum += 1

    result = list(best_set)
    result = [int(x) for x in result]
    return result
