import time
import math
import random

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # 145s deadline, 5s safety margin

    # 1. Generate a list of small primes up to 20000 for swap candidates
    def sieve(n):
        is_prime = [True] * (n + 1)
        is_prime[0] = is_prime[1] = False
        for i in range(2, int(n**0.5) + 1):
            if is_prime[i]:
                for j in range(i*i, n+1, i):
                    is_prime[j] = False
        return [i for i, v in enumerate(is_prime) if v]

    primes = sieve(20000)
    prime_index = {p: i for i, p in enumerate(primes)}

    # 2. Fast tau_perp for square-free numbers using bitmask
    def tau_perp_squarefree(prime_list):
        """Calculate tau_perp for a square-free number given its sorted prime factors."""
        k = len(prime_list)
        if k == 0:
            return 0

        # Generate all divisors with their prime-support bitmasks
        # divisor value and bitmask of which primes divide it
        divs = []
        for mask in range(1 << k):
            d = 1
            for i in range(k):
                if mask & (1 << i):
                    d *= prime_list[i]
            divs.append((d, mask))

        divs.sort(key=lambda x: x[0])

        # Count adjacent coprime pairs: gcd(d_i, d_{i+1}) == 1
        # For square-free, this means mask_i & mask_{i+1} == 0
        score = 0
        for i in range(len(divs) - 1):
            if (divs[i][1] & divs[i+1][1]) == 0:
                score += 1
        return score

    # 3. Find maximum number of distinct primes (primorial-based)
    max_k = 0
    prod = 1
    for p in primes:
        if prod * p > 10**18:
            break
        prod *= p
        max_k += 1

    # Initial candidate: first max_k primes
    initial_primes = primes[:max_k]
    best_score = tau_perp_squarefree(initial_primes)
    best_primes = list(initial_primes)

    # 4. Local search with hill climbing
    # Track current product
    current_prod = 1
    for p in best_primes:
        current_prod *= p

    # Precompute for faster lookup
    best_prime_set = set(best_primes)

    # Helper: evaluate a candidate prime set
    def evaluate(prime_set_list):
        return tau_perp_squarefree(prime_set_list)

    # Hill climbing loop
    improved = True
    iterations = 0
    while improved and time.time() < deadline - 15.0:
        improved = False
        iterations += 1
        best_primes.sort()
        best_prime_set = set(best_primes)

        # Try each position for swap
        for i in range(len(best_primes)):
            if time.time() > deadline - 15.0:
                break

            p_out = best_primes[i]
            max_p_in = int(10**18 * p_out / current_prod)

            # Candidate primes: not in set, <= max_p_in
            # Focus on primes that are "structurally different" - try a range
            # Use set-packing heuristic: prefer primes that create more disjoint pairs
            candidates = []
            for p in primes:
                if p > max_p_in:
                    break
                if p in best_prime_set:
                    continue
                candidates.append(p)

            if not candidates:
                continue

            # Evaluate candidates in order of potential improvement
            # Sort by a heuristic: primes that are "far" from existing ones may help
            # But first, try all candidates and take the best
            best_candidate = None
            best_candidate_score = best_score

            # Limit candidates for speed if too many
            if len(candidates) > 200:
                # Sample a subset: smallest, largest, and some random
                sampled = set()
                sampled.update(candidates[:50])
                sampled.update(candidates[-50:])
                random_sample = random.sample(candidates, min(100, len(candidates)))
                sampled.update(random_sample)
                candidates = list(sampled)

            for p_in in candidates:
                if time.time() > deadline - 15.0:
                    break

                new_primes = best_primes[:]
                new_primes[i] = p_in
                new_primes.sort()

                s = tau_perp_squarefree(new_primes)
                if s > best_candidate_score:
                    best_candidate_score = s
                    best_candidate = (p_in, s, new_primes)

            if best_candidate is not None:
                p_in, new_score, new_primes = best_candidate
                best_score = new_score
                best_primes = new_primes
                best_prime_set = set(best_primes)
                current_prod = current_prod * p_in // p_out
                improved = True
                break  # Restart with new state

    # 5. Random restarts and perturbations
    if time.time() < deadline - 10.0:
        # Try random perturbations of the best solution
        for _ in range(200):
            if time.time() > deadline - 10.0:
                break

            curr_primes = best_primes[:]
            curr_prod = current_prod
            curr_prime_set = set(curr_primes)

            # Do 3-8 random swaps
            num_swaps = random.randint(3, 8)
            valid = True
            for _ in range(num_swaps):
                i = random.randint(0, len(curr_primes) - 1)
                p_out = curr_primes[i]
                max_p_in = int(10**18 * p_out / curr_prod)

                # Find valid candidates
                candidates = []
                for p in primes:
                    if p > max_p_in:
                        break
                    if p in curr_prime_set:
                        continue
                    candidates.append(p)

                if not candidates:
                    valid = False
                    break

                p_in = random.choice(candidates)
                curr_primes[i] = p_in
                curr_prod = curr_prod * p_in // p_out
                curr_prime_set.remove(p_out)
                curr_prime_set.add(p_in)
                curr_primes.sort()

            if valid:
                s = tau_perp_squarefree(curr_primes)
                if s > best_score:
                    best_score = s
                    best_primes = curr_primes
                    current_prod = curr_prod

    # 6. Final polish: try single-prime swaps from the best solution
    if time.time() < deadline - 5.0:
        best_primes.sort()
        best_prime_set = set(best_primes)

        for i in range(len(best_primes)):
            if time.time() > deadline - 5.0:
                break

            p_out = best_primes[i]
            max_p_in = int(10**18 * p_out / current_prod)

            # Try all valid candidates
            for p_in in primes:
                if p_in > max_p_in:
                    break
                if p_in in best_prime_set:
                    continue

                new_primes = best_primes[:]
                new_primes[i] = p_in
                new_primes.sort()

                s = tau_perp_squarefree(new_primes)
                if s > best_score:
                    best_score = s
                    best_primes = new_primes
                    best_prime_set = set(best_primes)
                    current_prod = current_prod * p_in // p_out

    # Construct final number
    result = 1
    for p in best_primes:
        result *= p

    return result
