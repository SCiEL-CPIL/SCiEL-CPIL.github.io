import time
import math

def construct():
    start_time = time.time()
    deadline = start_time + 85  # safety margin before 90s kill

    # Precompute prime factors for all numbers up to 20000
    # pf[x] = set of prime factors of x
    def build_pf(limit):
        pf = [set() for _ in range(limit + 1)]
        for p in range(2, limit + 1):
            # check if p is prime
            is_p = True
            for d in range(2, int(math.isqrt(p)) + 1):
                if p % d == 0:
                    is_p = False
                    break
            if is_p:
                for m in range(p, limit + 1, p):
                    pf[m].add(p)
        return pf

    pf = build_pf(20000)

    def get_primes_for_sums(nums):
        all_primes = set()
        for i in range(len(nums)):
            for j in range(i + 1, len(nums)):
                all_primes.update(pf[nums[i] + nums[j]])
        return all_primes

    # Precomputed strong candidate: powers of 2 and 3×powers of 2
    # All pairwise sums have only primes {2, 3, 5}
    # Actually let's verify: 2^a + 2^b = 2^min(1+2^|a-b|), 2^a + 3·2^b = 2^min(2^|a-b|+3)
    # 1+2^k: k=1→3, k=2→5, k=3→9=3², k=4→17, k=5→33=3·11, ...
    # 2^k+3: k=1→5, k=2→7, k=3→11, k=4→19, ...
    # So this gives primes {2,3,5,7,11,17,19,...} - not just 2 primes

    # Better: pure powers of 2: 1,2,4,8,...,2048
    # 2^a + 2^b = 2^min(1+2^|a-b|)
    # Odd factors: 1+2^k for k=1..11: 3,5,9,17,33,65,129,257,513,1025,2049
    # = 3,5,3,17,3·11,5·13,3·43,257,3·171=3·3·19,5·7·29,3·683
    # Primes: {2,3,5,11,13,17,19,29,257,683} - 10 primes

    # The inspiration 1 candidate [1,2,4,6,8,12,16,24,32,48,64,96]:
    # Let's just use it and verify
    candidate = [1, 2, 4, 6, 8, 12, 16, 24, 32, 48, 64, 96]
    best_set = list(candidate)
    best_count = len(get_primes_for_sums(candidate))

    # Try smooth-set search for 2-prime solutions
    def smooth_numbers(primes, max_val):
        """Generate all numbers with only prime factors from 'primes' up to max_val."""
        result = set([1])
        for p in primes:
            new = set()
            for x in result:
                v = x
                while v * p <= max_val:
                    v *= p
                    new.add(v)
            result |= new
        return sorted(result)

    def try_smooth_solution(prime_set, deadline):
        """Try to find 12 distinct smooth numbers whose pairwise sums are also smooth."""
        ps = list(prime_set)
        max_sum = 20000
        smooth_sums = set(smooth_numbers(ps, max_sum))
        
        # Candidate numbers: must be positive, distinct, <=10000
        # and for any two candidates a,b: a+b must be in smooth_sums
        # Also a itself doesn't need to be smooth, only sums
        
        # Start with small candidates
        candidates = list(range(1, 101))
        # Filter: for pairs we'll check
        
        # Greedy: pick numbers one by one
        chosen = []
        for x in candidates:
            if len(chosen) >= 12:
                break
            ok = True
            for c in chosen:
                if (x + c) not in smooth_sums:
                    ok = False
                    break
            if ok:
                chosen.append(x)
        
        if len(chosen) == 12:
            return chosen
        return None

    # Try 2-prime sets
    small_primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97]
    
    if time.time() < deadline - 10:
        for i in range(len(small_primes)):
            for j in range(i + 1, len(small_primes)):
                if time.time() > deadline - 10:
                    break
                ps = (small_primes[i], small_primes[j])
                sol = try_smooth_solution(ps, deadline)
                if sol is not None:
                    c = len(get_primes_for_sums(sol))
                    if c < best_count:
                        best_count = c
                        best_set = sol
                    if best_count <= 2:
                        break
            if best_count <= 2:
                break

    # Local search: try to reduce prime count by replacing elements
    def local_search(start, max_time):
        current = list(start)
        current_primes = get_primes_for_sums(current)
        current_count = len(current_primes)
        
        improved = True
        while improved and time.time() < max_time:
            improved = False
            for idx in range(12):
                if time.time() > max_time:
                    break
                old_val = current[idx]
                # Try replacements in a neighborhood
                for offset in range(1, 200):
                    for new_val in [old_val + offset, old_val - offset]:
                        if new_val < 1 or new_val > 10000:
                            continue
                        if new_val in current and new_val != old_val:
                            continue
                        if new_val == old_val:
                            continue
                        
                        test = list(current)
                        test[idx] = new_val
                        test_primes = get_primes_for_sums(test)
                        
                        if len(test_primes) < current_count:
                            current = test
                            current_primes = test_primes
                            current_count = len(test_primes)
                            improved = True
                            break
                    if improved:
                        break
                if improved:
                    break
        
        return current, current_count

    if time.time() < deadline - 5:
        best_set, best_count = local_search(best_set, deadline - 5)

    # Final validation
    assert len(set(best_set)) == 12, f"Not 12 distinct: {best_set}"
    assert all(1 <= x <= 10000 for x in best_set), f"Out of range: {best_set}"

    return best_set
