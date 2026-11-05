import time
import math
from itertools import combinations
from bisect import bisect_left

def construct():
    start_time = time.time()
    deadline = start_time + 55  # safety margin before 60s kill

    N = 12
    MAXV = 10000
    MAXSUM = 2 * MAXV  # 20000

    # Precompute prime factors for all numbers up to 20000 using sieve
    def build_pf(limit):
        pf = [set() for _ in range(limit + 1)]
        for p in range(2, limit + 1):
            if not pf[p]:  # p is prime if no factor found yet
                for m in range(p, limit + 1, p):
                    pf[m].add(p)
        return pf

    pf = build_pf(MAXSUM)

    def get_primes_for_sums(nums):
        all_primes = set()
        for i in range(len(nums)):
            for j in range(i + 1, len(nums)):
                all_primes.update(pf[nums[i] + nums[j]])
        return all_primes

    # Start with a reasonable baseline
    baseline = list(range(1, 13))
    best_set = list(baseline)
    best_count = len(get_primes_for_sums(baseline))

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
        return result

    def find_smooth_solution(prime_set, deadline):
        """Find 12 distinct numbers <= MAXV whose pairwise sums are all S-smooth."""
        ps = list(prime_set)
        smooth_sums = smooth_numbers(ps, MAXSUM)
        
        # Candidate pool: numbers 1..600
        cand_range = list(range(1, 601))
        
        # Precompute compatibility: compat[x] = set of y in cand_range with x+y smooth
        compat = {}
        for x in cand_range:
            compat[x] = set()
            for y in cand_range:
                if x != y and (x + y) in smooth_sums:
                    compat[x].add(y)
        
        chosen = []
        
        def backtrack(start_idx):
            if time.time() > deadline:
                return False
            if len(chosen) == N:
                return True
            for i in range(start_idx, len(cand_range)):
                x = cand_range[i]
                # Check x compatible with all chosen
                ok = True
                for c in chosen:
                    if x not in compat[c]:
                        ok = False
                        break
                if not ok:
                    continue
                chosen.append(x)
                if backtrack(i + 1):
                    return True
                chosen.pop()
            return False
        
        if backtrack(0):
            return list(chosen)
        return None

    # Try 2-prime sets to find a 2-prime solution (optimal)
    small_primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29]
    two_prime_sets = list(combinations(small_primes, 2))
    
    for ps in two_prime_sets:
        if time.time() > deadline:
            break
        sol = find_smooth_solution(ps, deadline)
        if sol is not None:
            c = len(get_primes_for_sums(sol))
            if c < best_count:
                best_count = c
                best_set = sol
            if best_count <= 2:
                break

    # If no 2-prime solution found, try 3-prime sets
    if best_count > 2 and time.time() < deadline:
        three_prime_sets = list(combinations(small_primes[:8], 3))
        for ps in three_prime_sets:
            if time.time() > deadline:
                break
            sol = find_smooth_solution(ps, deadline)
            if sol is not None:
                c = len(get_primes_for_sums(sol))
                if c < best_count:
                    best_count = c
                    best_set = sol
                if best_count <= 3:
                    break

    # Local search to improve if not already optimal
    if best_count > 2 and time.time() < deadline - 5:
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
                    old_pair_primes = set()
                    for j in range(12):
                        if j != idx:
                            old_pair_primes.update(pf[old_val + current[j]])
                    
                    for offset in range(1, 200):
                        for new_val in [old_val + offset, old_val - offset]:
                            if new_val < 1 or new_val > 10000:
                                continue
                            if new_val == old_val:
                                continue
                            if any(new_val == current[j] for j in range(12) if j != idx):
                                continue
                            
                            new_pair_primes = set()
                            for j in range(12):
                                if j != idx:
                                    new_pair_primes.update(pf[new_val + current[j]])
                            
                            remaining = current_primes - old_pair_primes
                            test_primes = remaining | new_pair_primes
                            
                            if len(test_primes) < current_count:
                                current[idx] = new_val
                                current_primes = test_primes
                                current_count = len(test_primes)
                                improved = True
                                break
                        if improved:
                            break
                    if improved:
                        break
            
            return current, current_count
        
        best_set, best_count = local_search(best_set, deadline - 5)

    # Final validation
    assert len(set(best_set)) == 12, f"Not 12 distinct: {best_set}"
    assert all(1 <= x <= 10000 for x in best_set), f"Out of range: {best_set}"

    return best_set
