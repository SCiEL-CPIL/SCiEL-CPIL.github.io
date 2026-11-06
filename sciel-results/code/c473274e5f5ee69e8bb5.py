import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin for 90s limit
    
    best_set = []
    max_size = 48
    n_max = 512
    
    # Small primes for modular sieve
    primes = [3, 5, 7, 11, 13, 17, 19, 23, 29, 31]
    
    def _precheck_pair_mean(x, S, pair_sums_mod):
        """
        Returns True if x is NOT the mean of any pair in S (necessary condition).
        If for any prime p, (2*x) % p is not in the set of pair sums mod p,
        then 2*x != y+z for any y,z in S.
        """
        for p in primes:
            target = (2 * x) % p
            if target not in pair_sums_mod[p]:
                return False # x is a candidate for being a pair mean
        # If we reach here, 2*x matches pair sums mod all primes.
        # It MIGHT be a pair mean, but we need full check.
        # However, if it matches all, it's suspicious.
        # Actually, the logic is:
        # If 2x == y+z, then 2x % p == (y+z) % p for all p.
        # So if there is ANY p where 2x % p != (y+z) % p for all pairs, then 2x != y+z.
        # So if for ANY p, target NOT IN pair_sums_mod[p], then 2x != y+z.
        # So we can safely say "x is NOT a pair mean" (i.e., it's safe regarding pairs).
        # Wait, the function name is _precheck_pair_mean.
        # Let's rename to _is_pair_mean_safe.
        # If it returns True, it means x is safe from being a pair mean.
        # If it returns False, it means x MIGHT be a pair mean (needs full check) OR IS a pair mean.
        # Actually, if it returns False, it means we couldn't rule it out.
        # We want to FILTER OUT candidates that ARE pair means.
        # If 2x % p NOT IN pair_sums_mod[p] for some p, then 2x != y+z. So x is NOT a pair mean.
        # So if we find such a p, we can say "x is safe from pair means".
        # If for ALL p, 2x % p IN pair_sums_mod[p], then x is a "suspect".
        # We should still check it, but suspects are rare.
        
        # Let's invert logic for clarity in the loop:
        # We want to skip x if it's definitely a pair mean?
        # No, we want to add x if it's valid.
        # If x is a pair mean, it's invalid.
        # If we can prove it's NOT a pair mean via modular arithmetic, we save time?
        # No, the expensive part is the DP for ALL subset sizes.
        # The pair check is just k=2.
        # The DP handles all k.
        # So the pre-filter should eliminate candidates that are likely to fail the DP.
        # Most candidates will fail because they are means of some subset.
        # The modular check is a fast way to reject candidates that are means of PAIRS.
        # But if they are means of TRIPLES, the modular check might not catch it (unless we check triples mod p).
        
        # Let's stick to the pair check as a fast rejection for the most common case.
        # If x is a pair mean, it's invalid.
        # If we can prove x is NOT a pair mean, we still have to check triples, etc.
        # So the pre-filter doesn't help much unless we can prove it's INVALID.
        # We can't prove it's invalid via modular arithmetic easily (only prove it's not a pair mean).
        
        # Alternative: Use the modular check to order candidates?
        # Candidates that are "far" from pair sums are better?
        
        # Let's try a different structural change:
        # Instead of greedy, use a "Conflict Graph" approach?
        # Or just optimize the greedy with better candidate ordering.
        
        pass

    # Let's refine the strategy.
    # The bottleneck is the full DP check.
    # We can speed up the DP check by using Python integers as bitsets.
    # dp[k] is an integer where bit i is 1 if sum i is achievable with k elements.
    
    def _check_valid_addition(x, S, dp_S, dp_minus):
        """
        Checks if x can be added to S.
        dp_S: list of bitsets for S.
        dp_minus: dict mapping a in S to bitsets for S \ {a}.
        """
        n = len(S)
        
        # Check 1: Is x the mean of a subset of S?
        # Sum = k * x.
        for k in range(2, n + 1):
            target = k * x
            if target > 512 * 48:
                break
            if (dp_S[k] >> target) & 1:
                return False
        
        # Check 2: Is any a in S the mean of a subset of S U {x} \ {a}?
        # Subset must contain x.
        # T = {x} U U, U subset of S \ {a}.
        # Sum(T) = x + Sum(U) = (j+1) * a.
        # Sum(U) = (j+1) * a - x.
        # |U| = j. j >= 1.
        for a in S:
            dp_a = dp_minus[a]
            # j ranges from 1 to n-1
            for j in range(1, n):
                target = (j + 1) * a - x
                if target < 0:
                    continue
                if target > 512 * 48:
                    break
                if (dp_a[j] >> target) & 1:
                    return False
        return True

    def _update_dp(dp, x, n):
        """Update dp in place or return new dp."""
        new_dp = dp[:]
        for k in range(n, -1, -1):
            if new_dp[k]:
                new_dp[k+1] |= (new_dp[k] << x)
        return new_dp

    def _compute_dp_minus(S):
        """Precompute dp for S \ {a} for all a in S."""
        n = len(S)
        dp_minus = {}
        # This is O(n^2 * max_sum) which is slow.
        # Optimization: We can compute dp for S, and then "remove" a?
        # Bitset removal is hard.
        # Given n <= 48, and we do this once per addition attempt,
        # and we have ~500 candidates, it might be too slow.
        # 500 candidates * 48 elements * 48 steps * bitset ops.
        # Bitset ops are fast (C level).
        # Let's try it.
        
        for a in S:
            temp_S = [x for x in S if x != a]
            m = len(temp_S)
            dp_temp = [0] * (m + 1)
            dp_temp[0] = 1
            for elem in temp_S:
                for k in range(m - 1, -1, -1):
                    if dp_temp[k]:
                        dp_temp[k+1] |= (dp_temp[k] << elem)
            dp_minus[a] = dp_temp
        return dp_minus

    # Greedy with Random Restarts
    for restart in range(200):
        if time.time() > deadline:
            break
            
        S = []
        candidates = list(range(1, 513))
        random.shuffle(candidates)
        
        dp_S = [0] * (50)
        dp_S[0] = 1
        
        # Pre-compute pair sums mod p for fast filtering?
        # Let's just do the full check, but optimize the inner loop.
        
        found_any = True
        while len(S) < max_size and found_any:
            if time.time() > deadline:
                break
            
            n = len(S)
            
            # Precompute dp_minus for current S
            # This is expensive. Do we need it for every candidate?
            # Yes, because the condition depends on S.
            # But S doesn't change until we add an element.
            # So we compute it once per "round" of candidates.
            dp_minus = _compute_dp_minus(S)
            
            found = False
            for x in candidates:
                if time.time() > deadline:
                    break
                
                if x in S:
                    continue
                
                if _check_valid_addition(x, S, dp_S, dp_minus):
                    S.append(x)
                    # Update dp_S
                    dp_S = _update_dp(dp_S, x, n)
                    found = True
                    break
            
            if not found:
                # Stuck
                break
        
        if len(S) > len(best_set):
            best_set = S
            if len(best_set) == max_size:
                break

    return best_set
