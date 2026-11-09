# EVOLVE-BLOCK-START
"""Find Egyptian fraction of 5/31 minimizing the largest denominator.

Strategy:
1. Hard-code the best known 3-term solution [8, 31, 248] (max denom 248).
2. Run a bounded DFS to try to find a solution with max denom < 248.
   - Use a strict deadline guard (wall-clock) so we never exceed the 90 s budget.
   - Use integer arithmetic to avoid Fraction overhead.
   - Prune aggressively:
       * denominators must be strictly increasing (distinct, no repeats)
       * the remaining fraction must be representable with the remaining terms
         using denominators > last chosen and <= best_max_found
       * lower-bound check: even if we use the smallest possible next denominators,
         the sum of reciprocals must not exceed the remaining target.
3. If no better solution is found in time, return the hard-coded 248 solution.
"""

import time

TARGET_NUM = 5
TARGET_DEN = 31
MAX_DENOM = 10_000_000
MAX_TERMS = 256
DEADLINE_SECONDS = 85.0  # safety margin before 90 s kill

# Hard-coded best known solution
FALLBACK = [8, 31, 248]

def _verify(candidate):
    """Verify that candidate is a valid Egyptian fraction of 5/31."""
    if not candidate:
        return False
    if len(candidate) > MAX_TERMS:
        return False
    if len(set(candidate)) != len(candidate):
        return False
    if any(d < 1 or d > MAX_DENOM for d in candidate):
        return False
    # Exact rational check using integer arithmetic
    # Sum of 1/d for d in candidate == 5/31
    # We'll use Fraction for simplicity in verification (called rarely)
    from fractions import Fraction
    s = sum(Fraction(1, d) for d in candidate)
    return s == Fraction(TARGET_NUM, TARGET_DEN)

def _gcd(a, b):
    while b:
        a, b = b, a % b
    return a

def construct():
    start_time = time.time()
    
    # Verify and return fallback if valid
    if _verify(FALLBACK):
        best_solution = FALLBACK[:]
        best_max = max(FALLBACK)
    else:
        # Should never happen, but just in case
        best_solution = None
        best_max = MAX_DENOM + 1

    # Try to find a better solution via DFS
    # State: (remaining_num, remaining_den, min_denom, terms_used, current_terms)
    # We search for solutions with max denom < best_max
    # Pruning: if we have k terms left to place, the minimum possible sum
    # with denominators starting at min_denom is 1/min_denom + 1/(min_denom+1) + ...
    # We need this to be >= remaining fraction.
    
    deadline = start_time + DEADLINE_SECONDS
    found_better = False
    
    def dfs(rem_num, rem_den, min_denom, terms, current_terms):
        nonlocal best_solution, best_max, found_better
        
        if time.time() > deadline:
            return False  # timeout signal
        
        if rem_num == 0:
            if len(current_terms) >= 1:
                max_d = max(current_terms)
                if max_d < best_max:
                    best_max = max_d
                    best_solution = current_terms[:]
                    found_better = True
            return True  # found a solution, keep searching for better
        
        if len(terms) == 0:
            return False
        
        # Lower bound check: the sum of reciprocals of the smallest 'terms' denominators
        # starting from min_denom must be >= rem_num/rem_den
        # 1/min_denom + 1/(min_denom+1) + ... + 1/(min_denom+terms-1) >= rem_num/rem_den
        # For a quick check: terms / min_denom >= rem_num / rem_den
        # i.e., terms * rem_den >= rem_num * min_denom
        if terms * rem_den < rem_num * min_denom:
            return False
        
        # Upper bound check: even with the largest possible denominators,
        # we need the sum to be <= rem_num/rem_den
        # But since we're going to pick denominators >= min_denom, and we want sum = rem_num/rem_den,
        # the first denominator d must satisfy 1/d <= rem_num/rem_den, so d >= rem_den/rem_num.
        # Also d >= min_denom.
        # And d must be such that rem_num/rem_den - 1/d > 0 (or == 0 if last term).
        
        # The first denominator d must satisfy:
        # d >= ceil(rem_den / rem_num)  (so that 1/d <= rem_num/rem_den)
        # d >= min_denom
        # Also, we need d < best_max (since we want to improve)
        # And d must be such that the remaining fraction (rem_num/rem_den - 1/d) can be
        # represented with (terms-1) terms with denominators > d.
        
        # Lower bound for d:
        d_lower = max(min_denom, (rem_den + rem_num - 1) // rem_num)
        # Upper bound for d:
        # 1/d > rem_num/rem_den - (terms-1)/d_lower  ... this is complex.
        # Simpler: d < best_max
        d_upper = best_max - 1  # we want max denom < best_max
        
        if d_lower > d_upper:
            return False
        
        # Also, the remaining fraction after choosing d must be positive:
        # rem_num/rem_den - 1/d > 0  =>  d > rem_den/rem_num
        # This is already covered by d_lower.
        
        # Iterate d from d_lower to d_upper
        # But this could be a huge range. We need smarter pruning.
        # For each d, the new remaining fraction is:
        # (rem_num * d - rem_den) / (rem_den * d)
        # Let new_num = rem_num * d - rem_den, new_den = rem_den * d
        # We need new_num > 0 (guaranteed by d_lower)
        # And new_num/new_den must be representable with (terms-1) terms, each > d.
        
        # Quick prune: if terms == 1, then we need 1/d = rem_num/rem_den, so d = rem_den/rem_num exactly.
        if terms == 1:
            if rem_num * best_max > rem_den:  # 1/d > rem_num/rem_den for all d < best_max? No.
                pass
            # We need d such that 1/d = rem_num/rem_den => d = rem_den/rem_num
            if rem_den % rem_num == 0:
                d = rem_den // rem_num
                if d >= d_lower and d <= d_upper and d > min_denom - 1:  # d >= min_denom
                    if d > current_terms[-1] if current_terms else True:
                        # Valid
                        current_terms.append(d)
                        max_d = max(current_terms)
                        if max_d < best_max:
                            best_max = max_d
                            best_solution = current_terms[:]
                            found_better = True
                        current_terms.pop()
            return False
        
        # For terms > 1, iterate d. But the range [d_lower, d_upper] could be large.
        # We'll iterate but with a cap on iterations per node to avoid getting stuck.
        # Also, we can use the fact that d must be such that the new remaining fraction
        # is not too small to be represented with (terms-1) terms.
        
        # Max iterations per node to keep search tractable
        max_iters_per_node = 5000
        iters = 0
        for d in range(d_lower, d_upper + 1):
            if time.time() > deadline:
                return False
            if iters >= max_iters_per_node:
                break
            iters += 1
            
            # Check if d is distinct from previous terms (d > current_terms[-1] ensures this)
            if current_terms and d <= current_terms[-1]:
                continue
            
            # New remaining fraction
            new_num = rem_num * d - rem_den
            if new_num <= 0:
                continue  # 1/d >= rem_num/rem_den, not allowed (would make remaining <= 0)
            
            new_den = rem_den * d
            
            # Reduce fraction
            g = _gcd(new_num, new_den)
            new_num //= g
            new_den //= g
            
            # Prune: can new_num/new_den be represented with (terms-1) terms, each > d?
            # Lower bound: (terms-1) / (d+1) >= new_num/new_den
            # i.e., (terms-1) * new_den >= new_num * (d+1)
            if (terms - 1) * new_den < new_num * (d + 1):
                continue
            
            # Also, the first of the remaining terms must be > d, so:
            # (terms-1) / (d+1) >= new_num/new_den is a necessary condition.
            # Additionally, new_num/new_den > 0 is guaranteed.
            
            # Recurse
            current_terms.append(d)
            dfs(new_num, new_den, d + 1, terms - 1, current_terms)
            current_terms.pop()
            
            if found_better and best_max <= d_lower:
                # We found a solution with max denom <= d_lower, so no need to continue
                # (since further d values will have max denom >= d > d_lower >= best_max)
                break
        
        return found_better
    
    # Try different numbers of terms, starting from 2 (minimum for non-unit fraction)
    # We know 3 terms works. Try 2, then 3, then 4, etc.
    # For each term count, do DFS.
    for num_terms in range(2, min(8, MAX_TERMS + 1)):
        if time.time() > deadline:
            break
        # Try DFS with num_terms
        # Initial min_denom: the smallest possible denominator is ceil(31/5) = 7
        # But we need distinct denominators, so start from 1? No, 1/1 = 1 > 5/31, so min is 7.
        dfs(TARGET_NUM, TARGET_DEN, 1, num_terms, [])
        if found_better:
            break  # Found something better, stop searching more terms
    
    # Return the best solution found
    if best_solution is not None and _verify(best_solution):
        return best_solution
    elif _verify(FALLBACK):
        return FALLBACK
    else:
        # Last resort: greedy
        from fractions import Fraction
        r = Fraction(TARGET_NUM, TARGET_DEN)
        greedy_vals = []
        while r > 0 and len(greedy_vals) < MAX_TERMS:
            d = (r.denominator + r.numerator - 1) // r.numerator
            if d > MAX_DENOM:
                break
            greedy_vals.append(d)
            r -= Fraction(1, d)
        if r == 0 and _verify(greedy_vals):
            return greedy_vals
        return FALLBACK

# EVOLVE-BLOCK-END
