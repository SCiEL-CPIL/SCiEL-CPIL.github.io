"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
from fractions import Fraction
import math

def construct():
    """
    Returns 1..256 distinct integer denominators in [5, 10000000] 
    whose reciprocal sum is exactly 1; minimize their count.
    """
    
    def find_egyptian(remaining, min_denom, k):
        """
        Find k distinct integers d1 < d2 < ... < dk such that 
        sum(1/di) = remaining and di >= min_denom.
        Returns list of denominators or None.
        """
        if k == 0:
            return [] if remaining == 0 else None
        
        if k == 1:
            if remaining.denominator == 1:
                d = remaining.numerator
                if d >= min_denom and d <= 10000000:
                    return [d]
            return None

        # Lower bound for first denominator a:
        # 1/a <= remaining => a >= 1/remaining
        # Also a >= min_denom
        if remaining < 1:
            min_a = max(min_denom, math.ceil(remaining.denominator / remaining.numerator))
        else:
            min_a = min_denom
            
        # Upper bound for first denominator a:
        # remaining - 1/a must be <= sum of max possible (k-1) terms with denom > a
        # Max sum is 1/(a+1) + ... + 1/(a+k-1)
        # If remaining - 1/a > MaxSum, then a is too small? 
        # No, if a is small, 1/a is large, remaining - 1/a is small.
        # If a is large, 1/a is small, remaining - 1/a is large.
        # We need remaining - 1/a <= MaxSum(a).
        # So 1/a >= remaining - MaxSum(a).
        
        # Also, the minimum sum of (k-1) terms with denom > a is ~0.
        # But we need to ensure remaining - 1/a > 0 => 1/a < remaining => a > 1/remaining.
        # This is covered by min_a if we use strict inequality? 
        # If 1/a = remaining, then remaining - 1/a = 0. We need k-1 terms to sum to 0. Impossible for k>1.
        # So we need remaining - 1/a > 0 => a > 1/remaining.
        # So min_a should be at least floor(1/remaining) + 1 if 1/remaining is integer?
        # Let's just check remaining - 1/a > 0 in the loop.
        
        # To find an upper bound U for a:
        # We know a cannot be arbitrarily large.
        # If a is very large, 1/a is very small.
        # The problem is finding the right a.
        # For small k, a is usually not huge.
        # Let's set a practical upper bound. 
        # Since min_denom >= 5, and we want few terms, denominators won't be 10^7 immediately.
        # However, to be safe, we can bound a by the denominator of 'remaining' multiplied by k?
        # Or just iterate up to a limit like 10^5 or 10^6 first.
        
        # Let's refine the upper bound using the MaxSum condition.
        # Let f(a) = remaining - 1/a - sum_{i=1}^{k-1} 1/(a+i).
        # We need f(a) <= 0.
        # f(a) is decreasing? 
        # 1/a is decreasing. sum 1/(a+i) is decreasing.
        # So -1/a is increasing. -sum is increasing.
        # So f(a) is increasing.
        # If f(a) > 0, we need larger a? No.
        # Let's check values.
        # a small: 1/a large. remaining - 1/a small (or negative). Sum terms large. f(a) likely negative.
        # a large: 1/a small. remaining - 1/a ~ remaining. Sum terms small. f(a) ~ remaining > 0.
        # So there is a crossing point. We need f(a) <= 0.
        # So a must be <= a_max where f(a_max) <= 0 < f(a_max + 1).
        
        # We can binary search for a_max or just iterate.
        # Given k is small (likely < 10), iterating a from min_a upwards is fine if a_max is not too far from min_a.
        # If min_a is large, the range might be small.
        
        # Let's just iterate a from min_a.
        # We need an upper bound to stop.
        # If a > remaining.denominator * k, then 1/a < 1/(q*k).
        # The denominator of remaining - 1/a will be large.
        # A safe upper bound is 10000000, but that's too big to iterate.
        # However, if no solution is found for a < 10000, we might need to go higher.
        # But for "minimize count", the denominators are usually small.
        
        # Let's try iterating up to min(10000000, min_a + 100000) first.
        # If that fails, we might need a smarter search, but for this task, small solutions exist.
        
        # Optimization: 
        # For k=2, we have an exact formula.
        if k == 2:
            # 1/x + 1/y = remaining = p/q
            # y = x / (x*p/q - 1) = x*q / (x*p - q)
            # Let D = x*p - q. y = x*q / D.
            # D must divide x*q.
            # Also y > x => x*q/D > x => q/D > 1 => D < q.
            # And D > 0 => x*p > q => x > q/p.
            # So x is in (q/p, ...).
            # Also D = x*p - q.
            # Iterate x from max(min_denom, floor(q/p)+1) upwards.
            # Upper bound for x: 
            # y > x => D < q.
            # x*p - q < q => x*p < 2q => x < 2q/p.
            # So x is in [max(min_denom, floor(q/p)+1), floor(2q/p)-1].
            
            p = remaining.numerator
            q = remaining.denominator
            start_x = max(min_denom, q // p + 1)
            end_x = (2 * q) // p
            
            for x in range(start_x, end_x + 1):
                D = x * p - q
                if D <= 0:
                    continue
                if (x * q) % D == 0:
                    y = (x * q) // D
                    if y > x and y <= 10000000:
                        return [x, y]
            return None

        # For k > 2
        # We iterate a.
        # Upper bound estimation:
        # If we can't find a solution in the first 10000 candidates, something is wrong or a is large.
        # But let's try a bit wider.
        
        # To avoid infinite loops or huge searches, we set a hard cap on the number of iterations or range.
        # However, for k=3,4,5, the solutions are typically found quickly.
        
        # Let's determine a_max more carefully.
        # f(a) = remaining - 1/a - sum_{i=1}^{k-1} 1/(a+i)
        # We want f(a) <= 0.
        # Since f(a) is increasing, we can find the largest a such that f(a) <= 0.
        # If f(min_a) > 0, no solution.
        # Let's check f(min_a).
        
        # Calculate sum_{i=1}^{k-1} 1/(a+i)
        def max_sum(a, k_minus_1):
            s = Fraction(0, 1)
            for i in range(1, k_minus_1 + 1):
                s += Fraction(1, a + i)
            return s

        # Check if f(min_a) > 0
        ms = max_sum(min_a, k - 1)
        if remaining - Fraction(1, min_a) > ms:
            return None # Even the smallest possible a leaves a remainder that is too large to be formed by the next k-1 terms? 
            # Wait. If remaining - 1/a > MaxSum, it means the remainder is *larger* than the max possible sum of the next terms.
            # So we *cannot* form the remainder. We need a *smaller* 1/a? No, smaller a means larger 1/a, smaller remainder.
            # So if at min_a the remainder is too big, we need to increase a?
            # No. If a increases, 1/a decreases, remainder increases.
            # So if remainder > MaxSum at min_a, it will be even larger for larger a.
            # So no solution?
            # Let's re-verify.
            # We need Remainder <= MaxSum.
            # At a=min_a, if Remainder > MaxSum, then for any a > min_a, Remainder increases and MaxSum decreases.
            # So Remainder will be even more > MaxSum.
            # So yes, no solution.
            pass

        # Find a_max such that f(a_max) <= 0.
        # We can binary search or just iterate and break when f(a) > 0.
        
        # To speed up, we can estimate a_max.
        # For large a, 1/a ~ 0, MaxSum ~ (k-1)/a.
        # f(a) ~ remaining - (k-1)/a.
        # remaining - (k-1)/a <= 0 => a <= (k-1)/remaining.
        # So a_max is roughly (k-1)/remaining.
        
        # Let's iterate a from min_a to a_limit.
        # a_limit = min(10000000, int((k - 1) / remaining) + 1000)
        # But (k-1)/remaining might be huge if remaining is small.
        # However, if remaining is small, min_a is large.
        
        # Let's just iterate and break if a gets too large or f(a) > 0.
        
        # We will iterate a from min_a.
        # We will stop if a > 10000000.
        # We will also stop if remaining - 1/a > max_sum(a, k-1).
        
        # To avoid slow loops, we can skip a if it's not a candidate?
        # No, we just iterate.
        
        # Optimization: 
        # If k is small, the number of iterations is small.
        
        a = min_a
        # Upper bound for loop to prevent hanging if no solution
        # If min_a is close to 10^7, range is small.
        # If min_a is 5, range could be large.
        # But f(a) > 0 will trigger quickly if a is too large.
        
        # Let's add a check: if a > 1000000, we might be in trouble for small k?
        # No, for k=3, a can be large.
        
        # Let's just iterate.
        
        # To speed up, we can check if (remaining - 1/a) is 0? No, k>1.
        
        while a <= 10000000:
            # Check bound
            ms = max_sum(a, k - 1)
            rem = remaining - Fraction(1, a)
            
            if rem < 0:
                break # a is too small? No, a >= min_a ensures 1/a <= remaining? 
                # min_a was ceil(1/remaining). So 1/a <= remaining. rem >= 0.
                # If rem == 0, we need k-1 terms to sum to 0. Impossible.
                # So we need rem > 0.
                
            if rem == 0:
                a += 1
                continue
                
            if rem > ms:
                # As a increases, rem increases and ms decreases.
                # So rem > ms will hold for all larger a.
                break
                
            # Now try to find k-1 terms for 'rem' with min_denom = a+1
            res = find_egyptian(rem, a + 1, k - 1)
            if res is not None:
                return [a] + res
                
            a += 1
            
        return None

    target = Fraction(1, 1)
    
    # Try increasing number of terms
    for k in range(3, 20):
        result = find_egyptian(target, 5, k)
        if result is not None:
            # Verify
            s = sum(Fraction(1, d) for d in result)
            if s == target and len(set(result)) == len(result) and all(5 <= d <= 10000000 for d in result):
                return result
                
    raise RuntimeError("no representation found")
# EVOLVE-BLOCK-END
