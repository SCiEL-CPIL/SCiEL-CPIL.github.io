"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from fractions import Fraction
from math import ceil

def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # Safety margin before 60s limit
    
    # We need distinct integers in [5, 10000000] summing to 1.
    # Minimize count.
    # Max sum for k=6: 1/5+1/6+1/7+1/8+1/9+1/10 = 2131/2520 < 1.
    # So k >= 7.
    
    # Known solution with 7 terms:
    # 1/5 + 1/6 + 1/7 + 1/8 + 1/9 + 1/10 + 1/2520
    # Sum of first 6 is 2131/2520.
    # 1 - 2131/2520 = (2520 - 2131)/2520 = 389/2520.
    # Wait, 389/2520 is not a unit fraction.
    # Let's check: 389 is prime?
    # sqrt(389) ~ 19.7. Primes to check: 2, 3, 5, 7, 11, 13, 17, 19.
    # Not div by 2, 3 (sum=20), 5.
    # 389 = 7*55 + 4.
    # 389 = 11*35 + 4.
    # 389 = 13*29 + 12.
    # 389 = 17*22 + 15.
    # 389 = 19*20 + 9.
    # So 389 is prime.
    # Thus 389/2520 cannot be represented as 1/d.
    # So the specific set {5,6,7,8,9,10} does not work with a 7th term.
    
    # We need to search for a 7-term solution.
    # Strategy:
    # Fix the first few denominators (small ones) and solve for the rest.
    # Or use a DFS that is very fast.
    
    # Let's try a DFS for k=7.
    # To make it fast, we can use integer arithmetic with a common denominator.
    # However, the common denominator can get large.
    # Fractions are slower but handle gcd automatically.
    # Let's try to optimize the DFS with tight bounds.
    
    def dfs(start, remaining, k_left, chosen):
        if time.time() > deadline:
            return None
            
        if k_left == 0:
            if remaining == 0:
                return chosen[:]
            return None
            
        if k_left == 1:
            # remaining = 1/d
            if remaining > 0:
                num, den = remaining.numerator, remaining.denominator
                if den % num == 0:
                    d = den // num
                    if d > start and d >= 5 and d <= 10000000:
                        return chosen + [d]
            return None
            
        # Pruning
        # Max sum of k_left terms starting from start+1
        # We need to check if it's possible to reach 'remaining'
        # Lower bound for next denominator d:
        # 1/d <= remaining => d >= 1/remaining
        # Also, we need to fit k_left terms.
        # The smallest possible sum for k_left terms starting from d is approx k_left/d (if d is large)
        # Actually, sum_{i=0}^{k_left-1} 1/(d+i) >= remaining
        # This implies d cannot be too large.
        # Approximation: k_left/d >= remaining => d <= k_left/remaining
        
        # Upper bound for d:
        # We need sum_{i=0}^{k_left-1} 1/(d+i) >= remaining
        # A loose upper bound is d <= k_left / remaining
        # A tighter upper bound:
        # 1/d + 1/(d+1) + ... >= remaining
        # Since 1/(d+i) < 1/d, sum < k_left/d. So k_left/d > remaining => d < k_left/remaining.
        
        # Lower bound for d:
        # 1/d < remaining (strictly, because we have other positive terms)
        # So d > 1/remaining.
        # Also d > start.
        
        # Let's refine the bounds.
        # d_min = max(start + 1, ceil(1/remaining) + 1)
        # Wait, 1/d must be strictly less than remaining?
        # Yes, because remaining = 1/d + rest, and rest > 0.
        # So 1/d < remaining => d > 1/remaining.
        
        # d_max:
        # We need sum_{i=0}^{k_left-1} 1/(d+i) >= remaining.
        # Let's iterate d from d_min upwards.
        
        d_min = max(start + 1, int(1 / remaining) + 1)
        
        # Estimate d_max
        # If d is too large, the sum is too small.
        # sum_{i=0}^{k_left-1} 1/(d+i) >= remaining
        # Approx d_max = int(k_left / remaining)
        # But this can be a loose bound.
        # Let's use a safer upper bound.
        # If d > 10^7, we stop.
        d_max = min(10000000, int(k_left / remaining) + 1)
        
        if d_min > d_max:
            return None
            
        for d in range(d_min, d_max + 1):
            new_remaining = remaining - Fraction(1, d)
            if new_remaining < 0:
                break
            result = dfs(d, new_remaining, k_left - 1, chosen + [d])
            if result is not None:
                return result
        return None

    # Try k=7
    # To speed up, we can fix the first few denominators.
    # The smallest denominators are the most likely to be part of the solution.
    # Let's try fixing the first 3 denominators from {5, 6, 7, 8, 9, 10, 11, ...}
    # Actually, a full DFS from 5 might be fast enough if pruned well.
    # But Fraction arithmetic is slow.
    
    # Alternative: Use integer arithmetic with a fixed LCM?
    # LCM of 5..10 is 2520.
    # If we assume denominators are divisors of some number, it's hard.
    
    # Let's try the DFS with Fraction first.
    # If it's too slow, we will need a different approach.
    
    # Optimization: Precompute 1/d as Fractions? No, d varies.
    
    # Let's try to search for 7 terms.
    # We can restrict the search space by noting that denominators must be small.
    # If we have 7 terms summing to 1, the largest denominator cannot be too large?
    # Actually, it can be large if the others are close to 1.
    # But since min is 5, the sum of 6 smallest is < 1.
    # So the 7th must be significant.
    
    # Let's run the DFS.
    # To improve speed, we can try specific small sets first.
    
    # Known 7-term solution?
    # 1/5 + 1/6 + 1/7 + 1/8 + 1/9 + 1/10 + 1/2520 didn't work.
    # What if we replace 10 with something else?
    
    # Let's just run the DFS.
    # To make it faster, we can use a heuristic: try smaller d first.
    
    # One potential issue: Fraction reduction is slow.
    # We can try to use integers with a common denominator for the first few steps?
    # No, denominators change.
    
    # Let's try a different strategy:
    # Iterate over subsets of small denominators (5 to L) of size m.
    # Calculate their sum S.
    # If S < 1, we need to represent 1-S with k-m terms.
    # If k=7, m can be 3, 4, 5.
    # If m=5, we need 2 terms to represent 1-S.
    # 1-S = 1/a + 1/b => (a+b)/(ab) = 1-S => ab(1-S) = a+b => a(b(1-S)-1) = b => a = b / (b(1-S)-1).
    # This is a fast check.
    
    # Let's try m=5 (pick 5 from 5..20, say) and solve for 2 terms.
    # This reduces the search space significantly.
    
    # Step 1: Enumerate combinations of 5 denominators from [5, 30].
    # Step 2: For each, calculate remainder R = 1 - sum.
    # Step 3: Check if R can be written as 1/a + 1/b with a,b > last_den, distinct, in range.
    
    # 1/a + 1/b = R
    # (a+b)/(ab) = R
    # a+b = Rab
    # ab - b/R - a/R = 0
    # a(b/R - 1) = b/R
    # a = b / (b/R - 1) = bR / (b - R)
    # So for a given b, a is determined.
    # We need a to be an integer and a > b (to keep distinct and ordered, or just distinct).
    # Also a, b > last_den.
    
    # We can iterate b from last_den+1 to some limit.
    # Limit for b:
    # As b -> infinity, a -> R.
    # Since a > b, we need R > b? No.
    # a = bR / (b-R).
    # For a > b, we need bR / (b-R) > b => R / (b-R) > 1 (assuming b-R > 0) => R > b - R => 2R > b => b < 2R.
    # Wait, if b < R, then b-R is negative, a is negative.
    # So we need b > R.
    # And for a > b, we need b < 2R.
    # So b is in range (R, 2R).
    # If we allow a < b, then we just need to find integer solutions.
    # Generally, we can iterate b in a reasonable range and check if a is integer.
    
    # This is much faster than full DFS for the last 2 terms.
    
    # Let's implement this "Hybrid" approach.
    # 1. Try to find 5 denominators from [5, 50] such that the remainder can be split into 2 unit fractions.
    # 2. If not found, try 4 denominators from [5, 50] and solve for 3 terms (slower, but maybe necessary).
    # 3. Fallback to full DFS.
    
    def solve_two(rem, min_d):
        # rem = 1/a + 1/b
        # a = b*rem / (b - rem)
        # b must be > rem (since a,b > 0 implies b-rem > 0 if rem > 0? 
        # If rem < 1, b > rem is always true for integer b >= 5.
        # Actually, if rem is small, b can be large.
        # But a and b are symmetric. We can assume a <= b.
        # Then 1/a >= 1/b => 2/a >= rem => a <= 2/rem.
        # So we can iterate a from min_d to int(2/rem).
        # Then b = a*rem / (a - rem).
        # Check if b is integer and b >= a.
        
        # Using Fraction for exactness.
        # rem is a Fraction.
        # a - rem > 0 => a > rem.
        # b = a*rem / (a - rem)
        
        # Let's iterate a.
        # a_min = min_d
        # a_max = int(2 / rem)
        # If a_max is huge, this is slow.
        # But rem is 1 - sum of 5 terms.
        # Sum of 5 smallest (5,6,7,8,9) = 1/5+1/6+1/7+1/8+1/9
        # 1/5=0.2, 1/6=0.166, 1/7=0.142, 1/8=0.125, 1/9=0.111
        # Sum ~ 0.744.
        # Rem ~ 0.256.
        # 2/Rem ~ 7.8.
        # So a is small. This is very fast.
        
        if rem <= 0:
            return None
            
        a_min = min_d
        # a must satisfy a > rem (for b to be positive)
        # And a <= 2/rem (for b >= a)
        a_max = int(2 / rem) + 1
        if a_max < a_min:
            return None
            
        for a in range(a_min, a_max + 1):
            # b = a*rem / (a - rem)
            # a - rem must be > 0
            if a <= rem:
                continue
            num = a * rem.numerator
            den = a * rem.denominator - rem.numerator
            if den <= 0:
                continue
            if num % den == 0:
                b = num // den
                if b >= a and b >= 5 and b <= 10000000:
                    return [a, b]
        return None

    # Enumerate 5 denominators from [5, 30]
    # C(26, 5) is about 65780. Very small.
    # For each, check solve_two.
    
    from itertools import combinations
    
    found = None
    # Try range 5 to 30 first
    for combo in combinations(range(5, 31), 5):
        if time.time() > deadline:
            break
        s = Fraction(0, 1)
        for d in combo:
            s += Fraction(1, d)
        if s >= 1:
            continue
        rem = 1 - s
        res = solve_two(rem, combo[-1] + 1)
        if res:
            found = list(combo) + res
            break
            
    if found:
        # Validate
        s = Fraction(0, 1)
        for d in found:
            s += Fraction(1, d)
        if s == 1 and len(set(found)) == len(found) and all(5 <= d <= 10000000 for d in found):
            return found

    # If not found, try 4 denominators from [5, 30] and solve for 3 terms?
    # Or expand the range for 5 denominators.
    # Let's try range 5 to 50.
    # C(46, 5) is about 1.3 million. Might be too slow with Fraction.
    # Let's stick to the found solution or fallback.
    
    # If the 5+2 strategy fails, try full DFS for k=7 with a time limit.
    # But the 5+2 strategy is very likely to succeed.
    # 1/5+1/6+1/7+1/8+1/9 = 0.744. Rem = 0.256.
    # a, b around 4, 5? No, min_d is 10.
    # a_min = 10.
    # a=10: b = 10*0.256 / (10-0.256) = 2.56 / 9.744 ~ 0.26. Not integer.
    # a=11: b = 11*0.256 / (11-0.256) = 2.816 / 10.744 ~ 0.26.
    # Wait, my estimate of rem was rough.
    # Let's calculate exactly for 5,6,7,8,9.
    # Sum = 252/1260 + 210/1260 + 180/1260 + 157.5/1260? No.
    # LCM(5,6,7,8,9) = 2520.
    # 1/5 = 504/2520
    # 1/6 = 420/2520
    # 1/7 = 360/2520
    # 1/8 = 315/2520
    # 1/9 = 280/2520
    # Sum = (504+420+360+315+280)/2520 = 1879/2520.
    # Rem = (2520-1879)/2520 = 641/2520.
    # 641 is prime? sqrt(641) ~ 25.
    # 641 not div by 2,3,5.
    # 641 = 7*91 + 4.
    # 641 = 11*58 + 3.
    # 641 = 13*49 + 4.
    # 641 = 17*37 + 12.
    # 641 = 19*33 + 14.
    # 641 = 23*27 + 20.
    # So 641 is prime.
    # Rem = 641/2520.
    # We need 1/a + 1/b = 641/2520.
    # a = b*Rem / (b - Rem).
    # b must be > 641/2520 ~ 0.25.
    # min_d = 10.
    # a_max = int(2 / Rem) = int(5040/641) = 7.
    # But a_min = 10.
    # So a_max < a_min. No solution for this combo.
    
    # We need a larger remainder.
    # This means we need smaller denominators in the first 5?
    # No, smaller denominators mean larger sum, smaller remainder.
    # We need a LARGER remainder to allow a, b to be small enough?
    # Wait.
    # If Rem is large, 2/Rem is small.
    # If Rem is small, 2/Rem is large.
    # We need a >= 10.
    # So we need 2/Rem >= 10 => Rem <= 0.2.
    # Current Rem = 641/2520 ~ 0.254.
    # So Rem is too large.
    # We need to increase the sum of the first 5 terms.
    # So we should pick LARGER denominators? No, larger denominators mean smaller terms.
    # We need SMALLER denominators?
    # 5,6,7,8,9 are the smallest.
    # Their sum is 1879/2520 ~ 0.745.
    # Rem ~ 0.255.
    # This is the MAXIMUM remainder we can get with 5 distinct denominators >= 5.
    # So Rem is always >= 0.255?
    # No, if we pick larger denominators, the sum is smaller, so Rem is LARGER.
    # If we pick smaller denominators... we can't go smaller than 5.
    # So 5,6,7,8,9 gives the MINIMUM remainder?
    # Sum is max, so Rem is min.
    # Min Rem = 641/2520 ~ 0.254.
    # We need Rem <= 0.2 to have a solution with a >= 10.
    # But Rem >= 0.254.
    # So it is IMPOSSIBLE to have a solution of form 5+2 with denominators >= 5?
    # Let's re-verify the condition.
    # a = b*Rem / (b - Rem).
    # We need a >= 10.
    # As b -> infinity, a -> Rem.
    # So a is always > Rem.
    # If Rem = 0.254, then a > 0.254.
    # This doesn't constrain a to be < 10.
    # Wait.
    # a = b*Rem / (b - Rem).
    # If b is large, a ~ Rem.
    # If b is close to Rem, a is large.
    # We need a to be an integer.
    # My previous bound a <= 2/Rem was for the case a <= b.
    # If a <= b, then 1/a >= 1/b, so 2/a >= 1/a + 1/b = Rem => a <= 2/Rem.
    # So if we assume a <= b, then a must be <= 2/Rem.
    # If Rem = 0.254, 2/Rem = 7.8.
    # So a <= 7.
    # But we need a >= 10 (since min_d is 10).
    # Contradiction.
    # So there is NO solution with 5 terms from {5..} and 2 terms from {> last}.
    # Because the remainder is too large to be split into two unit fractions with denominators > 9.
    
    # So we must use more terms in the "small" part?
    # Or fewer?
    # If we use 4 small terms, we need 3 large terms.
    # Sum of 4 smallest (5,6,7,8) = 504+420+360+315 = 1599/2520.
    # Rem = 921/2520 = 307/840 ~ 0.365.
    # We need 1/a + 1/b + 1/c = Rem.
    # a,b,c > 8.
    # This is harder to solve analytically, but easier to search.
    
    # Let's try 4 small terms from [5, 20] and search for 3 large terms.
    # Or just use the full DFS but with better pruning.
    
    # Let's try the full DFS again, but optimized.
    # The previous DFS might have been slow due to Fraction overhead.
    # Let's try to limit the search space.
    
    # Actually, there is a known 7-term solution:
    # 1/5 + 1/6 + 1/7 + 1/8 + 1/9 + 1/10 + 1/2520 didn't work.
    # What about:
    # 1/5 + 1/6 + 1/7 + 1/8 + 1/9 + 1/11 + 1/x
    # Sum 5..9 = 1879/2520.
    # 1/11 = 229.09/2520? No.
    # LCM(5,6,7,8,9,11) = 27720.
    # 1/5 = 5544
    # 1/6 = 4620
    # 1/7 = 3960
    # 1/8 = 3465
    # 1/9 = 3080
    # 1/11 = 2520
    # Sum = 5544+4620+3960+3465+3080+2520 = 23189.
    # Rem = (27720 - 23189)/27720 = 4531/27720.
    # 4531 prime?
    # If not prime, it might be a unit fraction.
    # 4531 / 7 = 647.28
    # 4531 / 11 = 411.9
    # 4531 / 13 = 348.5
    # 4531 / 17 = 266.5
    # 4531 / 19 = 238.4
    # 4531 / 23 = 197
    # 4531 = 23 * 197.
    # 197 is prime.
    # So 4531/27720 is not a unit fraction.
    
    # Let's just run the DFS. It should find a solution if one exists.
    # To speed it up, we can try to fix the first 3 denominators.
    
    # Let's try a specific set that is known to work or likely to work.
    # 1/5 + 1/6 + 1/7 + 1/8 + 1/9 + 1/10 + 1/11 ... sum is too high?
    # Sum 5..11 = 23189/27720 ~ 0.836.
    # Rem ~ 0.164.
    # We need 1 term? No, we used 6 terms. We need 1 more.
    # 1/d = Rem.
    # d = 27720/4531 ~ 6.1.
    # But d must be > 11.
    # So this set is too "heavy".
    
    # We need lighter denominators.
    # Try replacing 5 with 12?
    # 1/6+1/7+1/8+1/9+1/10+1/11 = ?
    # LCM(6..11) = 27720.
    # 1/6 = 4620
    # 1/7 = 3960
    # 1/8 = 3465
    # 1/9 = 3080
    # 1/10 = 2772
    # 1/11 = 2520
    # Sum = 20417.
    # Rem = 7303/27720.
    # 7303 / 7 = 1043.2
    # 7303 / 11 = 663.9
    # 7303 / 13 = 561.7
    # 7303 / 17 = 429.5
    # 7303 / 19 = 384.3
    # 7303 / 23 = 317.5
    # 7303 / 29 = 251.8
    # 7303 / 31 = 235.5
    # 7303 / 37 = 197.3
    # 7303 / 41 = 178.1
    # 7303 / 43 = 169.8
    # 7303 / 47 = 155.3
    # 7303 / 53 = 137.7
    # 7303 / 59 = 123.7
    # 7303 / 61 = 119.7
    # 7303 / 67 = 109
    # 7303 = 67 * 109.
    # Not a unit fraction.
    
    # It seems finding a 7-term solution is non-trivial.
    # Maybe 8 terms?
    # The task says "minimize their count".
    # If 7 is impossible, we return 8.
    # But we should try 7 first.
    
    # Let's use the DFS with a time limit.
    # If it finds a 7-term solution, return it.
    # Else, try 8-term solution.
    
    # To make DFS faster, we can use a different representation.
    # But Fraction is safest.
    
    # Let's try to run the DFS for k=7.
    # If it times out, we fall back to k=8.
    
    # For k=8, we can use the 5+3 strategy?
    # Or just DFS.
    
    # Let's refine the DFS to be as fast as possible.
    
    def dfs_fast(start, remaining, k_left, chosen):
        if time.time() > deadline:
            return None
            
        if k_left == 0:
            if remaining == 0:
                return chosen[:]
            return None
            
        if k_left == 1:
            if remaining > 0:
                num, den = remaining.numerator, remaining.denominator
                if den % num == 0:
                    d = den // num
                    if d > start and d >= 5 and d <= 10000000:
                        return chosen + [d]
            return None
            
        d_min = max(start + 1, int(1 / remaining) + 1)
        d_max = min(10000000, int(k_left / remaining) + 1)
        
        if d_min > d_max:
            return None
            
        for d in range(d_min, d_max + 1):
            new_remaining = remaining - Fraction(1, d)
            if new_remaining < 0:
                break
            result = dfs_fast(d, new_remaining, k_left - 1, chosen + [d])
            if result is not None:
                return result
        return None

    # Try k=7
    result = dfs_fast(4, Fraction(1, 1), 7, [])
    if result:
        return result

    # Try k=8
    result = dfs_fast(4, Fraction(1, 1), 8, [])
    if result:
        return result

    # Fallback to DP
    target, reached = 2520, {0: []}
    for d in range(5, target + 1):
        if target % d:
            continue
        w = target // d
        for total, values in list(reached.items()):
            if total + w <= target and total + w not in reached:
                reached[total + w] = values + [d]
        if target in reached:
            return reached[target]
        if time.time() > deadline:
            break
    raise RuntimeError("no baseline representation")

# EVOLVE-BLOCK-END
