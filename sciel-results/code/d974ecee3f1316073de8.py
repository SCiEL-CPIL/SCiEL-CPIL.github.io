"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import math
import time

def construct():
    """
    Return a list of distinct integer denominators in [5, 10000000]
    whose reciprocal sum is exactly 1, minimizing the count.
    
    Uses a greedy-like search with backtracking to find the minimum number of terms.
    Since we need denominators >= 5, the minimum number of terms is at least 2
    (since 1/5 + 1/x = 1 => 1/x = 4/5 => x = 5/4, not integer).
    Actually, let's check small counts:
    Count 1: 1/d = 1 => d=1 (invalid, <5)
    Count 2: 1/a + 1/b = 1, a,b >= 5, a<b.
             1/b = 1 - 1/a = (a-1)/a => b = a/(a-1).
             For integer b, a-1 must divide a. gcd(a, a-1)=1, so a-1=1 => a=2 (invalid).
             So count 2 is impossible.
    Count 3: 1/a + 1/b + 1/c = 1, a<b<c, a>=5.
             Try a=5: 1/b + 1/c = 4/5.
             1/c = 4/5 - 1/b = (4b-5)/5b => c = 5b/(4b-5).
             4b-5 must divide 5b.
             4b-5 | 5b => 4b-5 | 5b - (4b-5) = 5.
             So 4b-5 is a divisor of 5.
             Divisors of 5: 1, 5.
             4b-5=1 => 4b=6 => b=1.5 (no)
             4b-5=5 => 4b=10 => b=2.5 (no)
             Try a=6: 1/b + 1/c = 5/6.
             1/c = 5/6 - 1/b = (5b-6)/6b => c = 6b/(5b-6).
             5b-6 | 6b => 5b-6 | 6b - (5b-6) = 6.
             5b-6 is a divisor of 6.
             5b-6=1 => 5b=7 (no)
             5b-6=2 => 5b=8 (no)
             5b-6=3 => 5b=9 (no)
             5b-6=6 => 5b=12 (no)
             Try a=7: 1/b + 1/c = 6/7.
             1/c = 6/7 - 1/b = (6b-7)/7b => c = 7b/(6b-7).
             6b-7 | 7b => 6b-7 | 7.
             6b-7=1 => 6b=8 (no)
             6b-7=7 => 6b=14 (no)
             Try a=8: 1/b + 1/c = 7/8.
             1/c = 7/8 - 1/b = (7b-8)/8b => c = 8b/(7b-8).
             7b-8 | 8b => 7b-8 | 8.
             7b-8=1 => 7b=9 (no)
             7b-8=2 => 7b=10 (no)
             7b-8=4 => 7b=12 (no)
             7b-8=8 => 7b=16 (no)
             Try a=9: 1/b + 1/c = 8/9.
             1/c = 8/9 - 1/b = (8b-9)/9b => c = 9b/(8b-9).
             8b-9 | 9b => 8b-9 | 9.
             8b-9=1 => 8b=10 (no)
             8b-9=3 => 8b=12 (no)
             8b-9=9 => 8b=18 (no)
             Try a=10: 1/b + 1/c = 9/10.
             1/c = 9/10 - 1/b = (9b-10)/10b => c = 10b/(9b-10).
             9b-10 | 10b => 9b-10 | 10.
             9b-10=1 => 9b=11 (no)
             9b-10=2 => 9b=12 (no)
             9b-10=5 => 9b=15 (no)
             9b-10=10 => 9b=20 (no)
             
             It seems count 3 is also impossible?
             Let's check a=5 again.
             1/5 + 1/b + 1/c = 1
             1/b + 1/c = 4/5
             (b-5)(c-5) = 25
             Factors of 25: (1,25), (5,5).
             b-5=1, c-5=25 => b=6, c=30.
             Check: 1/5 + 1/6 + 1/30 = 6/30 + 5/30 + 1/30 = 12/30 = 2/5 != 1.
             Wait.
             1/b + 1/c = 4/5
             (b+c)/(bc) = 4/5
             5(b+c) = 4bc
             4bc - 5b - 5c = 0
             16bc - 20b - 20c = 0
             (4b-5)(4c-5) = 25
             4b-5 and 4c-5 are divisors of 25.
             Divisors: 1, 5, 25.
             Since b>=6, 4b-5 >= 19.
             So 4b-5 must be 25.
             4b-5=25 => 4b=30 => b=7.5 (no).
             
             So count 3 is impossible with a=5.
             
             Let's try a general search for the minimum count.
             The known identity for 1 with distinct denominators >= 2 is:
             1 = 1/2 + 1/3 + 1/7 + 1/42
             But denominators must be >= 5.
             
             Let's look for a representation with 4 terms.
             1 = 1/5 + 1/6 + 1/7 + 1/x ... no.
             
             Actually, there is a known result for the minimum number of terms in an Egyptian fraction expansion of 1 with denominators >= k.
             For k=5, the minimum is 5?
             
             Let's try to find a 5-term solution.
             1 = 1/5 + 1/6 + 1/7 + 1/8 + 1/x ...
             Sum of 1/5+1/6+1/7+1/8 = (168+140+120+105)/840 = 533/840.
             1 - 533/840 = 307/840.
             1/x = 307/840 => x = 840/307 (not integer).
             
             Let's use a systematic search with a deadline.
    """
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin for 90s limit
    
    def solve(num, den, min_den, max_terms):
        """
        Find a set of distinct denominators >= min_den
        that sum to num/den.
        Returns a list of denominators or None.
        """
        if max_terms == 0:
            return None
        if num == den:
            return []
        if num > den:
            return None
            
        # Greedy lower bound for next denominator
        # We need 1/d <= num/den => d >= den/num
        # Also d >= min_den
        start_d = max(min_den, math.ceil(den / num))
        
        # Upper bound for d:
        # If we take 1/d, the remaining sum is (num*den - d) / (den*d).
        # The remaining max_terms-1 terms must sum to this.
        # The largest possible sum for max_terms-1 terms starting from d+1 is:
        # 1/(d+1) + 1/(d+2) + ...
        # A simple bound: (max_terms-1) / (d+1)
        # So we need (num*den - d) / (den*d) <= (max_terms-1) / (d+1)
        # (num*den - d)(d+1) <= den*d*(max_terms-1)
        # This is complex. Let's just iterate d from start_d upwards.
        
        # Another bound:
        # The remaining fraction is (num*den - d) / (den*d).
        # This must be positive, so d < num*den.
        # Also, the next term 1/d' must be <= remaining fraction.
        # d' >= den*d / (num*den - d)
        
        # Let's iterate d
        # To keep it fast, we limit the range of d.
        # If d is too large, the remaining fraction is small, requiring large denominators.
        
        # We can bound d by:
        # 1/d >= 1/(min_den + max_terms) is not right.
        
        # Let's just try d from start_d to some limit.
        # A reasonable limit: den//num + max_terms * 100?
        # Actually, if we pick d, the next min_den is d+1.
        
        # Optimization:
        # If num/den is small, d must be large.
        
        # Let's try a few values of d.
        # We can bound d by:
        # The maximum sum of max_terms-1 terms starting from d+1 is roughly (max_terms-1)/(d+1).
        # We need num/den - 1/d <= (max_terms-1)/(d+1)
        # This gives an upper bound on d.
        
        # Let's solve for d:
        # num/den - 1/d <= (max_terms-1)/(d+1)
        # num/den <= 1/d + (max_terms-1)/(d+1)
        # num/den <= (d+1 + d(max_terms-1)) / (d(d+1))
        # num/den <= (d*max_terms + 1) / (d^2 + d)
        # num(d^2 + d) <= den(d*max_terms + 1)
        # num*d^2 + num*d <= den*max_terms*d + den
        # num*d^2 + (num - den*max_terms)*d - den <= 0
        
        # Roots of num*d^2 + (num - den*max_terms)*d - den = 0
        # d = [-(num - den*max_terms) + sqrt((num - den*max_terms)^2 + 4*num*den)] / (2*num)
        
        b = num - den * max_terms
        discriminant = b*b + 4*num*den
        if discriminant < 0:
            return None
        sqrt_disc = math.isqrt(discriminant)
        if sqrt_disc * sqrt_disc < discriminant:
            sqrt_disc += 1 # Ceiling
        upper_d = (-(b) + sqrt_disc) // (2 * num)
        
        # Also, d must be such that the remaining fraction can be represented.
        # The remaining fraction is (num*den - d) / (den*d).
        # Let rem_num = num*den - d, rem_den = den*d.
        # We need rem_num > 0, so d < num*den.
        
        # So d is in [start_d, min(upper_d, num*den - 1)]
        
        end_d = min(upper_d, num * den - 1)
        
        if start_d > end_d:
            return None
            
        # Iterate d
        for d in range(start_d, end_d + 1):
            if time.time() > deadline:
                return None
                
            rem_num = num * den - d
            rem_den = den * d
            
            if rem_num <= 0:
                continue
                
            # Reduce fraction
            g = math.gcd(rem_num, rem_den)
            rn = rem_num // g
            rd = rem_den // g
            
            # Recursive call
            result = solve(rn, rd, d + 1, max_terms - 1)
            if result is not None:
                return [d] + result
                
    # Try increasing number of terms
    # We know 1, 2, 3 terms are impossible for min_den=5.
    # Start with 4? Or 5?
    # Let's try from 4 upwards.
    
    for terms in range(4, 10):
        if time.time() > deadline:
            break
        result = solve(1, 1, 5, terms)
        if result is not None:
            # Verify
            total = 0
            for d in result:
                total += 1/d
            if abs(total - 1.0) < 1e-9:
                # Check distinctness and range
                if len(set(result)) == len(result) and all(5 <= d <= 10000000 for d in result):
                    return result
                    
    # Fallback: Use a known valid solution if search fails
    # 1 = 1/5 + 1/6 + 1/7 + 1/42 + 1/140 + 1/210 + 1/420 + 1/840 + 1/1260 + 1/2520 ...
    # Let's construct one.
    # 1 = 1/2 + 1/3 + 1/7 + 1/42
    # Replace 1/2 with 1/5 + 1/6 + 1/30? No.
    
    # A simple valid solution:
    # 1 = 1/5 + 1/5 + 1/5 + 1/5 + 1/5 (not distinct)
    
    # Let's use the identity:
    # 1 = 1/2 + 1/3 + 1/6
    # 1/2 = 1/5 + 1/6 + 1/30 - 1/15? No.
    
    # Let's just return a valid set found by a quick check.
    # 1/5 + 1/6 + 1/7 + 1/14 + 1/21 + 1/42 + 1/105 + 1/210 + 1/420 + 1/840 + 1/1260 + 1/2520 + 1/5040 + 1/10080 + 1/20160 + 1/40320 + 1/80640 + 1/161280 + 1/322560 + 1/645120 + 1/1290240 + 1/2580480 + 1/5160960 + 1/10321920 + 1/20643840 + 1/41287680 + 1/82575360 + 1/165150720 + 1/330301440 + 1/660602880 + 1/1321205760 + 1/2642411520 + 1/5284823040 + 1/10569646080
    # This is getting long.
    
    # Let's try to find a shorter one.
    # 1 = 1/5 + 1/6 + 1/7 + 1/8 + 1/9 + 1/x ...
    
    # Actually, the search above should find a solution if one exists with <= 9 terms.
    # If it doesn't, we need a fallback.
    
    # A known 5-term solution?
    # 1/5 + 1/6 + 1/7 + 1/8 + 1/9 = 0.2 + 0.1666 + 0.1428 + 0.125 + 0.1111 = 0.7455
    # Remaining 0.2545 = 1/x + 1/y ...
    
    # Let's just return a valid 6-term solution if found.
    # If not, raise an error.
    
    raise RuntimeError("no baseline representation")
# EVOLVE-BLOCK-END
