"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import math
from fractions import Fraction

def construct():
    start_time = time.time()
    deadline = start_time + 50  # 50s safety margin before 60s limit
    
    # The minimal Egyptian fraction for 1 with denominators >= 5
    # 1/5 + 1/6 + 1/30 = 6/30 + 5/30 + 1/30 = 12/30 = 2/5... wait
    # Let's verify: 1/5 + 1/6 + 1/30 = 6/30 + 5/30 + 1/30 = 12/30 = 2/5. That's not 1.
    
    # Let me recalculate. We need sum of 1/d_i = 1 where d_i >= 5.
    # 1/5 + 1/5 + 1/5 = 3/5 (but denominators must be distinct)
    # 1/5 + 1/6 + 1/7 + 1/42 = 42/210 + 35/210 + 30/210 + 5/210 = 112/210 = 8/15. Not 1.
    
    # Known result: The minimal number of terms is 3.
    # 1/5 + 1/6 + 1/30 doesn't work as shown above.
    # Let's check: 1/5 + 1/6 + 1/x = 1 => 1/x = 1 - 1/5 - 1/6 = 30/30 - 6/30 - 5/30 = 19/30. x = 30/19, not integer.
    # 1/5 + 1/7 + 1/x = 1 => 1/x = 1 - 1/5 - 1/7 = 35/35 - 7/35 - 5/35 = 23/35. x = 35/23, not integer.
    # 1/5 + 1/8 + 1/x = 1 => 1/x = 1 - 1/5 - 1/8 = 40/40 - 8/40 - 5/40 = 27/40. x = 40/27, not integer.
    # 1/5 + 1/9 + 1/x = 1 => 1/x = 1 - 1/5 - 1/9 = 45/45 - 9/45 - 5/45 = 31/45. x = 45/31, not integer.
    # 1/5 + 1/10 + 1/x = 1 => 1/x = 1 - 1/5 - 1/10 = 10/10 - 2/10 - 1/10 = 7/10. x = 10/7, not integer.
    
    # Try 4 terms: 1/5 + 1/6 + 1/7 + 1/x = 1
    # 1/x = 1 - 1/5 - 1/6 - 1/7 = 210/210 - 42/210 - 35/210 - 30/210 = 103/210. x = 210/103, not integer.
    
    # Try: 1/5 + 1/6 + 1/8 + 1/x = 1
    # 1/x = 1 - 1/5 - 1/6 - 1/8 = 120/120 - 24/120 - 20/120 - 15/120 = 61/120. x = 120/61, not integer.
    
    # Let's use the known optimal solution. The minimum is 3 terms.
    # Actually, let me search more carefully.
    # 1/5 + 1/5 + 1/5 is not allowed (distinct).
    # 
    # Known: 1/5 + 1/6 + 1/30 is NOT 1.
    # 
    # Let me try: 1/5 + 1/6 + 1/7 + 1/42 = 1/5 + 1/6 + (1/7 + 1/42) = 1/5 + 1/6 + 1/6 = 1/5 + 1/3 = 8/15. No.
    # 
    # Actually 1/7 + 1/42 = 6/42 + 1/42 = 7/42 = 1/6. So 1/5 + 1/6 + 1/6 + 1/42 but 1/6 appears twice.
    # 
    # Let's try a different approach. The known minimal solution for denominators > 4 is:
    # 1/5 + 1/6 + 1/30 is not 1.
    # 
    # Let me compute: what 3-term Egyptian fractions equal 1?
    # 1/a + 1/b + 1/c = 1 with 5 <= a < b < c
    # 
    # a=5: 1/b + 1/c = 4/5. (b-5)(c-5) = 25. Pairs: (1,25),(5,5). 
    #   b-5=1, c-5=25 => b=6, c=30. Check: 1/5+1/6+1/30 = 6/30+5/30+1/30=12/30=2/5. NOT 1!
    #   Wait, I made an error. 1/b + 1/c = 4/5 means (b+c)/(bc) = 4/5, so 5(b+c) = 4bc, 
    #   4bc - 5b - 5c = 0, (4b-5)(4c-5) = 25. 
    #   4b-5=1 => b=6/4, not integer. 4b-5=5 => b=10/4, not integer. 4b-5=25 => b=30/4, not integer.
    #   So no 3-term solution with a=5?
    # 
    # a=6: 1/b + 1/c = 5/6. (6b-5)(6c-5) = 25. 
    #   6b-5=1 => b=1, too small. 6b-5=5 => b=10/6. 6b-5=25 => b=30/6=5. But b>a=6, so no.
    # 
    # a=7: 1/b + 1/c = 6/7. (7b-6)(7c-6) = 36.
    #   7b-6=1 => b=1. 7b-6=2 => b=8/7. 7b-6=3 => b=9/7. 7b-6=4 => b=10/7. 
    #   7b-6=6 => b=12/7. 7b-6=9 => b=15/7. 7b-6=12 => b=18/7. 7b-6=18 => b=24/7. 
    #   7b-6=36 => b=42/7=6. But b>a=7, so no.
    # 
    # Hmm, it seems there's no 3-term solution. Let me check 4 terms.
    # 
    # Actually, I recall that the answer is 3: [5, 6, 30] is often cited but let me recheck.
    # 1/5 + 1/6 + 1/30 = 6/30 + 5/30 + 1/30 = 12/30 = 2/5. Definitely not 1.
    # 
    # What about [3, 7, 42]? 1/3+1/7+1/42 = 14/42+6/42+1/42 = 21/42 = 1/2. No.
    # [3, 8, 24] = 8/24+3/24+1/24 = 12/24 = 1/2. No.
    # [2, 3, 6] = 1/2+1/3+1/6 = 3/6+2/6+1/6 = 1. But 2,3 < 5.
    # [2, 4, 4] not distinct. [2, 5, 10] = 5/10+2/10+1/10 = 8/10. No.
    # 
    # For 4 terms: [5, 6, 7, 42] = 42/210+35/210+30/210+5/210 = 112/210 = 8/15. No.
    # [5, 6, 8, 120/... let me try systematically.
    # 
    # Actually, the known minimal solution with all denominators >= 5 is 4 terms:
    # 1/5 + 1/6 + 1/7 + 1/42 doesn't work.
    # 
    # Let me try: 1/5 + 1/6 + 1/8 + 1/x = 1
    # 1/x = 1 - 1/5 - 1/6 - 1/8 = (120-24-20-15)/120 = 61/120. x=120/61, not integer.
    # 
    # 1/5 + 1/6 + 1/9 + 1/x = 1
    # 1/x = 1 - 1/5 - 1/6 - 1/9 = (90-18-15-10)/90 = 47/90. x=90/47, not integer.
    # 
    # 1/5 + 1/6 + 1/10 + 1/x = 1
    # 1/x = 1 - 1/5 - 1/6 - 1/10 = (30-6-5-3)/30 = 16/30 = 8/15. x=15/8, not integer.
    # 
    # 1/5 + 1/7 + 1/8 + 1/x = 1
    # 1/x = 1 - 1/5 - 1/7 - 1/8 = (280-56-40-35)/280 = 149/280. x=280/149, not integer.
    # 
    # 1/5 + 1/7 + 1/9 + 1/x = 1
    # 1/x = 1 - 1/5 - 1/7 - 1/9 = (315-63-45-35)/315 = 172/315. x=315/172, not integer.
    # 
    # 1/5 + 1/7 + 1/10 + 1/x = 1
    # 1/x = 1 - 1/5 - 1/7 - 1/10 = (70-14-10-7)/70 = 39/70. x=70/39, not integer.
    # 
    # 1/5 + 1/7 + 1/11 + 1/x = 1
    # 1/x = 1 - 1/5 - 1/7 - 1/11 = (385-77-55-35)/385 = 218/385. x=385/218, not integer.
    # 
    # 1/5 + 1/7 + 1/12 + 1/x = 1
    # 1/x = 1 - 1/5 - 1/7 - 1/12 = (420-84-60-35)/420 = 241/420. x=420/241, not integer.
    # 
    # This is taking too long. Let me just use the greedy algorithm with verification.
    # The greedy algorithm for Egyptian fractions is well-known and will find a valid solution.
    # We'll verify it with Fraction to ensure correctness.
    
    # Try Greedy Algorithm
    num, den = 1, 1
    used = set()
    result = []
    
    while num > 0:
        if time.time() > deadline:
            break
            
        min_d = (den + num - 1) // num
        if min_d < 5:
            min_d = 5
            
        d = min_d
        while d in used:
            d += 1
            
        if d > 10000000:
            break
            
        result.append(d)
        used.add(d)
        
        num = num * d - den
        den = den * d
        
        g = math.gcd(num, den)
        num //= g
        den //= g
        
    # Verify with exact arithmetic
    if num == 0 and len(result) > 0 and len(result) <= 256:
        s = sum(Fraction(1, x) for x in result)
        if s == 1:
            # Try to reduce count via merging pairs where 1/a + 1/b = 1/c
            improved = True
            while improved:
                improved = False
                n = len(result)
                for i in range(n):
                    if time.time() > deadline:
                        break
                    for j in range(i+1, n):
                        a, b = result[i], result[j]
                        # Check if 1/a + 1/b = 1/c for integer c >= 5
                        # (a+b)/(a*b) = 1/c => c = a*b/(a+b)
                        num_m = a * b
                        den_m = a + b
                        if num_m % den_m == 0:
                            c = num_m // den_m
                            if c >= 5 and c <= 10000000 and c not in used:
                                new_result = []
                                new_used = set()
                                for k in range(n):
                                    if k != i and k != j:
                                        new_result.append(result[k])
                                        new_used.add(result[k])
                                new_result.append(c)
                                new_used.add(c)
                                s_new = sum(Fraction(1, x) for x in new_result)
                                if s_new == 1:
                                    result = new_result
                                    used = new_used
                                    improved = True
                                    break
                    if improved:
                        break
            
            # Final verification
            s_final = sum(Fraction(1, x) for x in result)
            if s_final == 1 and 1 <= len(result) <= 256:
                return result
    
    # Fallback: use the DP approach on a known common denominator
    # 2520 = lcm(5,6,7,8,9,10,11,12,13,14,15) ... actually lcm of 1..10 is 2520
    target, reached = 2520, {0: []}
    for d in range(5, target + 1):
        if target % d:
            continue
        w = target // d
        for total, values in list(reached.items()):
            if total + w <= target and total + w not in reached:
                reached[total + w] = values + [d]
        if target in reached:
            s_fb = sum(Fraction(1, x) for x in reached[target])
            if s_fb == 1:
                return reached[target]
    
    raise RuntimeError("no valid representation")
# EVOLVE-BLOCK-END
