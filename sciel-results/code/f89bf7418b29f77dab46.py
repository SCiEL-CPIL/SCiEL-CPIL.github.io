import time
import math
from functools import reduce
from math import isqrt

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin for 120s limit

    # Strategy:
    # 1. Construct N = Q * M, where Q is a powerful number (specifically a square of a primorial-like product)
    #    and M is a product of distinct primes p_i ≡ 1 (mod 4).
    # 2. The number of representations of M as a sum of two squares is 4 * 2^(t-1) where t is the number of distinct primes.
    #    The number of pairs (a, b) with a <= b, a+b=N, a,b powerful is related to this count.
    #    Specifically, if N = Q * M, and we write a = Q * x, b = Q * y, then x + y = M? No.
    #    Actually, the standard construction is:
    #    Let N = 2 * Q * M.
    #    If M = prod p_i, the number of representations of M as sum of two squares is 2^t.
    #    The pairs (x, y) such that x^2 + y^2 = M (up to sign/order) are 2^(t-1).
    #    Then a = 2*Q*x^2, b = 2*Q*y^2?
    #    Let's check: a + b = 2*Q*(x^2 + y^2) = 2*Q*M = N.
    #    Are a and b powerful?
    #    a = 2 * Q * x^2.
    #    If Q is a square (e.g., Q = S^2), then a = 2 * S^2 * x^2 = 2 * (S*x)^2.
    #    For a to be powerful, the prime 2 must appear with exponent >= 2.
    #    So we need the exponent of 2 in a to be >= 2.
    #    a = 2 * (S*x)^2. The exponent of 2 in a is 1 + 2 * v_2(S*x).
    #    This is always odd. It is >= 2 if v_2(S*x) >= 0.5, i.e., v_2(S*x) >= 1? No.
    #    If v_2(S*x) = 0, exp = 1 (not powerful).
    #    If v_2(S*x) >= 1, exp >= 3 (powerful).
    #    So we need S*x to be even.
    #    
    #    Alternative construction:
    #    Let N = Q * M where Q is a square (powerful) and M is a sum of two squares.
    #    Actually, a simpler known result:
    #    If N = 2^k * m, and m is odd.
    #    Let's use the property:
    #    If N = A^2 * M, where M is a product of t distinct primes ≡ 1 mod 4,
    #    then the number of representations of N as a sum of two squares is 4 * 2^t.
    #    Wait, the task is sum of two POWERFUL numbers, not sum of two squares.
    #    
    #    Let's re-read the inspiration carefully:
    #    "Construct N = 2^k * M where M is a product of the first t primes congruent to 1 mod 4... 
    #     for each representation (x,y) of M as sum of two squares, generate pair (2^k x^2, 2^k y^2)"
    #     
    #    Check:
    #    a = 2^k * x^2
    #    b = 2^k * y^2
    #    a + b = 2^k * (x^2 + y^2) = 2^k * M = N.
    #    
    #    Is a powerful?
    #    a = 2^k * x^2.
    #    Prime factors of a come from 2 and x.
    #    If p | x, then p^2 | x^2, so p^2 | a.
    #    The only issue is the prime 2.
    #    v_2(a) = k + 2 * v_2(x).
    #    For a to be powerful, we need v_2(a) >= 2.
    #    So we need k + 2 * v_2(x) >= 2.
    #    If k >= 2, this is always true.
    #    If k = 1, we need 1 + 2 * v_2(x) >= 2 => 2 * v_2(x) >= 1 => v_2(x) >= 1 (x even).
    #    If k = 0, we need 2 * v_2(x) >= 2 => v_2(x) >= 1.
    #    
    #    So, if we choose k >= 2, ALL representations (x,y) of M as sum of two squares yield powerful a,b.
    #    If k = 1, we only get powerful a,b if x is even.
    #    
    #    How many pairs (a,b) do we get?
    #    The number of integer solutions (x,y) to x^2 + y^2 = M is 4 * 2^t (if M has t distinct prime factors ≡ 1 mod 4 and no prime factors ≡ 3 mod 4).
    #    Actually, the number of representations r_2(M) = 4 * (d_1(M) - d_3(M)).
    #    If M = prod_{i=1}^t p_i where p_i ≡ 1 mod 4, then d_1(M) = 2^t, d_3(M) = 0.
    #    So r_2(M) = 4 * 2^t.
    #    These 4*2^t solutions include signs and order.
    #    The number of distinct pairs { |x|, |y| } with x,y >= 0 is 2^{t-1} (for t>=1).
    #    Specifically, the pairs (x,y) with x>0, y>0, x!=y are 2^{t-1}.
    #    If x=y, then 2x^2 = M, so M must be 2 * square. Since M is product of odd primes, x!=y always (unless M=0, not possible).
    #    So there are 2^{t-1} distinct unordered pairs {x,y} of positive integers.
    #    For each such pair, we get a = 2^k x^2, b = 2^k y^2.
    #    Since x != y, a != b.
    #    So we get 2^{t-1} distinct pairs (a,b) with a < b.
    #    
    #    Wait, is that all?
    #    The problem asks for pairs (a,b) of POWERFUL numbers summing to N.
    #    Are there other powerful numbers a,b summing to N that are NOT of the form 2^k x^2?
    #    Yes, potentially. But the task says "score is the number of pairs... certified lower bound".
    #    So if we find 2^{t-1} pairs, the score is at least 2^{t-1}.
    #    We want to maximize this.
    #    
    #    Constraints:
    #    N <= 10^18.
    #    N = 2^k * M.
    #    M = prod_{i=1}^t p_i.
    #    We need 2^k * M <= 10^18.
    #    To maximize 2^{t-1}, we want to maximize t.
    #    We want to minimize M for a given t.
    #    So we should pick the smallest primes p ≡ 1 mod 4.
    #    Primes ≡ 1 mod 4: 5, 13, 17, 29, 37, 41, 53, 61, 73, 89, 97, 101, 109, 113, ...
    #    
    #    Let's estimate the max t.
    #    Product of first t such primes <= 10^18 / 2^k.
    #    Let's try k=2 (smallest k that guarantees powerfulness for all x).
    #    Limit for M: 10^18 / 4 = 2.5 * 10^17.
    #    
    #    Let's compute the product of the first few primes ≡ 1 mod 4.
    #    p1=5
    #    p2=13
    #    p3=17
    #    p4=29
    #    p5=37
    #    p6=41
    #    p7=53
    #    p8=61
    #    p9=73
    #    p10=89
    #    p11=97
    #    p12=101
    #    p13=109
    #    p14=113
    #    p15=137
    #    p16=149
    #    p17=157
    #    p18=173
    #    p19=181
    #    p20=193
    #    
    #    Let's calculate the product.
    
    primes_1_mod_4 = [5, 13, 17, 29, 37, 41, 53, 61, 73, 89, 97, 101, 109, 113, 137, 149, 157, 173, 181, 193, 197, 229, 233, 241, 257, 269, 277, 281, 293, 313, 317, 337, 349, 353, 373, 389, 397, 401, 409, 421, 433, 449, 457, 461, 509, 521, 541, 557, 569, 577, 593, 601, 613, 617, 641, 653, 661, 673, 677, 701, 709, 733, 757, 761, 769, 773, 797, 809, 821, 829, 853, 857, 877, 881, 929, 937, 941, 953, 977, 997]
    
    # Find max t such that product of first t primes <= 2.5 * 10^17
    M_limit = 10**18 // 4
    prod = 1
    t_max = 0
    M = 1
    for p in primes_1_mod_4:
        if prod * p > M_limit:
            break
        prod *= p
        M = prod
        t_max += 1
        
    # t_max is the number of primes we can use.
    # Score is 2^(t_max - 1).
    
    # We need to generate the pairs.
    # We need to find all representations of M as x^2 + y^2.
    # M is a product of t_max primes.
    # We can use a recursive algorithm to generate all (x,y).
    # Or, since t_max is likely around 15-20, the number of representations is 2^(t_max-1) ~ 10^5 to 10^6.
    # Generating them explicitly might be feasible if we do it efficiently.
    
    # However, we just need to RETURN the pairs.
    # We don't need to list ALL of them if we can't, but the score is the number of pairs in the list.
    # So we MUST list them.
    
    # Let's check t_max.
    # 5*13*17*29*37*41*53*61*73*89*97*101*109*113*137*149*157*173*181*193
    # Let's approximate.
    # log10(5)=0.7, 13=1.1, 17=1.2, 29=1.46, 37=1.57, 41=1.61, 53=1.72, 61=1.79, 73=1.86, 89=1.95
    # Sum first 10: ~14.4. 10^14.4.
    # 97=1.99, 101=2.0, 109=2.04, 113=2.05, 137=2.14, 149=2.17, 157=2.2, 173=2.24, 181=2.26, 193=2.29
    # Sum next 10: ~21.3. Total ~35.7. 10^35.7.
    # This is way bigger than 10^17.
    # So t_max is smaller.
    
    # Let's re-calculate.
    # M_limit = 2.5e17.
    # p1=5 (5)
    # p2=13 (65)
    # p3=17 (1105)
    # p4=29 (32045)
    # p5=37 (1185665)
    # p6=41 (48612265)
    # p7=53 (2576450045) ~ 2.5e9
    # p8=61 (157163452745) ~ 1.5e11
    # p9=73 (11472932050385) ~ 1.1e13
    # p10=89 (1021090952484265) ~ 1e15
    # p11=97 (99045822390973705) ~ 1e17
    # p12=101 (10003628061488344205) ~ 1e19 > 2.5e17
    
    # So t_max = 11.
    # Score = 2^(11-1) = 2^10 = 1024.
    
    # This is much better than 5.8.
    
    # Now, we need to generate the 1024 pairs.
    # We need to find all (x,y) such that x^2 + y^2 = M.
    # M = 5*13*17*29*37*41*53*61*73*89*97.
    
    # Algorithm to find all representations:
    # 1. Factor M (we know the factors).
    # 2. For each prime p = a^2 + b^2, we have a choice of factor (a+bi) or (a-bi).
    # 3. Multiply all chosen factors to get z = x + yi.
    # 4. Then x^2 + y^2 = M.
    # 5. We need to handle the Gaussian integers.
    
    # We need to find a, b for each prime p ≡ 1 mod 4 such that a^2 + b^2 = p.
    # We can precompute this or compute on the fly.
    
    def find_sum_of_squares_prime(p):
        # p is prime ≡ 1 mod 4
        # Find a, b such that a^2 + b^2 = p
        # Use Tonelli-Shanks or simple search since p is small
        # a^2 = (p - b^2). 
        # Or use the fact that -1 is a quadratic residue mod p.
        # Find x such that x^2 ≡ -1 mod p.
        # Then gcd(x + i, p) in Z[i] gives the factor.
        # Simpler: iterate a from 1 to isqrt(p)
        for a in range(1, isqrt(p) + 1):
            b2 = p - a*a
            b = isqrt(b2)
            if b*b == b2:
                return (a, b)
        return None

    # Precompute factors for the primes we use
    prime_factors = primes_1_mod_4[:t_max]
    gaussian_factors = []
    for p in prime_factors:
        a, b = find_sum_of_squares_prime(p)
        # We can choose (a, b) or (b, a) or (-a, b) etc.
        # The factor is (a + bi) or (a - bi).
        # We will store (a, b) and use both signs/choices in recursion.
        gaussian_factors.append((a, b))
        
    # Recursively generate all z = x + yi
    # z = prod (a_i + i b_i) or (a_i - i b_i)
    # There are 2^t combinations.
    # We only need x, y >= 0 and x != y.
    # Also, we must ensure a=2^k x^2 and b=2^k y^2 are powerful.
    # With k=2, this is always true.
    
    N = 4 * M  # k=2
    pairs = set()
    
    # Recursive generation
    def gen_reps(idx, x, y):
        if idx == len(gaussian_factors):
            # We have a representation x + yi
            # x^2 + y^2 = M
            # We need x, y >= 0
            # Since we multiply Gaussian integers, x and y can be negative.
            # We take abs(x), abs(y).
            ax = abs(x)
            ay = abs(y)
            if ax == 0 or ay == 0:
                return
            if ax == ay:
                # This would imply 2 ax^2 = M. M is odd product of primes, so impossible.
                pass
            # Generate pair
            a_val = 4 * ax * ax
            b_val = 4 * ay * ay
            if a_val > b_val:
                a_val, b_val = b_val, a_val
            pairs.add((a_val, b_val))
            return
            
        a, b = gaussian_factors[idx]
        # Option 1: multiply by (a + bi)
        # (x + yi)(a + bi) = (xa - yb) + i(xb + ya)
        x1 = x * a - y * b
        y1 = x * b + y * a
        gen_reps(idx + 1, x1, y1)
        
        # Option 2: multiply by (a - bi)
        # (x + yi)(a - bi) = (xa + yb) + i(ya - xb)
        x2 = x * a + y * b
        y2 = y * a - x * b
        gen_reps(idx + 1, x2, y2)
        
        if time.time() > deadline - 5:
            # Timeout guard
            pass # Can't easily break recursion, but t_max is small (11), so 2^11 calls is fine.
            
    gen_reps(0, 1, 0)
    
    # Convert set to list of lists
    final_pairs = [list(p) for p in pairs]
    
    # Check if we have any pairs
    if not final_pairs:
        # Fallback
        return {'N': 2, 'pairs': [[1, 1]]}
        
    return {'N': N, 'pairs': final_pairs}
