"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import math
import time
from functools import reduce

def _squarefree_kernel(n):
    """Return the squarefree kernel of n (product of primes with odd exponent)."""
    result = 1
    d = 2
    while d * d <= n:
        cnt = 0
        while n % d == 0:
            n //= d
            cnt += 1
        if cnt % 2 == 1:
            result *= d
        d += 1
    if n > 1:
        result *= n
    return result

def _is_square(n):
    if n < 0:
        return False
    r = math.isqrt(n)
    return r * r == n

def _verify(result, n=210, max_val=10210):
    """Verify the solution is valid."""
    if not isinstance(result, (list, tuple)):
        return False
    if len(result) < 1 or len(result) > 100:
        return False
    seen = set()
    for x in result:
        if not isinstance(x, int) or isinstance(x, bool):
            return False
        if x <= 210 or x > 10210:
            return False
        if x in seen:
            return False
        seen.add(x)
    prod = n
    for x in result:
        prod *= x
    return _is_square(prod)

def construct():
    deadline = time.time() + 80  # safety margin before 90s kill
    
    # Target: sf(210) = 210 (since 210 is squarefree)
    # We need subset of {211..M} with XOR of sf-kernels = 210
    
    # Represent sf-kernels as bit vectors over relevant primes
    # Primes up to 10210
    # sf(210) involves primes 2,3,5,7
    
    # For each number n in range, compute sf(n) as a set of primes (or frozenset)
    # Then do subset XOR search
    
    # Use Gaussian elimination approach:
    # Each number gives a vector in GF(2)^k where k = number of relevant primes
    # We need to find subset with XOR = target vector
    
    # First, determine which primes can appear in sf(n) for n in [211, 10210]
    # sf(n) is always squarefree, so its prime factors are primes <= n
    # For our purposes, we only care about primes that can appear
    
    # Let's collect all primes up to 10210
    def sieve(limit):
        is_p = [True] * (limit + 1)
        is_p[0] = is_p[1] = False
        for i in range(2, int(limit**0.5) + 1):
            if is_p[i]:
                for j in range(i*i, limit+1, i):
                    is_p[j] = False
        return [i for i in range(2, limit+1) if is_p[i]]
    
    primes = sieve(10210)
    prime_to_idx = {p: i for i, p in enumerate(primes)}
    num_primes = len(primes)
    
    # Target vector: sf(210) = 210 = 2*3*5*7
    target = frozenset({2, 3, 5, 7})
    
    # For each candidate M from 211 upward, check if we can form target from {211..M}
    # Use incremental Gaussian elimination
    
    best_result = None
    
    for M in range(211, 10211):
        if time.time() > deadline:
            break
        
        # We need to find subset of {211, ..., M} with XOR of sf-kernels = target
        # Use BFS/DFS or Gaussian elimination
        
        # For small ranges, we can try all subsets (but that's exponential)
        # Better: use the fact that we can pick numbers whose sf-kernel "cancels out"
        
        # Actually, let's use a different approach:
        # sf(a) * sf(b) = sf(a*b) if we consider multiplicative structure
        # No, that's not right. sf(a*b) != sf(a)*sf(b) in general.
        
        # Let me think again. We need product of selected numbers to have sf = 210.
        # Equivalent: for each prime p, the total exponent of p in the product is odd iff p in {2,3,5,7}.
        
        # Represent each number by its parity vector (exponent parity for each prime)
        # We need subset XOR = target parity vector
        
        # Use Gaussian elimination to find if target is in the span, and if so, find a small subset
        
        # For efficiency, let's try a greedy + local search approach:
        # 1. Try to find pairs (a,b) where sf(a) == sf(b) * 210 ... no
        # Actually: if sf(a) == sf(b), then a*b has sf=1, so a*b is a square times 1... 
        # Wait: sf(a*b) when sf(a)==sf(b): the odd primes in a and b cancel, so sf(a*b)=1
        # That means a*b is a perfect square!
        
        # So if I find a pair (a,b) with sf(a)==sf(b), then a*b is a square.
        # Then 210 * a * b has sf = sf(210) = 210, which is NOT a square.
        # That's wrong direction.
        
        # I need sf(product) = 210.
        # If I take one number n with sf(n) = 210, then sf(product) = 210. Done!
        # So: find smallest n > 210 with sf(n) = 210.
        
        # sf(n) = 210 means n = 210 * k^2 for some integer k, OR more generally
        # n has prime factorization where primes 2,3,5,7 have odd exponent and all other primes have even exponent.
        # So n = 2^a * 3^b * 5^c * 7^d * m^2 where a,b,c,d are odd positive integers, m is any integer.
        # Minimum such n > 210: try a=b=c=d=1, m=1: n=210 (not > 210)
        # m=2: n = 210*4 = 840. sf(840) = sf(2*3*5*7*2^2) = sf(2^3*3*5*7) = 2*3*5*7 = 210. Yes!
        
        # So 840 works! max = 840, value = 840-210 = 630.
        
        # Can we do better with multiple numbers?
        # We need subset with XOR = target. If we use 2 numbers a,b: sf(a)*sf(b) should give sf=210 in product sense...
        # Actually the XOR of parity vectors must equal target.
        # sf(a) XOR sf(b) = 210 means the symmetric difference of prime sets is {2,3,5,7}.
        
        # Let's just find the minimum M where ANY subset works.
        
        # Quick check: is there a single n in (210, 840) with sf(n)=210?
        # n = 2^a * 3^b * 5^c * 7^d * m^2, all exponents odd for 2,3,5,7
        # Min with a=b=c=d=1: 210*m^2. m=1 gives 210 (excluded). m=2 gives 840.
        # What about different odd exponents? a=3,b=1,c=1,d=1: 8*3*5*7=840 again.
        # a=1,b=3,c=1,d=1: 2*27*5*7=1890. Worse.
        # So 840 is the minimum single-element solution.
        
        # With 2 elements: need sf(a) XOR sf(b) = {2,3,5,7}
        # This means sf(a) and sf(b) differ exactly in primes {2,3,5,7}.
        # So sf(a) = S ∪ T, sf(b) = S ∪ U where T XOR U = {2,3,5,7} and T∩U=∅... 
        # Actually XOR of sets: sf(a) Δ sf(b) = {2,3,5,7}
        # So sf(a) and sf(b) agree on all primes except possibly {2,3,5,7}.
        # Let sf(a) = A, sf(b) = B. A Δ B = {2,3,5,7}.
        # So B = A Δ {2,3,5,7}.
        
        # We want to minimize max(a,b) where a,b > 210, a≠b, sf(a)=A, sf(b)=AΔ{2,3,5,7} for some squarefree A.
        
        # For a given squarefree A, min number with sf=A is A itself (if A>210) or A*k^2.
        # We need both A and AΔ{2,3,5,7} to be representable as numbers > 210.
        
        # Let's search: for squarefree A with A>210, compute B=AΔ{2,3,5,7}, check if both A and B can be small numbers >210.
        # min_num_with_sf(A) = A * k^2 where k is smallest such that A*k^2 > 210.
        
        # Try A containing some of {2,3,5,7} and some not.
        # If A contains exactly one of {2,3,5,7}, say A={2}∪X where X is disjoint from {2,3,5,7}:
        #   B = {3,5,7}∪X
        #   min a = 2*min(k^2 s.t. 2k^2>210) = 2*11^2=242? 2*121=242>210. sf(242)=sf(2*121)=sf(2*11^2)=2. Yes.
        #   min b = 3*5*7*min(k^2 s.t. 105k^2>210) = 105*1^2=105 (not>210), 105*4=420. sf(420)=sf(2^2*3*5*7)=3*5*7=105. Yes.
        #   max(242,420)=420. Value=420-210=210.
        
        # Can we do better? Try A={2,3}∪X, B={5,7}∪X:
        #   min a = 6*min(k^2 s.t. 6k^2>210)=6*6^2=216>210. sf(216)=sf(2^3*3^3)=2*3=6. Yes.
        #   min b = 35*min(k^2 s.t. 35k^2>210)=35*3^2=315>210. sf(315)=sf(3^2*5*7)=5*7=35. Yes.
        #   max(216,315)=315. Value=315-210=105.
        
        # Try A={2,5}∪X, B={3,7}∪X:
        #   min a = 10*min(k^2 s.t. 10k^2>210)=10*5^2=250. sf(250)=sf(2*5^3)=2*5=10.
        #   min b = 21*min(k^2 s.t. 21k^2>210)=21*4^2=336. sf(336)=sf(2^4*3*7)=3*7=21.
        #   max=336. Value=126.
        
        # Try A={2,7}∪X, B={3,5}∪X:
        #   min a = 14*min(k^2 s.t. 14k^2>210)=14*4^2=224. sf(224)=sf(2^5*7)=2*7=14.
        #   min b = 15*min(k^2 s.t. 15k^2>210)=15*4^2=240. sf(240)=sf(2^4*3*5)=3*5=15.
        #   max=240. Value=30!
        
        # Verify: 224=2^5*7, sf=14. 240=2^4*3*5, sf=15. 
        # 14 XOR 15: 14=2*7, 15=3*5. Symmetric diff = {2,3,5,7}. Yes!
        # 210*224*240 = 210*53760 = 11289600. sqrt = 3360? 3360^2=11289600. Yes!
        
        # Can we do even better?
        # Try A={3,5}∪X, B={2,7}∪X: same as above, just swap. max=240.
        
        # Try A={3,7}∪X, B={2,5}∪X: max=336.
        # Try A={5,7}∪X, B={2,3}∪X: max=315.
        
        # What about 3-element subsets?
        # Need sf(a) XOR sf(b) XOR sf(c) = {2,3,5,7}
        # This gives more freedom.
        
        # Let's try to find if max < 240 is possible.
        # We need all numbers > 210 and <= M where M < 240, i.e., M in {211,...,239}.
        # Numbers: 211,212,...,239.
        # 211 is prime, sf=211
        # 212=4*53, sf=53
        # 213=3*71, sf=213
        # 214=2*107, sf=214
        # 215=5*43, sf=215
        # 216=8*27=2^3*3^3, sf=6
        # 217=7*31, sf=217
        # 218=2*109, sf=218
        # 219=3*73, sf=219
        # 220=4*5*11, sf=55
        # 221=13*17, sf=221
        # 222=2*3*37, sf=222
        # 223 prime, sf=223
        # 224=32*7=2^5*7, sf=14
        # 225=9*25=3^2*5^2, sf=1
        # 226=2*113, sf=226
        # 227 prime, sf=227
        # 228=4*57=2^2*3*19, sf=57
        # 229 prime, sf=229
        # 230=2*5*23, sf=230
        # 231=3*7*11, sf=231
        # 232=8*29=2^3*29, sf=58
        # 233 prime, sf=233
        # 234=2*3^2*13, sf=26
        # 235=5*47, sf=235
        # 236=4*59, sf=59
        # 237=9*26=3^2*29... wait 237=3*79, sf=237
        # 238=2*7*17, sf=238
        # 239 prime, sf=239
        
        # I need subset with XOR = {2,3,5,7}. Looking at the sf values:
        # 6={2,3}, 14={2,7}, 55={5,11}, 57={3,19}, 58={2,29}, 26={2,13}
        # 6 XOR 14 = {3,7} (since 2 cancels). Then need XOR with {5}... but no element has sf={5}.
        # 6 XOR 55 = {2,3,5,11}. Not target.
        # This is getting complex. Let me just check if [224, 240] is valid and return it.
        
        result = [224, 240]
        if _verify(result):
            return result
        
        # Fallback: single element 840
        result = [840]
        if _verify(result):
            return result
        
        # Further fallback
        return [4 * 210]
# EVOLVE-BLOCK-END
