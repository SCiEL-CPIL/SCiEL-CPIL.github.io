def construct():
    """
    Returns the smallest integer n in [1, 1000000] such that phi(n) is divisible by 120.
    
    Mathematical Analysis:
    We seek min n such that 120 | phi(n).
    Since phi(n) < n for n > 1, if n < 120, phi(n) < 120, so phi(n) cannot be a multiple of 120 (as phi(n) > 0).
    Thus n >= 120.
    For n in [120, 142], we check if phi(n) = 120 (next multiple is 240, but phi(n) < n < 240 for n < 240? No, phi(n) can be up to n-1. But for n < 143, phi(n) < 143. So only phi(n)=120 is possible).
    Solutions to phi(n) = 120:
    Prime factors p of n must satisfy p-1 | 120.
    Possible primes: 2, 3, 5, 7, 11, 13, 31, 41, 61.
    Checking combinations for n < 143:
    - If 13 | n, n=13k. phi(n)=12phi(k)=120 => phi(k)=10 => k in {11, 22}. Min n = 13*11 = 143.
    - If 11 | n (and 13 not), n=11k. phi(n)=10phi(k)=120 => phi(k)=12 => k in {13, 21, 26, 28, 36, 42}. Min n = 11*13 = 143 (but 13 is excluded in this case? No, k=13 implies n=143 which has 13. If k must be coprime to 11, 13 is coprime. So n=143 is a solution here too).
    - If 7 | n (and 11, 13 not), n=7k. phi(n)=6phi(k)=120 => phi(k)=20 => k in {25, 33, ...}. Min n = 7*25 = 175 > 143.
    - If 5 | n (and 7, 11, 13 not), n=5k. phi(n)=4phi(k)=120 => phi(k)=30 => k in {31, 37, ...}. Min n = 5*31 = 155 > 143.
    - If no primes > 3, n=2^a 3^b. phi(n) is of form 2^x 3^y. 120 = 2^3 * 3 * 5. Contains factor 5. Impossible.
    Thus, 143 is the global minimum.
    """
    return 143
