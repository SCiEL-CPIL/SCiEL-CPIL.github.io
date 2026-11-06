"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
def construct():
    """
    Returns the smallest integer n in [1, 1000000] such that phi(n) is divisible by 120.
    
    The minimum is 143 = 11 * 13.
    phi(143) = phi(11) * phi(13) = 10 * 12 = 120, which is divisible by 120.
    
    Proof of minimality:
    - 120 = 2^3 * 3 * 5. The factor 5 in phi(n) requires:
      (a) a prime p ≡ 1 (mod 5) dividing n (smallest: 11, 31, 41, ...), or
      (b) 25 | n (since phi(25) = 20 contains factor 5).
    - Case (a) with p=11: n = 11k, gcd(k,11)=1. phi(n) = 10*phi(k).
      Need 10*phi(k) ≡ 0 (mod 120) => phi(k) ≡ 0 (mod 12).
      Smallest k coprime to 11 with phi(k) divisible by 12 is k=13 (phi(13)=12).
      So n = 11*13 = 143.
    - Case (a) with p=31: n = 31k, phi(n) = 30*phi(k).
      Need 30*phi(k) ≡ 0 (mod 120) => phi(k) ≡ 0 (mod 4).
      Smallest k coprime to 31 with phi(k) divisible by 4 is k=5 (phi(5)=4).
      So n = 31*5 = 155 > 143.
    - Case (b): n = 25k, gcd(k,25)=1. phi(n) = 20*phi(k).
      Need 20*phi(k) ≡ 0 (mod 120) => phi(k) ≡ 0 (mod 6).
      Smallest k coprime to 25 with phi(k) divisible by 6 is k=7 (phi(7)=6).
      So n = 25*7 = 175 > 143.
    
    Therefore, 143 is the minimum.
    """
    return 143
# EVOLVE-BLOCK-END
