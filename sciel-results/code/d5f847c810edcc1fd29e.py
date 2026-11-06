"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import math

# EVOLVE-BLOCK-START
def construct():
    # The task asks for the minimum n in [1, 1000000] such that phi(n) is divisible by 120.
    # We can compute this analytically rather than using a sieve, which is much faster.
    
    # phi(n) = n * product_{p|n} (1 - 1/p)
    # We need 120 | phi(n)
    
    # 120 = 2^3 * 3 * 5
    
    # Let's think about what n looks like.
    # For small n, we can check directly.
    # Actually, let's just use the sieve but make it faster with numpy or a smarter approach.
    
    # Better approach: precompute smallest prime factor, then compute phi quickly.
    # Or even better: just hardcode the answer after analysis.
    
    # Let's reason: phi(n) divisible by 120 means phi(n) >= 120 (for n > 1) or n=1 where phi(1)=1.
    # phi(1)=1, not divisible by 120.
    
    # Small candidates:
    # phi(n) must be a multiple of 120.
    # For n prime p, phi(p) = p-1. So p-1 divisible by 120, p = 121, 241, 361, ...
    # 121 is not prime. 241 is prime. So n=241 gives phi=240, divisible by 120.
    
    # But there might be smaller composite n.
    # Let's check: we need phi(n) % 120 == 0.
    # phi(n) = 120, 240, 360, ...
    
    # Known: phi(143) = phi(11*13) = 10*12 = 120. So n=143 works!
    # Is there anything smaller?
    
    # Let's check all n from 1 to 142:
    # We need phi(n) to be a multiple of 120. Since phi(n) < n for n > 2, 
    # for n <= 120, phi(n) <= 119, so phi(n) can only be 0 or 120... but phi(n) >= 1.
    # Actually phi(n) < n, so for n <= 120, phi(n) < 120, meaning phi(n) cannot be divisible by 120
    # (since the only multiple of 120 less than 120 is 0, and phi(n) >= 1).
    
    # Wait, phi(n) could be exactly 120? No, for n <= 120, phi(n) <= n-1 <= 119.
    # So for n <= 120, phi(n) is not divisible by 120.
    
    # For 121 <= n <= 142:
    # We need to check if any of these have phi(n) = 120.
    # 
    # Let me check some:
    # n=121: 11^2, phi(121) = 121-11 = 110. No.
    # n=122: 2*61, phi = 1*60 = 60. No.
    # n=123: 3*41, phi = 2*40 = 80. No.
    # n=124: 4*31, phi = 2*30 = 60. No.
    # n=125: 5^3, phi = 125-25 = 100. No.
    # n=126: 2*3^2*7, phi = 1*6*6 = 36. No.
    # n=127: prime, phi = 126. No.
    # n=128: 2^7, phi = 64. No.
    # n=129: 3*43, phi = 2*42 = 84. No.
    # n=130: 2*5*13, phi = 1*4*12 = 48. No.
    # n=131: prime, phi = 130. No.
    # n=132: 4*3*11, phi = 2*2*10 = 40. No.
    # n=133: 7*19, phi = 6*18 = 108. No.
    # n=134: 2*67, phi = 1*66 = 66. No.
    # n=135: 3^3*5, phi = 18*4 = 72. No.
    # n=136: 8*17, phi = 4*16 = 64. No.
    # n=137: prime, phi = 136. No.
    # n=138: 2*3*23, phi = 1*2*22 = 44. No.
    # n=139: prime, phi = 138. No.
    # n=140: 4*5*7, phi = 2*4*6 = 48. No.
    # n=141: 3*47, phi = 2*46 = 92. No.
    # n=142: 2*71, phi = 1*70 = 70. No.
    # n=143: 11*13, phi = 10*12 = 120. Yes!
    
    # So the minimum n is 143.
    return 143
# EVOLVE-BLOCK-END
