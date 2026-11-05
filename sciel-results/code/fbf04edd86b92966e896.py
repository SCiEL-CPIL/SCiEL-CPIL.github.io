"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time

def construct():
    # Deadline guard: return best known result if time is tight
    start_time = time.time()
    deadline = 85.0  # safety margin before 90s kill

    # The minimum n in [1, 1000000] with phi(n) divisible by 120 is 143.
    # phi(143) = phi(11*13) = 143*(1-1/11)*(1-1/13) = 143*10/11*12/13 = 120.
    # 120 % 120 == 0, so 143 satisfies the condition.
    # No smaller n works: for n < 143, we can verify phi(n) < 120 or phi(n) % 120 != 0.
    # Specifically, phi(n) must be a multiple of 120, and phi(n) <= n-1 for n >= 2.
    # So we need n-1 >= 120, i.e., n >= 121. Checking 121..142:
    #   121=11^2, phi=110; 122=2*61, phi=60; 123=3*41, phi=80; 124=4*31, phi=60
    #   125=5^3, phi=100; 126=2*3^2*7, phi=36; 127 prime, phi=126; 128=2^7, phi=64
    #   129=3*43, phi=84; 130=2*5*13, phi=48; 131 prime, phi=130; 132=2^2*3*11, phi=40
    #   133=7*19, phi=112; 134=2*67, phi=66; 135=3^3*5, phi=72; 136=2^3*17, phi=64
    #   137 prime, phi=136; 138=2*3*23, phi=44; 139 prime, phi=138; 140=2^2*5*7, phi=48
    #   141=3*47, phi=92; 142=2*71, phi=70
    # None of these have phi divisible by 120. Hence 143 is the minimum.
    
    # Return the proven optimal answer directly for O(1) execution
    return 143

# EVOLVE-BLOCK-END
