"""Return the smallest n in [1,1000000] with phi(n) divisible by 120."""

def construct():
    # 120 = 2^3 * 3 * 5
    # The smallest n is 11*13 = 143, since phi(143) = phi(11)*phi(13) = 10*12 = 120.
    # Verification that no smaller n works:
    #   n=1..142: none have phi(n) divisible by 120.
    #   The next candidates are 5*31=155 (phi=4*30=120), 3*7*11=231 (phi=2*6*10=120), etc.
    #   143 is strictly smaller than all of these.
    return 143
