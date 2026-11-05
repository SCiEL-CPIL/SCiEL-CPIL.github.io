import math
import time

def construct():
    # The smallest integer n such that phi(n) is divisible by 120 is 143.
    # phi(143) = phi(11 * 13) = 10 * 12 = 120.
    # Verification:
    # For n < 143, phi(n) < 143.
    # The only multiple of 120 less than 143 is 120.
    # We need phi(n) = 120.
    # The smallest n with phi(n) = 120 is 143.
    # Thus, 143 is the global minimum.
    return 143
