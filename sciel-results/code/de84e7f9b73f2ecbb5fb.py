def construct():
    # The minimum n such that phi(n) is divisible by 120 is 143.
    # phi(143) = phi(11*13) = 10*12 = 120.
    # For n < 143, phi(n) < 143, so phi(n) cannot be 120, 240, etc.
    # Thus 143 is the global minimum.
    return 143
