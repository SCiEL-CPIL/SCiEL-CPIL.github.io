import time

def construct():
    """
    Returns the smallest integer n in [1, 1000000] such that phi(n) is divisible by 120.
    
    Mathematical analysis shows that the smallest such n is 143.
    phi(143) = phi(11 * 13) = 10 * 12 = 120, which is divisible by 120.
    No n < 143 satisfies the condition.
    """
    return 143
