import time

_start_time = None
_DEADLINE = 330 * 0.95  # 313.5s

def priority(vector, n):
    """
    Priority function for greedy cap set construction in {0,1,2}^n.
    
    Structural change: Use a multiplicative hash on the base-3 representation
    of the vector to create a pseudo-random ordering that breaks lexicographic
    clustering. This shifts from statistical features (symbol counts, balance)
    to a geometric hash that directly controls the greedy evaluation order.
    """
    global _start_time
    if _start_time is None:
        _start_time = time.time()
    elif time.time() - _start_time > _DEADLINE:
        return 0.0

    # Convert vector to a base-3 number using modular arithmetic
    MOD = (1 << 61) - 1  # Mersenne prime for fast modulo
    num = 0
    for i in range(n - 1, -1, -1):
        num = (num * 3 + vector[i]) % MOD
    
    # Multiplicative hash (Knuth's constant) for pseudo-random spread
    h = (num * 2654435761) % MOD
    
    return float(h)
