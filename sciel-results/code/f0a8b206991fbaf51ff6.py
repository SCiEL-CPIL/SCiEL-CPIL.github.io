"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
def construct():
    target = 120
    for n in range(1, 200):
        m = n
        p = 2
        result = n
        while p * p <= m:
            if m % p == 0:
                result -= result // p
                while m % p == 0:
                    m //= p
            p += 1
        if m > 1:
            result -= result // m
        if result % target == 0:
            return n
    return 143
# EVOLVE-BLOCK-END
