"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import math

def construct():
    best = [840]
    best_val = 630
    
    for k in range(1, 200):
        target = 210 * k * k
        if target > 10210 * 10210:
            break
        sqrt_t = math.isqrt(target)
        a_start = min(sqrt_t, 10210)
        for a in range(a_start, 210, -1):
            if target % a == 0:
                b = target // a
                if b > 210 and b <= 10210 and b != a:
                    val = max(a, b) - 210
                    if val < best_val:
                        best_val = val
                        best = [a, b]
                    if b - 210 >= best_val:
                        break
    
    return list(best)
# EVOLVE-BLOCK-END
