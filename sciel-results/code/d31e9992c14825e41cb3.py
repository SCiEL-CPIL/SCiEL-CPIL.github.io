# EVOLVE-BLOCK-START
import time
from math import gcd

def construct():
    deadline = time.time() + 55  # safety margin before 90s limit
    
    # Known solution: 5/31 = 1/8 + 1/31 + 1/248, max = 248
    best = [8, 31, 248]
    best_max = 248
    
    def check_and_update(dens):
        nonlocal best, best_max
        mx = max(dens)
        if mx < best_max:
            best_max = mx
            best = list(dens)
    
    def frac_sub(p1, q1, p2, q2):
        p = p1 * q2 - p2 * q1
        q = q1 * q2
        if p == 0:
            return (0, 1)
        g = gcd(abs(p), q)
        return (p // g, q // g)
    
    # Try 3-term solutions
    for a in range(7, min(19, best_max)):
        if time.time() > deadline:
            break
        p_rem = 5 * a - 31
        q_rem = 31 * a
        if p_rem <= 0:
            continue
        g = gcd(p_rem, q_rem)
        p_rem //= g
        q_rem //= g
        
        b_lo = a + 1
        b_hi = min(best_max - 1, (2 * q_rem) // p_rem)
        for b in range(b_lo, b_hi + 1):
            p2 = p_rem * b - q_rem
            if p2 <= 0:
                continue
            q2 = q_rem * b
            if q2 % p2 != 0:
                continue
            c = q2 // p2
            if c > b and c < best_max:
                check_and_update([a, b, c])
    
    # Try 4-term solutions
    for a in range(7, min(25, best_max)):
        if time.time() > deadline:
            break
        p_rem = 5 * a - 31
        q_rem = 31 * a
        if p_rem <= 0:
            continue
        g = gcd(p_rem, q_rem)
        p_rem //= g
        q_rem //= g
        
        b_lo = a + 1
        b_hi = min(best_max - 1, (3 * q_rem) // p_rem)
        for b in range(b_lo, b_hi + 1):
            if time.time() > deadline:
                break
            p2 = p_rem * b - q_rem
            if p2 <= 0:
                continue
            q2 = q_rem * b
            g2 = gcd(p2, q2)
            p2 //= g2
            q2 //= g2
            
            c_lo = b + 1
            c_hi = min(best_max - 1, (2 * q2) // p2)
            for c in range(c_lo, c_hi + 1):
                if time.time() > deadline:
                    break
                p3 = p2 * c - q2
                if p3 <= 0:
                    continue
                q3 = q2 * c
                if q3 % p3 != 0:
                    continue
                d = q3 // p3
                if d > c and d < best_max:
                    check_and_update([a, b, c, d])
    
    # Try 5-term solutions
    for a in range(7, min(31, best_max)):
        if time.time() > deadline:
            break
        p_rem = 5 * a - 31
        q_rem = 31 * a
        if p_rem <= 0:
            continue
        g = gcd(p_rem, q_rem)
        p_rem //= g
        q_rem //= g
        
        b_lo = a + 1
        b_hi = min(best_max - 1, (4 * q_rem) // p_rem)
        for b in range(b_lo, b_hi + 1):
            if time.time() > deadline:
                break
            p2 = p_rem * b - q_rem
            if p2 <= 0:
                continue
            q2 = q_rem * b
            g2 = gcd(p2, q2)
            p2 //= g2
            q2 //= g2
            
            c_lo = b + 1
            c_hi = min(best_max - 1, (3 * q2) // p2)
            for c in range(c_lo, c_hi + 1):
                if time.time() > deadline:
                    break
                p3 = p2 * c - q2
                if p3 <= 0:
                    continue
                q3 = q2 * c
                g3 = gcd(p3, q3)
                p3 //= g3
                q3 //= g3
                
                d_lo = c + 1
                d_hi = min(best_max - 1, (2 * q3) // p3)
                for d in range(d_lo, d_hi + 1):
                    if time.time() > deadline:
                        break
                    p4 = p3 * d - q3
                    if p4 <= 0:
                        continue
                    q4 = q3 * d
                    if q4 % p4 != 0:
                        continue
                    e = q4 // p4
                    if e > d and e < best_max:
                        check_and_update([a, b, c, d, e])
    
    return best
# EVOLVE-BLOCK-END
