"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from math import gcd

def construct():
    start_time = time.time()
    deadline = start_time + 80  # Safety margin before 90s limit

    # Try to find minimal number of terms
    # First verify: no 2-term solution exists with denominators >= 5
    # 1/a + 1/b = 1 => (a-1)(b-1) = 1 => a=2, b=2 (both < 5, invalid)
    
    # Try 3-term: 1/a + 1/b + 1/c = 1, a < b < c, a >= 5
    # For each a >= 5: 1/b + 1/c = 1 - 1/a = (a-1)/a
    # (b+c)/(bc) = (a-1)/a => a(b+c) = (a-1)bc
    # (a-1)bc - ab - ac = 0 => (a-1)b(a-1)c - a(a-1)b - a(a-1)c = 0
    # Multiply by (a-1): (a-1)^2 bc - a(a-1)b - a(a-1)c = 0
    # (a(a-1)b - a(a-1))... let me use: (a-1)b(a-1)c - a(a-1)b - a(a-1)c = 0
    # Let me rearrange: (a-1)bc - ab - ac = 0
    # (a-1)bc - a(b+c) = 0
    # (a-1)bc - a(b+c) + a^2/(a-1) * ... 
    
    # Better: from 1/b + 1/c = (a-1)/a
    # a(b+c) = (a-1)bc
    # (a-1)bc - ab - ac = 0
    # (a-1)bc - a(b+c) = 0
    # Multiply both sides by (a-1):
    # (a-1)^2 bc - a(a-1)(b+c) = 0
    # (a(a-1)b - a^2 + a)(a(a-1)c - a^2 + a)... 
    
    # Simpler approach: iterate a, then for each a iterate b > a, compute c
    # 1/c = 1 - 1/a - 1/b = (ab - a - b)/(ab)
    # c = ab/(ab - a - b) = ab/((a-1)(b-1) - 1)
    
    found_3term = None
    max_a_3 = 100
    for a in range(5, max_a_3 + 1):
        if time.time() > deadline:
            break
        for b in range(a + 1, 10000):
            if time.time() > deadline:
                break
            num = a * b
            den = a * b - a - b
            if den <= 0:
                continue
            if num % den == 0:
                c = num // den
                if c > b and c >= 5 and c <= 10000000:
                    # Verify
                    if a * b * c == b * c + a * c + a * b:
                        found_3term = [a, b, c]
                        break
        if found_3term:
            break
    
    if found_3term:
        return found_3term

    # No 3-term found, try 4-term
    # 1/a + 1/b + 1/c + 1/d = 1
    # For small a, b, search c, d
    found_4term = None
    max_a_4 = 15
    max_b_4 = 100
    max_c_4 = 1000
    
    for a in range(5, max_a_4 + 1):
        if time.time() > deadline:
            break
        for b in range(a + 1, max_b_4 + 1):
            if time.time() > deadline:
                break
            rem_ab = 1 - 1/a - 1/b
            if rem_ab <= 0:
                continue
            # 1/c + 1/d = rem_ab = (ab - a - b)/(ab)
            # For each c > b: 1/d = rem_ab - 1/c
            # d = 1/(rem_ab - 1/c) = c/(c*rem_ab - 1)
            # c*rem_ab - 1 = c*(ab-a-b)/(ab) - 1 = (c*(ab-a-b) - ab)/(ab)
            # d = c*ab / (c*(ab-a-b) - ab)
            for c in range(b + 1, max_c_4 + 1):
                if time.time() > deadline:
                    break
                num_cd = c * a * b
                den_cd = c * (a * b - a - b) - a * b
                if den_cd <= 0:
                    continue
                if num_cd % den_cd == 0:
                    d = num_cd // den_cd
                    if d > c and d >= 5 and d <= 10000000:
                        # Verify
                        if (a * b * c * d == b * c * d + a * c * d + a * b * d + a * b * c):
                            found_4term = [a, b, c, d]
                            break
            if found_4term:
                break
        if found_4term:
            break
    
    if found_4term:
        return found_4term

    # Fallback: use the DP approach from the seed but with a better target
    # Actually, let's just return a known valid 5-term or use DP
    # Let's try a direct 5-term search for small values
    found_5term = None
    for a in range(5, 10):
        if time.time() > deadline:
            break
        for b in range(a + 1, 20):
            if time.time() > deadline:
                break
            for c in range(b + 1, 50):
                if time.time() > deadline:
                    break
                rem_abc = 1 - 1/a - 1/b - 1/c
                if rem_abc <= 0:
                    continue
                for d in range(c + 1, 200):
                    if time.time() > deadline:
                        break
                    rem_abcd = rem_abc - 1/d
                    if rem_abcd <= 0:
                        continue
                    # 1/e = rem_abcd, so e = 1/rem_abcd
                    # e = a*b*c*d / (a*b*c*d * rem_abcd)
                    # rem_abcd = 1 - 1/a - 1/b - 1/c - 1/d
                    # Let's compute directly
                    num_e = a * b * c * d
                    den_e = num_e - b * c * d - a * c * d - a * b * d - a * b * c
                    if den_e <= 0:
                        continue
                    if num_e % den_e == 0:
                        e = num_e // den_e
                        if e > d and e >= 5 and e <= 10000000:
                            # Verify all five sum to 1
                            if (a * b * c * d * e == 
                                b * c * d * e + a * c * d * e + a * b * d * e + a * b * c * e + a * b * c * d):
                                found_5term = [a, b, c, d, e]
                                break
                if found_5term:
                    break
            if found_5term:
                break
        if found_5term:
            break
    
    if found_5term:
        return found_5term

    # Ultimate fallback: DP approach with common denominator
    target = 2520
    reached = {0: []}
    for d in range(5, target + 1):
        if time.time() > deadline:
            break
        if target % d:
            continue
        w = target // d
        for total, values in list(reached.items()):
            if total + w <= target and total + w not in reached:
                reached[total + w] = values + [d]
        if target in reached:
            return reached[target]
    
    raise RuntimeError("no representation found")
# EVOLVE-BLOCK-END
