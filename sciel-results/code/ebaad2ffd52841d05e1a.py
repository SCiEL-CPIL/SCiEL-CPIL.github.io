from fractions import Fraction
import time

def construct():
    start_time = time.time()
    deadline = start_time + 50
    
    target = Fraction(5, 31)
    best_max = float('inf')
    best_sol = None
    
    # Meet-in-the-middle for 3-term solutions
    # Precompute all pairs (a, b) with a < b, a, b <= B
    # Store sum 1/a + 1/b in a dict keyed by the Fraction
    # Then for each c > b, check if 5/31 - 1/c exists in the dict
    
    B = 2000  # Upper bound for first two denominators
    
    # Precompute pairs
    pair_sums = {}
    for a in range(2, B + 1):
        if time.time() > deadline:
            break
        for b in range(a + 1, B + 1):
            s = Fraction(1, a) + Fraction(1, b)
            if s < target:
                # Store: for this sum, record (a, b)
                # We want to find c such that s + 1/c = target
                # i.e., 1/c = target - s
                # c = 1 / (target - s)
                rem = target - s
                if rem > 0:
                    c = 1 / rem
                    if c.denominator == 1:
                        c_int = c.numerator
                        if c_int > b and c_int <= 10000000:
                            max_den = max(a, b, c_int)
                            if max_den < best_max:
                                best_max = max_den
                                best_sol = [a, b, c_int]
    
    # Also try 4-term solutions with small denominators
    # Use a smarter approach: iterate a, b, c and solve for d
    for a in range(2, 50):
        if time.time() > deadline:
            break
        Ra = target - Fraction(1, a)
        if Ra <= 0:
            continue
        b_min = max(a + 1, int(1 / Ra) + 1)
        b_max = min(int(3 / Ra), 500)
        for b in range(b_min, b_max + 1):
            if time.time() > deadline:
                break
            Rab = Ra - Fraction(1, b)
            if Rab <= 0:
                continue
            c_min = max(b + 1, int(1 / Rab) + 1)
            c_max = min(int(2 / Rab), 1000)
            for c in range(c_min, c_max + 1):
                if time.time() > deadline:
                    break
                Rabc = Rab - Fraction(1, c)
                if Rabc <= 0:
                    continue
                d = 1 / Rabc
                if d.denominator == 1:
                    d_int = d.numerator
                    if d_int > c and d_int <= 10000000:
                        max_den = max(a, b, c, d_int)
                        if max_den < best_max:
                            best_max = max_den
                            best_sol = [a, b, c, d_int]
    
    if best_sol is None:
        return [8, 31, 248]
    return best_sol
