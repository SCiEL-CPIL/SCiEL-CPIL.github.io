from fractions import Fraction

def construct():
    # Search for 8-term solution
    # First 6 terms are small, last 2 are solved via Diophantine
    candidates = []
    # Generate combinations of 6 distinct denominators from 5 to 20
    # This is 16 choose 6 = 8008 combinations. Fast.
    import itertools
    
    for combo in itertools.combinations(range(5, 21), 6):
        s = sum(Fraction(1, d) for d in combo)
        rem = 1 - s
        if rem <= 0:
            continue
        num, den = rem.numerator, rem.denominator
        # Solve 1/x + 1/y = num/den
        # (num*x - den)(num*y - den) = den^2
        target = den * den
        found = False
        for x in range(1, int(target**0.5) + 1):
            if target % x == 0:
                y = target // x
                if (x + den) % num == 0:
                    d1 = (x + den) // num
                    if (y + den) % num == 0:
                        d2 = (y + den) // num
                        if d1 > combo[-1] and d2 > combo[-1] and d1 != d2:
                            # Check distinctness
                            if d1 not in combo and d2 not in combo:
                                res = list(combo) + [d1, d2]
                                # Verify
                                if sum(Fraction(1, d) for d in res) == 1:
                                    return res
                                found = True
                                break
        # If found is True but verification failed, continue to next combo
        # If not found, continue to next combo
        
    # Fallback: return a valid solution with more terms
    # Using a known identity or generating one
    # 1 = 1/5 + 1/6 + 1/7 + 1/8 + 1/9 + 1/10 + 1/11 + 1/12 + 1/13 + 1/14 + 1/15 + 1/16 + 1/17 + 1/18 + 1/19 + 1/20 + 1/21 + 1/22 + 1/23 + 1/24 + 1/25 + 1/26 + 1/27 + 1/28 + 1/29 + 1/30 + 1/31 + 1/32 + 1/33 + 1/34 + 1/35 + 1/36 + 1/37 + 1/38 + 1/39 + 1/40 + 1/41 + 1/42 + 1/43 + 1/44 + 1/45 + 1/46 + 1/47 + 1/48 + 1/49 + 1/50
    # This is not exact 1.
    
    # If no 8-term solution found, try 9 terms?
    # For now, raise error if not found, but it should be found.
    raise RuntimeError("No 8-term solution found")

def _search(terms_left, start_denom, remaining_sum):
    return None
