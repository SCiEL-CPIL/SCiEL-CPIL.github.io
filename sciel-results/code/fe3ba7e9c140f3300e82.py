import math

def construct():
    answers = []
    for n in range(3, 257):
        found = None
        # Try a few x values
        for x in [n+1, 2*n, 3*n, 4*n, 5*n, n+2, n+3, 2*n+1, 2*n+2, 3*n+1]:
            a = 4*x - n
            if a <= 0:
                continue
            b = n * x
            b_sq = b * b
            # Find divisors of b_sq
            # We only need to check u such that u+b is div by a
            # This is a modular arithmetic constraint
            # But iterating divisors is simpler if count is low
            # However, finding divisors takes O(b)
            # Let's try to find u directly?
            # u must be a divisor of b^2.
            # u = k * a - b.
            # So k*a - b must divide b^2.
            # k*a - b | b^2
            # Let d = k*a - b.
            # d | b^2 => d | b^2 mod d? No.
            # d | b^2 and d = k*a - b.
            # b^2 = q * (k*a - b)
            # b^2 + q*b = q*k*a
            # b(b+q) = q*k*a
            # This is a Diophantine equation.
            
            # Simpler: Just iterate divisors of b^2.
            # To speed up, we can iterate i from 1 to sqrt(b^2) = b.
            # But b can be ~ 10^6 if x is large.
            # With x=n+1, b ~ n^2 ~ 65000.
            # Iterating to 65000 is fine.
            
            divisors = []
            # Optimization: only add if (i+b) % a == 0 ? No, we need to check both u and v.
            # Actually, if u is a divisor, v = b^2/u is also a divisor.
            # We can just collect all divisors first.
            
            # Even better:
            # We need u | b^2 and (u+b) % a == 0 and (b^2//u + b) % a == 0.
            # Note that if (u+b) % a == 0, then u = m*a - b.
            # So we can iterate m?
            # u ~ b^2/v.
            # Range of u: 1 to b^2.
            # This is too large.
            
            # Let's stick to finding divisors of b^2.
            # For b=65000, b^2 = 4.2e9. sqrt is 65000.
            # Loop 1 to 65000.
            
            for i in range(1, int(math.isqrt(b_sq)) + 1):
                if b_sq % i == 0:
                    u1 = i
                    u2 = b_sq // i
                    # Check u1
                    if (u1 + b) % a == 0:
                        v1 = b_sq // u1
                        if (v1 + b) % a == 0:
                            y1 = (u1 + b) // a
                            z1 = (v1 + b) // a
                            if y1 > 0 and z1 > 0 and len(set([x, y1, z1])) == 3 and y1 <= 10**12 and z1 <= 10**12:
                                found = [x, y1, z1]
                                break
                    # Check u2
                    if found: break
                    if (u2 + b) % a == 0:
                        v2 = b_sq // u2
                        if (v2 + b) % a == 0:
                            y2 = (u2 + b) // a
                            z2 = (v2 + b) // a
                            if y2 > 0 and z2 > 0 and len(set([x, y2, z2])) == 3 and y2 <= 10**12 and z2 <= 10**12:
                                found = [x, y2, z2]
                                break
            if found:
                break
        
        answers.append(found)
    return answers
