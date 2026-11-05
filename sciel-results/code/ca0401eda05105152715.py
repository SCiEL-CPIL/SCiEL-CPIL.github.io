import random
import time

def count_nonzero_square(coeffs):
    """Compute the square of a polynomial given as coefficient list (ascending degree) and count nonzero terms."""
    n = len(coeffs)
    # Square via convolution
    sq = [0] * (2 * n - 1)
    for i in range(n):
        if coeffs[i] == 0:
            continue
        for j in range(n):
            if coeffs[j] == 0:
                continue
            sq[i + j] += coeffs[i] * coeffs[j]
    return sum(1 for c in sq if c != 0)

def construct():
    start_time = time.time()
    deadline = start_time + 85  # 5s safety margin
    
    best_count = 999
    best_poly = None
    
    # Try known constructions first
    # Construction 1: P = x^k * (sum of 12 monomials with special structure)
    # Try: P = (1+x^a+x^b+x^c)(1+x^d+x^e) where a,b,c,d,e chosen to give 12 terms
    # and maximize collisions in square
    
    # Let's try random search with structured candidates
    random.seed(42)
    
    # Try polynomials of form: sum of 12 terms at specific exponents
    # with coefficients +/-1 to maximize cancellation in square
    
    # First, try some known good candidates
    # The "Borwein" type: 1+x+x^3+x^4+x^6+x^7+x^9+x^10+x^12+x^13+x^15+x^16
    # This is related to (1+x^2+x^4)(1+x^6)(1+x^10) type structures
    
    candidates = []
    
    # Candidate: product of sparse factors
    # (1+x^a)(1+x^b)(1+x^c)(1+x^d)(1+x^e)(1+x^f) gives up to 64 terms
    # But if we use (1+x^a+x^b)(1+x^c)(1+x^d)(1+x^e) = up to 24 terms
    # (1+x^a+x^b+x^c)(1+x^d)(1+x^e) = up to 12 terms!
    
    # So P = (1+x^a+x^b+x^c)*(1+x^d)*(1+x^e) has at most 12 terms
    # if all exponents are distinct. P^2 = (1+x^a+x^b+x^c)^2 * (1+x^d)^2 * (1+x^e)^2
    
    # (1+x^a+x^b+x^c)^2 has at most 10 terms (4 terms squared)
    # (1+x^d)^2 = 1+2x^d+x^{2d} has 3 terms
    # (1+x^e)^2 = 1+2x^e+x^{2e} has 3 terms
    # Total: up to 10*3*3 = 90 terms, but with collisions
    
    # Let's search over small exponents
    for a in range(1, 20):
        for b in range(a+1, 20):
            for c in range(b+1, 20):
                for d in range(1, 20):
                    if d in (a, b, c):
                        continue
                    for e in range(1, 20):
                        if e in (a, b, c, d):
                            continue
                        # Build polynomial
                        exponents = set()
                        for x1 in [0, a, b, c]:
                            for x2 in [0, d]:
                                for x3 in [0, e]:
                                    exponents.add(x1 + x2 + x3)
                        if len(exponents) != 12:
                            continue
                        exponents = sorted(exponents)
                        # All coefficients are 1
                        coeffs = [0] * (exponents[-1] + 1)
                        for exp in exponents:
                            coeffs[exp] = 1
                        # Check length constraint
                        if len(coeffs) > 41:
                            continue
                        cnt = count_nonzero_square(coeffs)
                        if cnt < best_count:
                            best_count = cnt
                            best_poly = coeffs[:]
                            if best_count <= 5:
                                return best_poly
                if time.time() > deadline:
                    break
            if time.time() > deadline:
                break
        if time.time() > deadline:
            break
        if time.time() > deadline:
            break
    
    # Try random sparse polynomials
    for trial in range(50000):
        if time.time() > deadline:
            break
        # Random 12-term polynomial
        exponents = sorted(random.sample(range(0, 40), 12))
        coeffs = [0] * 41
        for exp in exponents:
            coeffs[exp] = random.choice([-1, 1])
        # Ensure last coefficient nonzero
        if coeffs[40] == 0 and exponents[-1] < 40:
            # Trim to actual degree
            pass
        # Compute actual length
        actual_len = exponents[-1] + 1
        if actual_len > 41:
            continue
        cnt = count_nonzero_square(coeffs[:actual_len])
        if cnt < best_count:
            best_count = cnt
            best_poly = coeffs[:actual_len]
            if best_count <= 5:
                break
    
    if best_poly is None:
        # Fallback: simple 12-term polynomial
        return [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1]
    
    # Trim trailing zeros (but keep last coefficient nonzero)
    while len(best_poly) > 1 and best_poly[-1] == 0:
        best_poly.pop()
    
    # Ensure length <= 41
    if len(best_poly) > 41:
        best_poly = best_poly[:41]
        while len(best_poly) > 1 and best_poly[-1] == 0:
            best_poly.pop()
    
    return best_poly
