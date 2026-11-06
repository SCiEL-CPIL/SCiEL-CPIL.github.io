import math

def construct():
    # Primes congruent to 1 mod 4
    primes = [5, 13, 17, 29, 37, 41, 53, 61, 73, 89, 97, 101, 109, 113, 137, 149, 157, 173, 181, 193, 197, 229, 233, 241, 257, 269, 277, 281, 293]
    
    # We need to find exponents e_i to maximize prod(e_i + 1)
    # subject to prod(p_i^e_i) <= 10^18
    
    best_score = 0
    best_exponents = []
    best_M = 1
    
    # Recursive search for exponents
    # e_0 >= e_1 >= ...
    def search(idx, current_prod, current_score, max_exp, exponents):
        nonlocal best_score, best_exponents, best_M
        if current_score > best_score:
            best_score = current_score
            best_exponents = exponents.copy()
            best_M = current_prod
            
        if idx >= len(primes):
            return
        
        p = primes[idx]
        # Try exponents from max_exp down to 0
        e = max_exp
        while e >= 0:
            # Check if p^e fits
            if math.log10(current_prod) + e * math.log10(p) > 18.0:
                e -= 1
                continue
                
            new_prod = current_prod * (p ** e)
            new_score = current_score * (e + 1)
            
            exponents.append(e)
            search(idx + 1, new_prod, new_score, e, exponents)
            exponents.pop()
            
            e -= 1
            
    # Initial call
    search(0, 1, 1, 40, [])
    
    # Now we have best_M and best_exponents
    # We need to find all pairs (X, Y) such that X^2 + Y^2 = best_M
    
    def find_ab(p):
        a = 1
        while a * a < p:
            b2 = p - a * a
            b = int(math.isqrt(b2))
            if b * b == b2:
                return a, b
            a += 1
        return None, None

    def mul_gauss(g1, g2):
        r1, i1 = g1
        r2, i2 = g2
        return (r1*r2 - i1*i2, r1*i2 + i1*r2)

    all_gaussians = [[] for _ in range(len(primes))]
    
    for i, e in enumerate(best_exponents):
        if e == 0:
            all_gaussians[i] = [(1, 0)]
            continue
            
        a, b = find_ab(primes[i])
        if a is None:
            all_gaussians[i] = [(1, 0)]
            continue
            
        z = (a, b)
        z_bar = (a, -b)
        
        z_powers = [(1, 0)]
        for _ in range(e):
            z_powers.append(mul_gauss(z_powers[-1], z))
            
        zb_powers = [(1, 0)]
        for _ in range(e):
            zb_powers.append(mul_gauss(zb_powers[-1], z_bar))
            
        for k in range(e + 1):
            term = mul_gauss(z_powers[k], zb_powers[e-k])
            all_gaussians[i].append(term)
            
    current_combinations = [(1, 0)]
    
    for gaussians in all_gaussians:
        if not gaussians:
            continue
        new_combinations = []
        for g1 in current_combinations:
            for g2 in gaussians:
                new_combinations.append(mul_gauss(g1, g2))
        current_combinations = new_combinations
        
    pairs = set()
    for r, i in current_combinations:
        X = abs(r)
        Y = abs(i)
        if X > 0 and Y > 0:
            if X > Y:
                X, Y = Y, X
            pairs.add((X, Y))
            
    final_pairs = [[X*X, Y*Y] for X, Y in pairs]
    
    N = best_M
    
    return {'N': N, 'pairs': final_pairs}
