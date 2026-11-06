import math
import time

def construct():
    start_time = time.time()
    deadline = start_time + 145.0
    
    def time_left():
        return deadline - time.time()
    
    # Primes 1 mod 4 for building M
    p_1mod4 = [5, 13, 17, 29, 37, 41, 53, 61, 73, 89, 97, 101, 109, 113, 137, 149, 157, 173, 181, 193, 197, 229, 233, 241, 257, 269, 277, 281, 293, 313, 317, 337, 349, 353, 373, 389, 397, 401, 409, 421, 433, 449, 457, 461, 509, 541, 557, 569, 577, 593, 601, 613, 617, 641, 653, 661, 673, 677, 701, 709, 733, 757, 761, 769, 773, 797, 809, 821, 829, 853, 857, 877, 881, 929, 937, 941, 953, 977]
    
    # Small primes for R (powerful core)  
    small_primes = [2, 3, 5, 7, 11, 13, 17, 19, 23]
    
    best_N = 0
    best_pairs = []
    
    # Find representation of p as sum of two squares
    def find_prime_rep(p):
        """Find a, b > 0 with a^2 + b^2 = p for prime p ≡ 1 (mod 4)"""
        for a in range(1, int(math.isqrt(p)) + 1):
            b2 = p - a * a
            b = math.isqrt(b2)
            if b * b == b2 and b > 0:
                return (a, b)
        return None
    
    # Gaussian integer multiplication
    def gauss_mult(z1, z2):
        a, b = z1
        c, d = z2
        return (a*c - b*d, a*d + b*c)
    
    # Gaussian integer power
    def gauss_pow(z, n):
        if n == 0:
            return (1, 0)
        if n == 1:
            return z
        if n % 2 == 0:
            half = gauss_pow(z, n // 2)
            return gauss_mult(half, half)
        else:
            return gauss_mult(z, gauss_pow(z, n - 1))
    
    # Generate all representations of M = prod p_i^e_i as sum of two squares
    # Using Gaussian integer factorization
    def generate_all_reps(M_factors, prime_reps):
        """
        M_factors: dict {p: e} for primes p ≡ 1 (mod 4)
        prime_reps: dict {p: (a, b)} where a^2 + b^2 = p
        Returns list of (x, y) with x >= 1, y >= 1, x <= y, x^2 + y^2 = M
        """
        # For each prime, we have π = a + bi and conj(π) = a - bi
        # A representation is: prod_i (π_i^{k_i} * conj(π_i)^{e_i - k_i}) * unit
        # where 0 <= k_i <= e_i
        # Units: 1, i, -1, -i
        # We'll generate all and filter
        
        primes = list(M_factors.keys())
        n = len(primes)
        
        if n == 0:
            # M = 1, only representation is (1, 0) but we need x >= 1, y >= 1
            return []
        
        # Precompute π_i and conj(π_i) for each prime
        pis = []
        conj_pis = []
        for p in primes:
            a, b = prime_reps[p]
            pis.append((a, b))
            conj_pis.append((a, -b))
        
        # Generate all combinations
        results = set()
        
        # Use iterative approach to avoid deep recursion
        # Start with [(1, 0)] and for each prime, expand
        current = [(1, 0)]
        
        for i, p in enumerate(primes):
            e = M_factors[p]
            pi = pis[i]
            conj_pi = conj_pis[i]
            
            new_current = []
            for (x, y) in current:
                for k in range(e + 1):
                    # Multiply by pi^k * conj_pi^(e-k)
                    # Compute pi^k * conj_pi^(e-k)
                    # = pi^k * conj(pi)^(e-k)
                    # Note: pi * conj(pi) = p, so conj_pi^(e-k) = conj(pi)^(e-k)
                    # Let's compute step by step
                    # Start with (1,0)
                    zx, zy = 1, 0
                    # Multiply by pi k times
                    for _ in range(k):
                        zx, zy = gauss_mult((zx, zy), pi)
                    # Multiply by conj_pi (e-k) times
                    for _ in range(e - k):
                        zx, zy = gauss_mult((zx, zy), conj_pi)
                    
                    # Now multiply (x, y) by (zx, zy)
                    new_x = x * zx - y * zy
                    new_y = x * zy + y * zx
                    new_current.append((new_x, new_y))
            
            current = new_current
        
        # Now apply units and filter
        for (x, y) in current:
            # Try all 4 units: 1, i, -1, -i
            # Unit 1: (x, y)
            # Unit i: (-y, x)
            # Unit -1: (-x, -y)
            # Unit -i: (y, -x)
            for (ux, uy) in [(x, y), (-y, x), (-x, -y), (y, -x)]:
                ax, ay = abs(ux), abs(uy)
                if ax >= 1 and ay >= 1:
                    if ax > ay:
                        ax, ay = ay, ax
                    # Check ax^2 + ay^2 == M
                    if ax * ax + ay * ay == M:
                        results.add((ax, ay))
        
        return list(results)
    
    # Try different R sizes
    r_candidates = [1, 2, 3, 4, 5, 6, 7]
    
    for r in r_candidates:
        if time_left() < 15:
            break
            
        # Compute R
        R = 1
        for i in range(r):
            R *= small_primes[i] ** 3
        
        M_max = 10**18 // R
        if M_max < 10:
            continue
        
        # Find best M <= M_max as product of (4k+1 primes) to maximize reps
        best_M = 1
        best_rep_count = 0
        best_M_factorization = {}
        
        n_primes = min(25, len(p_1mod4))
        search_primes = p_1mod4[:n_primes]
        
        def dfs(idx, current_M, current_prod, prev_exp, factorization):
            nonlocal best_M, best_rep_count, best_M_factorization
            
            if time_left() < 5:
                return
            
            if current_M > 1 and current_prod > best_rep_count:
                best_rep_count = current_prod
                best_M = current_M
                best_M_factorization = dict(factorization)
            
            if idx == len(search_primes):
                return
            
            p = search_primes[idx]
            
            val = 1
            for e in range(prev_exp + 1):
                next_M = current_M * val
                if next_M > M_max:
                    break
                factorization[p] = e
                dfs(idx + 1, next_M, current_prod * (e + 1), e, factorization)
                if e > 0:
                    del factorization[p]
                val *= p
                if val > M_max // max(current_M, 1):
                    break
        
        dfs(0, 1, 1, 50, {})
        
        M = best_M
        N = R * M
        
        if N < 2 or N > 10**18:
            continue
        
        # Get prime representations
        M_factors = best_M_factorization
        prime_reps = {}
        valid = True
        for p in M_factors:
            if M_factors[p] > 0:
                rep = find_prime_rep(p)
                if rep is None:
                    valid = False
                    break
                prime_reps[p] = rep
        
        if not valid:
            continue
        
        # Generate all representations
        reps = generate_all_reps(M_factors, prime_reps)
        
        pairs = []
        for (x, y) in reps:
            a = R * x * x
            b = R * y * y
            if a + b == N and a <= b and a >= 1 and b >= 1:
                pairs.append([a, b])
        
        # Remove duplicates
        seen = set()
        unique_pairs = []
        for p in pairs:
            key = (p[0], p[1])
            if key not in seen:
                seen.add(key)
                unique_pairs.append(p)
        pairs = unique_pairs
        
        if len(pairs) > len(best_pairs):
            best_pairs = pairs
            best_N = N
    
    # Fallback
    if not best_pairs:
        best_N = 5
        best_pairs = [[1, 4]]
    
    return {'N': best_N, 'pairs': best_pairs}

if __name__ == "__main__":
    res = construct()
    print(res['N'])
    print(len(res['pairs']))
