import math
import time

def _is_perfect_square(n):
    if n < 0:
        return False
    x = int(math.isqrt(n))
    return x * x == n

def _squarefree_kernel(n):
    """Return integer kernel (product of primes with odd exponent) of n."""
    f = n
    result = 1
    d = 2
    while d * d <= f:
        if f % d == 0:
            exp = 0
            while f % d == 0:
                f //= d
                exp += 1
            if exp % 2 == 1:
                result *= d
        d += 1 if d == 2 else 2
    if f > 1:
        result *= f
    return result

def construct():
    N = 210
    MAX = 10210
    MAX_SIZE = 100
    
    start_time = time.time()
    deadline = start_time + 55.0

    # 1. Find best pair (a, b)
    # a * b = 210 * s^2
    # 210 < a <= b
    # Minimize b
    
    best_pair = None
    min_b = float('inf')
    
    # s must be such that 210 * s^2 > 210^2 => s^2 > 210 => s >= 15
    # Also b >= sqrt(210 s^2) = s * sqrt(210)
    # If s * sqrt(210) >= min_b, we can stop
    s = 15
    while True:
        if time.time() > deadline - 10: break # Leave time for Gaussian
        
        # Check if further s can improve
        if s * math.sqrt(210) >= min_b:
            break
            
        P = 210 * s * s
        
        # Find divisors a of P such that 210 < a <= sqrt(P)
        # a must be integer
        # a > 210
        # a <= sqrt(P)
        # b = P / a
        # b >= a
        
        # We can iterate divisors up to sqrt(P)
        # But P can be large.
        # s starts at 15. P = 210 * 225 = 47250. sqrt ~ 217.
        # s increases.
        
        # Optimization: Only check s where P has small prime factors?
        # 210 = 2*3*5*7.
        # P = 2*3*5*7 * s^2.
        
        # Let's just iterate a from 211 to int(sqrt(P))
        # This is slow if P is large.
        # But we stop early if min_b is found.
        
        # Better: Iterate divisors of s?
        # No, just simple loop for now, s won't get too big.
        
        limit_a = int(math.isqrt(P))
        if limit_a <= 210:
            s += 1
            continue
            
        # We need a > 210
        # So start from 211
        # But a must divide P.
        
        # Instead of iterating all a, we can iterate factors of P?
        # Factoring P is fast.
        
        # Let's find all divisors of P
        divisors = []
        d = 2
        temp = P
        factors = {}
        while d * d <= temp:
            if temp % d == 0:
                exp = 0
                while temp % d == 0:
                    temp //= d
                    exp += 1
                factors[d] = exp
            d += 1 if d == 2 else 2
        if temp > 1:
            factors[temp] = 1
            
        # Generate divisors
        divs = [1]
        for p, e in factors.items():
            new_divs = []
            for d in divs:
                for i in range(e + 1):
                    new_divs.append(d * (p ** i))
            divs = new_divs
            
        for a in divs:
            if 210 < a <= limit_a:
                b = P // a
                if b >= a:
                    if b < min_b:
                        min_b = b
                        best_pair = (a, b)
                        
        s += 1

    if best_pair is None:
        # Fallback if pair search failed or timed out
        # Try a known solution
        return [224, 240]

    # 2. Gaussian Elimination to see if we can do better than min_b
    # Target: kernel of 210
    target_ker = _squarefree_kernel(N)
    
    # Map primes to indices
    prime_to_idx = {}
    next_prime_idx = 0
    
    def get_mask(ker):
        nonlocal prime_to_idx, next_prime_idx
        f = ker
        mask = 0
        d = 2
        while d * d <= f:
            if f % d == 0:
                exp = 0
                while f % d == 0:
                    f //= d
                    exp += 1
                if exp % 2 == 1:
                    if d not in prime_to_idx:
                        prime_to_idx[d] = next_prime_idx
                        next_prime_idx += 1
                    mask |= (1 << prime_to_idx[d])
            d += 1 if d == 2 else 2
        if f > 1:
            if f not in prime_to_idx:
                prime_to_idx[f] = next_prime_idx
                next_prime_idx += 1
            mask |= (1 << prime_to_idx[f])
        return mask

    target_mask = get_mask(target_ker)
    
    # Precompute masks for 211 to min_b
    # We only need to check up to min_b - 1 to see if a better solution exists
    # If we find one at T < min_b, it's better.
    # If we reach min_b, we return the pair.
    
    basis = {} # pivot -> (mask, rep_mask)
    
    def find_solution(target_m):
        current_rep = 0
        current_mask = target_m
        for p in sorted(basis.keys(), reverse=True):
            if current_mask & (1 << p):
                b_mask, b_rep = basis[p]
                current_mask ^= b_mask
                current_rep ^= b_rep
        if current_mask == 0:
            return current_rep
        return None

    # Precompute kernels for 211 to min_b
    # This is fast
    kernels = {}
    masks = {}
    for M in range(211, min_b + 1):
        ker = _squarefree_kernel(M)
        kernels[M] = ker
        masks[M] = get_mask(ker)

    for M in range(211, min_b + 1):
        if time.time() > deadline:
            return list(best_pair)
            
        mask = masks[M]
        rep = (1 << (M - 211))
        
        current_mask = mask
        current_rep = rep
        
        for p in sorted(basis.keys(), reverse=True):
            if current_mask & (1 << p):
                b_mask, b_rep = basis[p]
                current_mask ^= b_mask
                current_rep ^= b_rep
        
        if current_mask == 0:
            # Dependent
            continue
            
        pivot = current_mask.bit_length() - 1
        basis[pivot] = (current_mask, current_rep)
        
        for p in list(basis.keys()):
            if p != pivot:
                if basis[p][0] & (1 << pivot):
                    b_mask, b_rep = basis[p]
                    basis[p] = (b_mask ^ current_mask, b_rep ^ current_rep)
        
        # Check if target is solvable
        sol_rep = find_solution(target_mask)
        if sol_rep is not None:
            count = sol_rep.bit_count()
            if count <= MAX_SIZE and count > 0:
                selected = []
                for i in range(M - 211 + 1):
                    if (sol_rep >> i) & 1:
                        selected.append(211 + i)
                # Validate
                prod = 1
                for x in selected:
                    prod *= x
                if _is_perfect_square(prod * N):
                    return selected
                else:
                    return list(best_pair)
            else:
                # Solution too big, ignore for now, keep searching?
                # Or return pair?
                # If the only solution at this T is too big, we might find a smaller T later?
                # No, T is increasing.
                # But maybe a different combination at this T is smaller?
                # The Gaussian solution is one specific combination.
                # If it's too big, we can't easily find a smaller one.
                # So we should probably fall back to the pair if we can't get a small solution.
                # But wait, if T < min_b, and the solution is too big, it's still a valid solution
                # IF we can find one with size <= 100.
                # If we can't, we can't use it.
                # So we continue searching?
                # But if T increases, the solution might get even bigger?
                # Not necessarily.
                # However, the pair is a guaranteed valid solution with size 2.
                # If we find a T < min_b with a valid solution of size <= 100, it's better.
                # If we don't, we return the pair.
                pass

    # If we finish the loop without returning, return the pair
    return list(best_pair)
