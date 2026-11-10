import numpy as np
import time
import random
import heapq

def nikodym_set() -> np.ndarray:
    p = 7
    q = p * p  # 49
    deadline = time.time() + 580  # safety margin before 630s kill

    # Field F_49 = F_7[alpha], alpha^2 = 3
    k = 3  # alpha^2 = 3, 3 is a quadratic non-residue mod 7
    
    # Represent field elements as integers 0..48: element a + b*alpha = a + b*p
    # Field addition: (a1+b1*alpha) + (a2+b2*alpha) = (a1+a2) + (b1+b2)*alpha
    # Field multiplication: (a1+b1*alpha)(a2+b2*alpha) = a1*a2 + b1*b2*k + (a1*b2+a2*b1)*alpha
    
    def field_add(x, y):
        xa, xb = x % p, x // p
        ya, yb = y % p, y // p
        return (xa + ya) % p + ((xb + yb) % p) * p
    
    def field_mul(x, y):
        xa, xb = x % p, x // p
        ya, yb = y % p, y // p
        ra = (xa * ya + xb * yb * k) % p
        rb = (xa * yb + xb * ya) % p
        return ra + rb * p
    
    def field_sub(x, y):
        xa, xb = x % p, x // p
        ya, yb = y % p, y // p
        return (xa - ya) % p + ((xb - yb) % p) * p
    
    def field_div(x, y):
        # x / y = x * y^{-1}
        ya, yb = y % p, y // p
        # y^{-1} = conjugate(y) / Norm(y), conjugate(a+b*alpha) = a-b*alpha, Norm = a^2 - b^2*k
        norm = (ya * ya - yb * yb * k) % p
        # Inverse of norm in F_7
        norm_inv = pow(norm, p - 2, p)
        inv_a = (ya * norm_inv) % p
        inv_b = ((-yb) * norm_inv) % p
        return field_mul(x, inv_a + inv_b * p)
    
    # All field elements 0..48
    # All directions: (1, m) for m in 0..48, and (0, 1)
    # A point is (x, y) where x,y in 0..48
    # Point represented as x + y * q (so 0..2400)
    
    # For a set S of excluded points, S is valid iff:
    # - No two points in S have the same x (vertical line constraint)
    # - For each m, no two points in S have the same value of y - m*x (slope m constraint)
    #   i.e., the map x -> y - m*x is injective on S
    
    # Strategy: try to build S as a "permutation" {(x, f(x))} where f(x)-m*x is a permutation for all m
    # This gives |S| = 49, |N| = 2401 - 49 = 2352
    
    # Try various functions f
    # f(x) = x^2 + c for various c? No, x^2 isn't a permutation.
    
    # Let's try a greedy approach: pick points one at a time, ensuring constraints
    
    # Actually, let's first try to find a complete orthomorphism
    # f: F_49 -> F_49 such that f is a permutation AND f(x)-x is a permutation
    # (this handles m=0 and m=1; we need all m)
    
    # For a field F_q, a "complete mapping" is a permutation f such that f(x)+x is also a permutation.
    # We need the stronger property that f(x)-m*x is a permutation for ALL m.
    
    # This is related to "planar functions" or "perfect nonlinear functions".
    # In F_{p^2}, the function f(x) = x^{p+1} might work? Let's check.
    # f(x) = x^8 in F_49 (since p+1=8). x^8 = (x^7)^1 * x = x * (x^7). By Fermat, x^7 = 1 for x≠0.
    # So x^8 = x for x≠0, and 0^8=0. So f(x)=x, which is linear. Doesn't work.
    
    # Let me try a randomized greedy approach with backtracking
    # Or better: try all quadratic functions f(x) = ax^2 + bx + c and check
    
    # Actually, let me just try a simple construction and verify
    # Try f(x) = x^2 + x for all x, check if valid
    
    def check_permutation(vals):
        """Check if vals (list of field elements) is a permutation of F_49"""
        return len(set(vals)) == q
    
    def check_function(f_vals):
        """Check if f: x -> f_vals[x] gives a valid excluded set"""
        # f_vals is a list of length q, f_vals[x] = f(x)
        # Check f is a permutation (m=0 constraint)
        if len(set(f_vals)) != q:
            return False
        # Check f(x) - m*x is permutation for all m
        for m in range(q):
            vals = []
            for x in range(q):
                val = field_sub(f_vals[x], field_mul(m, x))
                vals.append(val)
            if len(set(vals)) != q:
                return False
        return True
    
    # Try to find a valid f by searching
    # First try simple polynomials
    found = None
    
    # Try f(x) = a*x^2 + b*x + c for various a,b,c
    # But x^2 isn't a permutation, so a*x^2 won't be a permutation unless a=0
    # So skip quadratics with a≠0
    
    # Try f(x) = a*x + b (linear): f(x)-m*x = (a-m)*x+b, permutation iff a≠m
    # Fails for m=a
    
    # Need a non-linear permutation with the stronger property
    # Let's try a randomized search over permutations
    
    random.seed(42)
    
    best_S = None
    
    # Try a few random permutations
    for trial in range(200):
        if time.time() > deadline - 100:
            break
        perm = list(range(q))
        random.shuffle(perm)
        if check_function(perm):
            found = perm
            break
    
    if found is None:
        # Fallback: use the original construction (smaller complement)
        # Original: exclude points on Hermitian unital with y not in special set
        y_special = set()
        for u in range(p):
            for v in range(p):
                y_special.add(u * u % p + v * p)
        
        excluded = set()
        for xa in range(p):
            for xb in range(p):
                x = xa + xb * p
                for ya in range(p):
                    for yb in range(p):
                        y = ya + yb * p
                        # Check if on unital: Trace(x) + Norm(y) = 0
                        # Trace(x) = x + x^7 = 2*xa (since x^7 = xa - xb*alpha for x=xa+xb*alpha... let me recompute)
                        # Actually Trace(x) = x + x^p. In F_{p^2}, x^p = conjugate(x) = xa - xb*alpha
                        # So Trace(x) = xa + xb*alpha + xa - xb*alpha = 2*xa
                        # Norm(y) = y * y^p = (ya+yb*alpha)(ya-yb*alpha) = ya^2 - yb^2*k
                        trace_x = (2 * xa) % p
                        norm_y = (ya * ya - yb * yb * k) % p
                        if (trace_x + norm_y) % p == 0:
                            if y not in y_special:
                                excluded.add(x + y * q)
        
        # Build N as complement
        all_points = set(range(q * q))
        N = all_points - excluded
        points = []
        for pt in sorted(N):
            y = pt // q
            x = pt % q
            xa, xb = x % p, x // p
            ya, yb = y % p, y // p
            points.append([[xa, xb], [ya, yb]])
        return np.array(points, dtype=np.int64)
    else:
        # Build S from f
        S = set()
        for x in range(q):
            y = found[x]
            S.add(x + y * q)
        
        all_points = set(range(q * q))
        N = all_points - S
        points = []
        for pt in sorted(N):
            y = pt // q
            x = pt % q
            xa, xb = x % p, x // p
            ya, yb = y % p, y // p
            points.append([[xa, xb], [ya, yb]])
        return np.array(points, dtype=np.int64)
