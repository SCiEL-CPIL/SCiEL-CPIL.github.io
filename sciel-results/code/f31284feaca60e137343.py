import time
import math
from itertools import permutations

def construct():
    divisors = [6, 10, 14, 15, 21, 22, 26, 33, 35, 39]
    n = len(divisors)
    
    # Precompute gcds for consistency checks
    # gcds[i][j] = gcd(divisors[i], divisors[j])
    gcds = [[0]*n for _ in range(n)]
    for i in range(n):
        for j in range(n):
            gcds[i][j] = math.gcd(divisors[i], divisors[j])
            
    # Sort indices to improve backtracking pruning
    # Divisors with more constraints (larger gcds with others) should be assigned earlier?
    # Or just sort by divisor value?
    # Let's sort by divisor descending to handle larger moduli first?
    # Actually, the constraint is on deltas.
    # Let's just use the original order or sort by number of neighbors with gcd > 1.
    
    # Calculate a "constraint score" for each index
    # Count how many other divisors share a gcd > 1
    scores = [0]*n
    for i in range(n):
        for j in range(n):
            if i != j and gcds[i][j] > 1:
                scores[i] += 1
                
    # Sort indices by score descending
    order = sorted(range(n), key=lambda i: -scores[i])
    # map original index to position in order
    # We will assign deltas to indices in this order.
    
    start_time = time.time()
    timeout = 85
    
    # Fallback
    best_result = divisors[:]
    best_span = max(divisors) - min(divisors)
    
    # We search for minimal W
    # Max W to check?
    # The trivial solution has span 33.
    # We can go up to 32.
    for W in range(9, best_span + 1):
        if time.time() - start_time > timeout:
            break
            
        # Backtracking over deltas
        # deltas[i] is the offset for divisor i
        # We need distinct deltas in [0, W]
        
        # Use a set for used deltas
        used_deltas = set()
        deltas = [None] * n
        
        # Consistency check function
        # Given current assignment of deltas for some indices,
        # check if adding delta_new for index idx_new is consistent.
        
        def is_consistent(idx_new, delta_new):
            # Check against all previously assigned indices
            for idx_old in range(n):
                if deltas[idx_old] is None:
                    continue
                g = gcds[idx_new][idx_old]
                if g > 1:
                    if (delta_new - deltas[idx_old]) % g != 0:
                        return False
            return True
            
        def backtrack(pos):
            if pos == n:
                # All deltas assigned
                # Solve for L using CRT
                # L = delta_0 * (-1) mod d_0 ? No.
                # d_i | (L + delta_i) => L + delta_i = k_i d_i => L = k_i d_i - delta_i
                # L = -delta_i mod d_i
                # We need to find L such that L = -delta_i mod d_i for all i.
                
                # Since we checked consistency, a solution exists.
                # Find the smallest positive L.
                # We can use a simple iterative CRT or just search?
                # Since L < LCM, and LCM is 30030, we could just search?
                # No, CRT is faster.
                
                # Let's implement a simple CRT solver for the system:
                # L = r_i mod m_i
                # where r_i = (-delta_i) % d_i, m_i = d_i
                
                r = (-deltas[0]) % divisors[0]
                m = divisors[0]
                
                for i in range(1, n):
                    r_i = (-deltas[i]) % divisors[i]
                    m_i = divisors[i]
                    
                    # Solve L = r mod m
                    #       L = r_i mod m_i
                    # Let L = r + k*m
                    # r + k*m = r_i mod m_i
                    # k*m = (r_i - r) mod m_i
                    # k = (r_i - r) * m^{-1} mod m_i / gcd(m, m_i)
                    
                    g = math.gcd(m, m_i)
                    diff = r_i - r
                    if diff % g != 0:
                        return False # Should not happen if consistency checked
                    
                    # Reduce
                    m1 = m // g
                    m2 = m_i // g
                    diff_g = diff // g
                    m1_inv = pow(m1, -1, m2)
                    
                    k = (diff_g * m1_inv) % m2
                    # New L = r + k*m
                    # New modulus = lcm(m, m_i)
                    L_new = r + k * m
                    m_new = m * m2 # lcm(m, m_i) = m * m_i / g = m * m2
                    
                    # Normalize
                    r = L_new % m_new
                    m = m_new
                    
                # r is the smallest non-negative solution modulo LCM
                # We need L >= 1
                if r == 0:
                    L_val = m # Next positive multiple? 
                    # Wait, L=0 is not positive.
                    # If r=0, L can be LCM, 2*LCM, etc.
                    # Smallest positive is LCM.
                    # But check if L=0 is allowed? "positive integers". So L>=1.
                    # If r=0, L=LCM is the first positive candidate?
                    # Actually, if r=0, L can be 0, LCM, 2LCM...
                    # So smallest positive is LCM.
                    L_val = m
                else:
                    L_val = r
                    
                # Check bounds
                # Values are L_val + deltas[i]
                # Max value = L_val + max(deltas) <= L_val + W
                # Min value = L_val + min(deltas) >= L_val
                # Need 1 <= min and max <= 1000000
                
                if L_val < 1:
                    # Should not happen as we found positive r or set to m
                    pass 
                
                max_val = L_val + max(deltas)
                min_val = L_val + min(deltas)
                
                if min_val >= 1 and max_val <= 1000000:
                    # Construct result
                    res = [L_val + deltas[i] for i in range(n)]
                    # Verify distinctness (guaranteed by distinct deltas)
                    # Verify divisibility (guaranteed by CRT)
                    # Verify bounds (checked)
                    return res
                    
                # If bounds not met, try adding LCM?
                # If L_val + max(deltas) > 1000000, we might need to adjust?
                # No, we want to MINIMIZE span, so we want SMALL L.
                # If smallest positive L is too big, maybe no solution for this W?
                # Or maybe we can pick a different L?
                # L can be r + k*LCM.
                # We want min(L) such that L >= 1 - min(deltas) ?
                # Actually, values are L + delta_i.
                # We need L + delta_i >= 1 => L >= 1 - min(delta_i).
                # We need L + delta_i <= 1000000 => L <= 1000000 - max(delta_i).
                # So L must be in [1 - min_delta, 1000000 - max_delta].
                # The solutions for L are r + k*LCM.
                # We can find the smallest k such that L >= 1 - min_delta.
                
                min_delta = min(deltas)
                max_delta = max(deltas)
                
                L_min_req = 1 - min_delta
                L_max_req = 1000000 - max_delta
                
                if L_val < L_min_req:
                    # Increase L_val by LCMs
                    k = (L_min_req - L_val + m - 1) // m
                    L_val += k * m
                    
                if L_val <= L_max_req:
                    res = [L_val + deltas[i] for i in range(n)]
                    return res
                else:
                    return None
            
            idx = order[pos]
            # Try all available deltas
            for d in range(W + 1):
                if d in used_deltas:
                    continue
                if is_consistent(idx, d):
                    used_deltas.add(d)
                    deltas[idx] = d
                    res = backtrack(pos + 1)
                    if res is not None:
                        return res
                    used_deltas.remove(d)
                    deltas[idx] = None
            return None
            
        result = backtrack(0)
        if result is not None:
            # Final check
            if len(set(result)) == n:
                if all(result[i] % divisors[i] == 0 for i in range(n)):
                    if all(1 <= x <= 1000000 for x in result):
                        return result
                        
    return best_result
