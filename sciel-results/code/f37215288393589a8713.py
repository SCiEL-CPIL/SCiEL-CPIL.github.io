import time
import math

def construct():
    start_time = time.time()
    deadline = start_time + 100  # Safety margin

    # Generate powerful numbers up to L
    # Powerful numbers are of the form x^2 * y^3 where y is square-free
    # Actually, standard form is product of prime powers >= 2.
    # Equivalent to x^2 * y where y is square-free? No.
    # a powerful number is a product of prime powers p^e with e >= 2.
    # We can generate them by iterating x (square part) and y (cube part of square-free?)
    # Simpler: iterate x from 1 to sqrt(L).
    # For each x, let x^2 be the square part.
    # We need x^2 * k <= L where k is such that x^2 * k is powerful.
    # This is not straightforward.
    
    # Better generation:
    # A number n is powerful if for every prime p dividing n, p^2 divides n.
    # We can generate them by:
    # 1. Iterate y (square-free part) from 1 to L^(1/3).
    #    Wait, the standard decomposition is n = s^2 * q where q is square-free?
    #    No, if n is powerful, n = u^2 * v where v is square-free?
    #    Example: 72 = 6^2 * 2. 2 is square-free.
    #    Example: 16 = 4^2 * 1. 1 is square-free.
    #    Example: 8 = 2^2 * 2. 2 is square-free.
    #    Yes, any powerful number is uniquely s^2 * q with q square-free.
    #    Proof: Let n = prod p_i^{e_i}.
    #    If e_i is even, p_i^{e_i} = (p_i^{e_i/2})^2.
    #    If e_i is odd, e_i >= 3. p_i^{e_i} = p_i^{e_i-1} * p_i = (p_i^{(e_i-1)/2})^2 * p_i.
    #    So n = (prod p_i^{floor(e_i/2)})^2 * (prod p_i^{e_i mod 2}).
    #    The second part is square-free.
    
    def generate_powerful(L):
        powerful = set()
        max_y = int(L ** (1/3)) + 1  # y^3 <= L => y <= L^(1/3)
        
        # Precompute square-free numbers up to max_y
        # Actually, we can just iterate y and check if it's square-free?
        # Or iterate x and y?
        # n = x^2 * y^3 ? No, that's for a different parametrization.
        # n = s^2 * q. q is square-free.
        # q <= L / s^2.
        # So for a fixed s, q can be any square-free number <= L/s^2.
        
        # Let's iterate s from 1 to sqrt(L).
        max_s = int(math.isqrt(L))
        powerful = set()
        for s in range(1, max_s + 1):
            s2 = s * s
            if s2 > L:
                break
            # q must be square-free and q <= L // s2
            max_q = L // s2
            # We need to iterate square-free numbers q up to max_q.
            # This is inefficient if we check each q.
            
            # Alternative: Iterate q (square-free) and s?
            # q is square-free.
            # s^2 * q <= L => s <= sqrt(L/q).
            # Max q is L (when s=1).
            # This is still hard to iterate square-free q up to L.
            
            # Let's go back to n = x^2 * y^3 ?
            # If n is powerful, n = prod p^{e}.
            # Let e = 2k or 2k+1.
            # n = (prod p^k)^2 * (prod p^{e mod 2}).
            # Let A = prod p^k. Let B = prod p^{e mod 2}.
            # B is square-free.
            # n = A^2 * B.
            # This doesn't restrict B to be a cube.
            
            # However, we can generate powerful numbers by:
            # Iterating x (the square root of the square part) is not enough.
            
            # Let's use the property:
            # Powerful numbers are sparse.
            # We can just iterate all numbers? No, L is too big.
            
            # Let's use the recursion or simple iteration for small L.
            # For L=10^12, we can generate them.
            
            # Efficient generation:
            # 1. Generate all square-free numbers up to L^(1/3)? No.
            # 2. Note that if n is powerful, n = a^2 b^3 where b is square-free?
            #    No, that's not unique.
            
            # Let's stick to the definition:
            # n is powerful if for all p|n, p^2|n.
            
            # We can generate by iterating the "cube part" y (square-free) and "square part" x.
            # Wait, if n = x^2 y^3, then n is powerful.
            # Is every powerful number of the form x^2 y^3?
            # 72 = 2^3 * 3^2.
            # x^2 y^3 = 72.
            # If y=2, y^3=8. x^2 = 9 => x=3. 3^2 * 2^3 = 9*8=72. Yes.
            # 16 = 2^4.
            # x^2 y^3 = 16.
            # If y=1, x^2=16 => x=4. 4^2 * 1^3 = 16. Yes.
            # 8 = 2^3.
            # x^2 y^3 = 8.
            # If y=2, y^3=8. x^2=1 => x=1. 1^2 * 2^3 = 8. Yes.
            # 9 = 3^2.
            # x^2 y^3 = 9.
            # If y=1, x^2=9 => x=3. 3^2 * 1^3 = 9. Yes.
            # 25 = 5^2.
            # x^2 y^3 = 25.
            # If y=1, x=5. Yes.
            # 36 = 2^2 * 3^2.
            # x^2 y^3 = 36.
            # If y=1, x=6. Yes.
            # 72 = 2^3 * 3^2.
            # If y=2, x=3. Yes.
            # What if n has a prime factor with exponent 5?
            # 32 = 2^5.
            # x^2 y^3 = 32.
            # If y=2, y^3=8. x^2 = 4 => x=2. 2^2 * 2^3 = 4*8=32. Yes.
            
            # So yes, every powerful number is of the form x^2 y^3 where y is square-free.
            # And this representation is unique?
            # If n = x^2 y^3 = u^2 v^3.
            # The exponent of p in y must be 0 or 1.
            # The exponent of p in n is 2*exp_p(x) + 3*exp_p(y).
            # Since exp_p(y) is 0 or 1, the exponent mod 3 is determined by exp_p(y).
            # If exp_p(n) % 3 == 0, exp_p(y)=0.
            # If exp_p(n) % 3 == 1, exp_p(y)=1? No, 2*0+3*1=3 (mod 3 = 0).
            # Wait.
            # If exp_p(y)=1, contribution is 3.
            # If exp_p(y)=0, contribution is 0.
            # So exp_p(n) = 2*exp_p(x) + 3*exp_p(y).
            # exp_p(n) % 2 = exp_p(y) % 2? No.
            # exp_p(n) % 3 = 2*exp_p(x) % 3.
            
            # Let's check 72 = 2^3 * 3^2.
            # exp_2(72)=3. exp_3(72)=2.
            # Form x^2 y^3.
            # For p=2: 2*exp_2(x) + 3*exp_2(y) = 3.
            # exp_2(y) must be 1 (since 3*1=3). Then 2*exp_2(x)=0 => exp_2(x)=0.
            # For p=3: 2*exp_3(x) + 3*exp_3(y) = 2.
            # exp_3(y) must be 0 (since 3*0=0). Then 2*exp_3(x)=2 => exp_3(x)=1.
            # So x = 3, y = 2.
            # x^2 y^3 = 9 * 8 = 72. Correct.
            
            # So we can generate powerful numbers by iterating y (square-free) and x.
            # y^3 <= L => y <= L^(1/3).
            # For each y, x^2 <= L/y^3 => x <= sqrt(L/y^3).
            
        powerful = set()
        max_y = int(L ** (1/3)) + 1
        
        # Generate square-free numbers up to max_y
        sq_free = []
        for y in range(1, max_y + 1):
            is_sq_free = True
            temp_y = y
            d = 2
            while d * d <= temp_y:
                if temp_y % (d * d) == 0:
                    is_sq_free = False
                    break
                d += 1
            if is_sq_free:
                sq_free.append(y)
                
        for y in sq_free:
            y_cubed = y ** 3
            if y_cubed > L:
                continue
            max_x = int(math.isqrt(L // y_cubed))
            for x in range(1, max_x + 1):
                val = x * x * y_cubed
                if val <= L:
                    powerful.add(val)
                    
        return sorted(powerful)

    # Try increasing L
    # Start with L = 10^12
    L = 10**12
    
    try:
        P = generate_powerful(L)
    except MemoryError:
        L = 10**10
        P = generate_powerful(L)

    K = len(P)
    
    # Count sums
    # Use a dictionary to store counts
    counts = {}
    P_set = set(P)
    
    # To speed up, we can break early if time is running out
    # But K^2 is the main cost.
    
    for i in range(K):
        a = P[i]
        for j in range(i, K):
            b = P[j]
            s = a + b
            if s > L: # We only care about N up to 2L? 
                # Actually, the task says N <= 10^18.
                # If L=10^12, max sum is 2*10^12.
                # This is well within 10^18.
                # But if we want to check larger N, we need larger L.
                # However, the heuristic suggests small N is better.
                # So we can just count all sums from this set.
                pass
            counts[s] = counts.get(s, 0) + 1
            
            if time.time() > deadline - 10:
                # Check current best
                if counts:
                    best_N = max(counts, key=counts.get)
                    best_count = counts[best_N]
                    # Retrieve pairs
                    pairs = []
                    for a2 in P:
                        if a2 > best_N:
                            break
                        b2 = best_N - a2
                        if b2 < a2:
                            continue
                        if b2 in P_set:
                            pairs.append([a2, b2])
                    return {'N': best_N, 'pairs': pairs}
                else:
                    return {'N': 2, 'pairs': [[1, 1]]}

    # Find best N
    best_N = max(counts, key=counts.get)
    best_count = counts[best_N]
    
    # Retrieve pairs
    pairs = []
    for a in P:
        if a > best_N:
            break
        b = best_N - a
        if b < a:
            continue
        if b in P_set:
            pairs.append([a, b])
            
    return {'N': best_N, 'pairs': pairs}
