"""Program using LFSR-generated binary sequence for signed step function to minimize C3 upper bound."""
import numpy as np

N = 400

# EVOLVE-BLOCK-START

def _gf2_poly_mul(a, b):
    """Multiply two polynomials over GF(2) represented as integers."""
    res = 0
    while b:
        if b & 1:
            res ^= a
        b >>= 1
        a <<= 1
    return res

def _gf2_poly_mod(a, m):
    """Compute a mod m over GF(2). m is the modulus polynomial (integer)."""
    deg_m = m.bit_length() - 1
    deg_a = a.bit_length() - 1
    while deg_a >= deg_m:
        if deg_a >= 0:
            a ^= (m << (deg_a - deg_m))
        deg_a = a.bit_length() - 1
    return a

def _gf2_poly_pow(base, exp, mod):
    """Compute base^exp mod mod over GF(2)."""
    result = 1
    base = _gf2_poly_mod(base, mod)
    while exp:
        if exp & 1:
            result = _gf2_poly_mod(_gf2_poly_mul(result, base), mod)
        base = _gf2_poly_mod(_gf2_poly_mul(base, base), mod)
        exp >>= 1
    return result

def _is_irreducible(m, p):
    """Check if polynomial p of degree m is irreducible over GF(2)."""
    # x^(2^m) == x mod p
    # and x^(2^(m/q)) != x mod p for all prime q dividing m
    import math
    # Get prime factors of m
    factors = set()
    temp = m
    d = 2
    while d * d <= temp:
        if temp % d == 0:
            factors.add(d)
            while temp % d == 0:
                temp //= d
        d += 1
    if temp > 1:
        factors.add(temp)
        
    # Check x^(2^m) == x
    # 2^m can be large, but we can do modular exponentiation
    # base = x (which is 2)
    # exp = 2^m
    # But 2^m is huge.
    # Note: x^(2^k) mod p can be computed by repeated squaring of x.
    # x^(2^m) = (...((x^2)^2)^2...)^2
    # Let's compute x^(2^m) mod p
    x_val = 2
    for _ in range(m):
        x_val = _gf2_poly_mod(_gf2_poly_mul(x_val, x_val), p)
    if x_val != 2:
        return False
        
    for q in factors:
        # Check x^(2^(m/q)) != x
        x_val = 2
        for _ in range(m // q):
            x_val = _gf2_poly_mod(_gf2_poly_mul(x_val, x_val), p)
        if x_val == 2:
            return False
            
    return True

def _is_primitive(m, p):
    """Check if irreducible polynomial p of degree m is primitive."""
    # Order is 2^m - 1
    # Check x^((2^m-1)/q) != 1 mod p for all prime q dividing 2^m-1
    P = (1 << m) - 1
    factors = set()
    temp = P
    d = 2
    while d * d <= temp:
        if temp % d == 0:
            factors.add(d)
            while temp % d == 0:
                temp //= d
        d += 1
    if temp > 1:
        factors.add(temp)
        
    for q in factors:
        exp = P // q
        # Compute x^exp mod p
        # base = x (2)
        val = _gf2_poly_pow(2, exp, p)
        if val == 1:
            return False
    return True

def get_primitive_polys(m):
    """Generate all primitive polynomials of degree m as lists of taps."""
    polys = []
    # Iterate over all polynomials of degree m with constant term 1
    # Mask must have bit m set and bit 0 set
    # Middle bits can be anything
    for mid in range(1 << (m - 1)):
        p = (1 << m) | mid | 1
        if _is_irreducible(m, p) and _is_primitive(m, p):
            # Extract taps
            taps = []
            for i in range(m):
                if (p >> i) & 1:
                    taps.append(i)
            polys.append(taps)
    return polys

def _generate_mls(taps, m):
    """Generate a maximal length sequence (MLS) of length 2^m - 1."""
    period = 2**m - 1
    # Initialize state with 1 followed by zeros
    state = np.zeros(m, dtype=np.int8)
    state[0] = 1
    
    seq = np.zeros(period, dtype=np.int8)
    
    for i in range(period):
        seq[i] = state[0]
        # Compute new bit
        # XOR of bits at tap positions
        new_bit = 0
        for t in taps:
            new_bit ^= state[t]
        
        # Shift left
        state[1:] = state[:-1]
        state[0] = new_bit
        
    return seq

def construct():
    best_score = float('inf')
    best_heights = None
    
    # Test m from 9 to 13
    for m in range(9, 14):
        polys = get_primitive_polys(m)
        for taps in polys:
            period = 2**m - 1
            if period < N:
                continue
                
            mls = _generate_mls(taps, m)
            # Map 0/1 to -1/1
            mls_pm1 = 2.0 * mls.astype(np.float64) - 1.0
            
            # Iterate over all possible starting phases
            for shift in range(period):
                # Extract window of length N
                idx = (np.arange(N) + shift) % period
                h = mls_pm1[idx]
                
                # Compute aperiodic autocorrelation
                ac = np.correlate(h, h, mode='full')
                
                # The peak is at index N-1
                # We want max abs of all other lags
                # Exclude the center
                ac_non_peak = np.concatenate([ac[:N-1], ac[N:]])
                
                max_ac = np.max(np.abs(ac_non_peak))
                
                # Score is the upper bound on C3, which is max_ac / N
                score = max_ac / N
                
                if score < best_score:
                    best_score = score
                    best_heights = h.copy()
                    
    if best_heights is None:
        # Fallback to random if no valid MLS found (should not happen)
        best_heights = np.random.choice([-1.0, 1.0], size=N)
        
    return best_heights

# EVOLVE-BLOCK-END
