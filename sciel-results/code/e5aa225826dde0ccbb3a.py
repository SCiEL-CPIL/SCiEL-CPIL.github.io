import numpy as np

N = 400

def construct():
    # Use a Legendre sequence of prime length P >= N
    # Find the smallest prime P >= N
    def is_prime(n):
        if n < 2:
            return False
        for i in range(2, int(np.sqrt(n)) + 1):
            if n % i == 0:
                return False
        return True

    P = N
    while not is_prime(P):
        P += 1
        
    # Generate Legendre symbols for 1..P-1
    # Legendre symbol (a/p) is 1 if a is a quadratic residue mod p, -1 otherwise.
    # For a=0, it is 0.
    # We need N values.
    # We will use indices 1 to N.
    # Since P >= N, 1..N are all < P.
    
    signs = np.zeros(N)
    for i in range(1, N + 1):
        # Compute i^2 mod P
        # If (i^2 mod P) is a square, it's a residue.
        # Actually, i is a residue if i^(P-1)/2 == 1 mod P.
        # Or simply check if i is in the set of squares.
        # Efficient way: 
        # Legendre symbol (i/P)
        # Using Euler's criterion: i^((P-1)/2) mod P
        # 1 if residue, P-1 if non-residue.
        
        # For small P, we can precompute squares.
        # But P is ~400, so it's fast.
        
        # Let's just use the property:
        # (i/P) = 1 if i is a QR, -1 if QNR.
        # We can compute this by checking if there exists x such that x^2 = i mod P.
        # Or use pow(i, (P-1)//2, P)
        
        val = pow(i, (P - 1) // 2, P)
        if val == 1:
            signs[i-1] = 1.0
        elif val == P - 1:
            signs[i-1] = -1.0
        else:
            # Should not happen for i < P and P prime
            signs[i-1] = 1.0
            
    return signs

