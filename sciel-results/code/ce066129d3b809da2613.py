"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import math

def construct():
    # Deadline guard: the evaluator kills at 630s wall-clock.
    # Our construction is fast (ms-scale), but we add a guard anyway.
    t_start = time.time()
    DEADLINE = 620.0  # 10s safety margin

    # The seed returns [4.0, 8.0, 12.0, ..., 48.0] (arithmetic progression, 12 terms).
    # Task: 12 double roots, m=12, n=1, dimension n=1.
    # Score = upper bound on C4 (lower is better). Score 1.0 = seed value.
    # We need to return 12 positive double-root locations for the Laguerre combination.

    # The Cohn-Goncalves refined construction for Fourier sign uncertainty
    # involves a polynomial P(x) = product (x - a_i)^2 where a_i are positive.
    # The bound C4 depends on the specific placement of roots.
    
    # For the Fourier sign uncertainty problem with m double roots and n=1,
    # the optimal root placements are known from the theory. The Cohn-Goncalves
    # construction uses roots related to Laguerre polynomial zeros.
    
    # Key insight: the "refined" construction with 12 double roots for n=1
    # typically places roots at values that minimize the L4 norm ratio.
    # The arithmetic progression 4*(i+1) gives a baseline.
    
    # Let's try the zeros of the Laguerre polynomial L_12(x), which are
    # known to be optimal for certain quadrature and approximation problems.
    # The zeros of L_n(x) are all positive and distinct.
    
    # Compute zeros of L_12(x) using numpy
    try:
        import numpy as np
        from numpy.polynomial import laguerre as nlaguerre
        
        # Get the roots of the 12th Laguerre polynomial
        # L_12(x) = sum_{k=0}^{12} (-1)^k * C(12+k, k) / k! * x^k
        # Use numpy's built-in if available, otherwise compute coefficients
        coeffs = nlaguerre.lagval(0, 1)  # just to import
        # Better: use the companion matrix or numpy.roots with explicit coefficients
        # L_n(x) coefficients: c_k = (-1)^k * binom(n+k, k) / k!
        n_lag = 12
        c = []
        for k in range(n_lag + 1):
            val = ((-1) ** k) * math.comb(n_lag + k, k) / math.factorial(k)
            c.append(val)
        # numpy.roots expects highest power first
        c_rev = list(reversed(c))
        roots = np.roots(c_rev)
        # Take real positive roots
        real_roots = [r.real for r in roots if abs(r.imag) < 1e-10 and r.real > 0]
        real_roots.sort()
        
        if len(real_roots) == 12:
            # Scale the roots. The seed uses 4,8,...,48 with mean ~26.
            # Laguerre zeros of L_12 range roughly from ~0.5 to ~40+.
            # We need to map them to a comparable range.
            # The score depends on the actual values, so we need to find
            # the optimal scaling. Without knowing the exact objective,
            # let's try scaling so that the range matches the seed's range
            # or try a few scalings.
            
            # Seed: min=4, max=48, range=44, mean=26
            lag_min = real_roots[0]
            lag_max = real_roots[-1]
            lag_range = lag_max - lag_min
            
            # Try scaling to match seed's range
            scale_factor = 44.0 / lag_range
            shifted_roots = [(r - lag_min) * scale_factor + 4.0 for r in real_roots]
            
            # Check deadline
            if time.time() - t_start > DEADLINE:
                return [4.0 * (i + 1) for i in range(12)]
            
            # Return the scaled Laguerre zeros
            return shifted_roots
    except Exception:
        pass

    # Fallback to seed if anything goes wrong
    if time.time() - t_start > DEADLINE:
        return [4.0 * (i + 1) for i in range(12)]
    
    return [4.0 * (i + 1) for i in range(12)]
# EVOLVE-BLOCK-END
