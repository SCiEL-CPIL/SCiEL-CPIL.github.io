import numpy as np
import math
import time

def construct():
    start_time = time.time()
    deadline = start_time + 580  # Safety margin before 630s kill

    m = 12
    # We are looking for P(x) = x^m + c_{m-1}x^{m-1} + ... + c_0
    # Minimize <P, P> = <x^m + C, x^m + C> where C is the lower part
    # <P, P> = <x^m, x^m> + 2 <C, x^m> + <C, C>
    # Let v = [c_0, c_1, ..., c_{m-1}]
    # <C, C> = v^T G v, where G_{ij} = <x^i, x^j>
    # <C, x^m> = v^T h, where h_i = <x^i, x^m>
    # Objective: v^T G v + 2 v^T h + const
    # Minimizer: G v = -h

    # Moments of Laguerre weight e^{-x} on [0, inf):
    # <x^i, x^j> = \int_0^\infty x^{i+j} e^{-x} dx = \Gamma(i+j+1) = (i+j)!
    
    # Construct G (m x m) and h (m)
    G = np.zeros((m, m))
    h = np.zeros(m)
    
    for i in range(m):
        for j in range(m):
            G[i, j] = math.factorial(i + j)
        h[i] = math.factorial(i + m)
        
    # Solve G v = -h
    try:
        v = np.linalg.solve(G, -h)
    except np.linalg.LinAlgError:
        # Fallback to pseudo-inverse if singular (shouldn't happen for Hilbert-like moment matrix)
        v = np.linalg.lstsq(G, -h, rcond=None)[0]

    # Construct the full polynomial coefficients in ascending order
    # p[0] = c_0, ..., p[m-1] = c_{m-1}, p[m] = 1
    p = np.zeros(m + 1)
    p[:m] = v
    p[m] = 1.0

    # Find roots of the polynomial
    # np.roots expects descending order
    p_desc = p[::-1]
    try:
        roots = np.roots(p_desc)
    except Exception:
        # Fallback if root finding fails
        return [4.0 * (2.0 ** i) for i in range(12)]

    # Filter for real, positive roots
    # The problem states "12 positive double-root locations"
    # The optimal monic polynomial for this inner product should have real positive roots
    # (related to orthogonal polynomials or similar structures)
    
    real_roots = []
    for r in roots:
        if abs(r.imag) < 1e-10:
            r_real = r.real
            if r_real > 0:
                real_roots.append(r_real)
            
    # Sort roots
    real_roots.sort()
    
    # We need exactly 12 roots. 
    # If we have fewer, it might be due to numerical instability or complex roots.
    # However, for the Laguerre moment problem, the optimal polynomial is expected to have real positive roots.
    # If we don't have 12, we might need to refine or fallback.
    # But given the algebraic nature, the roots should be well-behaved.
    
    if len(real_roots) != m:
        # If we don't have exactly m real positive roots, we can try to use the complex roots 
        # by taking their real parts if they are close, or just return the best subset.
        # However, the contract says "list of 12 positive double-root locations".
        # If the optimal polynomial has complex roots, the "sign change" logic might fail.
        # But the task implies a valid solution exists with 12 positive roots.
        
        # Let's check if we have 12 real roots (some might be negative? Unlikely for this weight)
        all_real_roots = [r.real for r in roots if abs(r.imag) < 1e-10]
        if len(all_real_roots) == m:
            # If all are real but some negative, we have a problem.
            # But for x^12 + ... with positive moments, the roots are likely positive.
            # Let's just return the real parts if we have 12 of them, assuming the problem setup ensures positivity.
            return [max(1e-6, r) for r in sorted(all_real_roots)]
        else:
            # Fallback to a known good heuristic if algebraic solution yields invalid root count
            # Geometric sequence as in the seed
            return [4.0 * (2.0 ** i) for i in range(12)]

    return real_roots
