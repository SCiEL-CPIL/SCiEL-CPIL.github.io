import numpy as np
from itertools import product

def beat_the_average_measure() -> tuple[np.ndarray, np.ndarray]:
    # Support points: 0, 1, M
    # Weights: a, b, c
    
    # Precompute monomial indices for a^i b^j c^k with i+j+k=4
    # We map (i, j, k) to an index
    monomials = []
    idx_map = {}
    for i in range(5):
        for j in range(5 - i):
            for k in range(5 - i - j):
                if i + j + k == 4:
                    idx = len(monomials)
                    monomials.append((i, j, k))
                    idx_map[(i, j, k)] = idx
    
    n_mono = len(monomials)
    
    # Grid for M
    # M > 1 is required for support {0, 1, M} to be distinct and ordered 0 < 1 < M
    # If M <= 1, the support is not {0, 1, M} in that order, or M=1 degenerates.
    # The problem implies distinct support points usually, but even if not, we can handle.
    # However, the condition s_i+s_j+s_k < 2s_l depends on values.
    # Let's search M in [1.1, 100].
    M_vals = np.concatenate([
        np.linspace(1.1, 5.0, 100),
        np.linspace(5.0, 20.0, 50),
        np.linspace(20.0, 100.0, 30),
        np.array([200.0, 500.0, 1000.0])
    ])
    
    # Grid for b (mass at 1)
    # b in [0, 1]
    b_vals = np.linspace(0.0, 1.0, 200)
    
    best_P = -1.0
    best_a, best_b, best_c, best_M = 0.0, 0.0, 0.0, 2.0
    
    for M in M_vals:
        s = np.array([0.0, 1.0, M])
        
        # Determine coefficients for P(a,b,c) = sum C_{ijk} a^i b^j c^k
        coeffs = np.zeros(n_mono)
        
        # Iterate over all (i, j, k, l)
        for i in range(3):
            for j in range(3):
                for k in range(3):
                    for l in range(3):
                        if s[i] + s[j] + s[k] < 2.0 * s[l]:
                            # This term contributes w_i w_j w_k w_l
                            # w = [a, b, c]
                            # w_i w_j w_k w_l is a monomial in a,b,c
                            # Count powers
                            p_a = (i == 0) + (j == 0) + (k == 0) + (l == 0)
                            p_b = (i == 1) + (j == 1) + (k == 1) + (l == 1)
                            p_c = (i == 2) + (j == 2) + (k == 2) + (l == 2)
                            # p_a + p_b + p_c = 4
                            idx = idx_map[(p_a, p_b, p_c)]
                            coeffs[idx] += 1.0
        
        # Now for each b in b_vals, find optimal a
        for b in b_vals:
            if b < 0 or b > 1:
                continue
            
            # P(a, b, 1-b-a) = sum C_{ijk} a^i b^j (1-b-a)^k
            # We want to maximize this over a in [0, 1-b]
            
            # Construct polynomial in a: Q(a) = sum_{m=0}^4 A_m a^m
            # A_m = sum_{i,j,k: i+j+k=4} C_{ijk} b^j [coefficient of a^m in (1-b-a)^k]
            
            # (1-b-a)^k = sum_{p=0}^k binom(k,p) (1-b)^{k-p} (-a)^p
            # Coeff of a^m is sum over k such that m <= k:
            # binom(k,m) (1-b)^{k-m} (-1)^m * (sum over i,j where i+j+k=4 and i+m = m_total? No.)
            
            # Let's just compute A_m directly.
            A = np.zeros(5) # a^0 to a^4
            
            for idx in range(n_mono):
                i, j, k = monomials[idx]
                C = coeffs[idx]
                if C == 0:
                    continue
                
                # Term: C * a^i * b^j * (1-b-a)^k
                # Expand (1-b-a)^k
                for p in range(k + 1):
                    # coeff of a^p in (1-b-a)^k is binom(k,p) (1-b)^{k-p} (-1)^p
                    # This contributes to a^{i+p}
                    m = i + p
                    if m > 4:
                        continue
                    binom_val = float(np.math.comb(k, p))
                    term_coeff = C * (b ** j) * (binom_val * ((1.0 - b) ** (k - p)) * ((-1.0) ** p))
                    A[m] += term_coeff
            
            # Q(a) = A[0] + A[1]a + A[2]a^2 + A[3]a^3 + A[4]a^4
            # Q'(a) = A[1] + 2A[2]a + 3A[3]a^2 + 4A[4]a^3
            
            # Find roots of Q'(a) = 0 in [0, 1-b]
            # Cubic: 4A[4]a^3 + 3A[3]a^2 + 2A[2]a + A[1] = 0
            
            a_max_range = 1.0 - b
            candidates = [0.0, a_max_range]
            
            # Coeffs for np.roots: [c3, c2, c1, c0]
            c3 = 4.0 * A[4]
            c2 = 3.0 * A[3]
            c1 = 2.0 * A[2]
            c0 = A[1]
            
            if abs(c3) > 1e-12:
                roots = np.roots([c3, c2, c1, c0])
                for r in roots:
                    if np.isreal(r):
                        r_real = np.real(r)
                        if -1e-9 <= r_real <= a_max_range + 1e-9:
                            candidates.append(np.clip(r_real, 0.0, a_max_range))
            elif abs(c2) > 1e-12:
                # Quadratic
                roots = np.roots([c2, c1, c0])
                for r in roots:
                    if np.isreal(r):
                        r_real = np.real(r)
                        if -1e-9 <= r_real <= a_max_range + 1e-9:
                            candidates.append(np.clip(r_real, 0.0, a_max_range))
            elif abs(c1) > 1e-12:
                r = -c0 / c1
                if -1e-9 <= r <= a_max_range + 1e-9:
                    candidates.append(np.clip(r, 0.0, a_max_range))
            
            # Evaluate Q(a) at candidates
            max_Q = -1.0
            best_a_local = 0.0
            for a in candidates:
                # Q(a) = sum A[m] a^m
                Q_val = 0.0
                for m in range(5):
                    Q_val += A[m] * (a ** m)
                if Q_val > max_Q:
                    max_Q = Q_val
                    best_a_local = a
            
            if max_Q > best_P:
                best_P = max_Q
                best_a = best_a_local
                best_b = b
                best_c = 1.0 - b - best_a
                best_M = M
    
    # Clean up
    best_c = max(0.0, best_c)
    best_a = max(0.0, best_a)
    best_b = max(0.0, best_b)
    s_sum = best_a + best_b + best_c
    if s_sum > 0:
        best_a /= s_sum
        best_b /= s_sum
        best_c /= s_sum
    else:
        best_a, best_b, best_c = 1.0, 0.0, 0.0

    support = np.array([0.0, 1.0, best_M], dtype=np.float64)
    weights = np.array([best_a, best_b, best_c], dtype=np.float64)
    
    return support, weights
