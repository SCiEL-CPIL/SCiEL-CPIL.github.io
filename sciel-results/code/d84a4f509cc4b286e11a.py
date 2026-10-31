"""Erdos minimum overlap: minimize max overlap of h with shifted 1-h."""
import time
import math
import numpy as np

N = 100


def _score(h):
    """Compute max overlap of h with shifted (1-h), treating h as zero outside [0,2]."""
    n = len(h)
    shifted = np.zeros(n)
    shifted[:-1] = h[1:]
    overlap = h * (1.0 - shifted)
    return float(np.max(overlap))


def _overlap_for_2block(k, a, b):
    """Compute max overlap for 2-block with first k elements = a, rest = b."""
    n = N
    if k <= 0 or k >= n:
        return float('inf')
    vals = []
    if k >= 2:
        vals.append(a * (1.0 - a))
    vals.append(a * (1.0 - b))
    if n - k >= 2:
        vals.append(b * (1.0 - b))
    vals.append(b)
    return max(vals)


def _find_optimal_2block(k):
    """Find optimal a for 2-block by analyzing critical points analytically."""
    n = N
    
    # b = (n/2 - k*a) / (n-k)
    # Candidates for max:
    #   f1 = a*(1-a)          (if k >= 2)
    #   f2 = a*(1-b)
    #   f3 = b*(1-b)          (if n-k >= 2)
    #   f4 = b
    
    # b(a) = (n/2 - k*a) / (n-k)
    # db/da = -k/(n-k)
    
    # f2(a) = a*(1 - (n/2 - k*a)/(n-k)) = a * (n-k - n/2 + k*a)/(n-k)
    #        = a * (n/2 - k + k*a)/(n-k)
    #        = (k*a^2 + (n/2-k)*a) / (n-k)
    # df2/da = (2k*a + (n/2-k)) / (n-k)
    
    # f3(a) = b*(1-b), where b = (n/2 - k*a)/(n-k)
    # df3/da = (1-2b) * db/da = (1-2b) * (-k/(n-k))
    
    # f4(a) = b = (n/2 - k*a)/(n-k)
    # df4/da = -k/(n-k)
    
    # f1(a) = a*(1-a), df1/da = 1-2a
    
    # Critical points: where two functions are equal and their max is minimized.
    # The optimal is at a point where the active constraints cross.
    
    # Let's enumerate pairs of functions that could be equal at the optimum:
    # For each pair (fi, fj), solve fi(a) = fj(a) and check if it's a valid local min.
    
    candidates = []
    
    # Helper: evaluate all candidate values at a given a
    def eval_at(a):
        if a < 0 or a > 1:
            return None
        b = (n/2.0 - k*a) / (n-k)
        if b < -1e-12 or b > 1.0 + 1e-12:
            return None
        b = max(0.0, min(1.0, b))
        vals = []
        if k >= 2:
            vals.append(('f1', a*(1.0-a)))
        vals.append(('f2', a*(1.0-b)))
        if n-k >= 2:
            vals.append(('f3', b*(1.0-b)))
        vals.append(('f4', b))
        return max(v for _, v in vals), b
    
    # Collect candidate a values:
    # 1. Where f1 = f2: a(1-a) = a(1-b) -> if a>0: 1-a = 1-b -> a=b -> a = (n/2 - k*a)/(n-k) -> a(n-k) = n/2 - k*a -> a*n = n/2 -> a = 1/2
    if k >= 2:
        candidates.append(0.5)
    
    # 2. Where f1 = f3: a(1-a) = b(1-b)
    #    b = (n/2 - k*a)/(n-k)
    #    This is a quadratic in a. Let's solve numerically.
    if k >= 2 and n-k >= 2:
        # a(1-a) = b(1-b) where b = (n/2 - k*a)/(n-k)
        # Let's use numpy roots
        # a - a^2 = b - b^2
        # a - a^2 - b + b^2 = 0
        # b = (n/2 - k*a)/(n-k)
        # Let c = n-k, d = n/2
        # b = (d - k*a)/c
        # a - a^2 - (d - k*a)/c + ((d - k*a)/c)^2 = 0
        # Multiply by c^2:
        # a*c^2 - a^2*c^2 - (d - k*a)*c + (d - k*a)^2 = 0
        # a*c^2 - a^2*c^2 - d*c + k*a*c + d^2 - 2*d*k*a + k^2*a^2 = 0
        # a^2*(-c^2 + k^2) + a*(c^2 + k*c - 2*d*k) + (-d*c + d^2) = 0
        c = n - k
        d = n / 2.0
        A_coef = -(c**2) + k**2
        B_coef = c**2 + k*c - 2*d*k
        C_coef = -d*c + d**2
        if abs(A_coef) > 1e-15:
            disc = B_coef**2 - 4*A_coef*C_coef
            if disc >= 0:
                sqrt_disc = math.sqrt(disc)
                a1 = (-B_coef + sqrt_disc) / (2*A_coef)
                a2 = (-B_coef - sqrt_disc) / (2*A_coef)
                candidates.extend([a1, a2])
        else:
            if abs(B_coef) > 1e-15:
                candidates.append(-C_coef / B_coef)
    
    # 3. Where f1 = f4: a(1-a) = b = (n/2 - k*a)/(n-k)
    if k >= 2:
        # a(1-a) = (n/2 - k*a)/(n-k)
        # a(n-k) - a^2(n-k) = n/2 - k*a
        # -a^2(n-k) + a(n-k) + k*a - n/2 = 0
        # -a^2(n-k) + a(n-k+k) - n/2 = 0
        # -a^2(n-k) + a*n - n/2 = 0
        # a^2(n-k) - a*n + n/2 = 0
        A_coef = n - k
        B_coef = -n
        C_coef = n / 2.0
        disc = B_coef**2 - 4*A_coef*C_coef
        if disc >= 0:
            sqrt_disc = math.sqrt(disc)
            a1 = (-B_coef + sqrt_disc) / (2*A_coef)
            a2 = (-B_coef - sqrt_disc) / (2*A_coef)
            candidates.extend([a1, a2])
    
    # 4. Where f2 = f3: a(1-b) = b(1-b)
    #    If b != 1: a = 1-b -> a + b = 1
    #    a + (n/2 - k*a)/(n-k) = 1
    #    a(n-k) + n/2 - k*a = n-k
    #    a*n + n/2 = n-k
    #    a = (n-k-n/2)/n = (n/2-k)/n = 1/2 - k/n
    if n-k >= 2:
        candidates.append(1.0/2.0 - k/n)
    
    # 5. Where f2 = f4: a(1-b) = b -> a = b/(1-b) (if b < 1)
    #    a(1-b) = b -> a - a*b = b -> a = b(1+a) -> a = b/(1-a)
    #    a(1-a) = b = (n/2 - k*a)/(n-k)
    #    This is same as f1 = f4! Already covered.
    
    # 6. Where f3 = f4: b(1-b) = b -> if b > 0: 1-b = 1 -> b = 0
    #    b = 0 -> n/2 - k*a = 0 -> a = n/(2k)
    if n-k >= 2:
        candidates.append(n / (2.0 * k))
    
    # 7. Where f2 is minimized: df2/da = 0 -> 2k*a + (n/2-k) = 0 -> a = (k - n/2)/(2k)
    #    This is negative for k < n/2, so likely not in range, but check.
    candidates.append((k - n/2.0) / (2.0*k))
    
    # 8. Where f3 is minimized: df3/da = 0 -> 1-2b = 0 -> b = 1/2
    #    (n/2 - k*a)/(n-k) = 1/2 -> n/2 - k*a = (n-k)/2 -> k*a = n/2 - (n-k)/2 = k/2 -> a = 1/2
    candidates.append(0.5)
    
    # 9. Boundary points
    a_min = max(0.0, (n/2.0 - n + k) / k)
    a_max = min(1.0, n / (2.0 * k))
    candidates.extend([a_min, a_max, a_min + 1e-10, a_max - 1e-10])
    
    # Evaluate all candidates
    best_score = float('inf')
    best_a = None
    
    for a in candidates:
        result = eval_at(a)
        if result is None:
            continue
        s, b = result
        if s < best_score:
            best_score = s
            best_a = a
    
    # Also do a fine grid search to catch any missed critical points
    if a_min < a_max:
        a_grid = np.linspace(a_min + 1e-10, a_max - 1e-10, 1000)
        for a in a_grid:
            result = eval_at(a)
            if result is None:
                continue
            s, b = result
            if s < best_score:
                best_score = s
                best_a = a
    
    # Refine around best_a with very fine grid
    if best_a is not None and a_min < a_max:
        step = (a_max - a_min) / 1000.0
        a_refine = np.linspace(max(a_min, best_a - 3*step), min(a_max, best_a + 3*step), 500)
        for a in a_refine:
            result = eval_at(a)
            if result is None:
                continue
            s, b = result
            if s < best_score:
                best_score = s
                best_a = a
    
    if best_a is None:
        return None, float('inf')
    
    b = (n/2.0 - k*best_a) / (n-k)
    b = max(0.0, min(1.0, b))
    
    return best_a, best_score


def construct():
    deadline = time.time() + 100.0  # safety margin before 120s kill

    best_h = None
    best_score = float('inf')

    # Try various splits k (number of elements in first block)
    for k in range(1, N):
        # Constraint: k*a + (N-k)*b = N/2
        # b = (N/2 - k*a) / (N-k)
        
        # Bounds on a:
        a_min = max(0.0, (N / 2.0 - N + k) / k)
        a_max = min(1.0, N / (2.0 * k))
        
        if a_min >= a_max:
            continue
        
        # Use analytical critical point analysis
        a_opt, s_exact = _find_optimal_2block(k)
        
        if a_opt is not None and s_exact < best_score:
            best_score = s_exact
            b_opt = (N / 2.0 - k * a_opt) / (N - k)
            b_opt = np.clip(b_opt, 0.0, 1.0)
            h = np.zeros(N)
            h[:k] = a_opt
            h[k:] = b_opt
            # Fix sum precisely
            s = h.sum()
            diff = N / 2.0 - s
            if abs(diff) > 1e-15:
                h[k - 1] += diff
                h = np.clip(h, 0.0, 1.0)
                s2 = h.sum()
                diff2 = N / 2.0 - s2
                if abs(diff2) > 1e-15:
                    h[k - 1] += diff2
                    h = np.clip(h, 0.0, 1.0)
            best_h = h.copy()
        
        if time.time() > deadline:
            break

    if best_h is None:
        # Fallback
        A = (3.0 - math.sqrt(5.0)) / 2.0
        B = 1.0 - A
        h = np.zeros(N)
        h[:N//2] = A
        h[N//2:] = B
        best_h = h

    return best_h
